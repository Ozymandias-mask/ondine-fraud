"""Ingestion en flux : consommateurs Kafka qui écrivent les messages en bronze.

Garanties :

- **aucune perte** : l'offset d'une partition n'est validé (commit) qu'après l'écriture du
  lot en bronze, l'envoi des messages invalides dans le topic d'erreurs et l'inscription du
  lot au registre. Un plantage entre les deux fait relire le lot au redémarrage ;
- **doublons maîtrisés** : un lot relu peut être écrit deux fois ; la couche silver
  dédoublonne sur l'identifiant métier (transaction_id, event_id, report_id) ;
- **passage à l'échelle** : les consommateurs d'un même groupe se répartissent les
  partitions (6 par topic) ; ajouter une réplique suffit à augmenter le débit ;
- **arrêt propre** : sur SIGTERM ou retrait de partitions, les lots en cours sont écrits et
  validés avant de rendre la main.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import psycopg
from prometheus_client import Counter, Gauge, Histogram, start_http_server

from ..lake import connect, with_retries
from ..settings import Settings
from .bronze_writer import DATASETS_BY_TOPIC, BatchResult, Message, write_batch

logger = logging.getLogger(__name__)

CONSUMED = Counter("stream_messages_consumed_total", "Messages lus dans Kafka", ["topic"])
WRITTEN = Counter("stream_rows_written_total", "Lignes écrites en bronze", ["topic"])
INVALID = Counter(
    "stream_messages_invalid_total", "Messages envoyés au topic d'erreurs", ["topic", "reason"]
)
FAILURES = Counter("stream_flush_failures_total", "Écritures de lot en échec définitif")
FLUSH_DURATION = Histogram(
    "stream_flush_duration_seconds",
    "Durée d'un cycle écriture + topic d'erreurs + registre + commit",
    buckets=(0.5, 1, 2.5, 5, 10, 20, 40, 80),
)
LATENCY = Histogram(
    "stream_ingestion_latency_seconds",
    "Délai entre l'arrivée d'un message dans Kafka et son commit après écriture en bronze",
    buckets=(5, 15, 30, 60, 90, 120, 180, 300, 600),
)
LAG = Gauge("stream_consumer_lag_messages", "Messages en attente", ["topic", "partition"])
ASSIGNED = Gauge("stream_assigned_partitions", "Partitions attribuées à ce consommateur")
LAST_COMMIT = Gauge("stream_last_commit_timestamp_seconds", "Horodatage du dernier commit")


@dataclass(frozen=True)
class StreamSettings:
    """Paramètres propres à l'ingestion en flux."""

    bootstrap: str
    group_id: str = "bronze-ingestion"
    consumer_id: str = "stream-ingest"
    flush_interval_s: float = 60.0
    max_batch: int = 50_000
    metrics_port: int = 8001
    topics: tuple[str, ...] = tuple(DATASETS_BY_TOPIC)

    @classmethod
    def from_env(cls) -> StreamSettings:
        """Lit KAFKA_BOOTSTRAP (obligatoire) et les réglages STREAM_* facultatifs."""
        bootstrap = os.environ.get("KAFKA_BOOTSTRAP", "")
        if not bootstrap:
            from ..settings import ConfigurationError

            raise ConfigurationError("Variable d'environnement manquante : KAFKA_BOOTSTRAP")
        return cls(
            bootstrap=bootstrap,
            group_id=os.environ.get("STREAM_GROUP_ID", "bronze-ingestion"),
            consumer_id=os.environ.get("HOSTNAME", "stream-ingest"),
            flush_interval_s=float(os.environ.get("STREAM_FLUSH_INTERVAL_S", "60")),
            max_batch=int(os.environ.get("STREAM_MAX_BATCH", "50000")),
            metrics_port=int(os.environ.get("STREAM_METRICS_PORT", "8001")),
        )


def dlq_topic(topic: str) -> str:
    """Topic d'erreurs (dead letter queue) associé à un topic source."""
    return f"{topic}.dlq"


def kafka_config(stream: StreamSettings) -> dict[str, Any]:
    """Configuration du consommateur : commit manuel, répartition coopérative."""
    return {
        "bootstrap.servers": stream.bootstrap,
        "group.id": stream.group_id,
        "client.id": stream.consumer_id,
        "enable.auto.commit": False,  # commit uniquement après écriture en bronze
        "auto.offset.reset": "earliest",  # nouveau groupe : rien n'est ignoré
        # Rééquilibrage incrémental : l'arrivée d'une réplique ne stoppe pas les autres.
        "partition.assignment.strategy": "cooperative-sticky",
        "max.poll.interval.ms": 600_000,
        "session.timeout.ms": 45_000,
    }


def producer_config(stream: StreamSettings) -> dict[str, Any]:
    """Producteur du topic d'erreurs : même fiabilité que le simulateur."""
    return {
        "bootstrap.servers": stream.bootstrap,
        "client.id": f"{stream.consumer_id}-dlq",
        "acks": "all",
        "enable.idempotence": True,
    }


class StreamIngestor:
    """Boucle lecture → lot → bronze → topic d'erreurs → registre → commit."""

    def __init__(
        self, settings: Settings, stream: StreamSettings, consumer: Any, producer: Any
    ) -> None:
        """Les clients Kafka sont injectés : les tests utilisent des doublures."""
        self.settings, self.stream = settings, stream
        self.consumer, self.producer = consumer, producer
        self.con = connect(settings)
        self.buffers: dict[tuple[str, int], list[Message]] = defaultdict(list)
        self.kafka_ts: dict[tuple[str, int], float] = {}
        self.buffered = 0
        self.last_flush = time.monotonic()
        self.assigned: set[tuple[str, int]] = set()
        self.stopping = False
        self.failed = False

    # ------------------------------------------------------------------ rééquilibrage
    def on_assign(self, _consumer: Any, partitions: list[Any]) -> None:
        """Partitions reçues (appelé par le client Kafka)."""
        self.assigned.update((p.topic, p.partition) for p in partitions)
        ASSIGNED.set(len(self.assigned))
        logger.info("Partitions attribuées : %s", sorted(self.assigned))

    def on_revoke(self, _consumer: Any, partitions: list[Any]) -> None:
        """Partitions retirées : écrire et valider leurs lots avant de les rendre."""
        revoked = {(p.topic, p.partition) for p in partitions}
        if not self.failed:  # après un échec, on s'arrête sans rien valider
            self.flush(only=revoked)
        self._forget(revoked)

    def on_lost(self, _consumer: Any, partitions: list[Any]) -> None:
        """Partitions perdues (session expirée) : un autre consommateur les relira."""
        lost = {(p.topic, p.partition) for p in partitions}
        for key in lost:
            self.buffered -= len(self.buffers.pop(key, []))
        self._forget(lost)

    def _forget(self, keys: set[tuple[str, int]]) -> None:
        self.assigned -= keys
        for topic, partition in keys:
            self.kafka_ts.pop((topic, partition), None)
            try:
                LAG.remove(topic, str(partition))
            except KeyError:
                pass
        ASSIGNED.set(len(self.assigned))

    # ------------------------------------------------------------------ boucle
    def run(self) -> None:
        """Consomme jusqu'à SIGTERM, puis écrit les derniers lots et ferme proprement."""
        self.consumer.subscribe(
            list(self.stream.topics),
            on_assign=self.on_assign,
            on_revoke=self.on_revoke,
            on_lost=self.on_lost,
        )
        try:
            while not self.stopping:
                self.poll_once()
            self.flush()
        finally:
            self.consumer.close()
            self.producer.flush(30)
            self.con.close()
            logger.info("Consommateur arrêté proprement")

    def poll_once(self, timeout_s: float = 1.0) -> None:
        """Lit un paquet de messages et déclenche l'écriture si le lot est prêt."""
        for msg in self.consumer.consume(num_messages=5_000, timeout=timeout_s):
            if msg.error():
                logger.warning("Erreur Kafka : %s", msg.error())
                continue
            key = (msg.topic(), msg.partition())
            self.buffers[key].append(Message(key[0], key[1], msg.offset(), msg.value()))
            _, ts_ms = msg.timestamp()
            self.kafka_ts.setdefault(key, ts_ms / 1000 if ts_ms and ts_ms > 0 else time.time())
            self.buffered += 1
            CONSUMED.labels(topic=key[0]).inc()
        due = time.monotonic() - self.last_flush >= self.stream.flush_interval_s
        if self.buffered >= self.stream.max_batch or (due and self.buffered):
            self.flush()
        elif due:
            self.last_flush = time.monotonic()
            self.update_lag()

    def flush(self, only: set[tuple[str, int]] | None = None) -> None:
        """Écrit les lots, puis valide leurs offsets. Toute erreur arrête le consommateur."""
        keys = [k for k, msgs in self.buffers.items() if msgs and (only is None or k in only)]
        if not keys:
            return
        started = time.perf_counter()
        try:
            results = {
                key: with_retries(
                    lambda key=key: write_batch(
                        self.con, self.settings, self.stream.consumer_id, self.buffers[key]
                    ),
                    attempts=3,
                    wait_s=10,
                )
                for key in keys
            }
            self._send_to_dlq(results)
            self._record(results)
            self._commit(keys)
        except Exception:
            self.failed = True
            FAILURES.inc()
            logger.exception("Lot non écrit : arrêt sans commit, reprise au dernier offset validé")
            raise
        committed_at = time.time()
        for key in keys:
            result, topic = results[key], key[0]
            WRITTEN.labels(topic=topic).inc(result.rows_valid)
            LATENCY.observe(max(0.0, committed_at - self.kafka_ts.pop(key, committed_at)))
            self.buffered -= len(self.buffers.pop(key))
        LAST_COMMIT.set(committed_at)
        FLUSH_DURATION.observe(time.perf_counter() - started)
        self.last_flush = time.monotonic()
        self.update_lag()
        logger.info(
            "%d lot(s) validé(s) : %d lignes en bronze, %d invalides, %.1f s",
            len(keys),
            sum(r.rows_valid for r in results.values()),
            sum(len(r.invalid) for r in results.values()),
            time.perf_counter() - started,
        )

    def _send_to_dlq(self, results: dict[tuple[str, int], BatchResult]) -> None:
        invalid = [item for r in results.values() for item in r.invalid]
        if not invalid:
            return
        for message, reason in invalid:
            self.producer.produce(
                dlq_topic(message.topic),
                value=message.value,
                headers={
                    "dq_reason": reason,
                    "source_partition": str(message.partition),
                    "source_offset": str(message.offset),
                    "consumer_id": self.stream.consumer_id,
                },
            )
            INVALID.labels(topic=message.topic, reason=reason.split(":")[0]).inc()
        remaining = self.producer.flush(30)
        if remaining:
            raise RuntimeError(f"{remaining} message(s) non livré(s) au topic d'erreurs")

    def _record(self, results: dict[tuple[str, int], BatchResult]) -> None:
        with psycopg.connect(self.settings.pg_dsn, connect_timeout=10) as pg, pg.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO ops.stream_batches (consumer_id, topic, kafka_partition, first_offset,
                    last_offset, rows_valid, rows_invalid, target_uris, min_event_time,
                    max_event_time, duration_ms)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (topic, kafka_partition, first_offset, last_offset) DO NOTHING
                """,
                [
                    (
                        self.stream.consumer_id,
                        topic,
                        partition,
                        self.buffers[(topic, partition)][0].offset,
                        self.buffers[(topic, partition)][-1].offset,
                        r.rows_valid,
                        len(r.invalid),
                        json.dumps(r.uris),
                        r.min_event_time,
                        r.max_event_time,
                        r.duration_ms,
                    )
                    for (topic, partition), r in results.items()
                ],
            )

    def _commit(self, keys: list[tuple[str, int]]) -> None:
        from confluent_kafka import TopicPartition

        offsets = [TopicPartition(t, p, self.buffers[(t, p)][-1].offset + 1) for t, p in keys]
        self.consumer.commit(offsets=offsets, asynchronous=False)

    def update_lag(self) -> None:
        """Retard par partition : dernier offset publié − prochain offset à lire."""
        from confluent_kafka import TopicPartition

        partitions = [TopicPartition(t, p) for t, p in sorted(self.assigned)]
        if not partitions:
            return
        try:
            positions = self.consumer.position(partitions)
            for tp in positions:
                _, high = self.consumer.get_watermark_offsets(tp, timeout=5, cached=False)
                if tp.offset >= 0 and high >= 0:
                    LAG.labels(topic=tp.topic, partition=str(tp.partition)).set(high - tp.offset)
        except Exception as exc:  # la supervision ne doit pas arrêter l'ingestion
            logger.warning("Retard non mesuré : %s", exc)

    def request_stop(self, signum: int, _frame: object) -> None:
        """Gestionnaire de signal : termine la boucle après le paquet en cours."""
        logger.info("Signal %s reçu : écriture des lots en cours puis arrêt", signum)
        self.stopping = True


def main(settings: Settings) -> None:
    """Point d'entrée du service stream-ingest (`ondine-pipelines stream`)."""
    from confluent_kafka import Consumer, Producer

    stream = StreamSettings.from_env()
    start_http_server(stream.metrics_port)
    ingestor = StreamIngestor(
        settings, stream, Consumer(kafka_config(stream)), Producer(producer_config(stream))
    )
    signal.signal(signal.SIGTERM, ingestor.request_stop)
    signal.signal(signal.SIGINT, ingestor.request_stop)
    logger.info(
        "Consommateur %s (groupe %s) : lots toutes les %.0f s ou %d messages",
        stream.consumer_id,
        stream.group_id,
        stream.flush_interval_s,
        stream.max_batch,
    )
    ingestor.run()
