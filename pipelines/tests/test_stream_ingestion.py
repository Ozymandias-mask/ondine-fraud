"""Ingestion en flux : doublures des clients Kafka, vrai data lake local et vraie base."""

import json
from dataclasses import replace
from datetime import datetime

import duckdb
import psycopg
import pytest
from ondine_pipelines.ingest.landing import ingest_dataset
from ondine_pipelines.silver.activity import build_month
from ondine_pipelines.stream import consumer as consumer_module
from ondine_pipelines.stream.consumer import StreamIngestor, StreamSettings

START_MS = int(datetime(2025, 2, 10, 12).timestamp() * 1000)


class FakeKafkaMessage:
    def __init__(self, topic, partition, offset, value, ts_ms=START_MS):
        self._t, self._p, self._o, self._v, self._ts = topic, partition, offset, value, ts_ms

    def error(self):
        return None

    def topic(self):
        return self._t

    def partition(self):
        return self._p

    def offset(self):
        return self._o

    def value(self):
        return self._v

    def timestamp(self):
        return (1, self._ts)


class FakeConsumer:
    """Rend les messages par paquets et enregistre les commits."""

    def __init__(self, messages, chunk=10_000):
        self.pending, self.chunk, self.commits, self.closed = list(messages), chunk, [], False

    def subscribe(self, topics, on_assign, on_revoke, on_lost):
        self.callbacks = (on_assign, on_revoke, on_lost)

    def consume(self, num_messages, timeout):
        batch, self.pending = self.pending[: self.chunk], self.pending[self.chunk :]
        return batch

    def commit(self, offsets, asynchronous):
        assert asynchronous is False
        self.commits.append({(tp.topic, tp.partition): tp.offset for tp in offsets})

    def position(self, partitions):
        return partitions

    def get_watermark_offsets(self, tp, timeout, cached):
        return 0, 0

    def close(self):
        self.closed = True


class FakeProducer:
    def __init__(self):
        self.sent = []

    def produce(self, topic, value, headers):
        self.sent.append((topic, value, headers))

    def flush(self, timeout):
        return 0


@pytest.fixture(scope="module")
def simulated_messages():
    """Messages réellement produits par le simulateur en mode flux (20 s à 400 tx/s)."""
    from ondine_sim.config import get_config
    from ondine_sim.stream import ControlState, StreamProducer
    from ondine_sim.stream import StreamSettings as SimStreamSettings

    class Capture:
        def __init__(self):
            self.messages = []

        def produce(self, topic, key, value, on_delivery):
            self.messages.append((topic, key, value))

        def poll(self, _timeout):
            return 0

    settings = SimStreamSettings(rate=400, report_delay_s=0, seed=7)
    capture = Capture()
    generator = StreamProducer(get_config("small"), settings, ControlState(settings), capture)
    for i in range(20):
        generator.tick(START_MS + i * 1000)
    offsets, out = {}, []
    for topic, key, value in capture.messages:
        partition = hash(key) % (3 if topic == "fraud-reports.raw" else 6)
        offset = offsets.get((topic, partition), 0)
        offsets[(topic, partition)] = offset + 1
        out.append(FakeKafkaMessage(topic, partition, offset, value))
    return out


def _ingestor(settings, messages, chunk=10_000, **stream_overrides):
    stream = replace(
        StreamSettings(bootstrap="fake:9092", consumer_id="test-consumer", flush_interval_s=0),
        **stream_overrides,
    )
    return StreamIngestor(settings, stream, FakeConsumer(messages, chunk), FakeProducer())


def _drain(ingestor):
    while ingestor.consumer.pending:
        ingestor.poll_once(timeout_s=0)
    ingestor.flush()


def test_stream_bronze_has_the_batch_schema(base_settings, fresh_settings, simulated_messages):
    for dataset in ("transactions", "app_events", "fraud_reports"):
        ingest_dataset(base_settings, dataset, limit=1)  # sans effet si déjà ingéré
    ingestor = _ingestor(fresh_settings, simulated_messages)
    _drain(ingestor)
    for dataset in ("transactions", "app_events", "fraud_reports"):

        def schema(settings, glob, dataset=dataset):
            base = settings.uri("bronze", dataset)
            return [
                (c[0], c[1])
                for c in duckdb.sql(
                    f"DESCRIBE SELECT * FROM read_parquet('{base}/**/{glob}', "
                    "hive_partitioning = false)"
                ).fetchall()
            ]

        batch = schema(base_settings, "*.parquet")
        assert schema(fresh_settings, "stream_*.parquet") == batch, dataset
    # Tous les messages valides ont été écrits, rien n'est parti en erreur.
    tx_sent = sum(m.topic() == "transactions.raw" for m in simulated_messages)
    written = duckdb.sql(
        f"SELECT count(*) FROM read_parquet('{fresh_settings.uri('bronze', 'transactions')}"
        "/**/stream_*.parquet')"
    ).fetchone()[0]
    assert written == tx_sent and not ingestor.producer.sent


def test_invalid_messages_go_to_dlq_and_offsets_are_committed_after_write(fresh_settings):
    good = json.dumps(
        {"transaction_id": "T1", "event_time": "2025-02-10T12:00:00Z", "amount": "12.5"}
    ).encode()
    messages = [
        FakeKafkaMessage("transactions.raw", 0, 40, good),
        FakeKafkaMessage("transactions.raw", 0, 41, b"{not json"),
        FakeKafkaMessage("transactions.raw", 0, 42, b'{"event_time": "2025-02-10T12:00:00Z"}'),
        FakeKafkaMessage(
            "transactions.raw", 0, 43, b'{"transaction_id": "T4", "event_time": "hier"}'
        ),
        FakeKafkaMessage(
            "transactions.raw",
            0,
            44,
            b'{"transaction_id": "T5", "event_time": "2025-02-10T12:00:00Z", "amount": "abc"}',
        ),
    ]
    ingestor = _ingestor(fresh_settings, messages)
    _drain(ingestor)

    reasons = {h["source_offset"]: h["dq_reason"] for _, _, h in ingestor.producer.sent}
    assert {t for t, _, _ in ingestor.producer.sent} == {"transactions.raw.dlq"}
    assert reasons == {
        "41": "invalid_json",
        "42": "missing_transaction_id",
        "43": "invalid_event_time",
        "44": "invalid_type:amount",
    }
    # Offset validé = dernier offset du lot + 1, une fois le lot écrit et enregistré.
    assert ingestor.consumer.commits == [{("transactions.raw", 0): 45}]
    with psycopg.connect(fresh_settings.pg_dsn) as con:
        row = con.execute(
            "SELECT rows_valid, rows_invalid, target_uris FROM ops.stream_batches"
            " WHERE consumer_id = 'test-consumer' AND first_offset = 40"
        ).fetchone()
    assert row[0] == 1 and row[1] == 4
    assert row[2][0].endswith("year=2025/month=02/stream_p00_000000000040-000000000044.parquet")


def test_nothing_is_committed_when_the_write_fails(fresh_settings, simulated_messages, monkeypatch):
    def broken(*_args, **_kwargs):
        raise RuntimeError("stockage indisponible")

    monkeypatch.setattr(consumer_module, "write_batch", broken)
    ingestor = _ingestor(fresh_settings, simulated_messages[:100], flush_interval_s=3600)
    ingestor.poll_once(timeout_s=0)
    with pytest.raises(RuntimeError):
        ingestor.flush()
    assert ingestor.consumer.commits == []
    assert ingestor.buffered == 100  # rien n'est perdu : le lot sera relu au redémarrage


def test_revoked_partitions_are_flushed_before_being_released(fresh_settings, simulated_messages):
    tx = [m for m in simulated_messages if m.topic() == "transactions.raw"][:500]
    ingestor = _ingestor(fresh_settings, tx, flush_interval_s=3600)
    ingestor.poll_once(timeout_s=0)
    assert not ingestor.consumer.commits

    class TP:
        def __init__(self, topic, partition):
            self.topic, self.partition = topic, partition

    partitions = sorted({(m.topic(), m.partition()) for m in tx})
    ingestor.on_assign(None, [TP(*p) for p in partitions])
    ingestor.on_revoke(None, [TP(*partitions[0])])
    assert list(ingestor.consumer.commits[0]) == [partitions[0]]
    assert partitions[0] not in ingestor.assigned and ingestor.buffered > 0


def test_replayed_batches_are_deduplicated_in_silver(fresh_settings, simulated_messages):
    ingest_dataset(fresh_settings, "transactions")
    tx = [m for m in simulated_messages if m.topic() == "transactions.raw"]
    # Premier passage : écrit mais « plantage » avant commit ; second passage découpé
    # autrement (autres noms de fichiers) : chaque message est présent deux fois en bronze.
    _drain(_ingestor(fresh_settings, tx, max_batch=len(tx) + 1))
    _drain(_ingestor(fresh_settings, tx, chunk=700))
    bronze = fresh_settings.uri("bronze", "transactions")
    copies = duckdb.sql(
        f"SELECT count(*) FROM read_parquet('{bronze}/year=2025/month=02/stream_*.parquet')"
    ).fetchone()[0]
    assert copies == 2 * len(tx)
    build_month(fresh_settings, "transactions", 2025, 2)
    silver = duckdb.sql(
        f"SELECT transaction_id FROM read_parquet("
        f"'{fresh_settings.uri('silver', 'transactions')}/year=2025/month=02/*.parquet')"
    ).df()["transaction_id"]
    stream_ids = {json.loads(m.value())["transaction_id"] for m in tx}
    assert silver.is_unique
    assert stream_ids <= set(silver)
