"""Mode flux : transactions, événements et déclarations envoyés en continu dans Kafka.

Le producteur fonctionne par « ticks » d'une seconde : à chaque tick il génère le
volume correspondant au débit cible, met en file les opérations futures (une attaque
dure plusieurs minutes) et publie celles dont l'heure est arrivée.

Pilotage pendant la démo (API HTTP, jeton obligatoire pour les modifications) :
    GET  /health                    état courant
    GET  /metrics                   métriques Prometheus
    POST /control/rate?value=500    change le débit cible
    POST /control/burst?enabled=1   active un pic de charge
    POST /control/drift?enabled=1   active le nouveau scénario de fraude (dérive)
"""

from __future__ import annotations

import json
import logging
import os
import signal
import threading
import time
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np
import orjson
import pandas as pd
from confluent_kafka import KafkaError, Message, Producer
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

from . import columns as col
from .behaviour import background_events, legit_transactions, linked_events
from .config import SimConfig
from .entities import cached_entities
from .formatting import dumps_lines, events_jsonl, transactions_frame
from .fraud import inject_fraud
from .identifiers import random_hex_ids
from .support import CLAIM_LABELS

logger = logging.getLogger(__name__)

TOPIC_TRANSACTIONS = "transactions.raw"
TOPIC_EVENTS = "app-events.raw"
TOPIC_REPORTS = "fraud-reports.raw"

RECORDS = Counter("sim_records_produced_total", "Messages acquittés par Kafka", ["topic"])
ERRORS = Counter("sim_delivery_errors_total", "Messages en échec de livraison", ["topic"])
INCIDENTS = Counter("sim_fraud_incidents_total", "Incidents de fraude injectés", ["fraud_type"])
TARGET_RATE = Gauge("sim_target_rate_tps", "Débit cible (transactions par seconde)")
ACTUAL_RATE = Gauge("sim_actual_rate_tps", "Débit réellement envoyé (transactions par seconde)")
DRIFT = Gauge("sim_drift_enabled", "1 si le scénario de dérive est actif")
BURST = Gauge("sim_burst_enabled", "1 si un pic de charge est en cours")
PENDING = Gauge("sim_pending_records", "Opérations futures en attente de publication")
TICK_LAG = Gauge("sim_tick_lag_seconds", "Retard du générateur sur le temps réel")
DELIVERY = Histogram(
    "sim_delivery_latency_seconds",
    "Délai entre envoi et acquittement Kafka",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5),
)


@dataclass
class StreamSettings:
    """Paramètres du mode flux."""

    bootstrap: str = "localhost:9092"
    rate: float = 300.0
    burst_rate: float = 2000.0
    drift: bool = False
    time_scale: float = 1 / 60
    metrics_port: int = 8000
    duration_s: float = 0.0
    tick_s: float = 1.0
    # Délai (secondes réelles) avant qu'une fraude soit déclarée par le client.
    report_delay_s: float = 300.0
    seed: int | None = None


class ControlState:
    """État modifiable à chaud, partagé entre la boucle de production et l'API HTTP."""

    def __init__(self, settings: StreamSettings) -> None:
        """Initialise l'état à partir des paramètres de lancement."""
        self._lock = threading.Lock()
        self.rate = settings.rate
        self.burst_rate = settings.burst_rate
        self.burst = False
        self.drift = settings.drift
        self.stop = threading.Event()

    def current_rate(self) -> float:
        """Débit à appliquer au prochain tick."""
        with self._lock:
            return self.burst_rate if self.burst else self.rate

    def update(self, name: str, value: str) -> dict[str, object]:
        """Applique une commande de pilotage ; lève ValueError si elle est invalide."""
        with self._lock:
            if name == "rate":
                rate = float(value)
                if not 0 < rate <= 20_000:
                    raise ValueError("le débit doit être compris entre 0 et 20 000 tx/s")
                self.rate = rate
            elif name in ("burst", "drift"):
                setattr(self, name, value.lower() in ("1", "true", "yes", "on"))
            else:
                raise ValueError(f"commande inconnue : {name}")
            logger.info("Pilotage : %s = %s", name, value)
            return self.snapshot()

    def snapshot(self) -> dict[str, object]:
        """État courant, pour /health."""
        return {
            "rate": self.rate,
            "burst_rate": self.burst_rate,
            "burst": self.burst,
            "drift": self.drift,
        }


def _handler_factory(state: ControlState, token: str | None) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def _reply(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 (nom imposé par http.server)
            path = urlparse(self.path).path
            if path == "/metrics":
                self._reply(HTTPStatus.OK, generate_latest(), CONTENT_TYPE_LATEST)
            elif path == "/health":
                self._reply(
                    HTTPStatus.OK,
                    orjson.dumps({"status": "ok", **state.snapshot()}),
                    "application/json",
                )
            else:
                self._reply(HTTPStatus.NOT_FOUND, b"{}", "application/json")

        def do_POST(self) -> None:  # noqa: N802
            # Les commandes modifient le comportement : jeton obligatoire s'il est configuré.
            if token and self.headers.get("X-Control-Token") != token:
                self._reply(
                    HTTPStatus.UNAUTHORIZED, b'{"error":"jeton invalide"}', "application/json"
                )
                return
            url = urlparse(self.path)
            if not url.path.startswith("/control/"):
                self._reply(HTTPStatus.NOT_FOUND, b"{}", "application/json")
                return
            params = parse_qs(url.query)
            value = (params.get("value") or params.get("enabled") or [""])[0]
            try:
                snapshot = state.update(url.path.removeprefix("/control/"), value)
            except ValueError as exc:
                self._reply(
                    HTTPStatus.BAD_REQUEST, orjson.dumps({"error": str(exc)}), "application/json"
                )
                return
            self._reply(HTTPStatus.OK, orjson.dumps(snapshot), "application/json")

        def log_message(self, fmt: str, *args: object) -> None:
            logger.debug("HTTP " + fmt, *args)

    return Handler


def _kafka_config(settings: StreamSettings) -> dict[str, object]:
    """Configuration du producteur : livraison fiable et idempotente.

    Des paramètres supplémentaires (TLS, SASL...) peuvent être passés par variables
    d'environnement SIM_KAFKA_<PARAMETRE>, ex. SIM_KAFKA_SECURITY_PROTOCOL=SASL_SSL.
    """
    config: dict[str, object] = {
        "bootstrap.servers": settings.bootstrap,
        "client.id": "ondine-simulator",
        "acks": "all",  # attendre les répliques synchronisées
        "enable.idempotence": True,  # pas de doublon en cas de nouvel essai
        "compression.type": "lz4",
        "linger.ms": 20,  # regroupe les messages : meilleur débit
        "delivery.timeout.ms": 120_000,
    }
    for key, value in os.environ.items():
        if key.startswith("SIM_KAFKA_"):
            config[key.removeprefix("SIM_KAFKA_").lower().replace("_", ".")] = value
    return config


class StreamProducer:
    """Génère et publie les messages tick après tick."""

    def __init__(
        self, cfg: SimConfig, settings: StreamSettings, state: ControlState, producer: Producer
    ) -> None:
        """Prépare le générateur (référentiels chargés une seule fois)."""
        self.cfg, self.settings, self.state, self.producer = cfg, settings, state, producer
        self.ent = cached_entities(cfg)
        self.rng = np.random.default_rng(settings.seed)
        self.pending_tx = col.empty(col.TX_COLUMNS, 0)
        self.pending_ev = col.empty(col.EVENT_COLUMNS, 0)
        self.pending_reports: list[tuple[float, bytes, str]] = []
        self.incident_counter = 0
        # Probabilité qu'une transaction « appartienne » à une nouvelle victime.
        self.victims_per_tx = cfg.fraud_victim_rate / cfg.avg_tx_per_client_month

    def tick(self, now_ms: int) -> int:
        """Génère un tick et publie ce qui est dû. Retourne le nombre de transactions publiées."""
        rng, ent, tick_ms = self.rng, self.ent, int(self.settings.tick_s * 1000)
        rate = self.state.current_rate()
        n_tx = rng.poisson(rate * self.settings.tick_s)

        # Clients tirés proportionnellement à leur activité.
        clients = np.searchsorted(ent.client_activity_cdf, rng.random(n_tx))
        clients = np.minimum(clients, ent.n_clients - 1)
        legit = legit_transactions(
            ent, rng, clients, now_ms + rng.integers(0, tick_ms, n_tx), schema_v2=True
        )
        n_victims = rng.poisson(n_tx * self.victims_per_tx)
        victims = rng.integers(0, ent.n_clients, n_victims)
        fraud = inject_fraud(
            ent,
            rng,
            victims,
            now_ms,
            now_ms + tick_ms,
            self.state.drift,
            incident_base=self.incident_counter,
            time_scale=self.settings.time_scale,
        )
        self.incident_counter += n_victims
        n_bg = rng.poisson(n_tx * 0.4)
        background = background_events(
            ent,
            rng,
            np.minimum(
                np.searchsorted(ent.client_activity_cdf, rng.random(n_bg)), ent.n_clients - 1
            ),
            now_ms + rng.integers(0, tick_ms, n_bg),
        )

        self.pending_tx = col.concat(col.TX_COLUMNS, [self.pending_tx, legit, fraud.tx])
        self.pending_ev = col.concat(
            col.EVENT_COLUMNS,
            [self.pending_ev, linked_events(ent, rng, legit), fraud.events, background],
        )

        due_tx = self._pop_due("tx", now_ms + tick_ms)
        due_ev = self._pop_due("ev", now_ms + tick_ms)
        published = self._publish_transactions(due_tx)
        self._publish_events(due_ev)
        self._publish_reports(time.time())
        PENDING.set(len(self.pending_tx["ts"]) + len(self.pending_ev["ts"]))
        for code in np.unique(fraud.tx["fraud_type"]):
            incidents = np.unique(fraud.tx["incident"][fraud.tx["fraud_type"] == code])
            INCIDENTS.labels(fraud_type=str(col.FRAUD_TYPES[code])).inc(len(incidents))
        return published

    def _pop_due(self, kind: str, limit_ms: int) -> dict[str, np.ndarray]:
        pending = self.pending_tx if kind == "tx" else self.pending_ev
        due = pending["ts"] < limit_ms
        ready, rest = col.sort_by_ts(col.take(pending, due)), col.take(pending, ~due)
        if kind == "tx":
            self.pending_tx = rest
        else:
            self.pending_ev = rest
        return ready

    def _produce(self, topic: str, key: str, value: bytes) -> None:
        sent_at = time.perf_counter()

        def on_delivery(err: KafkaError | None, _msg: Message) -> None:
            if err is not None:
                ERRORS.labels(topic=topic).inc()
                logger.error("Livraison impossible sur %s : %s", topic, err)
            else:
                RECORDS.labels(topic=topic).inc()
                DELIVERY.observe(time.perf_counter() - sent_at)

        while True:
            try:
                self.producer.produce(topic, key=key.encode(), value=value, on_delivery=on_delivery)
                return
            except BufferError:
                # File locale pleine (Kafka lent ou indisponible) : on attend (contre-pression).
                self.producer.poll(0.5)

    def _publish_transactions(self, tx: dict[str, np.ndarray]) -> int:
        n = len(tx["ts"])
        if n == 0:
            return 0
        frame, truth = transactions_frame(self.ent, self.rng, tx, schema_v2=True)
        keys = frame["card_id"].fillna(frame["client_id"]).tolist()
        for key, value in zip(keys, dumps_lines(frame), strict=True):
            self._produce(TOPIC_TRANSACTIONS, key, value)
        self._schedule_reports(truth)
        self.producer.poll(0)
        return n

    def _publish_events(self, ev: dict[str, np.ndarray]) -> None:
        if len(ev["ts"]) == 0:
            return
        lines, _ = events_jsonl(self.ent, self.rng, ev, app_v5_share=0.95)
        for key, value in zip(ev["client"].tolist(), lines, strict=True):
            self._produce(TOPIC_EVENTS, f"C{key:08d}", value)
        self.producer.poll(0)

    def _schedule_reports(self, truth: pd.DataFrame) -> None:
        """Programme la déclaration de 90 % des fraudes publiées, après un délai."""
        if truth.empty:
            return
        reported = self.rng.random(len(truth)) < self.cfg.fraud_report_rate
        delays = self.rng.uniform(0.5, 1.5, len(truth)) * self.settings.report_delay_s
        report_ids = random_hex_ids(self.rng, len(truth), 8)
        for row, is_reported, delay, rid in zip(
            truth.itertuples(), reported, delays, report_ids, strict=True
        ):
            if not is_reported:
                continue
            due = time.time() + delay
            record = {
                "report_id": f"R{rid}",
                "case_id": row.incident_id.replace("INC", "CASE"),
                "transaction_id": row.transaction_id,
                "client_id": row.client_id,
                "reported_at": pd.Timestamp(due, unit="s").strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                "report_channel": "app_claim",
                "claim_category": CLAIM_LABELS[row.fraud_type],
                "investigation_status": "confirmed",
            }
            self.pending_reports.append((due, orjson.dumps(record), row.client_id))

    def _publish_reports(self, now_s: float) -> None:
        due = [r for r in self.pending_reports if r[0] <= now_s]
        self.pending_reports = [r for r in self.pending_reports if r[0] > now_s]
        for _, value, key in due:
            self._produce(TOPIC_REPORTS, key, value)


def run_stream(cfg: SimConfig, settings: StreamSettings) -> None:
    """Lance la production en continu jusqu'à interruption (ou fin de `duration_s`)."""
    state = ControlState(settings)
    token = os.environ.get("SIM_CONTROL_TOKEN")
    if not token:
        logger.warning("SIM_CONTROL_TOKEN non défini : l'API de pilotage n'est pas protégée")
    server = ThreadingHTTPServer(
        ("0.0.0.0", settings.metrics_port),  # noqa: S104 (conteneur)
        _handler_factory(state, token),
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    logger.info("Métriques et pilotage sur le port %d", settings.metrics_port)

    def _stop(signum: int, _frame: object) -> None:
        logger.info("Signal %d reçu : arrêt propre", signum)
        state.stop.set()

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    producer = Producer(_kafka_config(settings))
    generator = StreamProducer(cfg, settings, state, producer)
    started = time.time()
    next_tick = started
    logger.info("Production vers %s : %s", settings.bootstrap, json.dumps(state.snapshot()))
    try:
        while not state.stop.is_set():
            if settings.duration_s and time.time() - started >= settings.duration_s:
                break
            tick_started = time.time()
            published = generator.tick(int(tick_started * 1000))
            TARGET_RATE.set(state.current_rate())
            ACTUAL_RATE.set(published / settings.tick_s)
            DRIFT.set(int(state.drift))
            BURST.set(int(state.burst))
            next_tick += settings.tick_s
            lag = time.time() - next_tick
            TICK_LAG.set(max(lag, 0.0))
            if lag > settings.tick_s:
                logger.warning("Générateur en retard de %.1f s : débit cible trop élevé", lag)
                next_tick = time.time()
            state.stop.wait(max(0.0, next_tick - time.time()))
    finally:
        remaining = producer.flush(30)
        if remaining:
            logger.error("%d message(s) non livrés à l'arrêt", remaining)
        server.shutdown()
        logger.info("Producteur arrêté")
