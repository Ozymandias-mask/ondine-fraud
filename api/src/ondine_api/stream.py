"""Alimentation du magasin en ligne depuis le flux Kafka.

Un consommateur dédié, dans son propre groupe (`online-features`), lit les mêmes topics que
l'ingestion du data lake. Les deux usages sont ainsi **indépendants** : une panne du magasin
en ligne n'arrête pas l'alimentation du lac, et inversement — chaque groupe garde sa position.

Garanties :

- *au moins une fois* : l'offset n'est validé qu'après l'écriture dans le magasin. Un
  message rejoué réécrit le même membre (clé = identifiant métier), donc sans effet de bord ;
- **retard mesuré** : l'écart entre l'horodatage du message et son écriture est exporté, car
  c'est lui qui détermine la fraîcheur des variables vues par l'API ;
- **messages invalides ignorés, jamais bloquants** : un message illisible est compté et passé,
  il est de toute façon tracé par l'ingestion du lac qui l'envoie en file d'erreurs.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import time

from confluent_kafka import Consumer, KafkaError
from prometheus_client import Counter, Gauge, Histogram, start_http_server

from .store import OnlineStore, epoch

logger = logging.getLogger(__name__)

APPLIED = Counter("online_store_messages_total", "Messages appliqués au magasin", ["topic"])
INVALID = Counter("online_store_invalid_total", "Messages illisibles ignorés", ["topic"])
FRESHNESS = Histogram(
    "online_store_freshness_seconds",
    "Retard entre l'horodatage d'un message et son écriture dans le magasin",
    buckets=(0.1, 0.5, 1, 2, 5, 10, 30, 60, 300),
)
LAST_WRITE = Gauge("online_store_last_write_timestamp_seconds", "Dernière écriture réussie")

TOPICS = ("transactions.raw", "app-events.raw")


def consume(store: OnlineStore, settings: dict, run_once: bool = False) -> int:
    """Boucle de consommation : lit, applique au magasin, valide l'offset."""
    consumer = Consumer(
        {
            "bootstrap.servers": settings["bootstrap"],
            "group.id": settings["group"],
            "auto.offset.reset": "latest",  # le magasin sert le présent, pas l'historique
            "enable.auto.commit": False,
            "partition.assignment.strategy": "cooperative-sticky",
        }
    )
    consumer.subscribe(list(TOPICS))
    running = True

    def stop(*_):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    applied = 0
    try:
        while running:
            message = consumer.poll(1.0)
            if message is None:
                if run_once:
                    break
                continue
            if message.error():
                if message.error().code() != KafkaError._PARTITION_EOF:
                    logger.warning("Erreur Kafka : %s", message.error())
                continue
            if apply(store, message.topic(), message.value()):
                applied += 1
            consumer.commit(message=message, asynchronous=False)
    finally:
        consumer.close()
    return applied


def apply(store: OnlineStore, topic: str, raw: bytes | None) -> bool:
    """Applique un message au magasin ; retourne faux si le message est inexploitable."""
    try:
        payload = json.loads((raw or b"").decode("utf-8"))
        if topic == "transactions.raw":
            store.record_transaction(payload)
        else:
            store.record_event(payload)
    except (ValueError, KeyError, TypeError) as error:
        INVALID.labels(topic).inc()
        logger.warning("Message ignoré sur %s : %s", topic, error)
        return False
    APPLIED.labels(topic).inc()
    moment = payload.get("event_time")
    if moment:
        FRESHNESS.observe(max(time.time() - epoch(moment), 0))
    LAST_WRITE.set(time.time())
    return True


def main() -> int:
    """Point d'entrée du service d'alimentation du magasin en ligne."""
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"), format="%(asctime)s %(message)s")
    start_http_server(int(os.environ.get("METRICS_PORT", "9109")))
    store = OnlineStore.from_url(
        os.environ.get("ONDINE_API_REDIS_URL", "redis://redis:6379/0"),
        os.environ["PSEUDONYMISATION_KEY"],
    )
    applied = consume(
        store,
        {
            "bootstrap": os.environ.get("KAFKA_BOOTSTRAP", "kafka-1:9092"),
            "group": os.environ.get("ONDINE_API_CONSUMER_GROUP", "online-features"),
        },
    )
    logger.info("Arrêt du service : %d messages appliqués.", applied)
    return 0
