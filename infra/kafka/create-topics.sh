#!/bin/bash
# Déclare les topics Kafka (idempotent : sans effet si le topic existe déjà).
# Réplication 3 et min.insync.replicas 2 : la perte d'un broker ne perd aucun message acquitté.
set -euo pipefail

KAFKA_TOPICS=/opt/kafka/bin/kafka-topics.sh

create_topic() {
  local name=$1 partitions=$2 retention_ms=$3
  "$KAFKA_TOPICS" --bootstrap-server "$BOOTSTRAP" --create --if-not-exists \
    --topic "$name" --partitions "$partitions" --replication-factor 3 \
    --config min.insync.replicas=2 \
    --config retention.ms="$retention_ms" \
    --config compression.type=producer
  echo "Topic prêt : $name (partitions=$partitions, rétention=${retention_ms} ms)"
}

# 6 partitions : permet jusqu'à 6 consommateurs en parallèle par groupe.
create_topic transactions.raw   6 86400000     # 24 h : le data lake est la source de vérité
create_topic app-events.raw     6 86400000     # 24 h
create_topic fraud-reports.raw  3 604800000    # 7 jours : labels peu volumineux

"$KAFKA_TOPICS" --bootstrap-server "$BOOTSTRAP" --describe
