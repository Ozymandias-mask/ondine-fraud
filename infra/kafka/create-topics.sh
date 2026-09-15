#!/bin/bash
# Déclare les topics Kafka (idempotent : sans effet si le topic existe déjà).
# Réplication 3 et min.insync.replicas 2 : la perte d'un broker ne perd aucun message acquitté.
set -euo pipefail

KAFKA_TOPICS=/opt/kafka/bin/kafka-topics.sh

KAFKA_CONFIGS=/opt/kafka/bin/kafka-configs.sh

# Crée le topic s'il n'existe pas, puis (ré)applique la rétention : le script peut être
# relancé pour mettre à jour la configuration d'un topic existant.
create_topic() {
  local name=$1 partitions=$2 retention_ms=$3 retention_bytes=$4
  "$KAFKA_TOPICS" --bootstrap-server "$BOOTSTRAP" --create --if-not-exists \
    --topic "$name" --partitions "$partitions" --replication-factor 3 \
    --config min.insync.replicas=2 \
    --config compression.type=producer
  "$KAFKA_CONFIGS" --bootstrap-server "$BOOTSTRAP" --alter --entity-type topics \
    --entity-name "$name" \
    --add-config retention.ms="$retention_ms",retention.bytes="$retention_bytes"
  echo "Topic prêt : $name (partitions=$partitions, rétention=${retention_ms} ms ou ${retention_bytes} o/partition)"
}

# Kafka est un tampon, pas un lieu de stockage : les données sont persistées dans le data
# lake. La rétention est bornée en durée ET en taille pour protéger le disque :
# au plus 512 Mo par partition, soit ~9 Go par topic de 6 partitions avec 3 répliques.
# 6 partitions : jusqu'à 6 consommateurs en parallèle par groupe.
create_topic transactions.raw   6 21600000  536870912   # 6 h
create_topic app-events.raw     6 21600000  536870912   # 6 h
create_topic fraud-reports.raw  3 604800000 268435456   # 7 jours : labels peu volumineux

"$KAFKA_TOPICS" --bootstrap-server "$BOOTSTRAP" --describe
