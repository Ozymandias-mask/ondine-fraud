#!/bin/sh
# Crée les buckets du data lake (idempotent : un bucket existant est ignoré).
#   landing          fichiers bruts déposés par les sources (purgés après ingestion)
#   bronze           données brutes converties en Parquet, historisées
#   silver           données nettoyées, contrôlées et pseudonymisées
#   gold             tables prêtes pour le modèle et les tableaux de bord
#   quarantine       enregistrements rejetés par les contrôles qualité
#   mlflow-artifacts modèles et artefacts d'entraînement
#   exports          exports des demandes d'accès RGPD (lecture par le seul compte pipeline)
#   backups          sauvegardes, verrouillage objet (non modifiables, non supprimables)
set -eu

MASTER=seaweed-master:9333

existing=$(echo "s3.bucket.list" | weed shell -master="$MASTER" 2>/dev/null || true)

for bucket in landing bronze silver gold quarantine mlflow-artifacts exports; do
  if echo "$existing" | grep -qw "$bucket"; then
    echo "Bucket déjà présent : $bucket"
  else
    echo "s3.bucket.create -name $bucket" | weed shell -master="$MASTER"
    echo "Bucket créé : $bucket"
  fi
done

if echo "$existing" | grep -qw backups; then
  echo "Bucket déjà présent : backups"
else
  echo "s3.bucket.create -name backups -withLock" | weed shell -master="$MASTER"
  echo "Bucket créé avec verrouillage objet : backups"
fi
