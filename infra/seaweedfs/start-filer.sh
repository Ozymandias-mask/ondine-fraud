#!/bin/sh
# Démarre le filer SeaweedFS avec l'API S3.
# 1. Génère la configuration des identités S3 à partir des variables d'environnement
#    (les secrets ne sont jamais écrits dans le dépôt).
# 2. Applique le moindre privilège : chaque identité n'accède qu'aux buckets nécessaires.
set -eu

for var in S3_ADMIN_ACCESS_KEY S3_ADMIN_SECRET_KEY S3_INGESTION_ACCESS_KEY S3_INGESTION_SECRET_KEY \
           S3_PIPELINE_ACCESS_KEY S3_PIPELINE_SECRET_KEY S3_ML_ACCESS_KEY S3_ML_SECRET_KEY; do
  eval "value=\${$var:-}"
  if [ -z "$value" ]; then
    echo "Variable manquante : $var (lancer scripts/generate_env.py)" >&2
    exit 1
  fi
done

mkdir -p /etc/seaweedfs
umask 077
cat > /etc/seaweedfs/s3.json <<JSON
{
  "identities": [
    {
      "name": "admin",
      "credentials": [{"accessKey": "${S3_ADMIN_ACCESS_KEY}", "secretKey": "${S3_ADMIN_SECRET_KEY}"}],
      "actions": ["Admin", "Read", "Write", "List", "Tagging"]
    },
    {
      "name": "ingestion",
      "credentials": [{"accessKey": "${S3_INGESTION_ACCESS_KEY}", "secretKey": "${S3_INGESTION_SECRET_KEY}"}],
      "actions": ["Write:landing", "List:landing", "Read:landing"]
    },
    {
      "name": "pipeline",
      "credentials": [{"accessKey": "${S3_PIPELINE_ACCESS_KEY}", "secretKey": "${S3_PIPELINE_SECRET_KEY}"}],
      "actions": ["Read:landing", "List:landing", "Write:landing",
                  "Read:bronze", "Write:bronze", "List:bronze",
                  "Read:silver", "Write:silver", "List:silver",
                  "Read:gold", "Write:gold", "List:gold",
                  "Read:quarantine", "Write:quarantine", "List:quarantine"]
    },
    {
      "name": "ml",
      "credentials": [{"accessKey": "${S3_ML_ACCESS_KEY}", "secretKey": "${S3_ML_SECRET_KEY}"}],
      "actions": ["Read:gold", "List:gold",
                  "Read:mlflow-artifacts", "Write:mlflow-artifacts", "List:mlflow-artifacts"]
    }
  ]
}
JSON

# -encryptVolumeData / -s3.encryptVolumeData : chaque fichier est chiffré en AES-256-GCM avec
# une clé propre, stockée dans les métadonnées du filer (les serveurs de volumes ne voient
# que des données chiffrées).
exec weed filer \
  -ip=seaweed-filer -ip.bind=0.0.0.0 -port=8888 \
  -master=seaweed-master:9333 \
  -defaultStoreDir=/data \
  -encryptVolumeData \
  -s3 -s3.port=8333 -s3.config=/etc/seaweedfs/s3.json -s3.encryptVolumeData \
  -metricsPort=9326
