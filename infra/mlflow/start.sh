#!/bin/sh
# Démarre le serveur MLflow : suivi dans PostgreSQL, artefacts dans le bucket S3.
# Le serveur relaie lui-même les artefacts (--serve-artifacts) : les clients n'ont donc
# aucune clé S3 à connaître, seul le serveur y accède (moindre privilège).
set -eu

mkdir -p /home/mlflow/auth
cat > /home/mlflow/auth/basic_auth.ini <<INI
[mlflow]
default_permission = READ
database_uri = sqlite:////home/mlflow/auth/basic_auth.db
admin_username = ${MLFLOW_ADMIN_USER}
admin_password = ${MLFLOW_ADMIN_PASSWORD}
authorization_function = mlflow.server.auth:authenticate_request_basic_auth
INI
export MLFLOW_AUTH_CONFIG_PATH=/home/mlflow/auth/basic_auth.ini

# MLflow refuse par défaut les requêtes dont l'en-tête Host n'est pas « localhost »
# (protection contre le DNS rebinding) : les autres conteneurs l'appellent par son nom de
# service, il faut donc l'autoriser explicitement.
exec mlflow server \
  --host 0.0.0.0 --port 5000 \
  --allowed-hosts "localhost,127.0.0.1,mlflow,mlflow:5000" \
  --cors-allowed-origins "http://localhost:5001,http://127.0.0.1:5001" \
  --app-name basic-auth \
  --backend-store-uri "postgresql+psycopg2://mlflow:${PG_MLFLOW_PASSWORD}@postgres:5432/mlflow" \
  --artifacts-destination s3://mlflow-artifacts/ \
  --serve-artifacts \
  --workers 2 \
  --gunicorn-opts "--timeout 120"
