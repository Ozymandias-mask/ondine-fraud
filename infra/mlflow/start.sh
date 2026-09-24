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

HELP=$(mlflow server --help 2>/dev/null || true)
OPTIONS=""

# MLflow 3.16 refuse (403) les requêtes dont l'en-tête Host n'est pas « localhost » :
# les autres conteneurs appellent le serveur par son nom de service, il faut l'autoriser.
# L'option est ajoutée seulement si la version installée la connaît.
case "$HELP" in
  *--allowed-hosts*) OPTIONS="$OPTIONS --allowed-hosts localhost,127.0.0.1,mlflow,mlflow:5000" ;;
esac
# Deux processus suffisent ici et tiennent dans la mémoire allouée au conteneur.
case "$HELP" in
  *--workers*) OPTIONS="$OPTIONS --workers 2" ;;
esac

echo "Démarrage de MLflow avec les options :$OPTIONS"
exec mlflow server \
  --host 0.0.0.0 --port 5000 \
  --app-name basic-auth \
  --backend-store-uri "postgresql+psycopg2://mlflow:${PG_MLFLOW_PASSWORD}@postgres:5432/mlflow" \
  --artifacts-destination s3://mlflow-artifacts/ \
  --serve-artifacts \
  $OPTIONS
