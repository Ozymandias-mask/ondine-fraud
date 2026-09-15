#!/bin/sh
# Rôles applicatifs (moindre privilège) et bases dédiées. Exécuté une seule fois,
# à la création du volume PostgreSQL. Les mots de passe viennent de l'environnement.
set -eu

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname ondine \
  -v ingestion_pwd="$PG_INGESTION_PASSWORD" -v api_pwd="$PG_API_PASSWORD" \
  -v analyst_pwd="$PG_ANALYST_PASSWORD" -v airflow_pwd="$PG_AIRFLOW_PASSWORD" \
  -v mlflow_pwd="$PG_MLFLOW_PASSWORD" <<'SQL'
-- Un rôle par usage : un composant compromis n'accède qu'à ce dont il a besoin.
CREATE ROLE ingestion_app LOGIN PASSWORD :'ingestion_pwd';
CREATE ROLE api_app       LOGIN PASSWORD :'api_pwd';
CREATE ROLE analyst_ro    LOGIN PASSWORD :'analyst_pwd';
CREATE ROLE airflow       LOGIN PASSWORD :'airflow_pwd';
CREATE ROLE mlflow        LOGIN PASSWORD :'mlflow_pwd';

CREATE DATABASE airflow OWNER airflow;
CREATE DATABASE mlflow  OWNER mlflow;

-- Personne ne se connecte par défaut : les droits sont accordés explicitement.
REVOKE CONNECT ON DATABASE ondine  FROM PUBLIC;
REVOKE CONNECT ON DATABASE airflow FROM PUBLIC;
REVOKE CONNECT ON DATABASE mlflow  FROM PUBLIC;
GRANT CONNECT ON DATABASE ondine TO ingestion_app, api_app, analyst_ro;
GRANT CONNECT ON DATABASE airflow TO airflow;
GRANT CONNECT ON DATABASE mlflow TO mlflow;
SQL
