# Image unique : orchestration Airflow, traitements du data lake et modèle de fraude.
# Contexte de construction = racine du dépôt (les paquets pipelines/ et ml/ y sont copiés).
FROM apache/airflow:3.3.1-python3.12

# LightGBM a besoin de la bibliothèque OpenMP (libgomp), absente de l'image Airflow.
USER root
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*
USER airflow

COPY pipelines/requirements.txt /tmp/requirements.txt
# apache-airflow est répété à sa version exacte : pip refuse alors toute mise à jour ou
# rétrogradation silencieuse d'Airflow causée par nos dépendances.
RUN pip install --no-cache-dir "apache-airflow==${AIRFLOW_VERSION}" -r /tmp/requirements.txt

COPY --chown=airflow:root pipelines/pyproject.toml pipelines/README.md /opt/ondine/pipelines/
COPY --chown=airflow:root pipelines/src /opt/ondine/pipelines/src
COPY --chown=airflow:root ml/pyproject.toml ml/README.md /opt/ondine/ml/
COPY --chown=airflow:root ml/src /opt/ondine/ml/src
RUN pip install --no-cache-dir --no-deps /opt/ondine/pipelines /opt/ondine/ml \
    # Extension S3 de DuckDB installée à la construction : rien n'est téléchargé en production.
    && python -c "import duckdb; duckdb.connect().execute('INSTALL httpfs')"
