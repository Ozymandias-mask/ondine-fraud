# Image unique des pipelines : sert à Airflow (orchestration) et à la ligne de commande.
FROM apache/airflow:3.3.1-python3.12

COPY requirements.txt /tmp/requirements.txt
# apache-airflow est répété à sa version exacte : pip refuse alors toute mise à jour ou
# rétrogradation silencieuse d'Airflow causée par nos dépendances.
RUN pip install --no-cache-dir "apache-airflow==${AIRFLOW_VERSION}" -r /tmp/requirements.txt

COPY --chown=airflow:root pyproject.toml README.md /opt/ondine/pipelines/
COPY --chown=airflow:root src /opt/ondine/pipelines/src
RUN pip install --no-cache-dir --no-deps /opt/ondine/pipelines \
    # Extension S3 de DuckDB installée à la construction : rien n'est téléchargé en production.
    && python -c "import duckdb; duckdb.connect().execute('INSTALL httpfs')"
