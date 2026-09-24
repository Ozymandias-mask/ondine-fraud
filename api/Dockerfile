# Image de service : seulement ce qu'il faut pour décider (pas d'Airflow, pas d'outillage).
# Contexte de construction = racine du dépôt (les paquets pipelines/, ml/ et api/ y sont copiés).
FROM python:3.12.14-slim

# LightGBM a besoin de la bibliothèque OpenMP.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

COPY pipelines/requirements.txt /tmp/pipelines.txt
COPY api/requirements.txt /tmp/api.txt
RUN pip install --no-cache-dir -r /tmp/pipelines.txt -r /tmp/api.txt

COPY pipelines/pyproject.toml pipelines/README.md /opt/ondine/pipelines/
COPY pipelines/src /opt/ondine/pipelines/src
COPY ml/pyproject.toml ml/README.md /opt/ondine/ml/
COPY ml/src /opt/ondine/ml/src
COPY api/pyproject.toml api/README.md /opt/ondine/api/
COPY api/src /opt/ondine/api/src
RUN pip install --no-cache-dir --no-deps /opt/ondine/pipelines /opt/ondine/ml /opt/ondine/api \
    # Extension S3 de DuckDB installée à la construction : rien n'est téléchargé en production.
    && python -c "import duckdb; duckdb.connect().execute('INSTALL httpfs')" \
    && useradd --create-home --uid 10002 ondine \
    && cp -r /root/.duckdb /home/ondine/.duckdb && chown -R ondine:ondine /home/ondine

# Moindre privilège : le service ne tourne pas en administrateur.
USER ondine
EXPOSE 8000
ENTRYPOINT ["ondine-api"]
CMD ["serve"]
