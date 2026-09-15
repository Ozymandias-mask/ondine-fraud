# Image du simulateur : légère, dépendances figées, exécution sans privilèges root.
FROM python:3.12.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Les dépendances sont installées avant le code : couche mise en cache tant
# que requirements.txt ne change pas (reconstructions rapides).
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-deps . \
    && useradd --create-home --uid 10001 simulator \
    && mkdir -p /data && chown simulator /data

USER simulator
EXPOSE 8000

ENTRYPOINT ["ondine-sim"]
CMD ["stream", "--bootstrap", "kafka-1:19092"]
