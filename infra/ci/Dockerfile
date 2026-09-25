# Image de test : l'image des traitements, plus les paquets qui ne servent qu'aux tests.
#
# Les tests tournent ainsi sur **exactement** les versions de bibliothèques installées en
# production (mêmes couches d'image), ce qu'un environnement Python reconstruit sur la
# machine d'intégration ne garantirait pas. Rien de ce fichier n'entre en production.
ARG BASE=ondine/pipelines:local
FROM ${BASE}

USER root
COPY requirements-dev.txt /tmp/requirements-dev.txt
COPY api/requirements.txt /tmp/requirements-api.txt
COPY simulator/requirements.txt /tmp/requirements-sim.txt
RUN pip install --no-cache-dir -r /tmp/requirements-dev.txt \
      -r /tmp/requirements-api.txt -r /tmp/requirements-sim.txt

COPY api/pyproject.toml api/README.md /opt/ondine/api/
COPY api/src /opt/ondine/api/src
COPY simulator/pyproject.toml simulator/README.md /opt/ondine/simulator/
COPY simulator/src /opt/ondine/simulator/src
RUN pip install --no-cache-dir --no-deps /opt/ondine/api /opt/ondine/simulator
USER airflow

WORKDIR /depot
ENTRYPOINT ["pytest"]
