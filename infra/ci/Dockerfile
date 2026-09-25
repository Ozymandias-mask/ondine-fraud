# Image de test : l'image des traitements, plus les paquets qui ne servent qu'aux tests.
#
# Les tests tournent ainsi sur **exactement** les versions de bibliothèques installées en
# production (mêmes couches d'image), ce qu'un environnement Python reconstruit sur la
# machine d'intégration ne garantirait pas. Rien de ce fichier n'entre en production.
#
# Tout se fait sous le compte `airflow` : l'image de base refuse `pip` en administrateur,
# et c'est une bonne règle — les paquets atterriraient dans un répertoire que le compte de
# service ne lit pas forcément.
ARG BASE=ondine/pipelines:local
FROM ${BASE}

COPY --chown=airflow:root requirements-dev.txt /tmp/requirements-dev.txt
COPY --chown=airflow:root api/requirements.txt /tmp/requirements-api.txt
COPY --chown=airflow:root simulator/requirements.txt /tmp/requirements-sim.txt
RUN pip install --no-cache-dir -r /tmp/requirements-dev.txt \
      -r /tmp/requirements-api.txt -r /tmp/requirements-sim.txt

COPY --chown=airflow:root api/pyproject.toml api/README.md /opt/ondine/api/
COPY --chown=airflow:root api/src /opt/ondine/api/src
COPY --chown=airflow:root simulator/pyproject.toml simulator/README.md /opt/ondine/simulator/
COPY --chown=airflow:root simulator/src /opt/ondine/simulator/src
RUN pip install --no-cache-dir --no-deps /opt/ondine/api /opt/ondine/simulator

# Le dépôt est monté ici par `docker compose` : les tests lisent le code tel qu'il est dans
# l'espace de travail, et les paquets tels qu'ils sont installés dans l'image.
WORKDIR /depot
ENTRYPOINT ["pytest"]
