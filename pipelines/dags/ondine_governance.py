"""Gouvernance automatisée des données personnelles.

- ondine_governance (toutes les 15 min) : exécution des demandes RGPD en attente,
  détection des accès anormaux dans le journal d'audit, publication des indicateurs
  (incidents ouverts, échéance CNIL 72 h, demandes en retard) vers Prometheus ;
- ondine_retention (chaque nuit) : suppression des fichiers bruts déjà ingérés de la zone
  d'arrivée, après le délai de conservation (LANDING_RETENTION_DAYS, 7 jours par défaut).
"""

from __future__ import annotations

import os
from datetime import UTC, datetime

from airflow.sdk import dag, get_current_context, task
from airflow.timetables.trigger import CronTriggerTimetable
from ondine_pipelines.orchestration import DEFAULT_ARGS, LAKE_POOL, orchestrator_id


def _settings():
    from ondine_pipelines.settings import Settings

    return Settings.from_env()


@dag(
    dag_id="ondine_governance",
    schedule="*/15 * * * *",
    start_date=datetime(2026, 9, 1, tzinfo=UTC),
    catchup=False,
    max_active_runs=1,
    default_args=DEFAULT_ARGS,
    tags=["ondine", "bloc3", "rgpd", "securite"],
    doc_md=__doc__,
)
def ondine_governance():
    """Droits RGPD, détection des accès anormaux, indicateurs de gouvernance."""

    @task(pool=LAKE_POOL)
    def process_data_subject_requests() -> dict[str, int]:
        """Accès, portabilité, opposition ; effacement une fois validé par le DPO."""
        from ondine_pipelines.governance.rights import process_requests

        return process_requests(_settings(), orchestrator_id(get_current_context()))

    @task
    def detect_abnormal_access() -> int:
        """Règles de détection sur le journal d'audit ; chaque écart devient un incident."""
        from ondine_pipelines.governance.security import detect_abnormal_access as detect

        incidents = detect(_settings())
        for incident in incidents:
            print("Nouvel incident de sécurité :", incident)
        return len(incidents)

    @task(trigger_rule="all_done")
    def publish_governance_metrics() -> dict:
        from ondine_pipelines.governance.monitoring import push_governance_metrics

        return push_governance_metrics(_settings())

    [process_data_subject_requests(), detect_abnormal_access()] >> publish_governance_metrics()


@dag(
    dag_id="ondine_retention",
    schedule=CronTriggerTimetable("0 3 * * *", timezone="Europe/Paris"),
    start_date=datetime(2026, 9, 1, tzinfo=UTC),
    catchup=False,
    max_active_runs=1,
    default_args=DEFAULT_ARGS,
    tags=["ondine", "bloc3", "rgpd"],
    doc_md=__doc__,
)
def ondine_retention():
    """Purge de la zone d'arrivée après le délai de conservation."""

    @task
    def purge_landing() -> dict[str, int]:
        from ondine_pipelines.ingest.retention import purge_landing as purge

        return purge(_settings(), int(os.environ.get("LANDING_RETENTION_DAYS", "7")))

    purge_landing()


ondine_governance()
ondine_retention()
