"""Paramètres communs aux DAG Airflow : relances, alerte d'échec, identifiant de run.

Les DAG ne contiennent que l'enchaînement des tâches ; toute la logique est dans ce paquet
(testée sans Airflow), ce qui permet aussi de lancer chaque traitement en ligne de commande.
"""

from __future__ import annotations

import logging
import time
from datetime import timedelta
from typing import Any

from prometheus_client import CollectorRegistry, Gauge, pushadd_to_gateway

from .settings import Settings

logger = logging.getLogger(__name__)

# Pool Airflow limitant les calculs DuckDB simultanés (mémoire du conteneur).
LAKE_POOL = "duckdb"


def notify_failure(context: dict[str, Any]) -> None:
    """Rappel d'échec d'une tâche (après épuisement des relances) : métrique d'alerte.

    Prometheus déclenche l'alerte TacheAirflowEnEchec ; le détail de l'erreur est dans les
    journaux Airflow et dans ops.pipeline_runs.
    """
    ti = context["ti"]
    logger.error("Échec définitif : %s.%s (run %s)", ti.dag_id, ti.task_id, context.get("run_id"))
    settings_url = Settings.from_env().pushgateway_url
    if not settings_url:
        return
    registry = CollectorRegistry()
    Gauge(
        "airflow_task_failure_timestamp_seconds",
        "Horodatage du dernier échec définitif d'une tâche Airflow",
        ["dag_id", "task_id"],
        registry=registry,
    ).labels(ti.dag_id, ti.task_id).set(time.time())
    try:
        pushadd_to_gateway(
            settings_url,
            job="airflow_failures",
            grouping_key={"dag_id": ti.dag_id, "task_id": ti.task_id},
            registry=registry,
            timeout=5,
        )
    except OSError as exc:
        logger.warning("Métrique d'échec non publiée : %s", exc)


DEFAULT_ARGS: dict[str, Any] = {
    "owner": "plateforme-data",
    # Reprise automatique : 3 essais espacés (2, 4 puis 8 min) avant de déclarer l'échec.
    "retries": 3,
    "retry_delay": timedelta(minutes=2),
    "retry_exponential_backoff": True,
    "max_retry_delay": timedelta(minutes=15),
    "execution_timeout": timedelta(hours=2),
    "on_failure_callback": notify_failure,
}


def orchestrator_id(context: dict[str, Any]) -> str:
    """Identifiant reliant un run du registre (ops.pipeline_runs) à la tâche Airflow."""
    ti = context["ti"]
    suffix = f"[{ti.map_index}]" if getattr(ti, "map_index", -1) >= 0 else ""
    return f"{ti.dag_id}/{context['run_id']}/{ti.task_id}{suffix}"
