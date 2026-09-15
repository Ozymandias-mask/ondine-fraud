"""Publication des métriques des traitements batch vers Prometheus (Pushgateway).

Un traitement batch est trop court pour être interrogé par Prometheus : il pousse ses
métriques en fin d'exécution. Si la passerelle est indisponible, l'échec est journalisé
mais le traitement n'est pas interrompu (la supervision ne doit pas casser la production).
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from prometheus_client import CollectorRegistry, Gauge, pushadd_to_gateway

if TYPE_CHECKING:
    from .registry import Run
    from .settings import Settings

logger = logging.getLogger(__name__)


def push_run(
    settings: Settings, run: Run, duration_s: float, extra: dict[str, float] | None = None
) -> None:
    """Pousse les métriques d'un run (volumes, débit, durée, statut, horodatage)."""
    if not settings.pushgateway_url:
        return
    registry = CollectorRegistry()
    labels = ["pipeline", "task"]
    values = [run.pipeline, run.task]

    def gauge(name: str, doc: str, value: float) -> None:
        Gauge(name, doc, labels, registry=registry).labels(*values).set(value)

    gauge("pipeline_rows_in", "Lignes lues", run.rows_in)
    gauge("pipeline_rows_out", "Lignes écrites", run.rows_out)
    gauge("pipeline_rows_rejected", "Lignes rejetées (quarantaine)", run.rows_rejected)
    gauge("pipeline_duration_seconds", "Durée du traitement", duration_s)
    gauge(
        "pipeline_rows_per_second",
        "Débit du traitement",
        run.rows_out / duration_s if duration_s > 0 else 0,
    )
    gauge(
        "pipeline_success",
        "1 si le dernier run a réussi, 0 sinon",
        1 if run.status == "success" else 0,
    )
    gauge("pipeline_last_run_timestamp_seconds", "Fin du dernier run", time.time())
    if run.status == "success":
        gauge("pipeline_last_success_timestamp_seconds", "Fin du dernier run réussi", time.time())
    for name, value in (extra or {}).items():
        gauge(f"pipeline_{name}", name, value)
    try:
        # pushadd : conserve les métriques absentes de ce push (ex. dernier succès).
        pushadd_to_gateway(
            settings.pushgateway_url, job=f"{run.pipeline}_{run.task}", registry=registry, timeout=5
        )
    except OSError as exc:
        logger.warning("Métriques non publiées (%s) : %s", settings.pushgateway_url, exc)
