"""Exécution d'une tâche planifiée : point d'entrée commun à la CLI et à Airflow."""

from __future__ import annotations

from .lake import with_retries
from .planning import PlannedTask
from .settings import Settings


def execute(
    settings: Settings, task: PlannedTask | dict, orchestrator_id: str | None = None
) -> dict[str, int]:
    """Lance la construction correspondant à la tâche (erreurs d'E/S transitoires relancées)."""
    t = task if isinstance(task, PlannedTask) else PlannedTask(**task)
    if t.layer == "silver":
        if t.dataset == "referentials":
            from .silver.referentials import build_referentials

            return with_retries(lambda: build_referentials(settings, orchestrator_id))
        if t.dataset == "fraud_labels":
            from .silver.labels import build_labels

            return with_retries(lambda: build_labels(settings, orchestrator_id))
        if t.dataset in ("transactions", "app_events"):
            from .silver.activity import build_month

            return with_retries(
                lambda: build_month(settings, t.dataset, t.year, t.month, orchestrator_id)
            )
        if t.dataset == "support_messages":
            from .silver.support import build_support_month

            return with_retries(
                lambda: build_support_month(settings, t.year, t.month, orchestrator_id)
            )
    if t.layer == "gold" and t.dataset == "transaction_features":
        from .gold.features import build_features_month

        return with_retries(
            lambda: build_features_month(settings, t.year, t.month, orchestrator_id)
        )
    if t.layer == "analytics" and t.dataset == "daily_kpis":
        from .analytics import refresh_daily_kpis

        return with_retries(lambda: refresh_daily_kpis(settings, t.year, t.month, orchestrator_id))
    raise ValueError(f"Tâche inconnue : {t}")
