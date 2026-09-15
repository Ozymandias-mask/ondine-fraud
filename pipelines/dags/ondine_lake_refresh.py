"""Mise à jour automatique du data lake : bronze -> silver -> gold -> indicateurs métier.

Déclenché dès que l'ingestion batch a écrit en bronze, et au moins toutes les heures pour
intégrer les données du flux Kafka. Seules les tâches dont les entrées ont changé sont
recalculées (voir ondine_pipelines.planning) ; une tâche en échec est relancée
automatiquement, puis replanifiée au passage suivant si elle échoue encore.
"""

from __future__ import annotations

from datetime import UTC, datetime

from airflow.sdk import Asset, dag, get_current_context, task
from airflow.timetables.assets import AssetOrTimeSchedule
from airflow.timetables.trigger import CronTriggerTimetable
from ondine_pipelines.orchestration import DEFAULT_ARGS, LAKE_POOL, orchestrator_id

# Écrit par les tâches d'ingestion du DAG ondine_ingestion.
BRONZE = Asset(name="ondine_bronze", uri="s3://bronze")


def _settings():
    from ondine_pipelines.settings import Settings

    return Settings.from_env()


def _registry():
    from ondine_pipelines.registry import Registry

    return Registry(_settings())


def _execute(planned: dict) -> dict[str, int]:
    from ondine_pipelines.tasks import execute

    return execute(_settings(), planned, orchestrator_id(get_current_context()))


@dag(
    dag_id="ondine_lake_refresh",
    schedule=AssetOrTimeSchedule(
        timetable=CronTriggerTimetable("5 * * * *", timezone="UTC"), assets=BRONZE
    ),
    start_date=datetime(2026, 9, 1, tzinfo=UTC),
    catchup=False,
    max_active_runs=1,
    default_args=DEFAULT_ARGS,
    tags=["ondine", "bloc3", "traitement"],
    doc_md=__doc__,
)
def ondine_lake_refresh():
    """Silver, gold et indicateurs, recalculés seulement là où les données ont changé."""

    @task
    def plan_silver_shared() -> list[dict]:
        """Référentiels et labels : prérequis des tâches mensuelles."""
        from ondine_pipelines.planning import plan_silver

        return [t.to_dict() for t in plan_silver(_registry()) if t.year is None]

    @task(pool=LAKE_POOL)
    def silver_shared(planned: dict) -> dict[str, int]:
        return _execute(planned)

    @task(trigger_rule="all_done")
    def plan_silver_months() -> list[dict]:
        from ondine_pipelines.planning import plan_silver

        return [t.to_dict() for t in plan_silver(_registry()) if t.year is not None]

    @task(pool=LAKE_POOL, max_active_tis_per_dagrun=2)
    def silver_month(planned: dict) -> dict[str, int]:
        return _execute(planned)

    @task(trigger_rule="all_done")
    def plan_gold() -> list[dict]:
        """Mois gold impactés par les nouvelles données silver (mois courant et suivant)."""
        from ondine_pipelines.planning import plan_gold

        return [t.to_dict() for t in plan_gold(_registry())]

    @task(pool=LAKE_POOL)
    def first_seen(gold_plan: list[dict]) -> dict[str, int]:
        """Tables de première apparition (historique complet), si un mois gold change."""
        from airflow.sdk.exceptions import AirflowSkipException
        from ondine_pipelines.gold.features import build_first_seen
        from ondine_pipelines.lake import with_retries

        if not gold_plan:
            raise AirflowSkipException("Aucun mois gold à recalculer")
        context = get_current_context()
        return with_retries(lambda: build_first_seen(_settings(), orchestrator_id(context)))

    # Fenêtres glissantes gourmandes en mémoire : un mois gold à la fois.
    @task(pool=LAKE_POOL, max_active_tis_per_dagrun=1)
    def gold_month(planned: dict) -> dict[str, int]:
        return _execute(planned)

    @task(trigger_rule="all_done")
    def plan_kpis() -> list[dict]:
        from ondine_pipelines.planning import plan_kpis

        return [t.to_dict() for t in plan_kpis(_registry())]

    @task(max_active_tis_per_dagrun=4)
    def daily_kpis(planned: dict) -> dict[str, int]:
        """Mise à jour des indicateurs métier dans PostgreSQL."""
        return _execute(planned)

    shared = silver_shared.expand(planned=plan_silver_shared())
    months_plan = plan_silver_months()
    shared >> months_plan
    months = silver_month.expand(planned=months_plan)

    gold_plan = plan_gold()
    months >> gold_plan
    seen = first_seen(gold_plan)
    gold = gold_month.expand(planned=gold_plan)
    seen >> gold

    kpis_plan = plan_kpis()
    gold >> kpis_plan
    daily_kpis.expand(planned=kpis_plan)


ondine_lake_refresh()
