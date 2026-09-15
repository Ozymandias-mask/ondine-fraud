"""Collecte automatique : zone d'arrivée -> bronze, toutes les 15 minutes.

Les livraisons de fichiers (manifestes) sont détectées à chaque passage ; chaque jeu de
données ayant de nouveaux fichiers est ingéré en parallèle. Les tâches réussies émettent
l'événement « bronze mis à jour », qui déclenche le DAG ondine_lake_refresh.
"""

from __future__ import annotations

from datetime import UTC, datetime

from airflow.sdk import Asset, dag, get_current_context, task
from ondine_pipelines.orchestration import DEFAULT_ARGS, LAKE_POOL, orchestrator_id

# Événement « bronze mis à jour » (même nom et même URI dans ondine_lake_refresh).
BRONZE = Asset(name="ondine_bronze", uri="s3://bronze")


@dag(
    dag_id="ondine_ingestion",
    schedule="*/15 * * * *",
    start_date=datetime(2026, 9, 1, tzinfo=UTC),
    catchup=False,
    max_active_runs=1,
    default_args=DEFAULT_ARGS,
    tags=["ondine", "bloc3", "collecte"],
    doc_md=__doc__,
)
def ondine_ingestion():
    """Détection des nouveaux fichiers puis ingestion parallèle par jeu de données."""

    @task
    def discover_new_files() -> list[str]:
        """Jeux de données ayant au moins un fichier livré et pas encore ingéré."""
        from ondine_pipelines.ingest.landing import DATASETS, discover
        from ondine_pipelines.registry import Registry
        from ondine_pipelines.settings import Settings

        settings = Settings.from_env()
        done = Registry(settings).ingested_files()
        pending = {
            f.dataset for f in discover(settings.landing_root) if (f.path, f.sha256) not in done
        }
        return [d for d in DATASETS if d in pending]

    @task(pool=LAKE_POOL, max_active_tis_per_dagrun=3, outlets=[BRONZE])
    def ingest(dataset: str) -> dict[str, int]:
        """Ingestion d'un jeu : intégrité, typage, quarantaine, lignage, audit."""
        from ondine_pipelines.ingest.landing import ingest_dataset
        from ondine_pipelines.settings import Settings

        return ingest_dataset(
            Settings.from_env(), dataset, orchestrator_id=orchestrator_id(get_current_context())
        )

    ingest.expand(dataset=discover_new_files())


ondine_ingestion()
