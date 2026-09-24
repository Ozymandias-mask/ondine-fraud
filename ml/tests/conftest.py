"""Fixtures : petit data lake complet (5 mois) et suivi MLflow local.

Les tests partent des vraies données du simulateur et des vrais traitements (ingestion,
silver, gold) : ce qui est vérifié ici est exactement ce qui tourne en production, en petit.
"""

from __future__ import annotations

import os
import re
import uuid
from pathlib import Path

import psycopg
import pytest
from ondine_pipelines.settings import Settings

ROOT = Path(__file__).resolve().parents[2]
ROLES = ("ingestion_app", "api_app", "analyst_ro", "airflow", "mlflow")


@pytest.fixture(scope="session")
def ml_database():
    """Base de registre temporaire, créée avec les vrais scripts SQL du dépôt."""
    admin_dsn = os.environ.get("TEST_PG_ADMIN_DSN")
    if not admin_dsn:
        pytest.skip("TEST_PG_ADMIN_DSN non défini")
    name = f"ondine_ml_{uuid.uuid4().hex[:8]}"
    with psycopg.connect(admin_dsn, autocommit=True) as con:
        for role in ROLES:
            con.execute(
                f"DO $$ BEGIN CREATE ROLE {role}; EXCEPTION WHEN duplicate_object THEN NULL; END $$"
            )
        con.execute(f"CREATE DATABASE {name}")
    dsn = re.sub(r"/[^/?]+(\?|$)", f"/{name}\\1", admin_dsn)
    files = [
        ROOT / "infra/postgres/init/02_governance_audit.sql",
        *sorted((ROOT / "infra/postgres/migrations").glob("*.sql")),
    ]
    with psycopg.connect(dsn, autocommit=True) as con:
        for sql_file in files:
            con.execute(
                "\n".join(
                    line
                    for line in sql_file.read_text(encoding="utf-8").splitlines()
                    if not line.startswith("\\connect")
                )
            )
    yield dsn
    with psycopg.connect(admin_dsn, autocommit=True) as con:
        con.execute(f"DROP DATABASE IF EXISTS {name} WITH (FORCE)")


@pytest.fixture(scope="session")
def lake(ml_database, tmp_path_factory) -> Settings:
    """Cinq mois de données passées par tout le pipeline, jusqu'à la couche gold."""
    from ondine_pipelines.cli import run_gold, run_silver
    from ondine_pipelines.ingest.landing import DATASETS, ingest_dataset
    from ondine_sim.batch import run_batch
    from ondine_sim.config import get_config

    root = tmp_path_factory.mktemp("ml")
    landing = root / "landing"
    run_batch(get_config("small", output_dir=landing, workers=2, months=5, n_clients=8000))
    settings = Settings(
        landing_root=landing,
        lake_root=str(root / "lake"),
        pg_dsn=ml_database,
        pseudonymisation_key="clé-de-test",
        duckdb_threads=2,
        duckdb_memory_limit="2GB",
    )
    for dataset in DATASETS:
        ingest_dataset(settings, dataset)
    assert run_silver(settings, "all", "all", resume=True) == []
    assert run_gold(settings, "all", resume=True) == []
    return settings


@pytest.fixture(scope="session")
def ml_settings(tmp_path_factory):
    """Paramètres d'entraînement adaptés à un petit jeu (suivi MLflow dans SQLite)."""
    from ondine_ml.settings import MlSettings

    store = tmp_path_factory.mktemp("mlflow")
    return MlSettings(
        tracking_uri=f"sqlite:///{store}/mlflow.db",
        experiment="tests",
        model_name="ondine-fraud-detector-test",
        negative_sampling_rate=0.3,
        num_boost_round=200,
        early_stopping_rounds=40,
        shap_sample=3000,
        train_bounds="2025-02..2025-03",
        valid_bounds="2025-04..2025-04",
        test_bounds="2025-05..2025-05",
    )


@pytest.fixture(scope="session")
def trained(lake, ml_settings):
    """Un entraînement complet, réutilisé par les tests (coûteux à refaire)."""
    from ondine_ml.train import train

    return train(lake, ml_settings)
