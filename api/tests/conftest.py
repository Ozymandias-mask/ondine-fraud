"""Fixtures de l'API : petit data lake réel, magasin en ligne en mémoire, modèle entraîné.

Les tests ne simulent pas les données : ils passent par le simulateur et les vrais
traitements jusqu'à la couche gold, puis rejouent **les mêmes messages bruts** que Kafka dans
le magasin en ligne. C'est ce qui permet de comparer, variable par variable, ce que calcule
l'API en production à ce qu'a calculé le SQL à l'entraînement.
"""

from __future__ import annotations

import os
import re
import uuid
from pathlib import Path

import fakeredis
import psycopg
import pytest
from ondine_api.store import OnlineStore
from ondine_pipelines.settings import Settings

ROOT = Path(__file__).resolve().parents[2]
ROLES = ("ingestion_app", "api_app", "analyst_ro", "airflow", "mlflow")
PSEUDO_KEY = "clé-de-test"


@pytest.fixture(scope="session")
def api_database():
    """Base de registre temporaire, créée avec les vrais scripts SQL du dépôt."""
    admin_dsn = os.environ.get("TEST_PG_ADMIN_DSN")
    if not admin_dsn:
        pytest.skip("TEST_PG_ADMIN_DSN non défini")
    name = f"ondine_api_{uuid.uuid4().hex[:8]}"
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
def lake(api_database, tmp_path_factory) -> Settings:
    """Quatre mois de données passées par tout le pipeline, jusqu'à la couche gold."""
    from ondine_pipelines.cli import run_gold, run_silver
    from ondine_pipelines.ingest.landing import DATASETS, ingest_dataset
    from ondine_sim.batch import run_batch
    from ondine_sim.config import get_config

    root = tmp_path_factory.mktemp("api")
    landing = root / "landing"
    run_batch(get_config("small", output_dir=landing, workers=2, months=4, n_clients=2500))
    settings = Settings(
        landing_root=landing,
        lake_root=str(root / "lake"),
        pg_dsn=api_database,
        pseudonymisation_key=PSEUDO_KEY,
        duckdb_threads=2,
        duckdb_memory_limit="2GB",
    )
    for dataset in DATASETS:
        ingest_dataset(settings, dataset)
    assert run_silver(settings, "all", "all", resume=True) == []
    assert run_gold(settings, "all", resume=True) == []
    return settings


@pytest.fixture
def store() -> OnlineStore:
    """Magasin en ligne isolé, en mémoire (même code Redis, sans serveur)."""
    return OnlineStore(fakeredis.FakeStrictRedis(decode_responses=True), PSEUDO_KEY)
