"""Fixtures : petite zone d'arrivée générée par le simulateur, data lake local, base de test.

La base PostgreSQL de test est créée à partir des vrais scripts d'initialisation et des
migrations du dépôt. Variable requise : TEST_PG_ADMIN_DSN (sinon les tests sont ignorés).
"""

from __future__ import annotations

import os
import re
import uuid
from dataclasses import replace
from pathlib import Path

import psycopg
import pytest
from ondine_pipelines.settings import Settings

ROOT = Path(__file__).resolve().parents[2]
ROLES = ("ingestion_app", "api_app", "analyst_ro", "airflow", "mlflow")


@pytest.fixture(scope="session")
def landing(tmp_path_factory) -> Path:
    from ondine_sim.batch import run_batch
    from ondine_sim.config import get_config

    root = tmp_path_factory.mktemp("sim") / "landing"
    run_batch(get_config("small", output_dir=root, workers=1))
    return root


@pytest.fixture(scope="session")
def pg_dsn():
    admin_dsn = os.environ.get("TEST_PG_ADMIN_DSN")
    if not admin_dsn:
        pytest.skip("TEST_PG_ADMIN_DSN non défini")
    name = f"ondine_test_{uuid.uuid4().hex[:8]}"
    with psycopg.connect(admin_dsn, autocommit=True) as con:
        for role in ROLES:
            con.execute(
                f"DO $$ BEGIN CREATE ROLE {role}; EXCEPTION WHEN duplicate_object THEN NULL; END $$"
            )
        con.execute(f"CREATE DATABASE {name}")
    dsn = re.sub(r"/[^/?]+(\?|$)", f"/{name}\\1", admin_dsn)
    sql_files = [
        ROOT / "infra/postgres/init/02_governance_audit.sql",
        *sorted((ROOT / "infra/postgres/migrations").glob("*.sql")),
    ]
    with psycopg.connect(dsn, autocommit=True) as con:
        for sql_file in sql_files:
            sql = "\n".join(
                line
                for line in sql_file.read_text(encoding="utf-8").splitlines()
                if not line.startswith("\\connect")
            )
            con.execute(sql)
    yield dsn
    with psycopg.connect(admin_dsn, autocommit=True) as con:
        con.execute(f"DROP DATABASE IF EXISTS {name} WITH (FORCE)")


@pytest.fixture(scope="session")
def base_settings(landing, pg_dsn, tmp_path_factory) -> Settings:
    return Settings(
        landing_root=landing,
        lake_root=str(tmp_path_factory.mktemp("lake")),
        pg_dsn=pg_dsn,
        pseudonymisation_key="clé-de-test",
        duckdb_threads=2,
        duckdb_memory_limit="1GB",
    )


@pytest.fixture()
def fresh_settings(base_settings, tmp_path) -> Settings:
    """Data lake vide propre au test (la base de registre reste partagée)."""
    return replace(base_settings, lake_root=str(tmp_path / "lake"))
