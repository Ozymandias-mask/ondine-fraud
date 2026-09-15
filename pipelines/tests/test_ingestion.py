import gzip
import shutil
from dataclasses import replace

import duckdb
import psycopg
import pytest
from ondine_pipelines.ingest.landing import (
    DATASETS,
    CompletenessError,
    IntegrityError,
    discover,
    ingest_dataset,
    pending_files,
)
from ondine_pipelines.registry import Registry


@pytest.fixture(scope="module")
def ingested(base_settings):
    return {d: ingest_dataset(base_settings, d) for d in DATASETS}


def test_discovery_lists_every_delivered_file(landing):
    files = discover(landing)
    assert {f.dataset for f in files} == set(DATASETS)
    assert all(len(f.sha256) == 64 and f.rows >= 0 for f in files)


def test_every_row_reaches_bronze(ingested, base_settings, landing):
    expected = {}
    for f in discover(landing):
        expected[f.dataset] = expected.get(f.dataset, 0) + f.rows
    for dataset, summary in ingested.items():
        assert summary["rows"] + summary["rejected"] == expected[dataset], dataset
        assert summary["rejected"] == 0


def test_bronze_schema_is_unified_across_versions(ingested, base_settings):
    con = duckdb.connect()
    path = f"{base_settings.uri('bronze', 'transactions')}/**/*.parquet"
    df = con.sql(f"SELECT * FROM read_parquet('{path}', hive_partitioning = true)").df()
    assert "wallet_type" in df.columns and "_source_sha256" in df.columns
    january = df[df["month"] == 1]
    assert january["wallet_type"].isna().all()  # schéma v1 : colonne absente à la source
    assert df["event_time"].notna().all()


def test_ingestion_is_incremental_and_traced(ingested, base_settings):
    registry = Registry(base_settings)
    assert pending_files(base_settings, registry, "transactions") == []
    assert ingest_dataset(base_settings, "transactions")["files"] == 0  # rien à refaire
    with psycopg.connect(base_settings.pg_dsn) as con:
        runs = con.execute(
            "SELECT status, rows_out, rows_per_s FROM ops.pipeline_runs "
            "WHERE pipeline = 'ingestion' AND task = 'transactions'"
        ).fetchall()
        audit = con.execute(
            "SELECT count(*) FROM audit.access_log WHERE action = 'ingest'"
        ).fetchone()
    assert runs[0][0] == "success" and runs[0][1] > 0 and runs[0][2] > 0
    assert audit[0] >= 2  # lecture des référentiels clients et des e-mails tracée


def test_emails_are_parsed(ingested, base_settings):
    path = f"{base_settings.uri('bronze', 'support_emails')}/**/*.parquet"
    row = duckdb.sql(
        f"SELECT sender_address, subject, body FROM read_parquet('{path}') LIMIT 1"
    ).fetchone()
    assert "@" in row[0] and row[2]


def test_corrupted_file_is_refused(fresh_settings, tmp_path, landing):
    copy = tmp_path / "landing"
    shutil.copytree(landing, copy)
    target = next((copy / "fraud_reports").rglob("*.csv.gz"))
    data = gzip.decompress(target.read_bytes()).replace(b"confirmed", b"CONFIRMED", 1)
    target.write_bytes(gzip.compress(data))
    settings = replace(fresh_settings, landing_root=copy)
    with psycopg.connect(settings.pg_dsn, autocommit=True) as con:
        con.execute("DELETE FROM ops.file_ingestions WHERE dataset = 'fraud_reports'")
    with pytest.raises(IntegrityError):
        ingest_dataset(settings, "fraud_reports")
    with psycopg.connect(settings.pg_dsn) as con:
        status = con.execute(
            "SELECT status, error FROM ops.pipeline_runs WHERE task = 'fraud_reports' "
            "ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
    assert status[0] == "failed" and status[1].startswith("IntegrityError")


def test_malformed_rows_go_to_quarantine(fresh_settings, tmp_path, landing):
    import hashlib
    import json

    copy = tmp_path / "landing"
    shutil.copytree(landing, copy)
    manifest_path = next((copy / "_manifests").glob("period=*/part-000.json"))
    manifest = json.loads(manifest_path.read_text())
    info = next(f for f in manifest["files"] if f["path"].startswith("support_tickets"))
    target = copy / info["path"]
    lines = gzip.decompress(target.read_bytes()).decode().splitlines()
    lines[1] = lines[1].replace(",email,", ",email,EXTRA,COLUMN,", 1)  # ligne cassée
    data = gzip.compress(("\n".join(lines) + "\n").encode())
    target.write_bytes(data)
    info.update(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    manifest_path.write_text(json.dumps(manifest))
    settings = replace(fresh_settings, landing_root=copy)
    with psycopg.connect(settings.pg_dsn, autocommit=True) as con:
        con.execute("DELETE FROM ops.file_ingestions WHERE dataset = 'support_tickets'")
    # 1 ligne sur une petite livraison dépasse le seuil de 1 % : la livraison est bloquée…
    if info["rows"] < 100:
        with pytest.raises(CompletenessError):
            ingest_dataset(settings, "support_tickets")
    else:
        assert ingest_dataset(settings, "support_tickets")["rejected"] == 1
    # … et la ligne fautive est conservée en quarantaine pour analyse.
    quarantine = list((tmp_path / "lake" / "quarantine" / "support_tickets").glob("*.parquet"))
    assert quarantine
