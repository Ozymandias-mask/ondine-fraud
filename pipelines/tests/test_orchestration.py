"""Automatisation de bout en bout : planification incrémentale, indicateurs, droits RGPD,
détection d'accès anormaux et rétention. Base de registre et data lake propres au module ;
les tests s'exécutent dans l'ordre du fichier (l'effacement RGPD vient en dernier)."""

import json
import os
import shutil
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import duckdb
import psycopg
import pytest
from ondine_pipelines.cli import run_gold, run_kpis, run_silver
from ondine_pipelines.governance.monitoring import collect
from ondine_pipelines.governance.rights import process_requests
from ondine_pipelines.governance.security import detect_abnormal_access
from ondine_pipelines.ingest.landing import DATASETS, ingest_dataset
from ondine_pipelines.ingest.retention import purge_landing
from ondine_pipelines.planning import plan_gold, plan_kpis, plan_silver
from ondine_pipelines.registry import Registry


@pytest.fixture(scope="module")
def lake(base_settings, module_pg_dsn, tmp_path_factory):
    """Lake et registre isolés, ingérés entièrement et construits jusqu'aux indicateurs."""
    root = tmp_path_factory.mktemp("orchestration")
    landing = root / "landing"
    shutil.copytree(base_settings.landing_root, landing)
    settings = replace(
        base_settings, landing_root=landing, lake_root=str(root / "lake"), pg_dsn=module_pg_dsn
    )
    for dataset in DATASETS:
        ingest_dataset(settings, dataset)
    return settings


def _sql(settings, query, params=()):
    with psycopg.connect(settings.pg_dsn, autocommit=True) as con:
        cur = con.execute(query, params)
        return cur.fetchall() if cur.description else []


def _parquet(settings, layer, dataset, where="true"):
    pattern = settings.uri(layer, dataset, "**", "*.parquet")
    return duckdb.sql(
        f"SELECT count(*) FROM read_parquet('{pattern}', union_by_name = true) WHERE {where}"
    ).fetchone()[0]


def test_incremental_planning_runs_only_what_changed(lake):
    registry = Registry(lake)
    first = {t.name for t in plan_silver(registry)}
    assert {"referentials", "fraud_labels", "transactions_2025-01", "app_events_2025-02"} <= first
    assert "support_messages_2025-01" in first

    assert run_silver(lake, "all", "all", resume=True) == []
    assert plan_silver(registry) == []
    assert {t.name for t in plan_gold(registry)} == {
        "transaction_features_2025-01",
        "transaction_features_2025-02",
    }
    assert run_gold(lake, "all", resume=True) == []
    assert plan_gold(registry) == []
    assert len(plan_kpis(registry)) == 2
    assert run_kpis(lake, "all", resume=True) == []
    assert plan_kpis(registry) == []

    # Un lot du flux Kafka arrive en février : seul ce mois est à refaire, en cascade.
    _sql(
        lake,
        "INSERT INTO ops.stream_batches (consumer_id, topic, kafka_partition, first_offset,"
        " last_offset, rows_valid, rows_invalid, min_event_time, max_event_time, duration_ms)"
        " VALUES ('test', 'transactions.raw', 0, 0, 9, 10, 0, '2025-02-10', '2025-02-10', 5)",
    )
    assert [t.name for t in plan_silver(registry)] == ["transactions_2025-02"]
    assert run_silver(lake, "all", "all", resume=True) == []
    assert [t.name for t in plan_gold(registry)] == ["transaction_features_2025-02"]

    # Un run en échec est replanifié même sans nouvelle donnée (reprise automatique).
    _sql(
        lake,
        "INSERT INTO ops.pipeline_runs (run_id, pipeline, task, status, finished_at)"
        " VALUES ('failed-run', 'silver', 'app_events_2025-01', 'failed', now())",
    )
    reasons = {t.name: t.reason for t in plan_silver(registry)}
    assert reasons == {"app_events_2025-01": "jamais réussie ou dernier run en échec"}
    assert run_silver(lake, "all", "all", resume=True) == []
    assert run_gold(lake, "all", resume=True) == []

    # Déclarations de fraude reçues en avril : labels à refaire pour février à avril seulement.
    _sql(
        lake,
        "INSERT INTO ops.stream_batches (consumer_id, topic, kafka_partition, first_offset,"
        " last_offset, rows_valid, rows_invalid, min_event_time, max_event_time, duration_ms)"
        " VALUES ('test', 'fraud-reports.raw', 0, 0, 0, 1, 0, '2025-04-02', '2025-04-02', 5)",
    )
    assert [t.name for t in plan_silver(registry)] == ["fraud_labels"]
    assert run_silver(lake, "all", "all", resume=True) == []
    assert [t.name for t in plan_gold(registry)] == ["transaction_features_2025-02"]
    assert run_gold(lake, "all", resume=True) == []


def test_daily_kpis_match_gold(lake):
    run_kpis(lake, "all", resume=True)
    gold = duckdb.sql(
        "SELECT count(*), count(*) FILTER (WHERE is_fraud), sum(amount) FROM read_parquet("
        f"'{lake.uri('gold', 'transaction_features')}/**/*.parquet')"
    ).fetchone()
    kpis = _sql(
        lake,
        "SELECT sum(transactions), sum(frauds), sum(amount_eur), count(DISTINCT day)"
        " FROM analytics.daily_kpis WHERE channel = 'all'",
    )[0]
    assert (kpis[0], kpis[1], kpis[2]) == gold
    assert kpis[3] >= 58  # janvier et février
    by_channel = _sql(
        lake, "SELECT sum(transactions) FROM analytics.daily_kpis WHERE channel <> 'all'"
    )[0][0]
    assert by_channel == gold[0]


def test_abnormal_access_is_detected_once(lake):
    sunday_night = datetime(2026, 9, 13, 1, 30, tzinfo=UTC)  # 3 h 30 à Paris, un dimanche
    rows = [
        ("intrus", "read", "s3://bronze/referentials/clients", "PRC-001", {}),
        ("analyste.fraude", "read", "s3://silver/transactions", "PRC-003", {}),
        ("dpo", "export", "s3://bronze/transactions", "RGPD-DROITS", {"rows": 25000}),
    ] + [("svc-ingestion", "ingest", f"s3://bronze/f{i}", "PRC-001", {}) for i in range(60)]
    with psycopg.connect(lake.pg_dsn, autocommit=True) as con, con.cursor() as cur:
        cur.executemany(
            "INSERT INTO audit.access_log (occurred_at, actor, action, resource, purpose, details)"
            " VALUES (%s, %s, %s, %s, %s, %s)",
            [(sunday_night, a, b, c, d, json.dumps(e)) for a, b, c, d, e in rows],
        )
    now = sunday_night + timedelta(minutes=20)
    found = {(i["rule"], i["actor"]) for i in detect_abnormal_access(lake, now=now)}
    assert found == {
        ("unknown_actor", "intrus"),
        ("purpose_not_allowed", "analyste.fraude"),
        ("off_hours_access", "analyste.fraude"),
        ("off_hours_access", "dpo"),
        ("mass_export", "dpo"),
        ("volume_spike", "svc-ingestion"),
    }
    assert detect_abnormal_access(lake, now=now) == []  # pas de doublon au passage suivant
    metrics = collect(lake)
    assert metrics["security_incidents_open"]["high"] == 3
    assert metrics["security_incidents_open"]["medium"] == 3


def test_landing_retention_keeps_recent_files_and_stays_in_landing(lake, tmp_path):
    outside = tmp_path / "outside.csv.gz"
    outside.write_bytes(b"x")
    assert purge_landing(lake, retention_days=7)["files"] == 0  # tout vient d'être ingéré
    _sql(
        lake,
        "INSERT INTO ops.file_ingestions (source_path, sha256, dataset, rows_expected,"
        " rows_loaded, rows_rejected, target_uri, run_id, ingested_at)"
        " SELECT %s, 'x', 'transactions', 0, 0, 0, 'x', run_id, now() - INTERVAL '30 days'"
        " FROM ops.pipeline_runs LIMIT 1",
        (os.path.relpath(outside, lake.landing_root),),
    )
    _sql(
        lake,
        "UPDATE ops.file_ingestions SET ingested_at = now() - INTERVAL '10 days'"
        " WHERE dataset = 'merchants'",
    )
    merchants = [p for p in lake.landing_root.rglob("merchants_*") if p.is_file()]
    result = purge_landing(lake, retention_days=7)
    assert result["files"] == len(merchants) > 0
    assert not any(p.exists() for p in merchants)
    assert outside.exists()  # jamais de suppression hors de la zone d'arrivée
    assert (lake.landing_root / "_manifests").exists()
    assert ingest_dataset(lake, "merchants")["files"] == 0  # pas de réingestion


def test_data_subject_requests(lake):
    client = _sql_client_with_history(lake)
    tx_rows = _parquet(lake, "bronze", "transactions", f"client_id = '{client}'")
    bronze_total = _parquet(lake, "bronze", "transactions")
    # Rejet d'ingestion contenant l'identifiant dans une ligne brute (sans colonne client_id).
    quarantine = Path(lake.uri("quarantine", "transactions", "manual_rejects.parquet"))
    quarantine.parent.mkdir(parents=True, exist_ok=True)
    duckdb.sql(
        f"COPY (SELECT 'T1,{client},12.5' AS csv_line UNION ALL SELECT 'T2,C99999999,1.0')"
        f" TO '{quarantine}' (FORMAT parquet)"
    )
    _sql(
        lake,
        "INSERT INTO governance.data_subject_requests (request_id, client_id, request_type,"
        " approved_by, received_at) VALUES"
        " ('DSR-1', %(c)s, 'access', NULL, now() - INTERVAL '3 days'),"
        " ('DSR-2', %(c)s, 'objection', NULL, now() - INTERVAL '2 days'),"
        " ('DSR-3', 'C00000001', 'erasure', NULL, now() - INTERVAL '2 days'),"
        " ('DSR-4', %(c)s, 'erasure', 'dpo', now() - INTERVAL '1 day'),"
        " ('DSR-5', %(c)s, 'rectification', NULL, now())",
        {"c": client},
    )
    summary = process_requests(lake)
    assert summary == {"completed": 3, "awaiting_approval": 1, "manual": 1, "failed": 0}

    # Accès : export JSON complet et manifeste.
    export = Path(lake.uri("exports", "dsr", "DSR-1"))
    exported = json.loads((export / "bronze_transactions.json").read_text(encoding="utf-8"))
    assert len(exported) == tx_rows and {r["client_id"] for r in exported} == {client}
    manifest = json.loads((export / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["rows_by_dataset"]["bronze_transactions"] == tx_rows

    # Opposition puis effacement : plus rien du client, rien d'autre de touché.
    assert _parquet(lake, "bronze", "transactions") == bronze_total - tx_rows
    for layer, dataset in [
        ("bronze", "referentials/clients"),
        ("bronze", "app_events"),
        ("silver", "transactions"),
        ("silver", "clients"),
        ("gold", "transaction_features"),
        ("gold", "first_seen"),
    ]:
        assert _parquet(lake, layer, dataset, f"client_id = '{client}'") == 0, dataset
    assert duckdb.sql(f"SELECT count(*) FROM '{quarantine}'").fetchone()[0] == 1
    assert (
        _sql(lake, "SELECT count(*) FROM governance.consents WHERE client_id = %s", (client,))[0][0]
        == 0
    )
    statuses = dict(_sql(lake, "SELECT request_id, status FROM governance.data_subject_requests"))
    assert statuses == {
        "DSR-1": "completed",
        "DSR-2": "completed",
        "DSR-3": "received",
        "DSR-4": "completed",
        "DSR-5": "received",
    }
    actions = _sql(lake, "SELECT action, count(*) FROM governance.data_subject_actions GROUP BY 1")
    assert dict(actions).keys() == {"export", "consent_withdrawn", "erase"}
    audited = _sql(lake, "SELECT action FROM audit.access_log WHERE actor = 'svc-rgpd'")
    assert {a for (a,) in audited} == {"export", "consent_withdrawn", "erase"}
    # Le service RGPD agit dans le cadre de sa finalité : aucun incident de sécurité créé.
    assert not [i for i in detect_abnormal_access(lake) if i["actor"] == "svc-rgpd"]
    assert collect(lake)["dsr_awaiting_approval"][""] == 1


def _sql_client_with_history(settings):
    """Client ayant des transactions, des messages au support et une déclaration de fraude."""
    return duckdb.sql(
        f"""
        SELECT r.client_id FROM read_parquet('{settings.uri("bronze", "fraud_reports")}/**/*.parquet') r
        JOIN read_parquet('{settings.uri("bronze", "support_tickets")}/**/*.parquet') t USING (client_id)
        GROUP BY 1 ORDER BY 1 LIMIT 1
        """
    ).fetchone()[0]
