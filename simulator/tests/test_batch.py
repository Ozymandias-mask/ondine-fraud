import gzip
import hashlib
import json
import mailbox

import pandas as pd
import pytest
from ondine_sim.batch import run_batch
from ondine_sim.formatting import TRANSACTION_SCHEMA_V1, TRANSACTION_SCHEMA_V2


@pytest.fixture(scope="module")
def generated(tmp_path_factory, small_cfg):
    from dataclasses import replace

    cfg = replace(small_cfg, output_dir=tmp_path_factory.mktemp("batch") / "landing")
    summary = run_batch(cfg)
    return cfg, summary


def _read(cfg, dataset, month):
    files = sorted((cfg.output_dir / dataset / f"year=2025/month={month:02d}").glob("*.gz"))
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def test_run_summary(generated):
    _, summary = generated
    assert summary["failed_parts"] == []
    assert summary["transactions"] > 30_000
    assert 0.001 < summary["fraud_transactions"] / summary["transactions"] < 0.01


def test_schema_evolution(generated):
    cfg, _ = generated
    assert list(_read(cfg, "transactions", 1).columns) == TRANSACTION_SCHEMA_V1
    assert list(_read(cfg, "transactions", 2).columns) == TRANSACTION_SCHEMA_V2


def test_ground_truth_is_not_leaked_in_landing(generated):
    cfg, _ = generated
    tx = _read(cfg, "transactions", 1)
    assert not {"fraud_type", "incident_id", "is_fraud"} & set(tx.columns)
    assert tx["transaction_id"].is_unique
    assert (cfg.output_dir.parent / "ground_truth").exists()


def test_manifests_match_files(generated):
    cfg, _ = generated
    manifest = json.loads((cfg.output_dir / "_manifests/period=2025-01/part-000.json").read_text())
    for info in manifest["files"]:
        path = cfg.output_dir / info["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == info["sha256"]
        assert path.stat().st_size == info["bytes"]


def test_events_are_valid_json_with_variable_schema(generated):
    cfg, _ = generated
    path = next((cfg.output_dir / "app_events/year=2025/month=02").glob("*.jsonl.gz"))
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        events = [json.loads(line) for line in handle]
    assert {"event_id", "event_type", "event_time", "client_id", "session", "payload"} <= set(
        events[0]
    )
    assert any("biometric_enabled" in e["session"] for e in events)  # app v5
    assert any("biometric_enabled" not in e["session"] for e in events)  # app v4
    assert any("geo" not in e for e in events)  # géoloc refusée


def test_reports_arrive_after_the_transaction(generated):
    cfg, _ = generated
    reports = _read(cfg, "fraud_reports", 1)
    tx = _read(cfg, "transactions", 1)[["transaction_id", "event_time"]]
    merged = reports.merge(tx, on="transaction_id")
    assert len(merged) == len(reports)
    assert (pd.to_datetime(merged["reported_at"]) > pd.to_datetime(merged["event_time"])).all()
    assert set(reports["investigation_status"]) <= {"confirmed", "rejected"}


def test_support_emails_are_parseable_and_contain_personal_data(generated, tmp_path):
    cfg, _ = generated
    path = next((cfg.output_dir / "support_emails/year=2025/month=01").glob("*.mbox.gz"))
    raw = tmp_path / "support.mbox"
    raw.write_bytes(gzip.decompress(path.read_bytes()))
    messages = list(mailbox.mbox(raw))
    tickets = _read(cfg, "support_tickets", 1)
    assert len(messages) == len(tickets) > 0
    bodies = " ".join(m.get_payload(decode=True).decode("utf-8") for m in messages)
    assert "FR76" in bodies or "+336" in bodies


def test_resume_skips_completed_parts(generated):
    cfg, _ = generated
    summary = run_batch(cfg, resume=True)
    assert summary["transactions"] == 0


def test_generation_is_reproducible_across_worker_counts(cfg_in):
    one = cfg_in(months=1, workers=1)
    run_batch(one)
    first = gzip.decompress(next((one.output_dir / "transactions").rglob("*.gz")).read_bytes())
    from dataclasses import replace

    two = replace(one, output_dir=one.output_dir.parent / "again", workers=2)
    run_batch(two)
    second = gzip.decompress(next((two.output_dir / "transactions").rglob("*.gz")).read_bytes())
    assert first == second
