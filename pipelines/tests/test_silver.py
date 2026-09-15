import duckdb
import psycopg
import pytest
from ondine_pipelines.ingest.landing import DATASETS, ingest_dataset
from ondine_pipelines.quality import QualityError
from ondine_pipelines.registry import Registry
from ondine_pipelines.silver.activity import build_month
from ondine_pipelines.silver.labels import build_labels
from ondine_pipelines.silver.referentials import build_referentials
from ondine_pipelines.silver.support import build_support_month


@pytest.fixture(scope="module")
def silver(base_settings):
    for dataset in DATASETS:
        ingest_dataset(base_settings, dataset)
    registry = Registry(base_settings)
    out = {"referentials": build_referentials(base_settings), "labels": build_labels(base_settings)}
    for period in registry.ingested_periods("transactions"):
        y, m = int(period[:4]), int(period[5:])
        out[f"tx-{period}"] = build_month(base_settings, "transactions", y, m)
        out[f"ev-{period}"] = build_month(base_settings, "app_events", y, m)
        out[f"support-{period}"] = build_support_month(base_settings, y, m)
    return out


def _read(settings, dataset):
    return duckdb.sql(
        f"SELECT * FROM read_parquet('{settings.uri('silver', dataset)}/**/*.parquet')"
    ).df()


def test_no_row_is_lost(silver, base_settings):
    with psycopg.connect(base_settings.pg_dsn) as con:
        failed = con.execute(
            "SELECT count(*) FROM ops.quality_results WHERE layer = 'silver'"
            " AND severity = 'error' AND NOT passed"
        ).fetchone()[0]
        bronze = con.execute(
            "SELECT sum(rows_loaded) FROM ops.file_ingestions WHERE dataset = 'transactions'"
        ).fetchone()[0]
    assert failed == 0
    assert len(_read(base_settings, "transactions")) == bronze


def test_personal_data_is_minimised(silver, base_settings):
    clients = _read(base_settings, "clients")
    assert not {"first_name", "last_name", "email", "phone", "iban", "birth_date"} & set(
        clients.columns
    )
    assert set(clients["age_band"]) <= {"18-25", "26-35", "36-50", "51-65", "66+"}
    tx = _read(base_settings, "transactions")
    assert not {"device_id", "ip_address", "beneficiary_iban"} & set(tx.columns)
    assert tx["device_hash"].dropna().str.fullmatch(r"[0-9a-f]{32}").all()


def test_pseudonyms_are_stable_and_joinable(silver, base_settings):
    tx = _read(base_settings, "transactions")
    events = _read(base_settings, "app_events")
    # Le même appareil a le même pseudonyme dans les transactions et dans les événements.
    assert len(set(tx["device_hash"].dropna()) & set(events["device_hash"].dropna())) > 100


def test_pseudonyms_depend_on_secret_key(base_settings):
    from ondine_pipelines.lake import enable_pseudonymisation

    a, b = duckdb.connect(), duckdb.connect()
    enable_pseudonymisation(a, "clé-1")
    enable_pseudonymisation(b, "clé-2")
    assert (
        a.sql("SELECT pseudo('dev_123')").fetchone() != b.sql("SELECT pseudo('dev_123')").fetchone()
    )


def test_json_is_flattened_with_schema_evolution(silver, base_settings):
    events = _read(base_settings, "app_events")
    assert {"device_os", "app_major_version", "sca_method", "geo_latitude"} <= set(events.columns)
    assert events["biometric_enabled"].isna().any() and events["biometric_enabled"].notna().any()
    assert (events["geo_latitude"].dropna().round(2) == events["geo_latitude"].dropna()).all()


def test_emails_are_masked(silver, base_settings):
    messages = _read(base_settings, "support_messages")
    text = " ".join(messages["body_masked"])
    assert "FR76" not in text and "+336" not in text and "@" not in text
    assert "[NOM]" in text
    assert messages["pii_phone"].sum() + messages["pii_iban"].sum() > 0  # il y en avait bien


def test_consents_are_synchronised(silver, base_settings):
    with psycopg.connect(base_settings.pg_dsn) as con:
        n = con.execute(
            "SELECT count(*) FROM governance.consents WHERE purpose = 'PRC-003'"
        ).fetchone()[0]
    assert n == 5_000


def test_invalid_rows_are_quarantined_and_blocking_checks_stop_publication(
    fresh_settings, tmp_path
):
    from dataclasses import replace

    import pandas as pd

    # Bronze fabriqué : 1 transaction valide et 1 sans montant.
    settings = replace(fresh_settings, lake_root=str(tmp_path / "lake"))
    target = tmp_path / "lake" / "bronze" / "transactions" / "year=2025" / "month=03"
    target.mkdir(parents=True)
    con = duckdb.connect()
    frame = pd.DataFrame(
        {
            "transaction_id": ["a", "b"],
            "event_time": pd.to_datetime(["2025-03-02", "2025-03-03"], utc=True),
            "client_id": ["C1", "C2"],
            "card_id": ["K1", "K2"],
            "merchant_id": ["M1", "M2"],
            "mcc": [5411, 5411],
            "merchant_country": ["FR", "FR"],
            "channel": ["pos_chip", "pos_chip"],
            "entry_mode": ["chip", "chip"],
            "amount": [10.0, None],
            "currency": ["EUR", "EUR"],
            "three_ds_authenticated": pd.Series([None, None], dtype="boolean"),
            "device_id": pd.Series([None, None], dtype="string"),
            "ip_address": pd.Series([None, None], dtype="string"),
            "latitude": [48.8, 48.8],
            "longitude": [2.3, 2.3],
            "beneficiary_iban": pd.Series([None, None], dtype="string"),
            "auth_status": ["approved", "approved"],
            "decline_reason": pd.Series([None, None], dtype="string"),
            "wallet_type": pd.Series([None, None], dtype="string"),
            "_source_file": ["f", "f"],
            "_source_sha256": ["s", "s"],
            "_run_id": ["r", "r"],
            "_ingested_at": pd.to_datetime(["2025-03-04", "2025-03-04"], utc=True),
        }
    )
    con.register("frame", frame)
    con.execute(f"COPY frame TO '{target / 'x.parquet'}' (FORMAT parquet)")
    # 1 ligne invalide sur 2 : taux de rejet 50 % > 0,1 % -> publication bloquée.
    with pytest.raises(QualityError):
        build_month(settings, "transactions", 2025, 3)
    assert (
        tmp_path
        / "lake"
        / "quarantine"
        / "transactions"
        / "year=2025"
        / "month=03"
        / "invalid.parquet"
    ).exists()
    assert not (tmp_path / "lake" / "silver" / "transactions" / "year=2025" / "month=03").exists()


def test_name_masking_avoids_false_positives_and_handles_accents():
    from ondine_pipelines.silver.support import mask_names

    text, residual = mask_names("Madame, Monsieur,\nCordialement,\nRémy Adam", "Rémy", "Adam")
    assert text == "Madame, Monsieur,\nCordialement,\n[NOM] [NOM]" and not residual
    text, _ = mask_names("Hervé\nHervé, hervé", "Emmanuel", "Hervé")
    assert "Hervé" not in text and "hervé" not in text
    text, _ = mask_names("Signé Jean-Pierre Le Roux", "Jean-Pierre", "Le Roux")
    assert text == "Signé [NOM] [NOM]"
