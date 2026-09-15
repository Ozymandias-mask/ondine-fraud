"""Transactions et événements applicatifs silver, traités mois par mois.

Le traitement d'un mois est indépendant des autres : les mois peuvent être calculés en
parallèle (montée en charge horizontale) et un mois en échec peut être relancé seul.
"""

from __future__ import annotations

import logging

import duckdb

from .. import quality
from ..lake import connect, enable_pseudonymisation, month_glob
from ..registry import Registry, Run
from ..settings import Settings
from .common import count, quarantine, require_input, validate_and_publish

logger = logging.getLogger(__name__)

CHANNELS = ("pos_chip", "pos_contactless", "ecommerce", "atm", "sepa_instant")
CARD_CHANNELS = ("pos_chip", "pos_contactless", "ecommerce", "atm")
EVENT_TYPES = (
    "app_open",
    "balance_view",
    "card_settings_view",
    "statement_download",
    "profile_update",
    "notification_opened",
    "login",
    "login_failed",
    "sca_challenge",
    "password_reset",
    "beneficiary_added",
    "sca_method_changed",
    "wallet_provisioning",
)


def _in_list(values: tuple[str, ...]) -> str:
    return "(" + ", ".join(f"'{v}'" for v in values) + ")"


# La partition silver correspond au mois de livraison. Une tolérance d'un jour est admise
# autour du mois : une connexion à 23 h 58 le 31 peut précéder un paiement livré le 1er.
def _load_month(
    con: duckdb.DuckDBPyConnection,
    settings: Settings,
    dataset: str,
    key: str,
    year: int,
    month: int,
    run: Run,
) -> None:
    """Charge le mois bronze et le dédoublonne (première ingestion conservée)."""
    pattern = month_glob(settings, "bronze", dataset, year, month)
    require_input(con, pattern)
    con.execute(
        f"CREATE OR REPLACE TEMP TABLE bronze_rows AS SELECT * FROM read_parquet('{pattern}',"
        " union_by_name = true)"
    )
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE deduplicated AS
        SELECT * EXCLUDE (rn) FROM (
            SELECT *, row_number() OVER (PARTITION BY {key} ORDER BY _ingested_at) AS rn
            FROM bronze_rows)
        WHERE rn = 1
    """)
    run.rows_in += count(con, "bronze_rows")
    run.inputs.append({"uri": pattern, "rows": count(con, "bronze_rows")})


def _transactions(
    con: duckdb.DuckDBPyConnection,
    settings: Settings,
    registry: Registry,
    run: Run,
    year: int,
    month: int,
) -> None:
    _load_month(con, settings, "transactions", "transaction_id", year, month, run)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE validated AS
        SELECT *,
            CASE
                WHEN transaction_id IS NULL THEN 'missing_transaction_id'
                WHEN event_time IS NULL THEN 'missing_event_time'
                WHEN event_time < TIMESTAMPTZ '{year}-{month:02d}-01' - INTERVAL 1 DAY
                     OR event_time >= TIMESTAMPTZ '{year}-{month:02d}-01'
                                      + INTERVAL 1 MONTH + INTERVAL 1 DAY
                    THEN 'outside_period'
                WHEN client_id IS NULL THEN 'missing_client_id'
                WHEN amount IS NULL OR amount <= 0 THEN 'invalid_amount'
                WHEN channel NOT IN {_in_list(CHANNELS)} THEN 'unknown_channel'
                WHEN channel IN {_in_list(CARD_CHANNELS)} AND card_id IS NULL THEN 'card_missing'
                WHEN channel = 'sepa_instant' AND beneficiary_iban IS NULL
                    THEN 'beneficiary_missing'
                WHEN auth_status NOT IN ('approved', 'declined') THEN 'invalid_auth_status'
            END AS dq_reason
        FROM deduplicated
    """)
    quarantine(
        con,
        settings,
        run,
        "SELECT * FROM validated WHERE dq_reason IS NOT NULL",
        settings.uri(
            "quarantine", "transactions", f"year={year}", f"month={month:02d}", "invalid.parquet"
        ),
    )
    # Pseudonymisation des identifiants indirects ; seuls les attributs utiles à la
    # détection sont conservés (pays et banque du bénéficiaire, pas son IBAN).
    con.execute("""
        CREATE OR REPLACE TEMP TABLE silver_rows AS
        SELECT
            transaction_id, event_time,
            CAST(event_time AS DATE) AS event_date,
            hour(event_time) AS event_hour,
            isodow(event_time) >= 6 AS is_weekend,
            client_id, card_id, merchant_id, mcc, merchant_country, channel, entry_mode,
            amount, currency, three_ds_authenticated,
            pseudo(device_id) AS device_hash,
            pseudo(ip_address) AS ip_hash,
            latitude AS merchant_latitude, longitude AS merchant_longitude,
            pseudo(beneficiary_iban) AS beneficiary_hash,
            left(beneficiary_iban, 2) AS beneficiary_country,
            substr(beneficiary_iban, 5, 5) AS beneficiary_bank_code,
            auth_status, decline_reason, wallet_type,
            _source_file, _run_id AS _bronze_run_id
        FROM validated WHERE dq_reason IS NULL
    """)
    checks = [
        quality.Check(
            "completeness",
            "SELECT (SELECT count(*) FROM bronze_rows) - (SELECT count(*) FROM silver_rows)"
            " - (SELECT count(*) FROM validated WHERE dq_reason IS NOT NULL)"
            " - ((SELECT count(*) FROM bronze_rows) - (SELECT count(*) FROM deduplicated))",
            "==",
            0,
            description="bronze = silver + quarantaine + doublons : aucune ligne perdue",
        ),
        quality.Check(
            "invalid_rate",
            "SELECT (SELECT count(*) FROM validated WHERE dq_reason IS NOT NULL)"
            " / greatest((SELECT count(*) FROM validated), 1)",
            "<=",
            0.001,
        ),
        quality.Check(
            "duplicate_rate",
            "SELECT 1 - (SELECT count(*) FROM deduplicated)"
            " / greatest((SELECT count(*) FROM bronze_rows), 1)",
            "<=",
            0.001,
            severity="warning",
        ),
        quality.Check(
            "pos_without_merchant",
            "SELECT count(*) FROM silver_rows WHERE channel IN"
            " ('pos_chip','pos_contactless','atm') AND merchant_id IS NULL",
            "==",
            0,
            severity="warning",
        ),
        quality.Check(
            "amount_p999_plausible",
            "SELECT quantile_cont(amount, 0.999) FROM silver_rows",
            "<=",
            20000,
            severity="warning",
            description="Montants extrêmes : dérive de la source ou erreur d'unité",
        ),
        quality.Check(
            "no_raw_identifier",
            "SELECT count(*) FROM (DESCRIBE silver_rows) WHERE"
            " column_name IN ('device_id','ip_address','beneficiary_iban')",
            "==",
            0,
            description="Pseudonymisation vérifiée sur le schéma publié",
        ),
        quality.Check("volume_not_empty", "SELECT count(*) FROM silver_rows", ">=", 1),
    ]
    validate_and_publish(
        con,
        settings,
        registry,
        run,
        "transactions",
        "silver_rows",
        settings.uri(
            "silver", "transactions", f"year={year}", f"month={month:02d}", "transactions.parquet"
        ),
        checks,
        order_by="event_time",
    )


def _events(
    con: duckdb.DuckDBPyConnection,
    settings: Settings,
    registry: Registry,
    run: Run,
    year: int,
    month: int,
) -> None:
    _load_month(con, settings, "app_events", "event_id", year, month, run)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE validated AS
        SELECT *,
            CASE
                WHEN event_id IS NULL THEN 'missing_event_id'
                WHEN event_time IS NULL THEN 'missing_event_time'
                WHEN event_time < TIMESTAMPTZ '{year}-{month:02d}-01' - INTERVAL 1 DAY
                     OR event_time >= TIMESTAMPTZ '{year}-{month:02d}-01'
                                      + INTERVAL 1 MONTH + INTERVAL 1 DAY
                    THEN 'outside_period'
                WHEN client_id IS NULL THEN 'missing_client_id'
                WHEN event_type IS NULL THEN 'missing_event_type'
            END AS dq_reason
        FROM deduplicated
    """)
    quarantine(
        con,
        settings,
        run,
        "SELECT * FROM validated WHERE dq_reason IS NOT NULL",
        settings.uri(
            "quarantine", "app_events", f"year={year}", f"month={month:02d}", "invalid.parquet"
        ),
    )
    # Aplatissement du JSON : chaque champ utile devient une colonne typée ; les champs
    # absents (anciennes versions de l'application, géolocalisation refusée) valent NULL.
    con.execute("""
        CREATE OR REPLACE TEMP TABLE silver_rows AS
        SELECT
            event_id, event_type, event_time, client_id,
            pseudo(session->>'device_id') AS device_hash,
            session->>'device_os' AS device_os,
            session->>'app_version' AS app_version,
            TRY_CAST(split_part(session->>'app_version', '.', 1) AS INTEGER) AS app_major_version,
            pseudo(session->>'ip_address') AS ip_hash,
            session->>'ip_country' AS ip_country,
            TRY_CAST(session->>'biometric_enabled' AS BOOLEAN) AS biometric_enabled,
            -- Géolocalisation arrondie à 2 décimales (~1 km) : suffisant pour la détection.
            round(TRY_CAST(geo->>'lat' AS DOUBLE), 2) AS geo_latitude,
            round(TRY_CAST(geo->>'lon' AS DOUBLE), 2) AS geo_longitude,
            TRY_CAST(geo->>'accuracy_m' AS INTEGER) AS geo_accuracy_m,
            payload->>'result' AS result,
            payload->>'method' AS sca_method,
            TRY_CAST(payload->>'amount_eur' AS DECIMAL(12, 2)) AS sca_amount,
            payload->>'wallet' AS wallet,
            payload->>'new_method' AS new_sca_method,
            _source_file, _run_id AS _bronze_run_id
        FROM validated WHERE dq_reason IS NULL
    """)
    checks = [
        quality.Check(
            "completeness",
            "SELECT (SELECT count(*) FROM bronze_rows) - (SELECT count(*) FROM silver_rows)"
            " - (SELECT count(*) FROM validated WHERE dq_reason IS NOT NULL)"
            " - ((SELECT count(*) FROM bronze_rows) - (SELECT count(*) FROM deduplicated))",
            "==",
            0,
        ),
        quality.Check(
            "invalid_rate",
            "SELECT (SELECT count(*) FROM validated WHERE dq_reason IS NOT NULL)"
            " / greatest((SELECT count(*) FROM validated), 1)",
            "<=",
            0.001,
        ),
        quality.Check(
            "unknown_event_type",
            f"SELECT count(*) FROM silver_rows WHERE event_type NOT IN {_in_list(EVENT_TYPES)}",
            "==",
            0,
            severity="warning",
            description="Nouveau type d'événement émis par l'application : à cataloguer",
        ),
        quality.Check(
            "device_present",
            "SELECT avg(CAST(device_hash IS NOT NULL AS INTEGER)) FROM silver_rows",
            ">=",
            0.99,
        ),
        quality.Check(
            "geo_precision_reduced",
            "SELECT count(*) FROM silver_rows WHERE"
            " geo_latitude IS NOT NULL AND round(geo_latitude, 2) <> geo_latitude",
            "==",
            0,
        ),
    ]
    validate_and_publish(
        con,
        settings,
        registry,
        run,
        "app_events",
        "silver_rows",
        settings.uri(
            "silver", "app_events", f"year={year}", f"month={month:02d}", "app_events.parquet"
        ),
        checks,
        order_by="event_time",
    )


BUILDERS = {"transactions": _transactions, "app_events": _events}


def build_month(
    settings: Settings, dataset: str, year: int, month: int, orchestrator_id: str | None = None
) -> dict[str, int]:
    """Construit un mois silver pour `transactions` ou `app_events`."""
    if dataset not in BUILDERS:
        raise ValueError(f"Jeu de données silver mensuel inconnu : {dataset}")
    registry = Registry(settings)
    con = connect(settings)
    enable_pseudonymisation(con, settings.pseudonymisation_key)
    try:
        with registry.run("silver", f"{dataset}_{year}-{month:02d}", orchestrator_id) as run:
            BUILDERS[dataset](con, settings, registry, run, year, month)
    finally:
        con.close()
    return {"rows": run.rows_out, "rejected": run.rows_rejected}
