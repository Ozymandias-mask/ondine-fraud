"""Référentiels silver : clients minimisés, cartes, marchands, synchronisation des consentements."""

from __future__ import annotations

import logging

import duckdb
import psycopg

from .. import quality
from ..lake import connect
from ..registry import Registry, Run
from ..settings import Settings
from .common import count, require_input, validate_and_publish

logger = logging.getLogger(__name__)


def _latest_snapshot(con: duckdb.DuckDBPyConnection, settings: Settings, dataset: str) -> str:
    """Requête sur la photo la plus récente du référentiel (partition snapshot_date=...)."""
    pattern = settings.uri("bronze", "referentials", dataset, "*", "*.parquet")
    require_input(con, pattern)
    latest = max(row[0] for row in con.execute("SELECT file FROM glob(?)", [pattern]).fetchall())
    snapshot = latest.split("snapshot_date=")[1].split("/")[0]
    return f"(SELECT *, DATE '{snapshot}' AS snapshot_date FROM read_parquet('{latest}'))"


def _clients(
    con: duckdb.DuckDBPyConnection, settings: Settings, registry: Registry, run: Run
) -> None:
    source = _latest_snapshot(con, settings, "clients")
    con.execute(f"CREATE OR REPLACE TEMP TABLE bronze_clients AS SELECT * FROM {source}")
    run.rows_in += count(con, "bronze_clients")
    # Minimisation : ni nom, ni e-mail, ni téléphone, ni adresse, ni IBAN, ni date de naissance.
    # L'identifiant client est un identifiant technique interne : la correspondance avec
    # l'identité reste dans la couche bronze, à accès restreint (pseudonymisation RGPD).
    con.execute("""
        CREATE OR REPLACE TEMP TABLE silver_clients AS
        SELECT DISTINCT ON (client_id)
            client_id,
            CASE
                WHEN age < 26 THEN '18-25'
                WHEN age < 36 THEN '26-35'
                WHEN age < 51 THEN '36-50'
                WHEN age < 66 THEN '51-65'
                ELSE '66+'
            END AS age_band,
            region, city, segment, income_band, kyc_level, is_pep,
            date_diff('day', created_at, snapshot_date) AS tenure_days,
            snapshot_date
        FROM (SELECT *, date_sub('year', birth_date, snapshot_date) AS age
              FROM bronze_clients)
        ORDER BY client_id
    """)
    checks = [
        quality.Check(
            "completeness",
            "SELECT (SELECT count(DISTINCT client_id) FROM bronze_clients)"
            " - (SELECT count(*) FROM silver_clients)",
            "==",
            0,
        ),
        quality.Check(
            "client_id_not_null",
            "SELECT count(*) FROM silver_clients WHERE client_id IS NULL",
            "==",
            0,
        ),
        # Anomalie réelle détectée sur les données sources (clients de moins de 18 ans à la date
        # de la photo) : signalée à l'équipe KYC, sans bloquer la chaîne de traitement.
        quality.Check(
            "adult_clients_only",
            "SELECT count(*) FROM bronze_clients"
            " WHERE birth_date > snapshot_date - INTERVAL 18 YEAR",
            "==",
            0,
            severity="warning",
            description="Ouverture de compte réservée aux majeurs : anomalie à transmettre au KYC",
        ),
        quality.Check(
            "no_direct_identifier",
            "SELECT count(*) FROM (DESCRIBE silver_clients) WHERE column_name IN "
            "('first_name','last_name','email','phone','street_address','iban','birth_date')",
            "==",
            0,
            description="Minimisation RGPD vérifiée sur le schéma publié",
        ),
    ]
    validate_and_publish(
        con,
        settings,
        registry,
        run,
        "clients",
        "silver_clients",
        settings.uri("silver", "clients", "clients.parquet"),
        checks,
    )
    registry.audit(
        "svc-pipeline-silver",
        "transform",
        "bronze/referentials/clients",
        "PRC-001",
        {"rows": count(con, "bronze_clients"), "output": "silver/clients (minimisé)"},
    )


def _sync_consents(con: duckdb.DuckDBPyConnection, settings: Settings) -> int:
    """Recopie les consentements marketing dans le registre de gouvernance (source unique)."""
    rows = con.execute("""
        SELECT client_id, consent_marketing, coalesce(consent_marketing_at, created_at)
        FROM bronze_clients WHERE consent_marketing IS NOT NULL
    """).fetchall()
    with psycopg.connect(settings.pg_dsn, autocommit=True) as pg, pg.cursor() as cur:
        cur.execute("CREATE TEMP TABLE consents_in (client_id TEXT, granted BOOLEAN, updated DATE)")
        with cur.copy("COPY consents_in FROM STDIN") as copy:
            for row in rows:
                copy.write_row(row)
        cur.execute("""
            INSERT INTO governance.consents (client_id, purpose, granted, source, updated_at)
            SELECT client_id, 'PRC-003', granted, 'referential', updated FROM consents_in
            ON CONFLICT (client_id, purpose) DO UPDATE
            SET granted = EXCLUDED.granted, updated_at = EXCLUDED.updated_at
            WHERE governance.consents.granted IS DISTINCT FROM EXCLUDED.granted
        """)
    return len(rows)


def _cards(
    con: duckdb.DuckDBPyConnection, settings: Settings, registry: Registry, run: Run
) -> None:
    source = _latest_snapshot(con, settings, "cards")
    con.execute(f"CREATE OR REPLACE TEMP TABLE bronze_cards AS SELECT * FROM {source}")
    run.rows_in += count(con, "bronze_cards")
    # Le jeton de carte et les 4 derniers chiffres ne servent pas à la détection : retirés.
    con.execute("""
        CREATE OR REPLACE TEMP TABLE silver_cards AS
        SELECT DISTINCT ON (card_id) card_id, client_id, card_type, network, issued_at,
               contactless_enabled, status
        FROM bronze_cards ORDER BY card_id
    """)
    checks = [
        quality.Check(
            "card_id_unique", "SELECT count(*) - count(DISTINCT card_id) FROM bronze_cards", "==", 0
        ),
        quality.Check(
            "orphan_cards",
            "SELECT count(*) FROM silver_cards c ANTI JOIN silver_clients k USING (client_id)",
            "==",
            0,
            description="Toute carte appartient à un client",
        ),
        quality.Check(
            "known_card_type",
            "SELECT count(*) FROM silver_cards WHERE card_type NOT IN ('debit','credit','virtual')",
            "==",
            0,
        ),
    ]
    validate_and_publish(
        con,
        settings,
        registry,
        run,
        "cards",
        "silver_cards",
        settings.uri("silver", "cards", "cards.parquet"),
        checks,
    )


def _merchants(
    con: duckdb.DuckDBPyConnection, settings: Settings, registry: Registry, run: Run
) -> None:
    source = _latest_snapshot(con, settings, "merchants")
    con.execute(f"CREATE OR REPLACE TEMP TABLE bronze_merchants AS SELECT * FROM {source}")
    run.rows_in += count(con, "bronze_merchants")
    con.execute("""
        CREATE OR REPLACE TEMP TABLE silver_merchants AS
        SELECT DISTINCT ON (merchant_id) merchant_id, category, mcc, is_online, city, country,
               latitude, longitude
        FROM bronze_merchants ORDER BY merchant_id
    """)
    checks = [
        quality.Check(
            "merchant_id_unique",
            "SELECT count(*) - count(DISTINCT merchant_id) FROM bronze_merchants",
            "==",
            0,
        ),
        quality.Check(
            "valid_coordinates",
            "SELECT count(*) FROM silver_merchants WHERE"
            " latitude NOT BETWEEN -90 AND 90 OR longitude NOT BETWEEN -180 AND 180",
            "==",
            0,
        ),
        quality.Check(
            "country_iso2",
            "SELECT count(*) FROM silver_merchants WHERE"
            " NOT regexp_full_match(country, '[A-Z]{2}')",
            "==",
            0,
        ),
    ]
    validate_and_publish(
        con,
        settings,
        registry,
        run,
        "merchants",
        "silver_merchants",
        settings.uri("silver", "merchants", "merchants.parquet"),
        checks,
    )


def build_referentials(settings: Settings, orchestrator_id: str | None = None) -> dict[str, int]:
    """Construit les trois référentiels silver et synchronise les consentements."""
    registry = Registry(settings)
    con = connect(settings)
    try:
        with registry.run("silver", "referentials", orchestrator_id) as run:
            _clients(con, settings, registry, run)
            _cards(con, settings, registry, run)
            _merchants(con, settings, registry, run)
            consents = _sync_consents(con, settings)
            logger.info("%d consentements synchronisés dans governance.consents", consents)
    finally:
        con.close()
    return {"rows": run.rows_out, "consents": consents}
