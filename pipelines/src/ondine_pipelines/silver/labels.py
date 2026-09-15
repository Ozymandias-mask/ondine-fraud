"""Déclarations de fraude silver : les labels qui serviront à entraîner et évaluer le modèle."""

from __future__ import annotations

from .. import quality
from ..lake import connect
from ..registry import Registry
from ..settings import Settings
from .common import count, quarantine, require_input, validate_and_publish


def build_labels(settings: Settings, orchestrator_id: str | None = None) -> dict[str, int]:
    """Nettoie l'ensemble des déclarations (volume faible : traitement complet à chaque fois)."""
    registry = Registry(settings)
    con = connect(settings)
    pattern = settings.uri("bronze", "fraud_reports", "*", "*", "*.parquet")
    try:
        with registry.run("silver", "fraud_labels", orchestrator_id) as run:
            require_input(con, pattern)
            con.execute(f"CREATE TEMP TABLE bronze_rows AS SELECT * FROM read_parquet('{pattern}')")
            run.rows_in = count(con, "bronze_rows")
            con.execute("""
                CREATE TEMP TABLE validated AS
                SELECT *,
                    CASE
                        WHEN transaction_id IS NULL THEN 'missing_transaction_id'
                        WHEN reported_at IS NULL THEN 'missing_reported_at'
                        WHEN investigation_status NOT IN ('confirmed', 'rejected')
                            THEN 'unknown_investigation_status'
                    END AS dq_reason
                FROM (SELECT DISTINCT ON (report_id) * FROM bronze_rows
                      ORDER BY report_id, _ingested_at)
            """)
            quarantine(
                con,
                settings,
                run,
                "SELECT * FROM validated WHERE dq_reason IS NOT NULL",
                settings.uri("quarantine", "fraud_labels", "invalid.parquet"),
            )
            con.execute("""
                CREATE TEMP TABLE silver_rows AS
                SELECT report_id, case_id, transaction_id, client_id, reported_at,
                       report_channel, claim_category, investigation_status,
                       investigation_status = 'confirmed' AS is_confirmed_fraud
                FROM validated WHERE dq_reason IS NULL
            """)
            checks = [
                quality.Check(
                    "report_id_unique",
                    "SELECT count(*) - count(DISTINCT report_id) FROM silver_rows",
                    "==",
                    0,
                ),
                quality.Check(
                    "one_case_per_transaction",
                    "SELECT count(*) FROM (SELECT transaction_id FROM silver_rows"
                    " WHERE is_confirmed_fraud GROUP BY 1 HAVING count(DISTINCT case_id) > 1)",
                    "==",
                    0,
                    severity="warning",
                ),
                quality.Check(
                    "rejected_claim_share",
                    "SELECT avg(CAST(NOT is_confirmed_fraud AS INTEGER)) FROM silver_rows",
                    "<=",
                    0.2,
                    severity="warning",
                    description="Hausse anormale des contestations infondées",
                ),
            ]
            validate_and_publish(
                con,
                settings,
                registry,
                run,
                "fraud_labels",
                "silver_rows",
                settings.uri("silver", "fraud_labels", "fraud_labels.parquet"),
                checks,
                order_by="reported_at",
            )
    finally:
        con.close()
    return {"rows": run.rows_out, "rejected": run.rows_rejected}
