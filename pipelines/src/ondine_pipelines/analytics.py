"""Indicateurs métier quotidiens (gold -> PostgreSQL analytics.daily_kpis).

Ils donnent aux équipes métier une vue directe de la fraude et de l'efficacité du moteur
de règles actuel (point de comparaison du futur modèle) : volumes, montants, fraudes
confirmées, fraudes stoppées, clients légitimes refusés à tort, délai de déclaration.

Le mois est remplacé en une transaction (suppression puis insertion) : un rafraîchissement
est idempotent et les tableaux de bord ne voient jamais un mois à moitié écrit.
"""

from __future__ import annotations

import psycopg

from .lake import connect, month_glob
from .registry import Registry
from .settings import Settings

KPI_SQL = """
SELECT
    event_date AS day,
    coalesce(channel, 'all') AS channel,
    count(*) AS transactions,
    sum(amount) AS amount_eur,
    count(*) FILTER (WHERE legacy_auth_status = 'declined') AS declined,
    count(*) FILTER (WHERE is_fraud) AS frauds,
    coalesce(sum(amount) FILTER (WHERE is_fraud), 0) AS fraud_amount_eur,
    count(*) FILTER (WHERE is_fraud AND legacy_auth_status = 'declined') AS frauds_declined,
    count(*) FILTER (WHERE NOT is_fraud AND legacy_auth_status = 'declined') AS legit_declined,
    round(avg(epoch(label_reported_at - event_time) / 3600) FILTER (WHERE is_fraud), 1)
        AS avg_report_delay_h
FROM read_parquet(?)
GROUP BY GROUPING SETS ((event_date, channel), (event_date))
ORDER BY 1, 2
"""

COLUMNS = (
    "day",
    "channel",
    "transactions",
    "amount_eur",
    "declined",
    "frauds",
    "fraud_amount_eur",
    "frauds_declined",
    "legit_declined",
    "avg_report_delay_h",
)


def refresh_daily_kpis(
    settings: Settings, year: int, month: int, orchestrator_id: str | None = None
) -> dict[str, int]:
    """Recalcule les indicateurs d'un mois à partir de gold et les publie dans PostgreSQL."""
    registry = Registry(settings)
    con = connect(settings)
    pattern = month_glob(settings, "gold", "transaction_features", year, month)
    try:
        with registry.run("analytics", f"daily_kpis_{year}-{month:02d}", orchestrator_id) as run:
            rows = con.execute(KPI_SQL, [pattern]).fetchall()
            run.rows_in = sum(r[2] for r in rows if r[1] == "all")
            run.inputs.append({"uri": pattern, "rows": run.rows_in})
            with psycopg.connect(settings.pg_dsn, connect_timeout=10) as pg, pg.cursor() as cur:
                cur.execute(
                    "DELETE FROM analytics.daily_kpis WHERE day >= make_date(%s, %s, 1)"
                    " AND day < make_date(%s, %s, 1) + INTERVAL '1 month'",
                    (year, month, year, month),
                )
                with cur.copy(f"COPY analytics.daily_kpis ({', '.join(COLUMNS)}) FROM STDIN") as cp:
                    for row in rows:
                        cp.write_row(row)
            run.rows_out = len(rows)
            run.outputs.append({"table": "analytics.daily_kpis", "rows": len(rows)})
    finally:
        con.close()
    return {"rows": len(rows), "transactions": run.rows_in}
