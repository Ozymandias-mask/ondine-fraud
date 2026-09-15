"""Outils partagés par les traitements silver."""

from __future__ import annotations

import duckdb

from .. import quality
from ..lake import copy_to_parquet, exists
from ..registry import Registry, Run
from ..settings import Settings


def count(con: duckdb.DuckDBPyConnection, table: str) -> int:
    """Nombre de lignes d'une table temporaire."""
    return int(con.execute(f"SELECT count(*) FROM {table}").fetchone()[0])


def validate_and_publish(
    con: duckdb.DuckDBPyConnection,
    settings: Settings,
    registry: Registry,
    run: Run,
    dataset: str,
    table: str,
    target: str,
    checks: list[quality.Check],
    order_by: str | None = None,
    layer: str = "silver",
) -> int:
    """Contrôle la table, enregistre les résultats puis, si tout est conforme, publie en Parquet.

    Les résultats qualité sont enregistrés même en cas d'échec (preuve et diagnostic).
    """
    results = quality.run_checks(con, checks)
    registry.record_quality(run, layer, dataset, results)
    quality.enforce(dataset, results)
    order = f" ORDER BY {order_by}" if order_by else ""
    written = copy_to_parquet(con, settings, f"SELECT * FROM {table}{order}", target)
    run.outputs.append({"uri": target, "rows": written})
    run.rows_out += written
    return written


def quarantine(
    con: duckdb.DuckDBPyConnection, settings: Settings, run: Run, query: str, target: str
) -> int:
    """Écrit les lignes invalides en quarantaine (seulement s'il y en a)."""
    n = int(con.execute(f"SELECT count(*) FROM ({query})").fetchone()[0])
    if n:
        copy_to_parquet(con, settings, query, target)
        run.outputs.append({"uri": target, "rows": n, "quarantine": True})
    run.rows_rejected += n
    return n


def require_input(con: duckdb.DuckDBPyConnection, pattern: str) -> None:
    """Échoue explicitement si la couche amont n'a encore rien produit."""
    if not exists(con, pattern):
        raise FileNotFoundError(f"Aucune donnée en entrée : {pattern}")
