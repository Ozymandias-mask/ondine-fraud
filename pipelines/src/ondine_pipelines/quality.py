"""Contrôles qualité déclaratifs, exécutés en SQL sur les tables produites.

Chaque contrôle calcule une valeur observée comparée à un seuil. Les résultats sont
enregistrés (ops.quality_results) et exposés en métriques. Un contrôle de sévérité
« error » en échec bloque la publication : les couches suivantes ne consomment jamais
une donnée non conforme.
"""

from __future__ import annotations

import logging
import operator
from dataclasses import dataclass

import duckdb

logger = logging.getLogger(__name__)

_OPERATORS = {"<=": operator.le, ">=": operator.ge, "==": operator.eq}


class QualityError(RuntimeError):
    """Au moins un contrôle bloquant a échoué."""


@dataclass(frozen=True)
class Check:
    """Contrôle : requête SQL renvoyant une valeur numérique, comparée à un seuil."""

    name: str
    sql: str
    op: str
    threshold: float
    severity: str = "error"
    description: str = ""


def run_checks(con: duckdb.DuckDBPyConnection, checks: list[Check]) -> list[dict[str, object]]:
    """Exécute les contrôles et retourne leurs résultats (sans lever d'exception)."""
    results = []
    for check in checks:
        value = con.execute(check.sql).fetchone()[0]
        observed = float(value) if value is not None else 0.0
        passed = _OPERATORS[check.op](observed, check.threshold)
        results.append(
            {
                "name": check.name,
                "severity": check.severity,
                "passed": passed,
                "observed": round(observed, 6),
                "threshold": check.threshold,
                "details": {
                    "rule": f"{check.op} {check.threshold}",
                    "description": check.description,
                },
            }
        )
        log = logger.info if passed else logger.warning
        log(
            "Contrôle %s : %s (observé %s %s %s)",
            check.name,
            "OK" if passed else "ÉCHEC",
            round(observed, 6),
            check.op,
            check.threshold,
        )
    return results


def enforce(dataset: str, results: list[dict[str, object]]) -> None:
    """Lève QualityError si un contrôle bloquant a échoué."""
    failed = [r["name"] for r in results if r["severity"] == "error" and not r["passed"]]
    if failed:
        raise QualityError(f"{dataset} : contrôles bloquants en échec : {', '.join(failed)}")
