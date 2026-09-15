"""Registre d'exécution et de lignage (PostgreSQL, schéma « ops »).

Chaque traitement est enregistré avec ses entrées, ses sorties, ses volumes et sa durée :
c'est la base de la traçabilité des données et du suivi des performances.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

import psycopg

from . import metrics
from .settings import Settings

logger = logging.getLogger(__name__)


@dataclass
class Run:
    """Exécution en cours d'un traitement ; les compteurs sont remplis par le traitement."""

    run_id: str
    pipeline: str
    task: str
    rows_in: int = 0
    rows_out: int = 0
    rows_rejected: int = 0
    inputs: list[dict[str, object]] = field(default_factory=list)
    outputs: list[dict[str, object]] = field(default_factory=list)
    status: str = "success"


class Registry:
    """Accès au schéma « ops »."""

    def __init__(self, settings: Settings) -> None:
        """Mémorise la chaîne de connexion (une connexion courte par opération)."""
        self.settings = settings

    def _connect(self) -> psycopg.Connection:
        return psycopg.connect(self.settings.pg_dsn, autocommit=True, connect_timeout=10)

    @contextmanager
    def run(self, pipeline: str, task: str, orchestrator_id: str | None = None) -> Iterator[Run]:
        """Enregistre le début, la fin, le statut et les volumes d'un traitement.

        En cas d'exception, le run est marqué « failed » avec le message d'erreur, les
        métriques d'échec sont publiées, puis l'exception est propagée (l'orchestrateur
        peut alors relancer la tâche).
        """
        run = Run(run_id=f"{pipeline}-{task}-{uuid.uuid4().hex[:12]}", pipeline=pipeline, task=task)
        started = time.perf_counter()
        with self._connect() as con:
            con.execute(
                "INSERT INTO ops.pipeline_runs (run_id, pipeline, task, orchestrator_id) "
                "VALUES (%s, %s, %s, %s)",
                (run.run_id, pipeline, task, orchestrator_id),
            )
        logger.info("Début %s", run.run_id)
        error: str | None = None
        try:
            yield run
        except Exception as exc:
            run.status = "failed"
            error = f"{type(exc).__name__}: {exc}"[:2000]
            raise
        finally:
            duration = time.perf_counter() - started
            rate = run.rows_out / duration if duration > 0 else None
            with self._connect() as con:
                con.execute(
                    "UPDATE ops.pipeline_runs SET finished_at = now(), status = %s, rows_in = %s,"
                    " rows_out = %s, rows_rejected = %s, duration_s = %s, rows_per_s = %s,"
                    " inputs = %s, outputs = %s, error = %s WHERE run_id = %s",
                    (
                        run.status,
                        run.rows_in,
                        run.rows_out,
                        run.rows_rejected,
                        round(duration, 3),
                        round(rate, 1) if rate else None,
                        json.dumps(run.inputs),
                        json.dumps(run.outputs),
                        error,
                        run.run_id,
                    ),
                )
            metrics.push_run(self.settings, run, duration)
            logger.info(
                "Fin %s : %s, %d lignes en %.1f s", run.run_id, run.status, run.rows_out, duration
            )

    def ingested_files(self) -> set[tuple[str, str]]:
        """Fichiers déjà ingérés, identifiés par (chemin, empreinte SHA-256)."""
        with self._connect() as con:
            rows = con.execute("SELECT source_path, sha256 FROM ops.file_ingestions").fetchall()
        return {(r[0], r[1]) for r in rows}

    def last_success(self, pipeline: str, task: str) -> bool:
        """Vrai si la dernière exécution d'une tâche a réussi (reprise sans tout refaire)."""
        with self._connect() as con:
            row = con.execute(
                "SELECT status FROM ops.pipeline_runs WHERE pipeline = %s AND task = %s"
                " AND status <> 'running' ORDER BY started_at DESC LIMIT 1",
                (pipeline, task),
            ).fetchone()
        return bool(row and row[0] == "success")

    def ingested_periods(self, dataset: str) -> list[str]:
        """Mois (AAAA-MM) pour lesquels des fichiers d'un jeu de données ont été ingérés."""
        with self._connect() as con:
            rows = con.execute(
                "SELECT DISTINCT period FROM ops.file_ingestions"
                " WHERE dataset = %s AND period IS NOT NULL ORDER BY 1",
                (dataset,),
            ).fetchall()
        return [r[0] for r in rows]

    def record_file(
        self,
        run: Run,
        source_path: str,
        sha256: str,
        dataset: str,
        period: str | None,
        rows_expected: int,
        rows_loaded: int,
        rows_rejected: int,
        target_uri: str,
    ) -> None:
        """Mémorise l'ingestion d'un fichier (lignage fichier source -> objet bronze)."""
        with self._connect() as con:
            con.execute(
                "INSERT INTO ops.file_ingestions (source_path, sha256, dataset, period,"
                " rows_expected, rows_loaded, rows_rejected, target_uri, run_id)"
                " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"
                " ON CONFLICT (source_path, sha256) DO UPDATE SET"
                " rows_loaded = EXCLUDED.rows_loaded, rows_rejected = EXCLUDED.rows_rejected,"
                " target_uri = EXCLUDED.target_uri,"
                " run_id = EXCLUDED.run_id, ingested_at = now()",
                (
                    source_path,
                    sha256,
                    dataset,
                    period,
                    rows_expected,
                    rows_loaded,
                    rows_rejected,
                    target_uri,
                    run.run_id,
                ),
            )

    def record_quality(self, run: Run, layer: str, dataset: str, results: list[dict]) -> None:
        """Enregistre les résultats des contrôles qualité d'un jeu de données."""
        with self._connect() as con, con.cursor() as cur:
            cur.executemany(
                "INSERT INTO ops.quality_results (run_id, layer, dataset, check_name,"
                " severity, passed, observed, threshold, details)"
                " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                [
                    (
                        run.run_id,
                        layer,
                        dataset,
                        r["name"],
                        r["severity"],
                        r["passed"],
                        r.get("observed"),
                        r.get("threshold"),
                        json.dumps(r.get("details", {})),
                    )
                    for r in results
                ],
            )

    def audit(
        self,
        actor: str,
        action: str,
        resource: str,
        purpose: str,
        details: dict[str, object] | None = None,
    ) -> None:
        """Ajoute une entrée au journal d'audit (accès à des données personnelles)."""
        with self._connect() as con:
            con.execute(
                "INSERT INTO audit.access_log (actor, action, resource, purpose, details)"
                " VALUES (%s, %s, %s, %s, %s)",
                (actor, action, resource, purpose, json.dumps(details or {})),
            )
