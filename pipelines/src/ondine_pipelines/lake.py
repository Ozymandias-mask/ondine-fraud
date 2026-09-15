"""Accès au data lake avec DuckDB (lecture et écriture de Parquet, local ou S3)."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

import duckdb

from .settings import Settings

logger = logging.getLogger(__name__)
T = TypeVar("T")


def _sql_literal(value: str) -> str:
    """Littéral SQL sûr (les apostrophes sont doublées)."""
    return "'" + value.replace("'", "''") + "'"


def connect(settings: Settings) -> duckdb.DuckDBPyConnection:
    """Ouvre une connexion DuckDB en mémoire configurée pour le data lake.

    Les ressources sont bornées (threads, mémoire) : au-delà, DuckDB déborde sur disque
    au lieu d'échouer, ce qui permet de traiter des volumes supérieurs à la RAM.
    """
    con = duckdb.connect(
        config={
            "threads": settings.duckdb_threads,
            "memory_limit": settings.duckdb_memory_limit,
        }
    )
    con.execute("SET TimeZone = 'UTC'")
    # L'ordre d'insertion n'a pas à être conservé (tri explicite à l'écriture) : DuckDB
    # peut alors traiter et écrire en flux, avec beaucoup moins de mémoire.
    con.execute("SET preserve_insertion_order = false")
    con.execute("SET temp_directory = '/tmp/duckdb_spill'")
    if settings.uses_s3:
        # Extension installée dans l'image Docker (aucun téléchargement à l'exécution).
        con.execute("LOAD httpfs")
        # Résilience réseau : une coupure brève du stockage (redémarrage, saturation) est
        # absorbée par des nouvelles tentatives espacées de façon exponentielle.
        con.execute("SET http_retries = 8")
        con.execute("SET http_retry_wait_ms = 1000")
        con.execute("SET http_retry_backoff = 2")
        con.execute("SET http_timeout = 120")
        con.execute(
            "CREATE OR REPLACE SECRET lake (TYPE s3, "
            f"KEY_ID {_sql_literal(settings.s3_access_key)}, "
            f"SECRET {_sql_literal(settings.s3_secret_key)}, "
            f"ENDPOINT {_sql_literal(settings.s3_endpoint)}, "
            "URL_STYLE 'path', USE_SSL false, REGION 'us-east-1')"
        )
    return con


def prepare_target(settings: Settings, uri: str) -> None:
    """Crée le dossier parent d'un fichier cible en mode local (inutile sur S3)."""
    if not settings.uses_s3:
        Path(uri).parent.mkdir(parents=True, exist_ok=True)


def copy_to_parquet(
    con: duckdb.DuckDBPyConnection, settings: Settings, query: str, uri: str
) -> int:
    """Écrit le résultat d'une requête dans un fichier Parquet (zstd). Retourne le nombre de lignes.

    Le nom du fichier cible est déterministe : relancer un traitement réécrit le même
    objet (idempotence), et l'écriture d'un objet S3 est atomique.
    """
    prepare_target(settings, uri)
    row = con.execute(
        f"COPY ({query}) TO {_sql_literal(uri)} "
        "(FORMAT parquet, COMPRESSION zstd, ROW_GROUP_SIZE 500000)"
    ).fetchone()
    return int(row[0]) if row else 0


def sql_literal(value: str) -> str:
    """Version publique de l'échappement des littéraux SQL."""
    return _sql_literal(value)


TRANSIENT_ERRORS = (duckdb.IOException, duckdb.HTTPException)


def with_retries(action: Callable[[], T], attempts: int = 3, wait_s: float = 20.0) -> T:
    """Relance une action après une erreur d'entrée-sortie transitoire (stockage indisponible).

    Les erreurs de logique ou de qualité ne sont jamais relancées : seules les erreurs réseau
    ou disque, susceptibles de disparaître d'elles-mêmes, le sont.
    """
    for attempt in range(1, attempts + 1):
        try:
            return action()
        except TRANSIENT_ERRORS as exc:
            if attempt == attempts:
                raise
            delay = wait_s * 2 ** (attempt - 1)
            logger.warning(
                "Erreur transitoire (%s), nouvel essai %d/%d dans %.0f s",
                exc,
                attempt + 1,
                attempts,
                delay,
            )
            time.sleep(delay)
    raise AssertionError("inaccessible")  # pragma: no cover


def enable_pseudonymisation(con: duckdb.DuckDBPyConnection, key: str) -> None:
    """Déclare la fonction SQL `pseudo(valeur)` : hachage SHA-256 avec clé secrète.

    Le même identifiant donne toujours le même pseudonyme (les jointures et les features
    restent possibles), mais sans la clé il est impossible de retrouver la valeur d'origine
    ni de recalculer le pseudonyme d'une valeur connue (attaque par dictionnaire).
    La clé est passée en paramètre lié : elle n'apparaît dans aucun texte de requête ni log.
    """
    con.execute("SET VARIABLE pseudo_key = ?", [key])
    con.execute(
        "CREATE OR REPLACE TEMP MACRO pseudo(v) AS CASE WHEN v IS NULL THEN NULL "
        "ELSE left(sha256(getvariable('pseudo_key') || '|' || CAST(v AS VARCHAR)), 32) END"
    )


def month_glob(settings: Settings, layer: str, dataset: str, year: int, month: int) -> str:
    """Motif des fichiers Parquet d'un mois (partitionnement year=/month=)."""
    return settings.uri(layer, dataset, f"year={year}", f"month={month:02d}", "*.parquet")


def exists(con: duckdb.DuckDBPyConnection, pattern: str) -> bool:
    """Vrai si au moins un fichier correspond au motif (local ou S3)."""
    return bool(con.execute("SELECT count(*) FROM glob(?)", [pattern]).fetchone()[0])
