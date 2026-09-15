"""Rétention de la zone d'arrivée : suppression des fichiers bruts déjà ingérés.

Minimisation (RGPD, article 5.1.e) : une fois converti en bronze et vérifié (empreinte
SHA-256 enregistrée), un fichier brut n'a plus d'utilité. Il est conservé quelques jours
pour permettre une réingestion en cas d'incident, puis supprimé. Les manifestes, sans
données personnelles, sont conservés pour la traçabilité.
"""

from __future__ import annotations

import logging

import psycopg

from ..registry import Registry
from ..settings import Settings

logger = logging.getLogger(__name__)


def purge_landing(settings: Settings, retention_days: int = 7) -> dict[str, int]:
    """Supprime les fichiers ingérés depuis plus de `retention_days` jours."""
    with psycopg.connect(settings.pg_dsn, connect_timeout=10) as pg:
        rows = pg.execute(
            "SELECT source_path FROM ops.file_ingestions"
            " WHERE ingested_at < now() - make_interval(days => %s)",
            (retention_days,),
        ).fetchall()
    root = settings.landing_root.resolve()
    removed, freed = 0, 0
    for (relative,) in rows:
        path = (root / relative).resolve()
        if root not in path.parents:  # chemin inattendu : jamais de suppression hors zone
            logger.warning("Chemin ignoré (hors zone d'arrivée) : %s", relative)
            continue
        if path.is_file():
            freed += path.stat().st_size
            path.unlink()
            removed += 1
    if removed:
        Registry(settings).audit(
            "svc-ingestion",
            "purge",
            str(settings.landing_root),
            "PRC-001",
            {"files": removed, "bytes": freed, "retention_days": retention_days},
        )
    logger.info("Zone d'arrivée : %d fichier(s) supprimé(s), %.1f Mo libérés", removed, freed / 1e6)
    return {"files": removed, "bytes": freed}
