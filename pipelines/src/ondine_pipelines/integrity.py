"""Contrôle d'intégrité du data lake : vérifier que chaque fichier est lisible en entier.

Un arrêt brutal (coupure, machine à bout de mémoire, conteneur tué) peut laisser un fichier
Parquet tronqué. Il reste alors présent, de taille plausible, et n'échoue qu'au moment où on
le relit vraiment — souvent bien plus tard, dans un traitement qui n'a rien à voir. C'est le
pire mode de panne d'un data lake : silencieux à l'écriture, bruyant au plus mauvais moment.

Ce contrôle décode **toutes les colonnes de toutes les lignes** de chaque fichier — et non ses
seules métadonnées, qui restent valides sur un fichier tronqué — puis nomme celui qui pose
problème, avec le mois à recalculer. Il s'exécute sans rien modifier, et son coût est celui
d'une lecture complète du lac : c'est le prix d'une réponse fiable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .lake import connect
from .settings import LAYERS, Settings

logger = logging.getLogger(__name__)


@dataclass
class FileReport:
    """Résultat de la relecture d'un fichier du lac."""

    uri: str
    rows: int | None
    error: str | None

    @property
    def ok(self) -> bool:
        """Vrai si le fichier a pu être relu intégralement."""
        return self.error is None


def check_layer(settings: Settings, layer: str) -> list[FileReport]:
    """Relit chaque fichier Parquet d'une couche et retourne un rapport par fichier."""
    if layer not in LAYERS:
        raise ValueError(f"Couche inconnue : {layer}")
    con = connect(settings)
    reports: list[FileReport] = []
    try:
        pattern = settings.uri(layer, "**", "*.parquet")
        files = [
            r[0] for r in con.execute("SELECT file FROM glob(?) ORDER BY 1", [pattern]).fetchall()
        ]
        for uri in files:
            try:
                # `hash(COLUMNS(*))` force la lecture de **toutes** les colonnes, ligne par
                # ligne. Un simple `count(*)`, même dans une sous-requête, est ramené par
                # l'optimiseur à une lecture des métadonnées — lesquelles survivent
                # précisément aux troncatures que ce contrôle doit détecter.
                rows = con.execute(
                    "SELECT count(*) FROM (SELECT hash(COLUMNS(*)) FROM read_parquet(?))", [uri]
                ).fetchone()[0]
                reports.append(FileReport(uri, int(rows), None))
            except Exception as error:  # noqa: BLE001 — on veut le motif, quel qu'il soit
                reports.append(FileReport(uri, None, str(error).splitlines()[0][:300]))
                logger.error("Fichier illisible : %s — %s", uri, str(error).splitlines()[0])
    finally:
        con.close()
    return reports


def summary(reports: list[FileReport]) -> dict[str, object]:
    """Résumé lisible : volumétrie totale et liste des fichiers à recalculer."""
    broken = [r for r in reports if not r.ok]
    return {
        "fichiers": len(reports),
        "fichiers_illisibles": len(broken),
        "lignes": sum(r.rows or 0 for r in reports),
        "a_recalculer": [{"fichier": r.uri, "motif": r.error} for r in broken],
    }
