"""Configuration des pipelines, lue dans les variables d'environnement.

Aucun secret n'a de valeur par défaut : une variable manquante fait échouer le démarrage
avec un message explicite plutôt qu'une connexion avec un mot de passe vide.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

LAYERS = ("bronze", "silver", "gold", "quarantine")


class ConfigurationError(RuntimeError):
    """Variable d'environnement obligatoire absente ou invalide."""


def _require(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        raise ConfigurationError(f"Variable d'environnement manquante : {name}")
    return value


@dataclass(frozen=True)
class Settings:
    """Paramètres d'exécution des pipelines."""

    landing_root: Path
    # "s3" : buckets SeaweedFS ; sinon chemin d'un dossier local (tests, développement).
    lake_root: str
    pg_dsn: str
    pseudonymisation_key: str = field(repr=False)
    s3_endpoint: str = "seaweed-filer:8333"
    s3_access_key: str = field(default="", repr=False)
    s3_secret_key: str = field(default="", repr=False)
    pushgateway_url: str = ""
    duckdb_threads: int = 4
    duckdb_memory_limit: str = "6GB"
    # Suppression des fichiers bruts une fois ingérés et vérifiés (minimisation RGPD).
    purge_landing: bool = False

    @property
    def uses_s3(self) -> bool:
        """Vrai si le data lake est sur le stockage objet S3."""
        return self.lake_root == "s3"

    def uri(self, layer: str, *parts: str) -> str:
        """Adresse d'un objet du data lake (bucket S3 ou dossier local de la couche)."""
        if layer not in LAYERS:
            raise ValueError(f"Couche inconnue : {layer}")
        suffix = "/".join(p.strip("/") for p in parts if p)
        if self.uses_s3:
            return f"s3://{layer}/{suffix}" if suffix else f"s3://{layer}"
        base = Path(self.lake_root) / layer
        return str(base / suffix) if suffix else str(base)

    @classmethod
    def from_env(cls) -> Settings:
        """Construit la configuration à partir de l'environnement."""
        lake_root = os.environ.get("LAKE_ROOT", "s3")
        uses_s3 = lake_root == "s3"
        return cls(
            landing_root=Path(os.environ.get("LANDING_ROOT", "/data/landing")),
            lake_root=lake_root,
            pg_dsn=_require("PIPELINE_PG_DSN"),
            pseudonymisation_key=_require("PSEUDONYMISATION_KEY"),
            s3_endpoint=os.environ.get("S3_ENDPOINT", "seaweed-filer:8333"),
            s3_access_key=_require("S3_PIPELINE_ACCESS_KEY") if uses_s3 else "",
            s3_secret_key=_require("S3_PIPELINE_SECRET_KEY") if uses_s3 else "",
            pushgateway_url=os.environ.get("PUSHGATEWAY_URL", ""),
            duckdb_threads=int(os.environ.get("DUCKDB_THREADS", "4")),
            duckdb_memory_limit=os.environ.get("DUCKDB_MEMORY_LIMIT", "6GB"),
            purge_landing=os.environ.get("PURGE_LANDING", "false").lower() == "true",
        )
