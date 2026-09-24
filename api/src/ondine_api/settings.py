"""Configuration de l'API de scoring, lue dans les variables d'environnement."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


class ConfigurationError(RuntimeError):
    """Variable d'environnement obligatoire absente : on s'arrête au démarrage, pas en vol."""


def _require(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise ConfigurationError(f"Variable d'environnement manquante : {name}")
    return value


@dataclass(frozen=True)
class ApiSettings:
    """Paramètres de l'API et du magasin de variables en ligne."""

    redis_url: str = "redis://redis:6379/0"
    # Clé de pseudonymisation : **la même** que les pipelines, sinon les clés du magasin
    # en ligne ne correspondraient pas à celles calculées par le flux.
    pseudonymisation_key: str = field(default="", repr=False)
    model_alias: str = "champion"
    # Délai maximal de chargement du modèle au démarrage (registre MLflow injoignable).
    load_timeout_s: float = 60.0
    # Au-delà, la réponse est renvoyée mais l'appel est compté comme hors budget.
    latency_budget_ms: float = 100.0
    # Décision retenue quand le modèle ou le magasin est indisponible : on laisse passer et
    # on trace, plutôt que de bloquer les paiements d'une banque entière.
    fail_open: bool = True
    explain_top_factors: int = 4
    # Recalage du seuil sur les scores récents : le budget d'alertes est une contrainte
    # d'organisation (un quantile), pas une probabilité figée.
    score_window: int = 200_000
    recalibration_interval_s: float = 300.0
    recalibration_min_samples: int = 20_000
    # Base de la file d'alertes (compte de l'API, au moindre privilège). Vide : l'interface
    # analyste est désactivée, l'API continue de décider.
    pg_dsn: str = field(default="", repr=False)
    alerts_page_size: int = 50

    @classmethod
    def from_env(cls) -> ApiSettings:
        """Construit la configuration à partir de l'environnement."""

        def value(name: str, default):
            raw = os.environ.get(name)
            return type(default)(raw) if raw not in (None, "") else default

        return cls(
            redis_url=os.environ.get("ONDINE_API_REDIS_URL", cls.redis_url),
            pseudonymisation_key=_require("PSEUDONYMISATION_KEY"),
            model_alias=os.environ.get("ONDINE_API_MODEL_ALIAS", cls.model_alias),
            load_timeout_s=value("ONDINE_API_LOAD_TIMEOUT_S", cls.load_timeout_s),
            latency_budget_ms=value("ONDINE_API_LATENCY_BUDGET_MS", cls.latency_budget_ms),
            fail_open=os.environ.get("ONDINE_API_FAIL_OPEN", "true").lower() == "true",
            explain_top_factors=value("ONDINE_API_EXPLAIN_FACTORS", cls.explain_top_factors),
            score_window=value("ONDINE_API_SCORE_WINDOW", cls.score_window),
            recalibration_interval_s=value(
                "ONDINE_API_RECALIBRATION_S", cls.recalibration_interval_s
            ),
            recalibration_min_samples=value(
                "ONDINE_API_RECALIBRATION_MIN", cls.recalibration_min_samples
            ),
            pg_dsn=os.environ.get("ONDINE_API_PG_DSN", cls.pg_dsn),
            alerts_page_size=value("ONDINE_API_ALERTS_PAGE_SIZE", cls.alerts_page_size),
        )
