"""Chargement d'un modèle du registre et usages hors ligne (évaluation, explication).

L'API de scoring utilise les mêmes fonctions : un modèle promu « champion » se charge par
son alias, jamais par un chemin de fichier, ce qui rend le changement de version immédiat
et traçable.
"""

from __future__ import annotations

import logging

import mlflow
import numpy as np
from ondine_pipelines.settings import Settings

from . import evaluate as ev
from . import scoring as sc
from .explain import explain_one, sentence
from .model import FraudModel
from .settings import MlSettings

logger = logging.getLogger(__name__)


def model_uri(ml: MlSettings, alias: str = "champion") -> str:
    """URI du modèle dans le registre MLflow (par alias)."""
    return f"models:/{ml.model_name}@{alias}"


def load_model(ml: MlSettings, alias: str = "champion") -> tuple[FraudModel, str]:
    """Charge le modèle et retourne (modèle, version du registre)."""
    mlflow.set_tracking_uri(ml.tracking_uri)
    uri = model_uri(ml, alias)
    loaded = mlflow.pyfunc.load_model(uri)
    model: FraudModel = loaded.unwrap_python_model()
    version = loaded.metadata.run_id[:8] if loaded.metadata.run_id else model.version
    logger.info("Modèle chargé : %s (version %s, seuil %.6f)", uri, version, model.threshold)
    return model, version


def registered_version(ml: MlSettings, alias: str = "champion") -> str | None:
    """Numéro de version du registre associé à un alias, s'il existe."""
    client = mlflow.MlflowClient(ml.tracking_uri)
    try:
        return client.get_model_version_by_alias(ml.model_name, alias).version
    except Exception:  # alias absent : aucun modèle promu
        return None


def evaluate_registered(settings: Settings, ml: MlSettings, alias: str = "champion") -> dict:
    """Réévalue un modèle du registre sur le jeu de test courant.

    Sert à comparer un candidat au modèle en production sur exactement les mêmes données, et
    avec exactement le même code que l'entraînement (lecture par lots, mémoire bornée).
    """
    model, version = load_model(ml, alias)
    test, _ = sc.score_range(settings, ml.split().test, model, ml.seed, ml.robustness_sample)
    scores = test.context["score"].to_numpy()
    y_true = test.target.to_numpy()
    amounts = test.context["amount"].astype(float).to_numpy()
    fairness = ev.fairness_report(test, scores, model.threshold)
    metrics = {
        **ev.quality_metrics(y_true, scores),
        **ev.threshold_metrics(y_true, scores, amounts, model.threshold),
        **ev.legacy_rules_metrics(test),
        **ev.fairness_gaps(fairness),
    }
    return {
        "alias": alias,
        "version": version,
        "threshold": model.threshold,
        "rows": int(len(y_true)),
        "metrics": {k: float(v) for k, v in metrics.items() if np.isfinite(float(v))},
    }


def explain_registered(
    settings: Settings, ml: MlSettings, alias: str = "champion", rank: int = 1
) -> dict:
    """Explique la n-ième transaction la plus risquée du jeu de test (démonstration)."""
    model, version = load_model(ml, alias)
    row, context, score, is_fraud = sc.riskiest(settings, ml.split().test, model, rank, ml.seed)
    factors = explain_one(model, row)
    return {
        "version": version,
        "transaction_id": str(context.get("transaction_id", "")),
        "montant": float(context.get("amount", 0)),
        "canal": str(context.get("channel", "")),
        "probabilite_fraude": score,
        "seuil": model.threshold,
        "fraude_confirmee": is_fraud,
        "facteurs": factors,
        "explication": sentence(factors),
    }
