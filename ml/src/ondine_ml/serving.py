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

from . import dataset as ds
from . import evaluate as ev
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

    Sert à comparer un candidat au modèle en production sur exactement les mêmes données.
    """
    model, version = load_model(ml, alias)
    bundle = ds.load(settings, ml.split(), negative_sampling_rate=1.0, seed=ml.seed)
    scores = model.probabilities(bundle.test.features)
    y_true = bundle.test.target.to_numpy()
    amounts = bundle.test.context["amount"].astype(float).to_numpy()
    fairness = ev.fairness_report(bundle.test, scores, model.threshold)
    metrics = {
        **ev.quality_metrics(y_true, scores),
        **ev.threshold_metrics(y_true, scores, amounts, model.threshold),
        **ev.legacy_rules_metrics(bundle.test),
        **ev.fairness_gaps(fairness),
    }
    return {
        "alias": alias,
        "version": version,
        "threshold": model.threshold,
        "metrics": {k: float(v) for k, v in metrics.items() if np.isfinite(float(v))},
    }


def explain_registered(
    settings: Settings, ml: MlSettings, alias: str = "champion", rank: int = 1
) -> dict:
    """Explique la n-ième transaction la plus risquée du jeu de test (démonstration)."""
    model, version = load_model(ml, alias)
    bundle = ds.load(settings, ml.split(), negative_sampling_rate=1.0, seed=ml.seed)
    scores = model.probabilities(bundle.test.features)
    index = int(np.argsort(scores)[::-1][max(rank - 1, 0)])
    row = bundle.test.features.iloc[[index]]
    factors = explain_one(model, row)
    context = bundle.test.context.iloc[index]
    return {
        "version": version,
        "transaction_id": str(context.get("transaction_id", "")),
        "montant": float(context.get("amount", 0)),
        "canal": str(context.get("channel", "")),
        "probabilite_fraude": float(scores[index]),
        "seuil": model.threshold,
        "fraude_confirmee": bool(bundle.test.target.iloc[index]),
        "facteurs": factors,
        "explication": sentence(factors),
    }
