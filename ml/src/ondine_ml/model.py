"""Modèle servi : prétraitement, prédiction calibrée et seuil de décision réunis.

L'API de scoring et les traitements d'évaluation chargent **le même objet** : le
prétraitement (catégories connues, types) et la correction du sous-échantillonnage ne
peuvent donc pas diverger entre l'entraînement et la production, cause classique d'écarts
inexplicables entre les deux.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import lightgbm as lgb
import mlflow.pyfunc
import numpy as np
import pandas as pd

from .evaluate import correct_sampling_bias


@dataclass
class FraudModel(mlflow.pyfunc.PythonModel):
    """Modèle LightGBM enrichi de son prétraitement, de sa calibration et de son seuil."""

    booster: lgb.Booster | None = None
    feature_names: list[str] = field(default_factory=list)
    categorical: list[str] = field(default_factory=list)
    categories: dict[str, list[str]] = field(default_factory=dict)
    negative_sampling_rate: float = 1.0
    threshold: float = 0.5
    version: str = "dev"

    def prepare(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Aligne les colonnes et les types sur ce que le modèle a vu à l'entraînement.

        Une colonne absente est ajoutée vide, une valeur de catégorie inconnue devient
        manquante : une entrée inattendue dégrade la prédiction, elle ne la fait pas échouer.
        """
        prepared = pd.DataFrame(index=frame.index)
        for name in self.feature_names:
            column = frame[name] if name in frame.columns else pd.Series(np.nan, index=frame.index)
            if name in self.categorical:
                # Une valeur inconnue devient manquante (LightGBM sait traiter les manquants).
                values = column.astype("object")
                values = values.where(values.isin(self.categories[name]), other=None)
                prepared[name] = pd.Categorical(values, categories=self.categories[name])
            elif column.dtype == "bool":
                prepared[name] = column.astype("float32")
            else:
                prepared[name] = pd.to_numeric(column, errors="coerce").astype("float32")
        return prepared

    def probabilities(self, frame: pd.DataFrame) -> np.ndarray:
        """Probabilité de fraude, corrigée du sous-échantillonnage (échelle réelle)."""
        raw = self.booster.predict(self.prepare(frame), num_iteration=self.booster.best_iteration)
        return correct_sampling_bias(np.asarray(raw, dtype=float), self.negative_sampling_rate)

    def decide(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Probabilité, décision et marge par rapport au seuil."""
        probabilities = self.probabilities(frame)
        return pd.DataFrame(
            {
                "probability": probabilities,
                "alert": probabilities >= self.threshold,
                "threshold": self.threshold,
            },
            index=frame.index,
        )

    # ----------------------------------------------------------------- interface MLflow
    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.DataFrame:  # noqa: ARG002, D102
        return self.decide(model_input)


def serving_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Version « telle qu'envoyée par l'API » : catégories en texte, pas de type pandas exotique.

    Sert d'exemple et de signature au modèle enregistré : ce que le modèle promet d'accepter
    est exactement ce que l'API lui enverra.
    """
    serving = frame.copy()
    for column in serving.columns:
        if str(serving[column].dtype) == "category":
            serving[column] = serving[column].astype("object").where(serving[column].notna(), None)
    return serving


def metadata(model: FraudModel) -> dict[str, object]:
    """Informations du modèle utiles au suivi et à l'API (versions, seuil, variables)."""
    return {
        "version": model.version,
        "threshold": model.threshold,
        "negative_sampling_rate": model.negative_sampling_rate,
        "features": model.feature_names,
        "categorical": model.categorical,
        "trees": model.booster.num_trees() if model.booster else 0,
    }
