"""Scoring d'un intervalle de mois par lots, à mémoire bornée.

Un mois de transactions représente plusieurs millions de lignes : le charger entièrement avec
ses variables ne tient pas en mémoire sur une machine de bureau, et n'est jamais nécessaire.
Ce module lit la couche gold par lots, score chaque lot, puis ne conserve que :

- pour **tout** l'intervalle : le score, le label et six colonnes d'évaluation (coût métier,
  équité, dérive, comparaison au moteur de règles) ;
- un **échantillon borné** avec ses variables, pour les tests de robustesse et l'explicabilité.

L'entraînement (`train.py`) et la réévaluation d'un modèle du registre (`serving.py`) passent
tous les deux par ici : les chiffres publiés sont donc produits par exactement le même code.
"""

from __future__ import annotations

import gc
import logging

import numpy as np
import pandas as pd
from ondine_pipelines.settings import Settings

from . import dataset as ds
from .model import FraudModel

logger = logging.getLogger(__name__)

# Colonnes de contexte conservées pour tout l'intervalle scoré. L'identifiant de transaction
# en est exclu : inutile pour les métriques, et c'est la colonne la plus lourde (du texte).
EVAL_CONTEXT = ("event_date", "amount", "legacy_auth_status", "age_band", "region", "segment")


def stream_scores(
    settings: Settings,
    bounds: tuple[str, str],
    model: FraudModel,
    seed: int = 42,
    context_column: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.Series | None]:
    """Score un intervalle complet et ne retourne que labels, probabilités, montants.

    Sert au choix du seuil de décision : celui-ci doit respecter un budget d'alertes exprimé
    en pourcentage des transactions **réelles**, donc sans sous-échantillonnage. Trois
    vecteurs suffisent, la mémoire reste négligeable. `context_column` ajoute une colonne de
    contexte (l'attribut d'équité, par exemple) quand l'appelant en a besoin.
    """
    targets, scores, amounts, context = [], [], [], []
    for chunk in _chunks(settings, bounds, model, seed):
        targets.append(chunk.target.to_numpy())
        scores.append(model.probabilities(chunk.features))
        amounts.append(chunk.context["amount"].astype("float32").to_numpy())
        if context_column and context_column in chunk.context.columns:
            context.append(chunk.context[context_column].astype("object"))
    groups = pd.concat(context, ignore_index=True) if context else None
    return np.concatenate(targets), np.concatenate(scores), np.concatenate(amounts), groups


def score_range(
    settings: Settings,
    bounds: tuple[str, str],
    model: FraudModel,
    seed: int = 42,
    sample_budget: int = 200_000,
) -> tuple[ds.Dataset, ds.Dataset]:
    """Score un intervalle et retourne (jeu complet sans variables, échantillon avec variables)."""
    contexts, targets, samples, sample_targets, sample_contexts = [], [], [], [], []
    per_chunk, kept, rows = max(sample_budget // 8, 1), 0, 0
    for chunk in _chunks(settings, bounds, model, seed):
        context = chunk.context[[c for c in EVAL_CONTEXT if c in chunk.context.columns]].copy()
        context["score"] = model.probabilities(chunk.features)
        contexts.append(context)
        targets.append(chunk.target)
        keep = min(per_chunk, len(chunk.target))
        index = chunk.features.sample(keep, random_state=seed).index
        samples.append(chunk.features.loc[index])
        sample_targets.append(chunk.target.loc[index])
        sample_contexts.append(chunk.context.loc[index])
        kept += keep
        if kept > 2 * sample_budget:  # l'échantillon ne grossit jamais indéfiniment
            samples, sample_targets, sample_contexts = trim(
                [samples, sample_targets, sample_contexts], sample_budget, seed
            )
            kept = len(samples[0])
        rows += len(chunk.target)
    samples, sample_targets, sample_contexts = trim(
        [samples, sample_targets, sample_contexts], sample_budget, seed
    )
    logger.info(
        "%s..%s scoré par lots : %d lignes (échantillon %d)", *bounds, rows, len(samples[0])
    )

    full = ds.Dataset(
        name="test",
        features=pd.DataFrame(),
        target=pd.concat(targets, ignore_index=True),
        context=pd.concat(contexts, ignore_index=True),
    )
    sample = ds.Dataset(
        name="test_sample",
        features=pd.concat(samples, ignore_index=True),
        target=pd.concat(sample_targets, ignore_index=True),
        context=pd.concat(sample_contexts, ignore_index=True),
    )
    return full, sample


def riskiest(
    settings: Settings,
    bounds: tuple[str, str],
    model: FraudModel,
    rank: int = 1,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.Series, float, bool]:
    """Retourne la n-ième transaction la plus risquée de l'intervalle (variables, contexte…).

    Seules les `rank` meilleures lignes de chaque lot sont retenues : trouver la transaction
    la plus risquée de douze millions ne demande pas de toutes les garder.
    """
    best_features, best_context, best_scores, best_targets = [], [], [], []
    for chunk in _chunks(settings, bounds, model, seed):
        scores = model.probabilities(chunk.features)
        top = np.argsort(scores)[::-1][:rank]
        best_features.append(chunk.features.iloc[top])
        best_context.append(chunk.context.iloc[top])
        best_scores.append(scores[top])
        best_targets.append(chunk.target.iloc[top])
    features = pd.concat(best_features, ignore_index=True)
    context = pd.concat(best_context, ignore_index=True)
    scores = np.concatenate(best_scores)
    targets = pd.concat(best_targets, ignore_index=True)
    order = np.argsort(scores)[::-1]
    if not len(order):
        raise ValueError(f"Aucune transaction sur {bounds[0]}..{bounds[1]}")
    index = int(order[min(max(rank, 1), len(order)) - 1])
    return (
        features.iloc[[index]],
        context.iloc[index],
        float(scores[index]),
        bool(targets.iloc[index]),
    )


def trim(parts: list[list], budget: int, seed: int) -> list[list]:
    """Ramène un échantillon à `budget` lignes, les collections restant alignées entre elles.

    Les positions conservées sont tirées une seule fois puis appliquées aux variables, aux
    labels et au contexte : aucune désynchronisation possible entre les trois.
    """
    merged = [pd.concat(part, ignore_index=True) for part in parts]
    if len(merged[0]) <= budget:
        return [[m] for m in merged]
    kept = (
        pd.Series(range(len(merged[0]))).sample(budget, random_state=seed).sort_values().to_numpy()
    )
    return [[m.iloc[kept].reset_index(drop=True)] for m in merged]


def _chunks(settings: Settings, bounds: tuple[str, str], model: FraudModel, seed: int):
    """Lots successifs alignés sur les colonnes et catégories vues par le modèle."""
    for chunk in ds.iter_chunks(
        settings,
        bounds,
        categories=model.categories,
        feature_names=model.feature_names,
        seed=seed,
    ):
        yield chunk
        del chunk
        gc.collect()
