"""Évaluation du modèle : qualité, coût métier, équité et robustesse.

La fraude est très déséquilibrée (0,2 %) : l'exactitude n'a aucun sens (tout prédire
« légitime » donnerait 99,8 %). On mesure donc :

- **AUC-PR** (précision-rappel), plus informative que l'AUC-ROC sur données déséquilibrées ;
- le **rappel à budget d'alertes fixé** : ce que l'équipe fraude peut réellement traiter ;
- le **coût métier** : montant fraudé non détecté + coût des blocages à tort, comparé au
  moteur de règles actuel — c'est le chiffre que comprend la direction ;
- l'**équité** : rappel et taux de fausses alertes par tranche d'âge et par région ;
- la **robustesse** : données manquantes, valeurs bruitées, catégories inconnues, dérive.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from .dataset import Dataset

# Coûts moyens constatés par l'équipe fraude (paramétrables) :
# une fraude non détectée coûte le montant de l'opération plus son traitement ;
# un blocage à tort coûte une réémission de carte, un appel au support et de l'insatisfaction.
COST_PER_UNDETECTED_FRAUD_FIXED = 25.0
COST_PER_FALSE_ALERT = 12.0
# Capacité de revue de l'équipe fraude : part des transactions qu'elle peut examiner.
ALERT_BUDGET = 0.001


def correct_sampling_bias(probabilities: np.ndarray, negative_rate: float) -> np.ndarray:
    """Corrige les probabilités du sous-échantillonnage des transactions légitimes.

    Sans cette correction, le modèle annonce des probabilités bien supérieures à la réalité
    (il a vu 20 fois trop de fraudes) : les seuils et les coûts seraient faux.
    """
    if negative_rate >= 1:
        return probabilities
    odds = probabilities / np.clip(1 - probabilities, 1e-9, None)
    return odds * negative_rate / (1 + odds * negative_rate)


@dataclass(frozen=True)
class ThresholdChoice:
    """Seuil de décision retenu et ce qu'il implique."""

    threshold: float
    alert_rate: float
    recall: float
    precision: float
    cost: float
    reason: str


def business_cost(y_true: np.ndarray, flagged: np.ndarray, amounts: np.ndarray) -> dict[str, float]:
    """Coût métier d'une politique de décision (montants en euros)."""
    missed = (y_true == 1) & (~flagged)
    false_alerts = (y_true == 0) & flagged
    caught = (y_true == 1) & flagged
    fraud_loss = float(amounts[missed].sum() + COST_PER_UNDETECTED_FRAUD_FIXED * missed.sum())
    friction = float(COST_PER_FALSE_ALERT * false_alerts.sum())
    return {
        "cost_total_eur": fraud_loss + friction,
        "cost_fraud_loss_eur": fraud_loss,
        "cost_friction_eur": friction,
        "fraud_amount_blocked_eur": float(amounts[caught].sum()),
        "alerts": float(flagged.sum()),
        "false_alerts": float(false_alerts.sum()),
        "missed_frauds": float(missed.sum()),
    }


def choose_threshold(
    y_true: np.ndarray, scores: np.ndarray, amounts: np.ndarray, budget: float = ALERT_BUDGET
) -> tuple[ThresholdChoice, ThresholdChoice]:
    """Deux seuils : celui qui minimise le coût, et celui qui tient le budget d'alertes.

    Le second est celui qu'on met en production : il garantit que l'équipe fraude peut
    traiter le volume, même si le coût théorique est un peu moins bon.
    """
    candidates = np.unique(np.quantile(scores, np.linspace(0.90, 0.99999, 400)))
    rows = []
    for threshold in candidates:
        flagged = scores >= threshold
        if flagged.sum() == 0:
            continue
        cost = business_cost(y_true, flagged, amounts)
        recall = float((y_true[flagged] == 1).sum() / max(y_true.sum(), 1))
        precision = float((y_true[flagged] == 1).sum() / flagged.sum())
        rows.append((threshold, flagged.mean(), recall, precision, cost["cost_total_eur"]))
    table = pd.DataFrame(rows, columns=["threshold", "alert_rate", "recall", "precision", "cost"])
    best = table.loc[table["cost"].idxmin()]
    affordable = table[table["alert_rate"] <= budget]
    chosen = affordable.loc[affordable["cost"].idxmin()] if len(affordable) else best
    return (
        ThresholdChoice(
            *best[["threshold", "alert_rate", "recall", "precision", "cost"]],
            reason="coût métier minimal",
        ),
        ThresholdChoice(
            *chosen[["threshold", "alert_rate", "recall", "precision", "cost"]],
            reason=f"budget d'alertes ≤ {budget:.3%}",
        ),
    )


def budget_threshold(scores: np.ndarray, budget: float = ALERT_BUDGET) -> float:
    """Seuil qui alerte exactement `budget` % des transactions d'une population donnée.

    Le budget d'alertes est une contrainte d'organisation (« l'équipe peut examiner une
    transaction sur mille »), donc un **quantile**, pas une probabilité fixe. Comparer ce
    seuil à celui retenu à l'entraînement mesure directement la dérive du modèle : si les
    scores baissent, un seuil figé fait chuter le volume d'alertes sans que la qualité du
    modèle ait bougé.
    """
    return float(np.quantile(scores, 1 - budget))


def quality_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """Métriques indépendantes du seuil."""
    return {
        "auc_pr": float(average_precision_score(y_true, scores)),
        "auc_roc": float(roc_auc_score(y_true, scores)),
        "brier": float(brier_score_loss(y_true, scores)),
        "mean_score": float(scores.mean()),
        "base_rate": float(y_true.mean()),
    }


def threshold_metrics(
    y_true: np.ndarray, scores: np.ndarray, amounts: np.ndarray, threshold: float
) -> dict[str, float]:
    """Métriques à seuil fixé, y compris le coût métier."""
    flagged = scores >= threshold
    true_positive = float(((y_true == 1) & flagged).sum())
    metrics = {
        "threshold": float(threshold),
        "alert_rate": float(flagged.mean()),
        "recall": true_positive / max(float(y_true.sum()), 1),
        "precision": true_positive / max(float(flagged.sum()), 1),
        "false_alarm_rate": float(((y_true == 0) & flagged).sum() / max((y_true == 0).sum(), 1)),
    }
    metrics["f1"] = (
        2 * metrics["recall"] * metrics["precision"] / (metrics["recall"] + metrics["precision"])
        if metrics["recall"] + metrics["precision"]
        else 0.0
    )
    metrics.update(business_cost(y_true, flagged, amounts))
    return metrics


def legacy_rules_metrics(dataset: Dataset) -> dict[str, float]:
    """Performance du moteur de règles actuel : la référence à battre."""
    flagged = (dataset.context["legacy_auth_status"] == "declined").to_numpy()
    y_true = dataset.target.to_numpy()
    amounts = dataset.context["amount"].astype(float).to_numpy()
    true_positive = float(((y_true == 1) & flagged).sum())
    metrics = {
        "legacy_alert_rate": float(flagged.mean()),
        "legacy_recall": true_positive / max(float(y_true.sum()), 1),
        "legacy_precision": true_positive / max(float(flagged.sum()), 1),
    }
    metrics.update({f"legacy_{k}": v for k, v in business_cost(y_true, flagged, amounts).items()})
    return metrics


def fairness_report(
    dataset: Dataset,
    scores: np.ndarray,
    threshold: float,
    attributes: tuple[str, ...] = ("age_band", "region"),
) -> pd.DataFrame:
    """Rappel, fausses alertes et taux d'alerte par groupe : détection d'un traitement inégal.

    Les attributs comparés ne sont jamais des variables d'entrée du modèle : un écart vient
    donc des comportements observés, pas d'une utilisation directe de l'attribut.
    """
    y_true = dataset.target.to_numpy()
    flagged = scores >= threshold
    rows = []
    for attribute in attributes:
        if attribute not in dataset.context:
            continue
        column = dataset.context[attribute]
        # Comparaison valeur par valeur : sur des dizaines de millions de lignes, un
        # `groupby` matérialiserait un index par groupe, pour un résultat identique.
        for value in sorted(v for v in column.dropna().unique()):
            mask = (column == value).to_numpy()
            positives = (y_true == 1) & mask
            negatives = (y_true == 0) & mask
            rows.append(
                {
                    "attribut": attribute,
                    "groupe": value,
                    "transactions": int(mask.sum()),
                    "fraudes": int(positives.sum()),
                    "rappel": float((flagged & positives).sum() / max(positives.sum(), 1)),
                    "taux_fausses_alertes": float(
                        (flagged & negatives).sum() / max(negatives.sum(), 1)
                    ),
                    "taux_alerte": float(flagged[mask].mean()) if mask.sum() else 0.0,
                }
            )
    report = pd.DataFrame(rows)
    return report.sort_values(["attribut", "groupe"]).reset_index(drop=True)


def fairness_gaps(
    report: pd.DataFrame, min_group: int = 500, min_frauds: int = 30
) -> dict[str, float]:
    """Écarts maximaux entre groupes (règle des 4/5 : un rapport < 0,8 doit alerter).

    Les groupes trop petits sont écartés : un rappel calculé sur 3 fraudes ne dit rien et
    ferait échouer la comparaison au hasard.
    """
    gaps: dict[str, float] = {}
    retained = report[(report["transactions"] >= min_group) & (report["fraudes"] >= min_frauds)]
    for attribute, group in retained.groupby("attribut"):
        recall = group["rappel"]
        false_alarm = group["taux_fausses_alertes"]
        gaps[f"fairness_{attribute}_recall_ratio"] = float(
            recall.min() / recall.max() if recall.max() else 1.0
        )
        gaps[f"fairness_{attribute}_false_alarm_ratio"] = float(
            false_alarm.max() / false_alarm.min() if false_alarm.min() else float("inf")
        )
    return gaps


def robustness_report(
    dataset: Dataset,
    predict,
    threshold: float,
    seed: int = 42,
    sample: int = 200_000,
    important_features: tuple[str, ...] = (
        "card_tx_count_1h",
        "amount_to_client_avg",
        "device_age_hours_for_client",
    ),
) -> pd.DataFrame:
    """Performance sous données dégradées : le modèle doit se dégrader, pas s'effondrer.

    Scénarios : valeurs manquantes (panne d'une source de features), montants bruités,
    catégorie jamais vue à l'entraînement, et transactions les plus récentes (dérive).
    """
    rng = np.random.default_rng(seed)
    # Échantillon : les scénarios dupliquent les données, inutile de le faire sur 12 millions
    # de lignes pour mesurer une dégradation.
    index = (
        dataset.features.sample(sample, random_state=seed).index
        if len(dataset.features) > sample
        else dataset.features.index
    )
    base = dataset.features.loc[index]
    y_true = dataset.target.loc[index].to_numpy()
    amounts = dataset.context.loc[index, "amount"].astype(float).to_numpy()
    scenarios: dict[str, pd.DataFrame] = {"référence": base}

    for share in (0.1, 0.5):
        degraded = base.copy()
        rows = rng.random(len(degraded)) < share
        for feature in important_features:
            if feature in degraded.columns:
                degraded.loc[rows, feature] = np.nan
        scenarios[f"{int(share * 100)} % de features manquantes"] = degraded

    if "amount" in base.columns:
        noisy = base.copy()
        noisy["amount"] = noisy["amount"] * rng.normal(1.0, 0.2, len(noisy)).astype("float32")
        scenarios["montants bruités (±20 %)"] = noisy

    categorical = [c for c in base.columns if str(base[c].dtype) == "category"]
    if categorical:
        unknown = base.copy()
        column = categorical[0]
        unknown[column] = pd.Categorical(
            [None] * len(unknown), categories=unknown[column].cat.categories
        )
        scenarios[f"catégorie inconnue ({column})"] = unknown

    rows_out = []
    for name, features in scenarios.items():
        scores = predict(features)
        metrics = quality_metrics(y_true, scores)
        at_threshold = threshold_metrics(y_true, scores, amounts, threshold)
        rows_out.append(
            {
                "scénario": name,
                "auc_pr": metrics["auc_pr"],
                "rappel": at_threshold["recall"],
                "précision": at_threshold["precision"],
                "taux_alerte": at_threshold["alert_rate"],
                "coût_eur": at_threshold["cost_total_eur"],
            }
        )
    return pd.DataFrame(rows_out)


def monthly_report(dataset: Dataset, scores: np.ndarray, threshold: float) -> pd.DataFrame:
    """Performance mois par mois : met en évidence une dérive (nouveau mode opératoire)."""
    # Seules trois colonnes sont nécessaires : recopier tout le contexte doublerait la
    # mémoire sur un jeu de test de plusieurs millions de lignes.
    frame = pd.DataFrame(
        {
            # Type « période » (entier) plutôt que texte : 8 octets par ligne au lieu de ~60.
            "mois": pd.to_datetime(dataset.context["event_date"]).dt.to_period("M"),
            "score": scores,
            "is_fraud": dataset.target.to_numpy(),
        }
    )
    rows = []
    for month, group in frame.groupby("mois"):
        flagged = group["score"].to_numpy() >= threshold
        y_true = group["is_fraud"].to_numpy()
        detected = float(((y_true == 1) & flagged).sum())
        rows.append(
            {
                "mois": str(month),
                "transactions": len(group),
                "fraudes": int(y_true.sum()),
                "rappel": detected / max(float(y_true.sum()), 1),
                "précision": detected / max(float(flagged.sum()), 1),
                "taux_alerte": float(flagged.mean()),
                "score_moyen": float(group["score"].mean()),
            }
        )
    return pd.DataFrame(rows)
