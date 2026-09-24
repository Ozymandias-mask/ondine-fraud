"""Explicabilité : pourquoi cette transaction a-t-elle été signalée.

Deux niveaux, exigés par l'AI Act pour un système à haut risque et par l'équipe fraude :

- **global** : quelles variables pèsent le plus dans les décisions du modèle (valeurs SHAP
  moyennes sur un échantillon) — sert à valider que le modèle apprend des signaux de fraude
  plausibles et non des artefacts ;
- **local** : pour une transaction, les variables qui ont le plus poussé la décision, avec
  une phrase lisible par un analyste (et opposable au client).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .dataset import Dataset
from .model import FraudModel

logger = logging.getLogger(__name__)

# Formulation métier de chaque variable : l'analyste ne lit pas des noms de colonnes.
LABELS: dict[str, str] = {
    "card_tx_count_1h": "nombre de paiements de la carte dans l'heure",
    "card_tx_count_24h": "nombre de paiements de la carte sur 24 h",
    "card_amount_sum_24h": "montant cumulé sur la carte en 24 h",
    "card_distinct_merchants_1h": "marchands différents dans l'heure",
    "card_small_online_tx_1h": "petits achats en ligne dans l'heure",
    "card_declined_1h": "refus sur la carte dans l'heure",
    "seconds_since_prev_card_tx": "délai depuis le paiement précédent",
    "client_tx_count_30d": "paiements du client sur 30 jours",
    "amount_to_client_avg": "montant rapporté à l'habitude du client",
    "client_distinct_countries_7d": "pays différents sur 7 jours",
    "client_transfers_24h": "virements du client sur 24 h",
    "client_transfer_amount_24h": "montant viré sur 24 h",
    "km_from_prev_physical": "distance depuis le dernier paiement physique",
    "travel_speed_kmh": "vitesse de déplacement implicite",
    "device_age_hours_for_client": "ancienneté de l'appareil pour ce client",
    "is_new_beneficiary": "bénéficiaire jamais utilisé",
    "beneficiary_new_senders_30d": "bénéficiaire récemment utilisé par d'autres clients",
    "failed_logins_24h": "échecs de connexion sur 24 h",
    "password_resets_24h": "réinitialisations de mot de passe sur 24 h",
    "beneficiaries_added_24h": "bénéficiaires ajoutés sur 24 h",
    "foreign_ip_logins_24h": "connexions depuis l'étranger sur 24 h",
    "new_device_logins_24h": "connexions depuis un nouvel appareil sur 24 h",
    "sms_otp_challenges_1h": "codes SMS demandés dans l'heure",
    "sca_method_changes_7d": "changements de méthode d'authentification sur 7 jours",
    "wallet_provisionings_7d": "ajouts de la carte à un portefeuille mobile sur 7 jours",
    "amount": "montant",
    "is_night": "paiement de nuit",
    "is_foreign_merchant": "marchand étranger",
    "is_online_merchant": "paiement en ligne",
    "three_ds_authenticated": "authentification 3-D Secure",
    "merchant_category": "secteur du marchand",
    "merchant_country": "pays du marchand",
    "channel": "canal de paiement",
    "entry_mode": "mode de saisie",
    "wallet_type": "portefeuille mobile",
    "beneficiary_country": "pays du bénéficiaire",
}


def readable(feature: str) -> str:
    """Nom lisible d'une variable (le nom technique sinon)."""
    return LABELS.get(feature, feature.replace("_", " "))


def shap_values(model: FraudModel, features: pd.DataFrame) -> np.ndarray:
    """Contributions SHAP exactes de LightGBM (dernière colonne : valeur de base)."""
    prepared = model.prepare(features)
    return model.booster.predict(
        prepared, pred_contrib=True, num_iteration=model.booster.best_iteration
    )


def global_importance(
    model: FraudModel, dataset: Dataset, sample: int = 20_000, seed: int = 42
) -> pd.DataFrame:
    """Importance globale : contribution SHAP absolue moyenne, par variable."""
    frame = dataset.features
    if len(frame) > sample:
        frame = frame.sample(sample, random_state=seed)
    contributions = shap_values(model, frame)[:, :-1]
    importance = pd.DataFrame(
        {
            "variable": model.feature_names,
            "libellé": [readable(f) for f in model.feature_names],
            "contribution_moyenne": np.abs(contributions).mean(axis=0),
        }
    )
    total = importance["contribution_moyenne"].sum() or 1.0
    importance["part_%"] = 100 * importance["contribution_moyenne"] / total
    return importance.sort_values("contribution_moyenne", ascending=False).reset_index(drop=True)


def explain_one(model: FraudModel, row: pd.DataFrame, top: int = 4) -> list[dict[str, object]]:
    """Facteurs ayant le plus poussé la décision pour une transaction.

    Chaque facteur porte son sens : « augmente » ou « diminue » le risque, sa valeur, et une
    formulation lisible. C'est ce que l'API renvoie et ce qu'affiche l'interface analyste.
    """
    contributions = shap_values(model, row)[0][:-1]
    order = np.argsort(np.abs(contributions))[::-1][:top]
    prepared = model.prepare(row).iloc[0]
    factors = []
    for index in order:
        name = model.feature_names[index]
        value = prepared[name]
        factors.append(
            {
                "variable": name,
                "libellé": readable(name),
                "valeur": None
                if pd.isna(value)
                else (str(value) if isinstance(value, str) else float(value)),
                "effet": "augmente le risque" if contributions[index] > 0 else "diminue le risque",
                "poids": round(float(contributions[index]), 4),
            }
        )
    return factors


def sentence(factors: list[dict[str, object]]) -> str:
    """Résume les facteurs en une phrase destinée à l'analyste et au client."""
    raising = [f for f in factors if f["effet"] == "augmente le risque"]
    if not raising:
        return "Aucun facteur de risque marquant sur cette opération."
    parts = [
        f"{f['libellé']} ({f['valeur']})" if f["valeur"] is not None else f["libellé"]
        for f in raising[:3]
    ]
    return "Signalée principalement sur : " + ", ".join(parts) + "."
