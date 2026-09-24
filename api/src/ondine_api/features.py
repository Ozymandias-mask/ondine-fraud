"""Calcul des variables du modèle à partir du magasin en ligne.

Ce module est le **miroir exact** de `FEATURES_SQL` (couche gold) : mêmes fenêtres, mêmes
bornes, mêmes valeurs manquantes. C'est le point le plus délicat de toute mise en production
d'un modèle : si une variable est calculée différemment à l'entraînement et en production
(*training/serving skew*), le modèle voit des données qu'il n'a jamais apprises et ses scores
n'ont plus de sens — sans qu'aucune erreur ne soit levée.

Deux garde-fous : les définitions sont écrites ici une seule fois, et un test rejoue de vraies
données dans le magasin puis compare, variable par variable, ce qu'obtient l'API à ce qu'a
calculé le SQL de la couche gold.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime

from .store import DAY, HOUR, OnlineStore, State, epoch

EARTH_RADIUS_KM = 6371.0

# Types d'événements applicatifs utilisés, et variable alimentée par chacun.
LOGIN_TYPES = ("login", "login_failed")


def _window(rows: list[tuple[float, list[str]]], moment: float, seconds: float):
    """Lignes de l'historique comprises dans [t - seconds, t] (la transaction en cours exclue)."""
    return [row for row in rows if row[0] >= moment - seconds]


def _seconds(start: float, end: float) -> int:
    """Écart en secondes entières, exactement comme `date_diff('second', …)` en SQL.

    Les deux horodatages sont tronqués à la seconde *avant* la soustraction : une différence
    d'arrondi ici suffirait à décaler une variable entre l'entraînement et la production.
    """
    return int(end) - int(start)


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance en kilomètres entre deux points (même formule que la couche gold)."""
    d_lat, d_lon = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = (
        math.sin(d_lat / 2) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(d_lon / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def _sum_or_none(rows: list, flag) -> float | None:
    """Somme SQL : `NULL` si la fenêtre est vide, comme `sum(...) OVER (...)` en SQL.

    Ce détail compte : le modèle a appris à distinguer « aucune activité dans la fenêtre »
    (valeur manquante) de « de l'activité, mais aucun incident » (zéro).
    """
    return None if not rows else float(sum(flag(row) for row in rows))


def compute(transaction: dict, state: State) -> dict[str, object]:
    """Variables d'entrée du modèle pour une transaction, à l'instant de la décision."""
    moment = epoch(transaction["event_time"])
    when = datetime.fromtimestamp(moment, tz=UTC)
    amount = float(transaction.get("amount") or 0.0)
    iban = transaction.get("beneficiary_iban") or ""
    country = transaction.get("merchant_country")

    features: dict[str, object] = {
        # ---------------------------------------------------- transaction et référentiel
        "amount": amount,
        "log_amount": math.log1p(amount),
        "channel": transaction.get("channel"),
        "entry_mode": transaction.get("entry_mode"),
        "mcc": transaction.get("mcc"),
        "merchant_category": state.merchant.get("category") or None,
        "merchant_country": country,
        "is_foreign_merchant": bool(country != "FR") if country is not None else False,
        "is_online_merchant": state.merchant.get("is_online") == "1",
        "event_hour": when.hour,
        "is_weekend": when.isoweekday() >= 6,
        "is_night": 0 <= when.hour <= 5,
        "three_ds_authenticated": transaction.get("three_ds_authenticated"),
        "wallet_type": transaction.get("wallet_type") or "none",
        "beneficiary_country": iban[:2] or None,
        "beneficiary_bank_code": iban[4:9] or None,
    }
    features |= _card_features(state, moment)
    features |= _client_features(state, moment, amount)
    features |= _travel_features(state, moment, transaction)
    features |= _relationship_features(state, moment, transaction)
    features |= _session_features(state, moment)
    return features


def _card_features(state: State, moment: float) -> dict[str, object]:
    """Vélocité de la carte : test de carte, achats en rafale, refus successifs."""
    hour_rows, day_rows = _window(state.card, moment, HOUR), _window(state.card, moment, DAY)
    amounts = [float(row[1][1] or 0) for row in day_rows]
    last = max((row[0] for row in state.card), default=None)
    return {
        "card_tx_count_1h": len(hour_rows),
        "card_tx_count_24h": len(day_rows),
        "card_amount_sum_24h": float(sum(amounts)),
        "card_distinct_merchants_1h": len({row[1][2] for row in hour_rows}),
        "card_small_online_tx_1h": sum(
            1 for row in hour_rows if row[1][3] == "ecommerce" and float(row[1][1] or 0) < 5
        ),
        "card_declined_1h": sum(1 for score in state.card_declines if score >= moment - HOUR),
        "seconds_since_prev_card_tx": None if last is None else _seconds(last, moment),
    }


def _client_features(state: State, moment: float, amount: float) -> dict[str, object]:
    """Habitudes du client : volume, montant inhabituel, virements instantanés récents."""
    month_rows = _window(state.client, moment, 30 * DAY)
    week_rows = _window(state.client, moment, 7 * DAY)
    day_rows = _window(state.client, moment, DAY)
    amounts = [float(row[1][1] or 0) for row in month_rows]
    average = sum(amounts) / len(amounts) if amounts else None
    transfers = [row for row in day_rows if row[1][2] == "sepa_instant"]
    return {
        "client_tx_count_30d": len(month_rows),
        "client_avg_amount_30d": average,
        "client_max_amount_30d": max(amounts) if amounts else None,
        "amount_to_client_avg": amount / average if average else None,
        "client_distinct_countries_7d": len(
            {row[1][3] for row in week_rows if row[1][3] not in (None, "")}
        ),
        "client_transfers_24h": len(transfers),
        "client_transfer_amount_24h": float(sum(float(row[1][1] or 0) for row in transfers)),
    }


def _travel_features(state: State, moment: float, transaction: dict) -> dict[str, object]:
    """Déplacement depuis le dernier paiement physique : une carte clonée voyage trop vite.

    Comme dans la couche gold, ces variables n'existent que pour un paiement physique (le
    marchand a des coordonnées) précédé d'un autre paiement physique.
    """
    latitude, longitude = transaction.get("latitude"), transaction.get("longitude")
    if latitude is None or longitude is None or state.previous_physical is None:
        return {"km_from_prev_physical": None, "travel_speed_kmh": None}
    previous_time, previous_lat, previous_lon = state.previous_physical
    distance = _haversine(previous_lat, previous_lon, float(latitude), float(longitude))
    hours = _seconds(previous_time, moment) / 3600.0
    return {
        "km_from_prev_physical": distance,
        "travel_speed_kmh": distance / max(hours, 1.0 / 60),
    }


def _relationship_features(state: State, moment: float, transaction: dict) -> dict[str, object]:
    """Appareil et bénéficiaire : prise de contrôle de compte, comptes « mules »."""
    device_age = None
    if transaction.get("device_id"):
        first = state.device_first_seen
        # Appareil inconnu : c'est sa première utilisation, comme dans la couche gold où le
        # minimum des dates inclut la transaction en cours.
        device_age = 0.0 if first is None else max(_seconds(first, moment), 0) / 3600.0
    if not transaction.get("beneficiary_iban"):
        return {
            "device_age_hours_for_client": device_age,
            "is_new_beneficiary": None,
            "beneficiary_new_senders_30d": None,
        }
    return {
        "device_age_hours_for_client": device_age,
        "is_new_beneficiary": state.beneficiary_first_seen is None
        or state.beneficiary_first_seen >= moment,
        "beneficiary_new_senders_30d": state.beneficiary_new_senders_30d,
    }


def _session_features(state: State, moment: float) -> dict[str, object]:
    """Signaux de session applicative avant le paiement (SIM swap, réinitialisation, wallet).

    La fenêtre porte sur la chronologie complète du client — événements **et** transactions —
    exactement comme la table `timeline` de la couche gold : une fenêtre sans aucune ligne
    donne une valeur manquante, une fenêtre sans incident donne zéro.
    """
    windows = {}
    for name, seconds in (("h1", HOUR), ("h24", DAY), ("d7", 7 * DAY)):
        events = _window(state.events, moment, seconds)
        # Les transactions du client comptent comme des lignes de la chronologie (valeur 0).
        populated = bool(events) or bool(_window(state.client, moment, seconds))
        windows[name] = (events, populated)

    def total(window: str, flag) -> float | None:
        events, populated = windows[window]
        if not populated:
            return None
        return float(sum(flag(row[1]) for row in events))

    def is_type(*types: str):
        return lambda parts: parts[1] in types

    return {
        "failed_logins_24h": total("h24", is_type("login_failed")),
        "password_resets_24h": total("h24", is_type("password_reset")),
        "beneficiaries_added_24h": total("h24", is_type("beneficiary_added")),
        "foreign_ip_logins_24h": total(
            "h24", lambda p: p[1] in LOGIN_TYPES and p[2] not in ("FR", "", None)
        ),
        "new_device_logins_24h": total("h24", lambda p: p[1] in LOGIN_TYPES and p[4] == "1"),
        "sms_otp_challenges_1h": total(
            "h1", lambda p: p[1] == "sca_challenge" and p[3] == "sms_otp"
        ),
        "sca_method_changes_7d": total("d7", is_type("sca_method_changed")),
        "wallet_provisionings_7d": total("d7", is_type("wallet_provisioning")),
    }


def features_for(store: OnlineStore, transaction: dict) -> tuple[dict[str, object], State]:
    """Lit le magasin puis calcule les variables : le chemin complet d'une décision."""
    state = store.state(transaction)
    return compute(transaction, state), state
