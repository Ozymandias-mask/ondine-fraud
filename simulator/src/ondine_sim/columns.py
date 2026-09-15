"""Colonnes internes des transactions et événements, et codes des valeurs catégorielles.

Pendant la génération, chaque colonne est un tableau numpy compact (entiers,
flottants). La conversion en chaînes lisibles ne se fait qu'à l'export
(`writers.py`), ce qui divise la mémoire utilisée par ~5.
"""

from __future__ import annotations

import numpy as np

from . import reference as ref

CHANNELS = np.array(["pos_chip", "pos_contactless", "ecommerce", "atm", "sepa_instant"])
CH_CHIP, CH_CONTACTLESS, CH_ECOM, CH_ATM, CH_TRANSFER = range(5)

ENTRY_MODES = np.array(
    ["chip", "contactless", "magstripe", "card_not_present", "token", "transfer"]
)
EM_CHIP, EM_CONTACTLESS, EM_MAGSTRIPE, EM_CNP, EM_TOKEN, EM_TRANSFER = range(6)

STATUSES = np.array(["approved", "declined"])
DECLINE_REASONS = np.array(
    ["", "insufficient_funds", "do_not_honor", "suspected_fraud", "invalid_cvv"]
)
DR_NONE, DR_FUNDS, DR_DNH, DR_SUSPECTED, DR_CVV = range(5)

WALLETS = np.array(["none", "apple_pay", "google_pay"])

FRAUD_TYPES = np.array(
    [
        "legit",
        "card_testing",
        "cloned_card",
        "account_takeover",
        "sim_swap",
        "app_scam",
        "wallet_enrolment",
    ]
)
FT_LEGIT, FT_CARD_TESTING, FT_CLONED, FT_ATO, FT_SIM_SWAP, FT_APP_SCAM, FT_WALLET = range(7)

EVENT_TYPES = np.array(
    [
        "app_open",
        "balance_view",
        "card_settings_view",
        "statement_download",
        "profile_update",
        "notification_opened",
        "login",
        "login_failed",
        "sca_challenge",
        "password_reset",
        "beneficiary_added",
        "sca_method_changed",
        "wallet_provisioning",
    ]
)
(
    EV_APP_OPEN,
    EV_BALANCE,
    EV_CARD_SETTINGS,
    EV_STATEMENT,
    EV_PROFILE,
    EV_NOTIF,
    EV_LOGIN,
    EV_LOGIN_FAILED,
    EV_SCA,
    EV_PWD_RESET,
    EV_BENEFICIARY,
    EV_SCA_CHANGED,
    EV_WALLET,
) = range(13)

SCA_METHODS = np.array(["biometric", "app_push", "sms_otp"])
RESULTS = np.array(["success", "failure"])
OS_NAMES = np.array(["android", "ios"])

# Pays des adresses IP et des marchands : 0 = France, puis voyages, puis pays à risque.
COUNTRIES = np.array(
    ["FR"] + [c[0] for c in ref.TRAVEL_COUNTRIES] + [c[0] for c in ref.RISKY_COUNTRIES]
)
COUNTRY_LAT = np.array(
    [46.6] + [c[1] for c in ref.TRAVEL_COUNTRIES] + [c[1] for c in ref.RISKY_COUNTRIES]
)
COUNTRY_LON = np.array(
    [2.3] + [c[2] for c in ref.TRAVEL_COUNTRIES] + [c[2] for c in ref.RISKY_COUNTRIES]
)
FIRST_TRAVEL = 1
FIRST_RISKY = 1 + len(ref.TRAVEL_COUNTRIES)
N_COUNTRIES = len(COUNTRIES)

# Premiers octets d'adresses IP : box internet, réseau mobile, VPN / hébergeurs.
MOBILE_IP_OCTETS = np.array([37, 176, 188])
RISKY_IP_OCTETS = np.array([45, 91, 185, 193])

NEW_DEVICE_OFFSET = 10**12  # clés d'appareils inconnus (fraudeurs)
NEW_BENEFICIARY_OFFSET = 10**10

TX_COLUMNS: dict[str, tuple[type, object]] = {
    "ts": (np.int64, 0),
    "client": (np.int64, 0),
    "card": (np.int64, -1),
    "merchant": (np.int64, -1),
    "channel": (np.int8, 0),
    "amount": (np.float64, 0.0),
    "entry_mode": (np.int8, 0),
    "three_ds": (np.int8, -1),
    "device_key": (np.int64, -1),
    "ip_octet": (np.int16, -1),
    "ip_key": (np.int64, 0),
    "ip_country": (np.int8, -1),
    "beneficiary_key": (np.int64, -1),
    "beneficiary_is_mule": (np.bool_, False),
    "status": (np.int8, 0),
    "decline_reason": (np.int8, 0),
    "wallet": (np.int8, 0),
    "fraud_type": (np.int8, FT_LEGIT),
    "incident": (np.int64, -1),
}

EVENT_COLUMNS: dict[str, tuple[type, object]] = {
    "ts": (np.int64, 0),
    "client": (np.int64, 0),
    "event_type": (np.int8, 0),
    "device_key": (np.int64, 0),
    "os": (np.int8, 0),
    "ip_octet": (np.int16, 0),
    "ip_key": (np.int64, 0),
    "ip_country": (np.int8, 0),
    "has_geo": (np.bool_, False),
    "lat": (np.float32, np.nan),
    "lon": (np.float32, np.nan),
    "sca_method": (np.int8, -1),
    "result": (np.int8, -1),
    "amount": (np.float32, np.nan),
    "wallet": (np.int8, 0),
    "fraud_type": (np.int8, FT_LEGIT),
    "incident": (np.int64, -1),
}


def empty(schema: dict[str, tuple[type, object]], n: int) -> dict[str, np.ndarray]:
    """Crée `n` lignes remplies des valeurs par défaut du schéma."""
    return {name: np.full(n, default, dtype=dtype) for name, (dtype, default) in schema.items()}


def concat(
    schema: dict[str, tuple[type, object]], parts: list[dict[str, np.ndarray]]
) -> dict[str, np.ndarray]:
    """Concatène plusieurs blocs de colonnes ayant le même schéma."""
    parts = [p for p in parts if len(p["ts"])]
    if not parts:
        return empty(schema, 0)
    return {name: np.concatenate([p[name] for p in parts]) for name in schema}


def take(cols: dict[str, np.ndarray], mask_or_index: np.ndarray) -> dict[str, np.ndarray]:
    """Sélectionne des lignes (masque booléen ou indices)."""
    return {name: arr[mask_or_index] for name, arr in cols.items()}


def sort_by_ts(cols: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Trie toutes les colonnes par horodatage."""
    return take(cols, np.argsort(cols["ts"], kind="stable"))
