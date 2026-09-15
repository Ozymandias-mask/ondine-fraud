"""Conversion des colonnes internes en enregistrements lisibles (CSV et JSON).

C'est ici que sont fixés les schémas des données brutes livrées dans la zone
d'arrivée : ils constituent le contrat avec les pipelines d'ingestion.
"""

from __future__ import annotations

import numpy as np
import orjson
import pandas as pd

from . import columns as col
from .entities import CAT_MCC, Entities
from .identifiers import (
    external_ibans,
    hashed_hex_ids,
    ipv4_from_keys,
    random_hex_ids,
    sequential_ids,
)

TRANSACTION_SCHEMA_V1 = [
    "transaction_id",
    "event_time",
    "client_id",
    "card_id",
    "merchant_id",
    "mcc",
    "merchant_country",
    "channel",
    "entry_mode",
    "amount",
    "currency",
    "three_ds_authenticated",
    "device_id",
    "ip_address",
    "latitude",
    "longitude",
    "beneficiary_iban",
    "auth_status",
    "decline_reason",
]
# v2 (application 5.0) : ajout du moyen de paiement mobile utilisé.
TRANSACTION_SCHEMA_V2 = [*TRANSACTION_SCHEMA_V1, "wallet_type"]


def iso_times(ts_ms: np.ndarray) -> np.ndarray:
    """Horodatages ISO 8601 en UTC, précision milliseconde."""
    return np.char.add(ts_ms.astype("datetime64[ms]").astype(str), "Z")


def _nullable_str(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = values.astype(object)
    out[~mask] = None
    return out


def transactions_frame(
    ent: Entities, rng: np.random.Generator, tx: dict[str, np.ndarray], schema_v2: bool
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retourne (transactions brutes, vérité terrain).

    La vérité terrain (type de fraude réel) n'est jamais livrée avec les données
    brutes : dans la réalité, la banque ne l'apprend que par les déclarations.
    """
    n = len(tx["ts"])
    tx_ids = random_hex_ids(rng, n)
    has_merchant = tx["merchant"] >= 0
    merchant = np.maximum(tx["merchant"], 0)
    has_card = tx["card"] >= 0
    has_device = tx["device_key"] >= 0
    has_ip = tx["ip_octet"] >= 0
    has_benef = tx["beneficiary_key"] >= 0
    card_present = has_merchant & ~ent.merchant_online[merchant]

    benef_keys = np.maximum(tx["beneficiary_key"], 0)
    benef_iban = np.where(
        tx["beneficiary_is_mule"],
        ent.mule_ibans[np.minimum(benef_keys, len(ent.mule_ibans) - 1)],
        external_ibans(benef_keys),
    )
    three_ds = np.array([None, False, True], dtype=object)[tx["three_ds"] + 1]

    frame = pd.DataFrame(
        {
            "transaction_id": tx_ids,
            "event_time": iso_times(tx["ts"]),
            "client_id": sequential_ids(tx["client"], "C", 8),
            "card_id": _nullable_str(sequential_ids(np.maximum(tx["card"], 0), "K", 9), has_card),
            "merchant_id": _nullable_str(sequential_ids(merchant, "M", 7), has_merchant),
            "mcc": pd.Series(CAT_MCC[ent.merchant_category[merchant]], dtype="Int64").mask(
                ~has_merchant
            ),
            "merchant_country": _nullable_str(
                col.COUNTRIES[ent.merchant_country[merchant]], has_merchant
            ),
            "channel": col.CHANNELS[tx["channel"]],
            "entry_mode": col.ENTRY_MODES[tx["entry_mode"]],
            "amount": tx["amount"],
            "currency": "EUR",
            "three_ds_authenticated": three_ds,
            "device_id": _nullable_str(
                hashed_hex_ids(np.maximum(tx["device_key"], 0), "dev_"), has_device
            ),
            "ip_address": _nullable_str(
                ipv4_from_keys(tx["ip_key"], np.maximum(tx["ip_octet"], 0)), has_ip
            ),
            "latitude": np.where(card_present, ent.merchant_lat[merchant].round(5), np.nan),
            "longitude": np.where(card_present, ent.merchant_lon[merchant].round(5), np.nan),
            "beneficiary_iban": _nullable_str(benef_iban, has_benef),
            "auth_status": col.STATUSES[tx["status"]],
            "decline_reason": _nullable_str(
                col.DECLINE_REASONS[tx["decline_reason"]], tx["decline_reason"] > 0
            ),
        }
    )
    if schema_v2:
        frame["wallet_type"] = col.WALLETS[tx["wallet"]]

    fraud = tx["fraud_type"] > 0
    truth = pd.DataFrame(
        {
            "transaction_id": tx_ids[fraud],
            "client_id": frame["client_id"].to_numpy()[fraud],
            "event_time": frame["event_time"].to_numpy()[fraud],
            "amount": tx["amount"][fraud],
            "fraud_type": col.FRAUD_TYPES[tx["fraud_type"][fraud]],
            "incident_id": sequential_ids(tx["incident"][fraud], "INC", 9),
        }
    )
    return frame, truth


_PAYLOAD_SECTIONS = ["limits", "pin", "contactless", "online_payments", "travel_mode"]
_PROFILE_FIELDS = ["address", "email", "phone", "income"]
_CAMPAIGNS = ["security_tip", "monthly_summary", "new_feature", "payment_received"]


def events_jsonl(
    ent: Entities, rng: np.random.Generator, ev: dict[str, np.ndarray], app_v5_share: float
) -> tuple[list[bytes], np.ndarray]:
    """Sérialise les événements en JSON (une ligne par événement).

    Le schéma est volontairement variable (champs optionnels, charge utile propre
    à chaque type d'événement, champs ajoutés par l'application v5) : c'est la
    source semi-structurée du projet.
    Retourne les lignes JSON et les identifiants d'événements.
    """
    n = len(ev["ts"])
    if n == 0:
        return [], np.array([], dtype=str)
    event_ids = random_hex_ids(rng, n)
    times = iso_times(ev["ts"])
    client_ids = sequential_ids(ev["client"], "C", 8)
    device_ids = hashed_hex_ids(ev["device_key"], "dev_")
    ips = ipv4_from_keys(ev["ip_key"], ev["ip_octet"])
    v5 = rng.random(n) < app_v5_share
    patch = rng.integers(0, 10, n)
    minor = 8 + (ev["ts"] // (30 * 86_400_000)) % 12
    versions = np.where(
        v5,
        np.char.add("5.0.", patch.astype(str)),
        np.char.add(np.char.add("4.", minor.astype(str)), np.char.add(".", patch.astype(str))),
    )
    biometric = rng.random(n) < 0.7
    accuracy = rng.integers(5, 120, n)
    aux = rng.integers(0, 1_000_000, n)

    # Conversion en listes Python : orjson et les accès par index sont bien plus
    # rapides sur des types natifs que sur des scalaires numpy.
    cols = {
        "etype_code": ev["event_type"].tolist(),
        "etype": col.EVENT_TYPES[ev["event_type"]].tolist(),
        "os": col.OS_NAMES[ev["os"]].tolist(),
        "country": col.COUNTRIES[ev["ip_country"]].tolist(),
        "event_id": event_ids.tolist(),
        "time": times.tolist(),
        "client": client_ids.tolist(),
        "device": device_ids.tolist(),
        "ip": list(ips),
        "version": versions.tolist(),
        "v5": v5.tolist(),
        "biometric": biometric.tolist(),
        "accuracy": accuracy.tolist(),
        "aux": aux.tolist(),
        "has_geo": ev["has_geo"].tolist(),
        "lat": np.round(ev["lat"].astype(np.float64), 4).tolist(),
        "lon": np.round(ev["lon"].astype(np.float64), 4).tolist(),
        "result": col.RESULTS[np.maximum(ev["result"], 0)].tolist(),
        "sca_method": col.SCA_METHODS[np.maximum(ev["sca_method"], 0)].tolist(),
        "amount": np.round(ev["amount"].astype(np.float64), 2).tolist(),
        "wallet": col.WALLETS[ev["wallet"]].tolist(),
    }
    lines: list[bytes] = []
    for i in range(n):
        session: dict[str, object] = {
            "device_id": cols["device"][i],
            "device_os": cols["os"][i],
            "app_version": cols["version"][i],
            "ip_address": cols["ip"][i],
            "ip_country": cols["country"][i],
        }
        if cols["v5"][i]:
            session["biometric_enabled"] = cols["biometric"][i]
        record: dict[str, object] = {
            "event_id": cols["event_id"][i],
            "event_type": cols["etype"][i],
            "event_time": cols["time"][i],
            "client_id": cols["client"][i],
            "session": session,
        }
        if cols["has_geo"][i]:
            geo: dict[str, object] = {"lat": cols["lat"][i], "lon": cols["lon"][i]}
            if cols["v5"][i]:
                geo["accuracy_m"] = cols["accuracy"][i]
            record["geo"] = geo
        record["payload"] = _payload(cols["etype_code"][i], cols, i)
        lines.append(orjson.dumps(record))
    return lines, event_ids


def _payload(et: int, cols: dict[str, list], i: int) -> dict[str, object]:
    """Charge utile propre à chaque type d'événement."""
    aux = cols["aux"][i]
    if et in (col.EV_LOGIN, col.EV_LOGIN_FAILED):
        return {"result": cols["result"][i]}
    if et == col.EV_SCA:
        payload: dict[str, object] = {"method": cols["sca_method"][i], "result": cols["result"][i]}
        amount = cols["amount"][i]
        if amount == amount:  # NaN != NaN : montant absent
            payload["amount_eur"] = amount
        return payload
    if et == col.EV_PWD_RESET:
        return {"channel": "email"}
    if et == col.EV_BENEFICIARY:
        return {"beneficiary_iban_masked": f"FR76 **** **** **** {aux % 10_000:04d}"}
    if et == col.EV_SCA_CHANGED:
        return {"new_method": cols["sca_method"][i]}
    if et == col.EV_WALLET:
        return {"wallet": cols["wallet"][i], "card_last4": f"{aux % 10_000:04d}"}
    if et == col.EV_APP_OPEN:
        return {"cold_start": bool(aux % 2)}
    if et == col.EV_CARD_SETTINGS:
        return {"section": _PAYLOAD_SECTIONS[aux % len(_PAYLOAD_SECTIONS)]}
    if et == col.EV_STATEMENT:
        return {"format": "pdf"}
    if et == col.EV_PROFILE:
        return {"field": _PROFILE_FIELDS[aux % len(_PROFILE_FIELDS)]}
    if et == col.EV_NOTIF:
        return {"campaign": _CAMPAIGNS[aux % len(_CAMPAIGNS)]}
    return {}


def dumps_lines(frame: pd.DataFrame) -> list[bytes]:
    """Sérialise un DataFrame en lignes JSON (mode flux)."""
    records = frame.replace({np.nan: None}).to_dict(orient="records")
    return [orjson.dumps(r) for r in records]
