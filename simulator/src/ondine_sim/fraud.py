"""Scénarios de fraude injectés dans les données.

Chaque scénario reproduit un mode opératoire connu, avec ses signaux caractéristiques
(et parfois l'absence de signal, pour que la détection ne soit pas triviale) :

1. card_testing      : petits achats de test en ligne puis gros achats (carte volée).
2. cloned_card       : carte copiée utilisée physiquement dans un pays à risque.
3. account_takeover  : prise de contrôle du compte depuis un nouvel appareil, virements.
4. sim_swap          : détournement de la ligne mobile pour intercepter les SMS.
5. app_scam          : faux conseiller ; la victime fait elle-même le virement.
6. wallet_enrolment  : carte enrôlée dans un wallet mobile frauduleux puis paiements
                       sans contact en France. Apparaît au mois de dérive seulement.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import columns as col
from .behaviour import _fill_client_context, sample_merchants
from .entities import CAT_INDEX, N_CITIES, Entities
from .reference import FRAUD_HOUR_PROBS, LEGIT_HOUR_PROBS

_WEIGHTS_BEFORE_DRIFT = {
    col.FT_CARD_TESTING: 0.27,
    col.FT_CLONED: 0.18,
    col.FT_ATO: 0.25,
    col.FT_SIM_SWAP: 0.10,
    col.FT_APP_SCAM: 0.20,
}
_WALLET_SHARE_AFTER_DRIFT = 0.30

MINUTE = 60_000
HOUR = 60 * MINUTE


@dataclass
class FraudOutput:
    """Transactions et événements frauduleux générés pour un lot de victimes."""

    tx: dict[str, np.ndarray]
    events: dict[str, np.ndarray]


def scenario_weights(drift_active: bool) -> tuple[np.ndarray, np.ndarray]:
    """Types de fraude et probabilités, avant ou après l'apparition de la dérive."""
    types = list(_WEIGHTS_BEFORE_DRIFT)
    probs = list(_WEIGHTS_BEFORE_DRIFT.values())
    if drift_active:
        probs = [p * (1 - _WALLET_SHARE_AFTER_DRIFT) for p in probs]
        types.append(col.FT_WALLET)
        probs.append(_WALLET_SHARE_AFTER_DRIFT)
    return np.array(types), np.array(probs) / sum(probs)


def inject_fraud(
    ent: Entities,
    rng: np.random.Generator,
    victims: np.ndarray,
    start_ms: int,
    end_ms: int,
    drift_active: bool,
    incident_base: int,
    time_scale: float = 1.0,
) -> FraudOutput:
    """Génère les incidents de fraude des `victims` sur la période donnée.

    `time_scale` < 1 compresse la durée des incidents (utilisé en mode flux pour
    qu'une attaque de plusieurs heures se déroule en quelques minutes de démo).
    `incident_base` garantit des identifiants d'incident uniques entre lots.
    """
    types, probs = scenario_weights(drift_active)
    scenario = rng.choice(types, size=len(victims), p=probs)
    incident_ids = incident_base + np.arange(len(victims))

    builders = {
        col.FT_CARD_TESTING: _card_testing,
        col.FT_CLONED: _cloned_card,
        col.FT_ATO: _account_takeover,
        col.FT_SIM_SWAP: _sim_swap,
        col.FT_APP_SCAM: _app_scam,
        col.FT_WALLET: _wallet_enrolment,
    }
    tx_parts, ev_parts = [], []
    for ftype, builder in builders.items():
        mask = scenario == ftype
        if not mask.any():
            continue
        v = victims[mask]
        hours = LEGIT_HOUR_PROBS if ftype == col.FT_APP_SCAM else FRAUD_HOUR_PROBS
        span = max(end_ms - start_ms - int(16 * HOUR * time_scale), 1)
        t0 = _incident_start(rng, len(v), start_ms, span, hours)
        tx, ev = builder(ent, rng, v, t0, time_scale)
        for part in (tx, ev):
            # Les builders stockent l'index local de la victime dans « incident » :
            # on le remplace ici par l'identifiant global de l'incident.
            part["fraud_type"][:] = ftype
            part["incident"] = incident_ids[mask][part["incident"]]
        tx_parts.append(tx)
        ev_parts.append(ev)

    events = col.concat(col.EVENT_COLUMNS, ev_parts)
    _fill_client_context(ent, rng, events)
    return FraudOutput(tx=col.concat(col.TX_COLUMNS, tx_parts), events=events)


# --------------------------------------------------------------------------- outils


def _incident_start(
    rng: np.random.Generator, n: int, start_ms: int, span_ms: int, hour_probs: np.ndarray
) -> np.ndarray:
    if span_ms < 86_400_000:  # fenêtre courte (mode flux) : pas de profil horaire
        return start_ms + rng.integers(0, span_ms, n)
    days = span_ms // 86_400_000
    hour = rng.choice(24, size=n, p=hour_probs)
    return start_ms + rng.integers(0, days, n) * 86_400_000 + hour * HOUR + rng.integers(0, HOUR, n)


def _repeat(victims: np.ndarray, k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Répète chaque victime k fois ; retourne (index victime, rang dans l'incident)."""
    owner = np.repeat(np.arange(len(victims)), k)
    rank = np.arange(len(owner)) - np.repeat(np.cumsum(k) - k, k)
    return owner, rank


def _gaps(
    rng: np.random.Generator, owner: np.ndarray, low_ms: float, high_ms: float, scale: float
) -> np.ndarray:
    """Écarts cumulés aléatoires entre opérations successives d'un même incident."""
    gaps = rng.uniform(low_ms, high_ms, len(owner)) * scale
    cum = np.cumsum(gaps)
    first = np.r_[True, owner[1:] != owner[:-1]]
    group_start = np.maximum.accumulate(np.where(first, np.arange(len(owner)), 0))
    return (cum - cum[group_start] + gaps[group_start]).astype(np.int64)


def _tx(n: int, client: np.ndarray, owner: np.ndarray, ts: np.ndarray) -> dict[str, np.ndarray]:
    tx = col.empty(col.TX_COLUMNS, n)
    tx["client"], tx["ts"], tx["incident"] = client, ts, owner
    return tx


def _ev(
    n: int, client: np.ndarray, owner: np.ndarray, ts: np.ndarray, event_type: int
) -> dict[str, np.ndarray]:
    ev = col.empty(col.EVENT_COLUMNS, n)
    ev["client"], ev["ts"], ev["incident"] = client, ts, owner
    ev["event_type"][:] = event_type
    return ev


def _decline(rng: np.random.Generator, tx: dict[str, np.ndarray], prob: float, reason: int) -> None:
    """Une partie des fraudes est déjà bloquée par les règles actuelles de la banque."""
    declined = rng.random(len(tx["ts"])) < prob
    tx["status"][declined] = 1
    tx["decline_reason"][declined] = reason


def _fraud_device(rng: np.random.Generator, n: int) -> np.ndarray:
    return col.NEW_DEVICE_OFFSET + rng.integers(0, 2**40, n)


def _set_ip(rng: np.random.Generator, obj: dict[str, np.ndarray], kind: str) -> None:
    n = len(obj["ts"])
    if kind == "risky":
        obj["ip_octet"] = col.RISKY_IP_OCTETS[rng.integers(0, len(col.RISKY_IP_OCTETS), n)]
        obj["ip_country"] = rng.integers(col.FIRST_RISKY, col.N_COUNTRIES, n).astype(np.int8)
    else:  # réseau mobile français : aucun signal géographique
        obj["ip_octet"] = col.MOBILE_IP_OCTETS[rng.integers(0, len(col.MOBILE_IP_OCTETS), n)]
        obj["ip_country"][:] = 0
    obj["ip_key"] = rng.integers(0, 2**40, n)


def _to_mule(
    ent: Entities, rng: np.random.Generator, tx: dict[str, np.ndarray], low: float, high: float
) -> None:
    n = len(tx["ts"])
    tx["channel"][:] = col.CH_TRANSFER
    tx["entry_mode"][:] = col.EM_TRANSFER
    tx["amount"] = np.round(rng.uniform(low, high, n), 2)
    tx["beneficiary_key"] = rng.integers(0, len(ent.mule_ibans), n)
    tx["beneficiary_is_mule"][:] = True


def _session_events(
    rng: np.random.Generator,
    client: np.ndarray,
    t0: np.ndarray,
    device: np.ndarray,
    steps: list[tuple[int, int, dict[str, object]]],
    ip_kind: str,
) -> dict[str, np.ndarray]:
    """Suite d'événements d'une session frauduleuse (même appareil, même IP)."""
    parts = []
    ip_template = col.empty(col.EVENT_COLUMNS, len(client))
    ip_template["ts"] = t0
    _set_ip(rng, ip_template, ip_kind)
    for offset_ms, event_type, extra in steps:
        ev = _ev(len(client), client, np.arange(len(client)), t0 + offset_ms, event_type)
        ev["device_key"] = device
        for name in ("ip_octet", "ip_key", "ip_country"):
            ev[name] = ip_template[name]
        for name, value in extra.items():
            ev[name][:] = value
        parts.append(ev)
    return col.concat(col.EVENT_COLUMNS, parts)


# --------------------------------------------------------------------------- scénarios


def _card_testing(
    ent: Entities, rng: np.random.Generator, victims: np.ndarray, t0: np.ndarray, scale: float
) -> tuple[dict, dict]:
    n_tests = rng.integers(4, 13, len(victims))
    n_big = rng.integers(1, 4, len(victims))
    k = n_tests + n_big
    owner, rank = _repeat(victims, k)
    n = len(owner)
    card = ent.client_card_start[victims] + (ent.client_card_count[victims] - 1)  # dernière carte
    tx = _tx(n, victims[owner], owner, t0[owner] + _gaps(rng, owner, 20_000, 120_000, scale))

    is_test = rank < n_tests[owner]
    test_category = np.where(rng.random(n) < 0.6, CAT_INDEX["digital_goods"], CAT_INDEX["gambling"])
    big_category = rng.choice(
        [CAT_INDEX["electronics"], CAT_INDEX["jewelry"], CAT_INDEX["airline"]], n
    )
    category = np.where(is_test, test_category, big_category)
    tx["merchant"] = sample_merchants(ent, rng, category, np.zeros(n, dtype=np.int64), "online")
    # Les gros achats arrivent 10 à 60 minutes après les tests.
    pause = (rng.uniform(10, 60, n) * MINUTE * scale).astype(np.int64)
    tx["ts"] = tx["ts"] + np.where(is_test, 0, pause)
    tx["amount"] = np.round(
        np.where(is_test, rng.uniform(0.5, 2.99, n), np.exp(rng.normal(6.3, 0.5, n))), 2
    )
    tx["card"] = card[owner]
    tx["channel"][:] = col.CH_ECOM
    tx["entry_mode"][:] = col.EM_CNP
    tx["three_ds"][:] = 0  # petits montants exemptés, marchands peu regardants
    tx["device_key"] = _fraud_device(rng, len(victims))[owner]
    _set_ip(rng, tx, "risky")
    _decline(rng, tx, 0.30, col.DR_CVV)
    return tx, col.empty(col.EVENT_COLUMNS, 0)


def _cloned_card(
    ent: Entities, rng: np.random.Generator, victims: np.ndarray, t0: np.ndarray, scale: float
) -> tuple[dict, dict]:
    k = rng.integers(2, 7, len(victims))
    owner, _ = _repeat(victims, k)
    n = len(owner)
    offsets = _gaps(rng, owner, 5 * MINUTE, 90 * MINUTE, scale)
    tx = _tx(n, victims[owner], owner, t0[owner] + offsets)
    category = rng.choice(
        [CAT_INDEX["atm_cash"], CAT_INDEX["jewelry"], CAT_INDEX["electronics"]],
        n,
        p=[0.5, 0.25, 0.25],
    )
    tx["merchant"] = sample_merchants(ent, rng, category, np.zeros(n, dtype=np.int64), "risky")
    is_atm = category == CAT_INDEX["atm_cash"]
    tx["channel"] = np.where(is_atm, col.CH_ATM, col.CH_CHIP).astype(np.int8)
    tx["amount"] = np.round(
        np.where(is_atm, rng.integers(4, 11, n) * 50.0, np.exp(rng.normal(6.0, 0.6, n))), 2
    )
    tx["card"] = ent.client_card_start[victims][owner]
    tx["entry_mode"] = np.where(rng.random(n) < 0.7, col.EM_MAGSTRIPE, col.EM_CHIP).astype(np.int8)
    _decline(rng, tx, 0.35, col.DR_DNH)
    return tx, col.empty(col.EVENT_COLUMNS, 0)


def _account_takeover(
    ent: Entities, rng: np.random.Generator, victims: np.ndarray, t0: np.ndarray, scale: float
) -> tuple[dict, dict]:
    nv = len(victims)
    device = _fraud_device(rng, nv)
    m = int(MINUTE * scale)
    events = _session_events(
        rng,
        victims,
        t0,
        device,
        [
            (0, col.EV_LOGIN_FAILED, {"result": 1}),
            (m, col.EV_LOGIN_FAILED, {"result": 1}),
            (2 * m, col.EV_PWD_RESET, {"result": 0}),
            (4 * m, col.EV_LOGIN, {"result": 0}),
            (6 * m, col.EV_BENEFICIARY, {"result": 0}),
            (8 * m, col.EV_SCA, {"result": 0, "sca_method": 2}),
        ],
        ip_kind="risky",
    )
    k = rng.integers(1, 4, nv)
    owner, _ = _repeat(victims, k)
    tx = _tx(
        len(owner),
        victims[owner],
        owner,
        t0[owner] + 9 * m + _gaps(rng, owner, 2 * MINUTE, 40 * MINUTE, scale),
    )
    _to_mule(ent, rng, tx, 800, 5_000)
    tx["device_key"] = device[owner]
    ip_by_victim = {name: events[name][:nv] for name in ("ip_octet", "ip_key", "ip_country")}
    for name, values in ip_by_victim.items():
        tx[name] = values[owner]
    _decline(rng, tx, 0.15, col.DR_DNH)
    return tx, events


def _sim_swap(
    ent: Entities, rng: np.random.Generator, victims: np.ndarray, t0: np.ndarray, scale: float
) -> tuple[dict, dict]:
    nv = len(victims)
    device = _fraud_device(rng, nv)
    m = int(MINUTE * scale)
    events = _session_events(
        rng,
        victims,
        t0,
        device,
        [
            (0, col.EV_LOGIN, {"result": 0, "sca_method": 2}),
            (3 * m, col.EV_SCA_CHANGED, {"result": 0, "sca_method": 2}),
            (5 * m, col.EV_BENEFICIARY, {"result": 0}),
            (7 * m, col.EV_SCA, {"result": 0, "sca_method": 2}),
        ],
        ip_kind="mobile",
    )
    k = rng.integers(1, 3, nv)
    owner, _ = _repeat(victims, k)
    tx = _tx(
        len(owner),
        victims[owner],
        owner,
        t0[owner] + 8 * m + _gaps(rng, owner, 3 * MINUTE, 30 * MINUTE, scale),
    )
    _to_mule(ent, rng, tx, 1_500, 9_000)
    tx["device_key"] = device[owner]
    for name in ("ip_octet", "ip_key", "ip_country"):
        tx[name] = events[name][:nv][owner]
    _decline(rng, tx, 0.10, col.DR_DNH)
    return tx, events


def _app_scam(
    ent: Entities, rng: np.random.Generator, victims: np.ndarray, t0: np.ndarray, scale: float
) -> tuple[dict, dict]:
    # La victime agit depuis son propre appareil et sa connexion habituelle :
    # seuls le nouveau bénéficiaire et le montant inhabituel trahissent la fraude.
    nv = len(victims)
    m = int(MINUTE * scale)
    device = victims * 4
    home_ip = {
        "ip_octet": ent.client_isp_octet[victims],
        "ip_key": victims,
        "ip_country": np.zeros(nv, dtype=np.int8),
    }
    parts = []
    for offset, etype, extra in (
        (0, col.EV_LOGIN, {"result": 0}),
        (4 * m, col.EV_BENEFICIARY, {"result": 0}),
        (6 * m, col.EV_SCA, {"result": 0, "sca_method": 0}),
    ):
        ev = _ev(nv, victims, np.arange(nv), t0 + offset, etype)
        ev["device_key"] = device
        for name, values in home_ip.items():
            ev[name] = values.astype(ev[name].dtype)
        for name, value in extra.items():
            ev[name][:] = value
        parts.append(ev)
    events = col.concat(col.EVENT_COLUMNS, parts)

    k = rng.integers(1, 3, nv)
    owner, _ = _repeat(victims, k)
    tx = _tx(
        len(owner),
        victims[owner],
        owner,
        t0[owner] + 7 * m + _gaps(rng, owner, 5 * MINUTE, 60 * MINUTE, scale),
    )
    _to_mule(ent, rng, tx, 1_000, 6_000)
    tx["device_key"] = device[owner]
    for name, values in home_ip.items():
        tx[name] = values[owner].astype(tx[name].dtype)
    _decline(rng, tx, 0.05, col.DR_SUSPECTED)
    return tx, events


def _wallet_enrolment(
    ent: Entities, rng: np.random.Generator, victims: np.ndarray, t0: np.ndarray, scale: float
) -> tuple[dict, dict]:
    # Nouveau mode opératoire : paiements en magasin, en France, via un wallet
    # enrôlé sur le téléphone du fraudeur. Les signaux « pays » et « IP » sont absents.
    nv = len(victims)
    device = _fraud_device(rng, nv)
    wallet = rng.integers(1, 3, nv).astype(np.int8)
    m = int(MINUTE * scale)
    events = _session_events(
        rng,
        victims,
        t0,
        device,
        [
            (0, col.EV_SCA, {"result": 0, "sca_method": 2}),
            (2 * m, col.EV_WALLET, {"result": 0}),
        ],
        ip_kind="mobile",
    )
    events["wallet"] = np.tile(wallet, 2)

    k = rng.integers(3, 9, nv)
    owner, _ = _repeat(victims, k)
    n = len(owner)
    tx = _tx(
        n,
        victims[owner],
        owner,
        t0[owner] + 30 * m + _gaps(rng, owner, 10 * MINUTE, 90 * MINUTE, scale),
    )
    category = rng.choice(
        [
            CAT_INDEX["electronics"],
            CAT_INDEX["clothing"],
            CAT_INDEX["jewelry"],
            CAT_INDEX["grocery"],
        ],
        n,
        p=[0.4, 0.3, 0.15, 0.15],
    )
    city = rng.integers(0, N_CITIES, nv)[owner]
    tx["merchant"] = sample_merchants(ent, rng, category, city, "local")
    tx["channel"][:] = col.CH_CONTACTLESS
    tx["entry_mode"][:] = col.EM_TOKEN
    tx["wallet"] = wallet[owner]
    tx["card"] = ent.client_card_start[victims][owner]
    tx["amount"] = np.round(np.exp(rng.normal(5.2, 0.5, n)), 2)
    _decline(rng, tx, 0.03, col.DR_DNH)
    return tx, events
