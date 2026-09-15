"""Comportement légitime des clients : transactions et événements applicatifs."""

from __future__ import annotations

import numpy as np

from . import columns as col
from .entities import (
    CAT_INDEX,
    CAT_MU,
    CAT_ONLINE_SHARE,
    CAT_SIGMA,
    CAT_WEIGHT,
    N_CATEGORIES,
    N_CITIES,
    Entities,
)
from .reference import LEGIT_HOUR_PROBS

# Répartition des canaux par segment (chip, sans contact, e-commerce, DAB, virement).
_CHANNEL_PROBS = np.array(
    [
        [0.18, 0.30, 0.36, 0.06, 0.10],  # étudiant
        [0.20, 0.32, 0.28, 0.05, 0.15],  # jeune actif
        [0.28, 0.27, 0.23, 0.07, 0.15],  # actif
        [0.42, 0.16, 0.15, 0.12, 0.15],  # senior
    ]
)
_CHANNEL_CDF = np.cumsum(_CHANNEL_PROBS, axis=1)

_ATM = CAT_INDEX["atm_cash"]
_POS_WEIGHTS = CAT_WEIGHT * (1 - CAT_ONLINE_SHARE)
_POS_WEIGHTS[_ATM] = 0.0
_ONLINE_WEIGHTS = CAT_WEIGHT * CAT_ONLINE_SHARE

CONTACTLESS_LIMIT_EUR = 50.0  # plafond du sans contact en France (hors wallet mobile)
SCA_THRESHOLD_EUR = 30.0  # exemption DSP2 des paiements en ligne de faible montant


def month_timestamps(
    rng: np.random.Generator,
    n: int,
    start_ms: int,
    end_ms: int,
    hour_probs: np.ndarray = LEGIT_HOUR_PROBS,
) -> np.ndarray:
    """Horodatages (ms) répartis sur la période avec un profil horaire réaliste."""
    n_days = max(1, (end_ms - start_ms) // 86_400_000)
    day = rng.integers(0, n_days, n)
    hour = rng.choice(24, size=n, p=hour_probs)
    ms = day * 86_400_000 + hour * 3_600_000 + rng.integers(0, 3_600_000, n)
    return start_ms + np.minimum(ms, end_ms - start_ms - 1)


def sample_merchants(
    ent: Entities, rng: np.random.Generator, category: np.ndarray, city: np.ndarray, pool: str
) -> np.ndarray:
    """Choisit un marchand par transaction avec repli si la combinaison est vide.

    `pool` : "local" (commerce physique de la ville), "travel" (commerce physique à
    l'étranger), "risky" (pays à risque) ou "online".
    """
    if pool == "local":
        merchant = ent.offline_fr_by_city_cat.sample(city * N_CATEGORIES + category, rng)
    elif pool == "travel":
        merchant = ent.offline_travel_by_cat.sample(category, rng)
    elif pool == "risky":
        merchant = ent.offline_risky_by_cat.sample(category, rng)
    else:
        merchant = ent.online_by_cat.sample(category, rng)
    missing = merchant < 0
    if missing.any():
        fallback = ent.online_by_cat if pool == "online" else ent.offline_fr_by_cat
        merchant[missing] = fallback.sample(category[missing], rng)
    still = merchant < 0
    if still.any():
        merchant[still] = rng.integers(0, len(ent.merchant_category), still.sum())
    return merchant


def legit_transactions(
    ent: Entities, rng: np.random.Generator, client: np.ndarray, ts: np.ndarray, schema_v2: bool
) -> dict[str, np.ndarray]:
    """Transactions légitimes pour une liste (client, horodatage)."""
    n = len(client)
    tx = col.empty(col.TX_COLUMNS, n)
    tx["ts"], tx["client"] = ts, client
    if n == 0:
        return tx

    segment = ent.client_segment[client]
    channel = (rng.random(n)[:, None] > _CHANNEL_CDF[segment]).sum(axis=1).astype(np.int8)
    channel = np.minimum(channel, col.CH_TRANSFER)

    category = np.full(n, -1, dtype=np.int64)
    is_pos = (channel == col.CH_CHIP) | (channel == col.CH_CONTACTLESS)
    is_ecom = channel == col.CH_ECOM
    is_atm = channel == col.CH_ATM
    is_transfer = channel == col.CH_TRANSFER
    category[is_pos] = rng.choice(N_CATEGORIES, is_pos.sum(), p=_POS_WEIGHTS / _POS_WEIGHTS.sum())
    category[is_ecom] = rng.choice(
        N_CATEGORIES, is_ecom.sum(), p=_ONLINE_WEIGHTS / _ONLINE_WEIGHTS.sum()
    )
    category[is_atm] = _ATM

    # Marchand : commerce local (92 %), autre ville française (6 %), voyage (2 %).
    card_present = is_pos | is_atm
    u = rng.random(n)
    home_city = ent.client_city[client].astype(np.int64)
    other_city = rng.integers(0, N_CITIES, n)
    local = card_present & (u < 0.92)
    elsewhere = card_present & (u >= 0.92) & (u < 0.98)
    travel = card_present & (u >= 0.98)
    for mask, city, pool in (
        (local, home_city, "local"),
        (elsewhere, other_city, "local"),
        (travel, home_city, "travel"),
    ):
        if mask.any():
            tx["merchant"][mask] = sample_merchants(ent, rng, category[mask], city[mask], pool)
    if is_ecom.any():
        tx["merchant"][is_ecom] = sample_merchants(
            ent, rng, category[is_ecom], home_city[is_ecom], "online"
        )

    # Montants : log-normale par catégorie ; retraits arrondis ; virements plus élevés.
    amount = np.exp(rng.normal(CAT_MU[np.maximum(category, 0)], CAT_SIGMA[np.maximum(category, 0)]))
    amount[is_atm] = np.clip(np.round(amount[is_atm] / 20) * 20, 20, 1000)
    amount[is_transfer] = np.exp(rng.normal(4.8, 1.0, is_transfer.sum()))
    amount = np.clip(np.round(amount, 2), 0.5, 15_000)

    # Au-delà du plafond sans contact, le paiement bascule en insertion de la puce,
    # sauf s'il passe par un wallet mobile (authentification sur le téléphone).
    wallet = np.zeros(n, dtype=np.int8)
    is_contactless = channel == col.CH_CONTACTLESS
    if schema_v2:
        uses_wallet = is_contactless & (rng.random(n) < 0.30)
        wallet[uses_wallet] = np.where(ent.client_os[client[uses_wallet]] == 1, 1, 2)
    over_limit = is_contactless & (wallet == 0) & (amount > CONTACTLESS_LIMIT_EUR)
    channel[over_limit] = col.CH_CHIP
    is_contactless &= ~over_limit

    entry = np.full(n, col.EM_CHIP, dtype=np.int8)
    entry[is_contactless] = np.where(wallet[is_contactless] > 0, col.EM_TOKEN, col.EM_CONTACTLESS)
    entry[is_ecom] = col.EM_CNP
    entry[is_transfer] = col.EM_TRANSFER
    entry[travel & (rng.random(n) < 0.05)] = col.EM_MAGSTRIPE

    # Carte : physique en magasin ; physique ou virtuelle en ligne ; aucune pour un virement.
    first_card = ent.client_card_start[client]
    has_virtual = ent.client_card_count[client] > 1
    use_virtual = is_ecom & has_virtual & (rng.random(n) < 0.5)
    tx["card"] = np.where(is_transfer, -1, first_card + use_virtual.astype(np.int64))

    # 3-D Secure (DSP2) : obligatoire au-dessus de 30 EUR, sauf exemptions.
    three_ds = np.full(n, -1, dtype=np.int8)
    sca_prob = np.where(amount >= SCA_THRESHOLD_EUR, 0.95, 0.20)
    three_ds[is_ecom] = (rng.random(is_ecom.sum()) < sca_prob[is_ecom]).astype(np.int8)

    # Appareil et IP pour les opérations à distance (e-commerce et virement).
    remote = is_ecom | is_transfer
    tx["device_key"] = np.where(remote, client * 4 + (rng.random(n) < 0.05), -1)
    mobile = rng.random(n) < 0.25
    tx["ip_octet"] = np.where(
        remote,
        np.where(mobile, col.MOBILE_IP_OCTETS[rng.integers(0, 3, n)], ent.client_isp_octet[client]),
        -1,
    )
    tx["ip_key"] = np.where(mobile, rng.integers(0, 2**40, n), client)
    tx["ip_country"] = np.where(remote, 0, -1)

    # Bénéficiaires habituels (3 par client) ou nouveau bénéficiaire (10 %).
    new_benef = rng.random(n) < 0.10
    tx["beneficiary_key"] = np.where(
        is_transfer,
        np.where(
            new_benef,
            col.NEW_BENEFICIARY_OFFSET + rng.integers(0, 2**40, n),
            client * 8 + rng.integers(0, 3, n),
        ),
        -1,
    )

    # Refus légitimes : fonds insuffisants, et faux positifs des règles actuelles.
    status = np.zeros(n, dtype=np.int8)
    reason = np.zeros(n, dtype=np.int8)
    v = rng.random(n)
    funds = v < 0.015
    suspected = (v >= 0.015) & (v < 0.0165) & ~is_transfer
    status[funds | suspected] = 1
    reason[funds] = col.DR_FUNDS
    reason[suspected] = col.DR_SUSPECTED

    tx.update(
        channel=channel,
        amount=amount,
        entry_mode=entry,
        three_ds=three_ds,
        status=status,
        decline_reason=reason,
        wallet=wallet,
    )
    return tx


def linked_events(
    ent: Entities, rng: np.random.Generator, tx: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Événements précédant les opérations à distance : connexion et 3-D Secure."""
    remote = np.flatnonzero((tx["channel"] == col.CH_ECOM) | (tx["channel"] == col.CH_TRANSFER))
    with_sca = np.flatnonzero((tx["three_ds"] == 1) | (tx["channel"] == col.CH_TRANSFER))

    logins = col.empty(col.EVENT_COLUMNS, len(remote))
    _copy_session(logins, tx, remote)
    logins["ts"] = tx["ts"][remote] - rng.integers(20_000, 600_000, len(remote))
    logins["event_type"][:] = col.EV_LOGIN
    logins["result"][:] = 0

    sca = col.empty(col.EVENT_COLUMNS, len(with_sca))
    _copy_session(sca, tx, with_sca)
    sca["ts"] = tx["ts"][with_sca] - rng.integers(3_000, 60_000, len(with_sca))
    sca["event_type"][:] = col.EV_SCA
    sca["sca_method"] = np.where(rng.random(len(with_sca)) < 0.7, 0, 1).astype(np.int8)
    sca["result"] = (rng.random(len(with_sca)) < 0.03).astype(np.int8)
    sca["amount"] = tx["amount"][with_sca].astype(np.float32)

    events = col.concat(col.EVENT_COLUMNS, [logins, sca])
    _fill_client_context(ent, rng, events)
    return events


def background_events(
    ent: Entities, rng: np.random.Generator, client: np.ndarray, ts: np.ndarray
) -> dict[str, np.ndarray]:
    """Usage courant de l'application (consultation du solde, ouverture, etc.)."""
    n = len(client)
    ev = col.empty(col.EVENT_COLUMNS, n)
    ev["ts"], ev["client"] = ts, client
    ev["event_type"] = rng.choice(
        [
            col.EV_APP_OPEN,
            col.EV_BALANCE,
            col.EV_CARD_SETTINGS,
            col.EV_STATEMENT,
            col.EV_PROFILE,
            col.EV_NOTIF,
        ],
        size=n,
        p=[0.45, 0.30, 0.08, 0.05, 0.02, 0.10],
    ).astype(np.int8)
    ev["device_key"] = client * 4 + (rng.random(n) < 0.05)
    mobile = rng.random(n) < 0.35
    ev["ip_octet"] = np.where(
        mobile, col.MOBILE_IP_OCTETS[rng.integers(0, 3, n)], ent.client_isp_octet[client]
    )
    ev["ip_key"] = np.where(mobile, rng.integers(0, 2**40, n), client)
    _fill_client_context(ent, rng, ev)
    return ev


def _copy_session(ev: dict[str, np.ndarray], tx: dict[str, np.ndarray], idx: np.ndarray) -> None:
    for name in ("client", "device_key", "ip_octet", "ip_key", "fraud_type", "incident"):
        ev[name] = tx[name][idx].astype(ev[name].dtype)
    ev["ip_country"] = np.maximum(tx["ip_country"][idx], 0).astype(np.int8)


def _fill_client_context(
    ent: Entities, rng: np.random.Generator, ev: dict[str, np.ndarray]
) -> None:
    """Système d'exploitation et géolocalisation (seulement si le client l'a autorisée)."""
    client = ev["client"]
    n = len(client)
    ev["os"] = ent.client_os[client]
    # ~60 % des clients autorisent la géolocalisation (choix stable par client).
    ev["has_geo"] = (client % 10) < 6
    home = ev["ip_country"] == 0
    ev["lat"] = np.where(
        home, ent.client_lat[client] + rng.normal(0, 0.01, n), col.COUNTRY_LAT[ev["ip_country"]]
    ).astype(np.float32)
    ev["lon"] = np.where(
        home, ent.client_lon[client] + rng.normal(0, 0.01, n), col.COUNTRY_LON[ev["ip_country"]]
    ).astype(np.float32)
