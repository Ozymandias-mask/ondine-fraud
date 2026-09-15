"""Génération des référentiels : clients, cartes, marchands, comptes mules.

Les entités sont d'abord produites sous forme de tableaux numpy compacts
(`Entities`), utilisés par les générateurs de transactions. Les DataFrames
complets (avec les données personnelles fictives) ne sont construits qu'au
moment de l'export, via `clients_frame`, `cards_frame` et `merchants_frame`.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from datetime import timedelta
from functools import lru_cache

import numpy as np
import pandas as pd
from faker import Faker

from . import reference as ref
from .columns import COUNTRIES, COUNTRY_LAT, COUNTRY_LON, FIRST_RISKY, FIRST_TRAVEL
from .config import SimConfig
from .identifiers import (
    external_ibans,
    french_ibans,
    hashed_hex_ids,
    sequential_ids,
    splitmix64,
)

ONDINE_BANK_CODE = 99001
FRENCH_ISP_FIRST_OCTETS = np.array([37, 78, 82, 86, 90, 92, 109, 176])

N_CATEGORIES = len(ref.MERCHANT_CATEGORIES)
CAT_MU = np.array([c[2] for c in ref.MERCHANT_CATEGORIES])
CAT_SIGMA = np.array([c[3] for c in ref.MERCHANT_CATEGORIES])
CAT_ONLINE_SHARE = np.array([c[4] for c in ref.MERCHANT_CATEGORIES])
CAT_WEIGHT = np.array([c[5] for c in ref.MERCHANT_CATEGORIES])
CAT_NAMES = np.array([c[0] for c in ref.MERCHANT_CATEGORIES])
CAT_MCC = np.array([c[1] for c in ref.MERCHANT_CATEGORIES])
CAT_INDEX = {name: i for i, name in enumerate(CAT_NAMES)}

CITY_NAMES = np.array([c[0] for c in ref.CITIES])
CITY_POSTAL = np.array([c[1] for c in ref.CITIES])
CITY_REGION = np.array([c[2] for c in ref.CITIES])
CITY_LAT = np.array([c[3] for c in ref.CITIES])
CITY_LON = np.array([c[4] for c in ref.CITIES])
CITY_WEIGHT = np.array([c[5] for c in ref.CITIES]) / sum(c[5] for c in ref.CITIES)
N_CITIES = len(ref.CITIES)


@dataclass
class MerchantIndex:
    """Index de marchands triés par clé, pour un tirage vectorisé par groupe."""

    order: np.ndarray  # indices de marchands triés par clé
    start: np.ndarray  # position de départ de chaque clé dans `order`
    count: np.ndarray  # nombre de marchands par clé

    def sample(self, keys: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Tire un marchand par clé ; retourne -1 si la clé n'a aucun marchand."""
        counts = self.count[keys]
        offsets = (rng.random(len(keys)) * np.maximum(counts, 1)).astype(np.int64)
        picked = self.order[np.minimum(self.start[keys] + offsets, len(self.order) - 1)]
        return np.where(counts > 0, picked, -1)


def _build_index(keys: np.ndarray, members: np.ndarray, n_keys: int) -> MerchantIndex:
    order = members[np.argsort(keys[members], kind="stable")]
    count = np.bincount(keys[members], minlength=n_keys)
    start = np.concatenate(([0], np.cumsum(count)[:-1]))
    return MerchantIndex(order=order, start=start, count=count)


@dataclass
class Entities:
    """Référentiels sous forme de tableaux numpy (une ligne = une entité)."""

    cfg: SimConfig
    # Clients
    client_segment: np.ndarray
    client_age: np.ndarray
    client_city: np.ndarray
    client_lat: np.ndarray
    client_lon: np.ndarray
    client_activity: np.ndarray
    client_activity_cdf: np.ndarray
    client_os: np.ndarray
    client_isp_octet: np.ndarray
    client_card_start: np.ndarray
    client_card_count: np.ndarray
    # Cartes
    card_client: np.ndarray
    card_type: np.ndarray
    # Marchands
    merchant_category: np.ndarray
    merchant_city: np.ndarray
    merchant_online: np.ndarray
    merchant_country: np.ndarray
    merchant_lat: np.ndarray
    merchant_lon: np.ndarray
    offline_fr_by_city_cat: MerchantIndex
    offline_fr_by_cat: MerchantIndex
    offline_travel_by_cat: MerchantIndex
    offline_risky_by_cat: MerchantIndex
    online_by_cat: MerchantIndex
    # Comptes mules (IBAN externes recevant l'argent des fraudes)
    mule_ibans: np.ndarray

    @property
    def n_clients(self) -> int:
        """Nombre de clients."""
        return len(self.client_segment)

    @property
    def n_cards(self) -> int:
        """Nombre de cartes."""
        return len(self.card_client)


def _entity_rng(cfg: SimConfig) -> np.random.Generator:
    # Flux aléatoire dédié aux référentiels : indépendant des flux mensuels.
    return np.random.default_rng([cfg.seed, 1_000_003])


def build_entities(cfg: SimConfig) -> Entities:
    """Construit les référentiels de manière déterministe à partir de la graine."""
    rng = _entity_rng(cfg)
    n = cfg.n_clients

    seg_weights = np.array([s[1] for s in ref.SEGMENTS])
    segment = rng.choice(len(ref.SEGMENTS), size=n, p=seg_weights / seg_weights.sum())
    age_min = np.array([s[2] for s in ref.SEGMENTS])[segment]
    age_max = np.array([s[3] for s in ref.SEGMENTS])[segment]
    age = rng.integers(age_min, age_max + 1)

    city = rng.choice(N_CITIES, size=n, p=CITY_WEIGHT)
    lat = CITY_LAT[city] + rng.normal(0, 0.03, n)
    lon = CITY_LON[city] + rng.normal(0, 0.03, n)

    # Activité : distribution log-normale (beaucoup de petits utilisateurs, quelques
    # très gros), modulée par le segment puis normalisée à une moyenne de 1.
    seg_mult = np.array([s[4] for s in ref.SEGMENTS])[segment]
    activity = rng.lognormal(0.0, 0.6, n) * seg_mult
    activity /= activity.mean()
    cdf = np.cumsum(activity)
    cdf /= cdf[-1]

    # Cartes : une carte physique par client, une carte virtuelle pour la moitié.
    card_count = 1 + (rng.random(n) < 0.5).astype(np.int64)
    card_start = np.concatenate(([0], np.cumsum(card_count)[:-1]))
    card_client = np.repeat(np.arange(n), card_count)
    slot = np.arange(len(card_client)) - card_start[card_client]
    card_type = np.where(slot == 0, np.where(rng.random(len(card_client)) < 0.15, 1, 0), 2)

    merchants = _build_merchants(cfg, rng)

    mule_ibans = external_ibans(np.arange(cfg.n_mule_accounts, dtype=np.int64) + 7_000_000_000)

    return Entities(
        cfg=cfg,
        client_segment=segment.astype(np.int8),
        client_age=age.astype(np.int16),
        client_city=city.astype(np.int16),
        client_lat=lat.astype(np.float32),
        client_lon=lon.astype(np.float32),
        client_activity=activity.astype(np.float32),
        client_activity_cdf=cdf,
        client_os=(rng.random(n) < 0.42).astype(np.int8),
        client_isp_octet=FRENCH_ISP_FIRST_OCTETS[rng.integers(0, 8, n)],
        client_card_start=card_start,
        client_card_count=card_count,
        card_client=card_client,
        card_type=card_type.astype(np.int8),
        mule_ibans=mule_ibans,
        **merchants,
    )


def _build_merchants(cfg: SimConfig, rng: np.random.Generator) -> dict[str, object]:
    m = cfg.n_merchants
    category = rng.choice(N_CATEGORIES, size=m, p=CAT_WEIGHT / CAT_WEIGHT.sum())
    online = rng.random(m) < CAT_ONLINE_SHARE[category]

    # Pays : les marchands physiques sont à 96 % en France, 3 % dans des pays de
    # voyage, 1 % dans des pays à risque ; les marchands en ligne à 8 % à l'étranger.
    u = rng.random(m)
    n_travel = len(ref.TRAVEL_COUNTRIES)
    n_risky = len(ref.RISKY_COUNTRIES)
    travel_country = FIRST_TRAVEL + rng.integers(0, n_travel, m)
    risky_country = FIRST_RISKY + rng.integers(0, n_risky, m)
    country = np.where(
        online,
        np.where(u < 0.08, travel_country, 0),
        np.where(u < 0.03, travel_country, np.where(u < 0.04, risky_country, 0)),
    )

    # Garantit au moins un marchand par combinaison utile, sinon les petits
    # préréglages de test pourraient avoir des trous dans les tirages.
    required: list[tuple[int, bool, int]] = []
    for cat in range(N_CATEGORIES):
        if CAT_ONLINE_SHARE[cat] < 1:
            required.append((cat, False, 0))
            required.append((cat, False, FIRST_TRAVEL))
        if CAT_ONLINE_SHARE[cat] > 0:
            required.append((cat, True, 0))
    for name in ("atm_cash", "jewelry", "electronics"):
        required.append((CAT_INDEX[name], False, FIRST_RISKY))
    for pos, (cat, is_online, ctry) in enumerate(required[:m]):
        category[pos], online[pos], country[pos] = cat, is_online, ctry

    city = rng.choice(N_CITIES, size=m, p=CITY_WEIGHT)
    domestic = country == 0
    lat = np.where(domestic, CITY_LAT[city], COUNTRY_LAT[country]) + rng.normal(0, 0.02, m)
    lon = np.where(domestic, CITY_LON[city], COUNTRY_LON[country]) + rng.normal(0, 0.02, m)

    idx = np.arange(m)
    offline_fr = idx[~online & domestic]
    offline_travel = idx[~online & (country >= FIRST_TRAVEL) & (country < FIRST_RISKY)]
    offline_risky = idx[~online & (country >= FIRST_RISKY)]
    return {
        "merchant_category": category.astype(np.int16),
        "merchant_city": city.astype(np.int16),
        "merchant_online": online,
        "merchant_country": country.astype(np.int8),
        "merchant_lat": lat.astype(np.float32),
        "merchant_lon": lon.astype(np.float32),
        "offline_fr_by_city_cat": _build_index(
            city * N_CATEGORIES + category, offline_fr, N_CITIES * N_CATEGORIES
        ),
        "offline_fr_by_cat": _build_index(category, offline_fr, N_CATEGORIES),
        "offline_travel_by_cat": _build_index(category, offline_travel, N_CATEGORIES),
        "offline_risky_by_cat": _build_index(category, offline_risky, N_CATEGORIES),
        "online_by_cat": _build_index(category, idx[online], N_CATEGORIES),
    }


@lru_cache(maxsize=4)
def cached_entities(cfg: SimConfig) -> Entities:
    """Référentiels mis en cache par processus (évite de les recalculer par mois)."""
    return build_entities(cfg)


# --------------------------------------------------------------------------- export


def _ascii_slug(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return "".join(ch for ch in normalized.lower() if ch.isalnum())


@lru_cache(maxsize=2)
def name_pools(seed: int) -> dict[str, np.ndarray]:
    """Réservoirs de noms fictifs générés une fois avec Faker (fr_FR)."""
    fake = Faker("fr_FR")
    fake.seed_instance(seed)
    first = np.array(sorted({fake.first_name() for _ in range(6_000)}))
    last = np.array(sorted({fake.last_name() for _ in range(9_000)}))
    return {
        "first": first,
        "first_slug": np.array([_ascii_slug(x) for x in first]),
        "last": last,
        "last_slug": np.array([_ascii_slug(x) for x in last]),
        "street": np.array(sorted({fake.street_name() for _ in range(6_000)})),
        "company": np.array(sorted({fake.company() for _ in range(8_000)})),
    }


def _name_indices(cfg: SimConfig, client_idx: np.ndarray) -> tuple[np.ndarray, ...]:
    pools = name_pools(cfg.seed)
    h = splitmix64(client_idx.astype(np.uint64) + np.uint64(cfg.seed))
    first = (h % np.uint64(len(pools["first"]))).astype(np.int64)
    last = ((h >> np.uint64(20)) % np.uint64(len(pools["last"]))).astype(np.int64)
    return first, last, h


def client_contacts(cfg: SimConfig, client_idx: np.ndarray) -> dict[str, np.ndarray]:
    """Coordonnées fictives d'un ensemble de clients, dérivées de façon stable de l'index.

    Utilisé par le référentiel clients et par les e-mails du support, qui doivent
    citer les mêmes noms, e-mails et IBAN.
    """
    pools = name_pools(cfg.seed)
    first_i, last_i, h = _name_indices(cfg, client_idx)
    suffix = np.char.zfill(((h >> np.uint64(40)) % np.uint64(100)).astype(str), 2)
    domains = np.array(ref.EMAIL_DOMAINS)
    domain = domains[((h >> np.uint64(48)) % np.uint64(len(domains))).astype(np.int64)]
    email = np.char.add(np.char.add(pools["first_slug"][first_i], "."), pools["last_slug"][last_i])
    email = np.char.add(np.char.add(np.char.add(email, suffix), "@"), domain)
    phone = np.char.add(
        "+336", np.char.zfill(((h >> np.uint64(8)) % np.uint64(10**8)).astype(str), 8)
    )
    return {
        "client_id": sequential_ids(client_idx, "C", 8),
        "first_name": pools["first"][first_i],
        "last_name": pools["last"][last_i],
        "email": email,
        "phone": phone,
        "iban": french_ibans(client_idx, ONDINE_BANK_CODE),
    }


def clients_frame(ent: Entities) -> pd.DataFrame:
    """Référentiel clients complet, avec données personnelles fictives."""
    cfg = ent.cfg
    rng = np.random.default_rng([cfg.seed, 1_000_033])
    n = ent.n_clients
    idx = np.arange(n)
    pools = name_pools(cfg.seed)
    contacts = client_contacts(cfg, idx)
    street_num = rng.integers(1, 180, n).astype(str)
    street = pools["street"][rng.integers(0, len(pools["street"]), n)]

    start = pd.Timestamp(cfg.start_date)
    birth = start - pd.to_timedelta(
        ent.client_age.astype(np.int64) * 365 + rng.integers(0, 365, n), unit="D"
    )
    created = pd.Timestamp("2018-01-01") + pd.to_timedelta(
        rng.integers(0, (start - pd.Timestamp("2018-01-01")).days, n), unit="D"
    )
    consent = rng.random(n) < 0.55
    consent_at = created + pd.to_timedelta(rng.integers(0, 30, n), unit="D")

    seg_names = np.array([s[0] for s in ref.SEGMENTS])
    income_probs = np.array(
        [
            [0.70, 0.25, 0.04, 0.01],
            [0.25, 0.50, 0.18, 0.07],
            [0.15, 0.45, 0.25, 0.15],
            [0.30, 0.45, 0.17, 0.08],
        ]
    )
    u = rng.random(n)
    income = (u[:, None] > np.cumsum(income_probs[ent.client_segment], axis=1)).sum(axis=1)
    income_labels = np.array(["<1500", "1500-3000", "3000-5000", ">5000"])

    return pd.DataFrame(
        {
            "client_id": contacts["client_id"],
            "first_name": contacts["first_name"],
            "last_name": contacts["last_name"],
            "birth_date": birth.date,
            "email": contacts["email"],
            "phone": contacts["phone"],
            "street_address": np.char.add(np.char.add(street_num, " "), street),
            "postal_code": CITY_POSTAL[ent.client_city],
            "city": CITY_NAMES[ent.client_city],
            "region": CITY_REGION[ent.client_city],
            "iban": contacts["iban"],
            "segment": seg_names[ent.client_segment],
            "income_band": income_labels[np.minimum(income, 3)],
            "kyc_level": np.where(rng.random(n) < 0.1, "enhanced", "standard"),
            "is_pep": rng.random(n) < 0.002,
            "created_at": created.date,
            "consent_marketing": consent,
            "consent_marketing_at": np.where(consent, consent_at.astype(str), ""),
        }
    )


def cards_frame(ent: Entities) -> pd.DataFrame:
    """Référentiel cartes. Le numéro de carte (PAN) n'est jamais stocké : jeton uniquement."""
    cfg = ent.cfg
    rng = np.random.default_rng([cfg.seed, 1_000_037])
    n = ent.n_cards
    idx = np.arange(n)
    issued = pd.Timestamp("2019-01-01") + pd.to_timedelta(
        rng.integers(0, (pd.Timestamp(cfg.start_date) - pd.Timestamp("2019-01-01")).days, n),
        unit="D",
    )
    expiry = issued + pd.DateOffset(years=4)
    type_labels = np.array(["debit", "credit", "virtual"])
    return pd.DataFrame(
        {
            "card_id": sequential_ids(idx, "K", 9),
            "client_id": sequential_ids(ent.card_client, "C", 8),
            "pan_token": hashed_hex_ids(idx.astype(np.uint64) + np.uint64(3 << 40), "tok_"),
            "last4": np.char.zfill((splitmix64(idx) % np.uint64(10_000)).astype(str), 4),
            "card_type": type_labels[ent.card_type],
            "network": np.where(rng.random(n) < 0.8, "VISA", "MASTERCARD"),
            "issued_at": issued.date,
            "expiry": expiry.strftime("%m/%y"),
            "contactless_enabled": rng.random(n) < 0.93,
            "status": np.where(rng.random(n) < 0.97, "active", "blocked"),
        }
    )


def merchants_frame(ent: Entities) -> pd.DataFrame:
    """Référentiel marchands (personnes morales : pas de donnée personnelle)."""
    cfg = ent.cfg
    rng = np.random.default_rng([cfg.seed, 1_000_039])
    pools = name_pools(cfg.seed)
    m = len(ent.merchant_category)
    idx = np.arange(m)
    domestic = ent.merchant_country == 0
    created = pd.Timestamp("2010-01-01") + pd.to_timedelta(rng.integers(0, 5_000, m), unit="D")
    return pd.DataFrame(
        {
            "merchant_id": sequential_ids(idx, "M", 7),
            "merchant_name": pools["company"][rng.integers(0, len(pools["company"]), m)],
            "category": CAT_NAMES[ent.merchant_category],
            "mcc": CAT_MCC[ent.merchant_category],
            "is_online": ent.merchant_online,
            "city": np.where(domestic, CITY_NAMES[ent.merchant_city], ""),
            "postal_code": np.where(domestic, CITY_POSTAL[ent.merchant_city], ""),
            "country": COUNTRIES[ent.merchant_country],
            "latitude": ent.merchant_lat.round(5),
            "longitude": ent.merchant_lon.round(5),
            "created_at": np.minimum(
                created, pd.Timestamp(cfg.start_date) - timedelta(days=1)
            ).date,
        }
    )


def month_bounds(cfg: SimConfig, month_index: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Début (inclus) et fin (exclue) du mois `month_index`."""
    start = pd.Timestamp(cfg.start_date) + pd.DateOffset(months=month_index)
    return start, start + pd.DateOffset(months=1)
