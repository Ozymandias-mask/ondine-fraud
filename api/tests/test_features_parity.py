"""Cohérence entre les variables calculées en production et celles vues à l'entraînement.

C'est le test le plus important de la mise en service : il rejoue dans le magasin en ligne
les **messages bruts** (couche bronze, identique à ce que transporte Kafka), dans l'ordre
chronologique, puis compare pour chaque transaction du dernier mois les variables calculées
par l'API à celles calculées en SQL par la couche gold.

Un écart ici signifierait que le modèle voit en production des données différentes de celles
sur lesquelles il a appris — la panne la plus coûteuse et la plus silencieuse d'un système de
détection de fraude.
"""

from __future__ import annotations

import math
from decimal import Decimal

import fakeredis
import pytest
from ondine_api.features import compute
from ondine_api.store import OnlineStore
from ondine_pipelines.lake import connect

# Variables comparées : toutes celles que l'API doit reconstituer sans le data lake.
COMPARED = (
    "amount",
    "log_amount",
    "channel",
    "entry_mode",
    "mcc",
    "merchant_category",
    "merchant_country",
    "is_foreign_merchant",
    "is_online_merchant",
    "event_hour",
    "is_weekend",
    "is_night",
    "three_ds_authenticated",
    "wallet_type",
    "beneficiary_country",
    "beneficiary_bank_code",
    "card_tx_count_1h",
    "card_tx_count_24h",
    "card_amount_sum_24h",
    "card_distinct_merchants_1h",
    "card_small_online_tx_1h",
    "card_declined_1h",
    "seconds_since_prev_card_tx",
    "client_tx_count_30d",
    "client_avg_amount_30d",
    "client_max_amount_30d",
    "amount_to_client_avg",
    "client_distinct_countries_7d",
    "client_transfers_24h",
    "client_transfer_amount_24h",
    "km_from_prev_physical",
    "travel_speed_kmh",
    "device_age_hours_for_client",
    "is_new_beneficiary",
    "beneficiary_new_senders_30d",
    "failed_logins_24h",
    "password_resets_24h",
    "beneficiaries_added_24h",
    "foreign_ip_logins_24h",
    "new_device_logins_24h",
    "sms_otp_challenges_1h",
    "sca_method_changes_7d",
    "wallet_provisionings_7d",
)


def _rows(con, query: str) -> list[dict]:
    result = con.execute(query)
    columns = [d[0] for d in result.description]
    return [dict(zip(columns, row, strict=True)) for row in result.fetchall()]


def _equal(online, offline) -> bool:
    """Égalité tolérante aux arrondis, stricte sur les valeurs manquantes."""
    if offline is None or (isinstance(offline, float) and math.isnan(offline)):
        return online is None
    if online is None:
        return False
    if isinstance(offline, bool) or isinstance(online, bool):
        return bool(online) == bool(offline)
    if isinstance(offline, Decimal):
        offline = float(offline)
    if isinstance(offline, int | float) and isinstance(online, int | float):
        return math.isclose(float(online), float(offline), rel_tol=1e-6, abs_tol=1e-6)
    return str(online) == str(offline)


@pytest.fixture(scope="module")
def replayed(lake):
    """Rejoue bronze dans le magasin, en s'arrêtant sur chaque transaction du dernier mois."""
    store = OnlineStore(fakeredis.FakeStrictRedis(decode_responses=True), lake.pseudonymisation_key)
    con = connect(lake)
    try:
        merchants = _rows(
            con,
            f"SELECT merchant_id, category, is_online"
            f" FROM read_parquet('{lake.uri('silver', 'merchants', 'merchants.parquet')}')",
        )
        store.load_merchants(merchants)
        # Les messages bruts effectivement retenus : ceux que la couche silver a validés et
        # dédoublonnés. L'API et le consommateur appliquent la même validation avant d'écrire
        # dans le magasin ; les autres partent en file d'erreurs, ici comme en production.
        transactions = _rows(
            con,
            "SELECT b.* FROM read_parquet"
            f"('{lake.uri('bronze', 'transactions', '**', '*.parquet')}') b"
            " SEMI JOIN read_parquet"
            f"('{lake.uri('silver', 'transactions', '**', '*.parquet')}') s"
            " USING (transaction_id) ORDER BY b.event_time",
        )
        events = _rows(
            con,
            "SELECT b.* FROM read_parquet"
            f"('{lake.uri('bronze', 'app_events', '**', '*.parquet')}') b"
            " SEMI JOIN read_parquet"
            f"('{lake.uri('silver', 'app_events', '**', '*.parquet')}') s"
            " USING (event_id) ORDER BY b.event_time",
        )
        last_month = max(t["event_time"].strftime("%Y-%m") for t in transactions)
        gold = {
            row["transaction_id"]: row
            for row in _rows(
                con,
                "SELECT * FROM read_parquet"
                f"('{lake.uri('gold', 'transaction_features', '**', '*.parquet')}')"
                f" WHERE strftime(event_date, '%Y-%m') = '{last_month}'",
            )
        }
    finally:
        con.close()

    timeline = sorted(
        [(t["event_time"], 0, t) for t in transactions] + [(e["event_time"], 1, e) for e in events],
        key=lambda item: (item[0], item[1]),
    )
    compared, mismatches = 0, {}
    for _, kind, record in timeline:
        if kind == 1:
            store.record_event(record)
            continue
        expected = gold.get(record["transaction_id"])
        if expected is not None:
            online = compute(record, store.state(record))
            compared += 1
            for name in COMPARED:
                if not _equal(online.get(name), expected.get(name)):
                    mismatches.setdefault(name, []).append(
                        (record["transaction_id"], online.get(name), expected.get(name))
                    )
        store.record_transaction(record)
    return {"compared": compared, "mismatches": mismatches, "features": COMPARED}


def test_online_features_match_the_training_features(replayed):
    """Aucune variable ne diffère entre le calcul en ligne et le calcul de la couche gold."""
    assert replayed["compared"] > 200, "trop peu de transactions comparées pour conclure"
    details = {
        name: (len(cases), cases[:3]) for name, cases in sorted(replayed["mismatches"].items())
    }
    assert not details, f"variables divergentes entre production et entraînement : {details}"


def test_every_model_feature_is_covered(replayed, lake):
    """Le test couvre bien *toutes* les variables d'entrée du modèle, sans oubli."""
    from ondine_ml.dataset import EXCLUDED, gold_pattern

    con = connect(lake)
    try:
        columns = [
            row[0]
            for row in con.execute(
                f"DESCRIBE SELECT * FROM read_parquet('{gold_pattern(lake)}')"
            ).fetchall()
        ]
    finally:
        con.close()
    expected = {c for c in columns if c not in EXCLUDED}
    assert expected == set(replayed["features"])
