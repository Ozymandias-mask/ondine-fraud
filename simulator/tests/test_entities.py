import re

import numpy as np
from ondine_sim.entities import (
    CAT_ONLINE_SHARE,
    N_CATEGORIES,
    build_entities,
    cards_frame,
    client_contacts,
    clients_frame,
    merchants_frame,
)


def test_entities_are_reproducible(small_cfg, entities):
    again = build_entities(small_cfg)
    assert np.array_equal(again.client_city, entities.client_city)
    assert np.array_equal(again.merchant_category, entities.merchant_category)


def test_every_client_has_one_or_two_cards(entities):
    assert entities.n_clients == 5_000
    assert set(np.unique(entities.client_card_count)) <= {1, 2}
    assert entities.n_cards == entities.client_card_count.sum()


def test_merchant_pools_cover_each_category_and_channel(entities):
    for cat in range(N_CATEGORIES):
        if CAT_ONLINE_SHARE[cat] > 0:
            assert entities.online_by_cat.count[cat] > 0
        if CAT_ONLINE_SHARE[cat] < 1:
            assert entities.offline_fr_by_cat.count[cat] > 0


def test_client_referential_quality(entities):
    df = clients_frame(entities)
    assert df["client_id"].is_unique and df["iban"].is_unique
    assert df["email"].str.fullmatch(r"[a-z]+\.[a-z]+\d{2}@[a-z.]+").all()
    assert df["phone"].str.fullmatch(r"\+336\d{8}").all()
    # Le consentement marketing a une date si et seulement s'il est donné.
    assert ((df["consent_marketing_at"] != "") == df["consent_marketing"]).all()


def test_contacts_match_referential(small_cfg, entities):
    df = clients_frame(entities)
    contacts = client_contacts(small_cfg, np.array([10, 20]))
    assert list(contacts["email"]) == list(df.loc[[10, 20], "email"])
    assert list(contacts["iban"]) == list(df.loc[[10, 20], "iban"])


def test_cards_never_expose_a_card_number(entities):
    df = cards_frame(entities)
    assert "pan" not in df.columns
    assert df["pan_token"].str.fullmatch(r"tok_[0-9a-f]{16}").all()
    assert not df.astype(str).apply(lambda col: col.str.fullmatch(r"\d{16}")).any().any()


def test_foreign_merchants_have_no_french_city(entities):
    df = merchants_frame(entities)
    foreign = df[df["country"] != "FR"]
    assert (foreign["city"] == "").all()
    assert re.fullmatch(r"M\d{7}", df["merchant_id"].iloc[0])
