import numpy as np
from ondine_sim import columns as col
from ondine_sim.behaviour import (
    CONTACTLESS_LIMIT_EUR,
    background_events,
    legit_transactions,
    linked_events,
    month_timestamps,
)


def _legit(entities, schema_v2=True, n=20_000):
    rng = np.random.default_rng(1)
    clients = rng.integers(0, entities.n_clients, n)
    ts = month_timestamps(rng, n, 0, 30 * 86_400_000)
    return legit_transactions(entities, rng, clients, ts, schema_v2)


def test_timestamps_stay_in_period():
    ts = month_timestamps(np.random.default_rng(0), 10_000, 1_000, 1_000 + 86_400_000 * 28)
    assert ts.min() >= 1_000 and ts.max() < 1_000 + 86_400_000 * 28


def test_contactless_limit_is_respected_without_wallet(entities):
    tx = _legit(entities)
    plain = tx["entry_mode"] == col.EM_CONTACTLESS
    assert plain.any()
    assert (tx["amount"][plain] <= CONTACTLESS_LIMIT_EUR).all()


def test_channel_consistency(entities):
    tx = _legit(entities)
    transfer = tx["channel"] == col.CH_TRANSFER
    ecom = tx["channel"] == col.CH_ECOM
    assert (tx["card"][transfer] == -1).all() and (tx["merchant"][transfer] == -1).all()
    assert (tx["beneficiary_key"][transfer] >= 0).all()
    assert (tx["device_key"][ecom] >= 0).all() and (tx["ip_octet"][ecom] >= 0).all()
    assert (tx["three_ds"][~ecom] == -1).all()
    assert (tx["fraud_type"] == col.FT_LEGIT).all()


def test_wallets_only_after_schema_change(entities):
    assert (_legit(entities, schema_v2=False)["wallet"] == 0).all()
    assert (_legit(entities, schema_v2=True)["wallet"] > 0).any()


def test_events_precede_remote_operations(entities):
    tx = _legit(entities, n=2_000)
    ev = linked_events(entities, np.random.default_rng(2), tx)
    remote = (tx["channel"] == col.CH_ECOM) | (tx["channel"] == col.CH_TRANSFER)
    assert (ev["event_type"] == col.EV_LOGIN).sum() == remote.sum()
    bg = background_events(entities, np.random.default_rng(3), np.arange(100), np.arange(100))
    assert len(bg["ts"]) == 100
