import numpy as np
import pytest
from ondine_sim import columns as col
from ondine_sim.fraud import inject_fraud, scenario_weights

DAY = 86_400_000


def _inject(entities, drift, n_victims=400, seed=0):
    rng = np.random.default_rng(seed)
    victims = rng.choice(entities.n_clients, n_victims, replace=False)
    return victims, inject_fraud(entities, rng, victims, 0, 30 * DAY, drift, incident_base=1_000)


def test_drift_scenario_only_when_active():
    types, probs = scenario_weights(False)
    assert col.FT_WALLET not in types and probs.sum() == pytest.approx(1)
    types, probs = scenario_weights(True)
    assert col.FT_WALLET in types and probs.sum() == pytest.approx(1)


def test_all_scenarios_generated_and_labelled(entities):
    _, out = _inject(entities, drift=True)
    assert set(np.unique(out.tx["fraud_type"])) == set(range(1, 7))
    assert (out.tx["incident"] >= 1_000).all()
    assert out.tx["ts"].max() < 30 * DAY


def test_no_wallet_fraud_before_drift(entities):
    _, out = _inject(entities, drift=False)
    assert col.FT_WALLET not in out.tx["fraud_type"]


def test_scenario_signatures(entities):
    victims, out = _inject(entities, drift=True, n_victims=600, seed=4)
    tx = out.tx
    ato = tx["fraud_type"] == col.FT_ATO
    assert (tx["beneficiary_is_mule"][ato]).all()
    assert (tx["device_key"][ato] >= col.NEW_DEVICE_OFFSET).all()

    scam = tx["fraud_type"] == col.FT_APP_SCAM
    # Faux conseiller : la victime utilise son propre appareil, depuis la France.
    assert (tx["device_key"][scam] == tx["client"][scam] * 4).all()
    assert (tx["ip_country"][scam] == 0).all()

    cloned = tx["fraud_type"] == col.FT_CLONED
    assert (entities.merchant_country[tx["merchant"][cloned]] >= col.FIRST_RISKY).all()

    wallet = tx["fraud_type"] == col.FT_WALLET
    assert (tx["entry_mode"][wallet] == col.EM_TOKEN).all() and (tx["wallet"][wallet] > 0).all()

    # Chaque incident appartient à une seule victime.
    for incident in np.unique(tx["incident"])[:50]:
        assert len(np.unique(tx["client"][tx["incident"] == incident])) == 1
    assert set(np.unique(tx["client"])) <= set(victims)


def test_time_compression_for_streaming(entities):
    rng = np.random.default_rng(5)
    out = inject_fraud(entities, rng, np.arange(50), 0, 1_000, True, 0, time_scale=1 / 60)
    # Une attaque de plusieurs heures tient en quelques minutes en mode flux.
    assert out.tx["ts"].max() < 20 * 60_000
