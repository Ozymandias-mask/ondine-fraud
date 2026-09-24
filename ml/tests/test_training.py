"""Entraînement : absence de fuite, reproductibilité, enregistrement, équité, robustesse."""

import mlflow
import numpy as np
import pandas as pd
import pytest
from ondine_ml import dataset as ds
from ondine_ml import evaluate as ev
from ondine_ml.explain import explain_one, global_importance, sentence
from ondine_ml.model import serving_frame
from ondine_ml.serving import load_model
from ondine_ml.train import train


def test_features_contain_no_leak_and_splits_are_chronological(lake, ml_settings):
    bundle = ds.load(
        lake, ml_settings.split(), ml_settings.negative_sampling_rate, ml_settings.seed
    )
    forbidden = {
        "is_fraud",
        "label_reported_at",
        "fraud_claim_category",
        "has_rejected_claim",
        "legacy_auth_status",
        "legacy_decline_reason",
        "transaction_id",
        "client_id",
        "card_id",
        "age_band",
        "region",
        "segment",
        "tenure_days",
        "event_time",
        "event_date",
    }
    assert not forbidden & set(bundle.feature_names)
    assert {"amount", "card_tx_count_1h", "channel"} <= set(bundle.feature_names)

    def bounds(part):
        dates = pd.to_datetime(part.context["event_date"])
        return dates.min(), dates.max()

    train_max = bounds(bundle.train)[1]
    valid_min, valid_max = bounds(bundle.valid)
    assert train_max < valid_min and valid_max < bounds(bundle.test)[0]
    # Le sous-échantillonnage ne touche que l'entraînement.
    assert bundle.train.rows_before_sampling > len(bundle.train.target)
    assert bundle.valid.rows_before_sampling == len(bundle.valid.target)
    assert bundle.train.fraud_rate > bundle.valid.fraud_rate  # fraudes toutes conservées


def test_training_registers_a_usable_model_with_its_report(trained, ml_settings):
    assert trained.metrics["auc_pr"] > 0.5
    assert trained.metrics["alert_rate"] <= 2 * ml_settings.alert_budget
    assert trained.metrics["recall"] > trained.metrics["legacy_recall"]
    assert trained.metrics["precision"] > trained.metrics["legacy_precision"]

    mlflow.set_tracking_uri(ml_settings.tracking_uri)
    client = mlflow.MlflowClient(ml_settings.tracking_uri)
    version = client.get_model_version_by_alias(ml_settings.model_name, "challenger")
    assert version.version == trained.model_version
    artifacts = {a.path for a in client.list_artifacts(trained.run_id, "rapport")}
    assert {
        "rapport/courbe_precision_rappel.png",
        "rapport/equite_par_groupe.csv",
        "rapport/robustesse.csv",
        "rapport/metriques.json",
    } <= artifacts
    run = client.get_run(trained.run_id)
    assert run.data.params["data_sha256_16"] == trained.fingerprint["sha256_16"]
    assert run.data.params["split_train"] == ml_settings.split().as_dict()["train"]


def test_same_seed_gives_the_same_model(lake, ml_settings, trained):
    again = train(lake, ml_settings, register=False)
    assert again.threshold == pytest.approx(trained.threshold, rel=1e-9)
    for metric in ("auc_pr", "recall", "precision", "cost_total_eur"):
        assert again.metrics[metric] == pytest.approx(trained.metrics[metric], rel=1e-9)


def test_sampling_correction_restores_the_real_fraud_rate(lake, ml_settings, trained):
    model, _ = load_model(ml_settings, "challenger")
    bundle = ds.load(lake, ml_settings.split(), negative_sampling_rate=1.0, seed=ml_settings.seed)
    probabilities = model.probabilities(bundle.test.features)
    # Sans correction, la probabilité moyenne serait proche du taux de l'entraînement (~1 %).
    assert probabilities.mean() < 5 * bundle.test.fraud_rate
    assert trained.metrics["brier"] < 0.01


def test_reports_cover_fairness_robustness_and_drift(lake, ml_settings):
    model, _ = load_model(ml_settings, "challenger")
    bundle = ds.load(lake, ml_settings.split(), negative_sampling_rate=1.0, seed=ml_settings.seed)
    scores = model.probabilities(bundle.test.features)

    fairness = ev.fairness_report(bundle.test, scores, model.threshold)
    assert set(fairness["attribut"]) == {"age_band", "region"}
    assert (fairness["taux_alerte"] <= 1).all()

    robustness = ev.robustness_report(
        bundle.test, model.probabilities, model.threshold, ml_settings.seed, sample=5000
    )
    reference = robustness.loc[robustness["scénario"] == "référence", "auc_pr"].item()
    degraded = robustness[robustness["scénario"] != "référence"]
    assert len(degraded) >= 3
    assert (degraded["auc_pr"] <= reference + 1e-9).all()  # dégradation, jamais amélioration
    assert (degraded["auc_pr"] > 0.1).all()  # mais le modèle ne s'effondre pas

    monthly = ev.monthly_report(bundle.test, scores, model.threshold)
    assert len(monthly) >= 1 and (monthly["transactions"] > 0).all()


def test_explanations_name_the_deciding_factors(lake, ml_settings):
    model, _ = load_model(ml_settings, "challenger")
    bundle = ds.load(lake, ml_settings.split(), negative_sampling_rate=1.0, seed=ml_settings.seed)
    scores = model.probabilities(bundle.test.features)
    riskiest = bundle.test.features.iloc[[int(np.argmax(scores))]]

    factors = explain_one(model, riskiest)
    assert len(factors) == 4
    assert all(
        f["libellé"] and f["effet"] in ("augmente le risque", "diminue le risque") for f in factors
    )
    assert abs(factors[0]["poids"]) >= abs(factors[-1]["poids"])
    assert sentence(factors)

    importance = global_importance(model, bundle.test, sample=2000, seed=ml_settings.seed)
    assert importance["part_%"].sum() == pytest.approx(100, rel=1e-6)
    assert importance.iloc[0]["contribution_moyenne"] > 0


def test_model_from_registry_handles_imperfect_input(lake, ml_settings):
    model, _ = load_model(ml_settings, "challenger")
    bundle = ds.load(lake, ml_settings.split(), negative_sampling_rate=1.0, seed=ml_settings.seed)
    sample = bundle.test.features.head(200)
    reference = model.probabilities(sample)

    # Ce que l'API enverra : catégories en texte, colonnes dans le désordre.
    payload = serving_frame(sample).sample(frac=1, axis=1, random_state=0)
    assert model.probabilities(payload) == pytest.approx(reference, rel=1e-9)

    degraded = payload.drop(columns=["card_tx_count_1h"]).copy()
    degraded["channel"] = "canal_inconnu"
    scores = model.probabilities(degraded)
    assert np.isfinite(scores).all() and ((scores >= 0) & (scores <= 1)).all()

    decisions = model.decide(sample)
    assert set(decisions.columns) == {"probability", "alert", "threshold"}
    assert decisions["alert"].equals(decisions["probability"] >= model.threshold)
