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
    # Le sous-échantillonnage touche les jeux d'apprentissage, jamais le test : ce sont les
    # chiffres du test qui sont annoncés.
    assert bundle.train.rows_before_sampling > len(bundle.train.target)
    assert bundle.valid.rows_before_sampling > len(bundle.valid.target)
    assert bundle.test.rows_before_sampling == len(bundle.test.target)
    assert bundle.train.fraud_rate > bundle.test.fraud_rate  # fraudes toutes conservées


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


def test_promotion_applies_the_written_criteria(trained, ml_settings):
    """Un candidat n'est promu que s'il passe les critères ; un refus est tracé et motivé."""
    import mlflow
    from ondine_ml.promote import check, promote, rollback

    mlflow.set_tracking_uri(ml_settings.tracking_uri)
    client = mlflow.MlflowClient(ml_settings.tracking_uri)

    decision = promote(ml_settings)
    assert decision.promoted, decision.reasons
    assert client.get_model_version_by_alias(ml_settings.model_name, "champion").version == (
        decision.version
    )

    # Les critères refusent bien un candidat insuffisant, en nommant chaque motif.
    mauvais = {
        "auc_pr": 0.1,
        "alert_rate": 0.05,
        "cost_total_eur": 900_000.0,
        "fairness_age_band_recall_ratio": 0.4,
    }
    motifs = check(mauvais, {"cost_total_eur": 100_000.0}, ml_settings)
    assert len(motifs) == 4
    assert any("AUC-PR" in m for m in motifs) and any("4/5" in m for m in motifs)

    # Le retour arrière est possible même quand aucun champion précédent n'existe : il le dit.
    retour = rollback(ml_settings)
    assert retour.promoted is False and "aucun champion précédent" in retour.reasons[0]


def test_every_command_of_the_cli_runs(trained, ml_settings, lake, monkeypatch):
    """Chaque sous-commande démarre réellement : une commande livrée cassée est inacceptable.

    Les commandes sont le seul point de contact avec la solution en exploitation ; un import
    manquant ne doit pas se découvrir en production.
    """
    from ondine_ml.cli import main

    monkeypatch.setenv("MLFLOW_TRACKING_URI", ml_settings.tracking_uri)
    monkeypatch.setenv("ONDINE_ML_MODEL_NAME", ml_settings.model_name)
    monkeypatch.setenv("ONDINE_ML_EXPERIMENT", ml_settings.experiment)
    monkeypatch.setenv("LAKE_ROOT", lake.lake_root)
    monkeypatch.setenv("LANDING_ROOT", str(lake.landing_root))
    monkeypatch.setenv("PIPELINE_PG_DSN", lake.pg_dsn)
    monkeypatch.setenv("PSEUDONYMISATION_KEY", "clé-de-test")
    monkeypatch.setenv("ONDINE_ML_TEST", ml_settings.test_bounds)
    monkeypatch.setenv("ONDINE_ML_VALID", ml_settings.valid_bounds)
    monkeypatch.setenv("ONDINE_ML_TRAIN", ml_settings.train_bounds)

    assert main(["metrics", "--alias", "challenger"]) == 0
    assert main(["report", "--alias", "challenger", "--name", "equite"]) == 0
    assert main(["explain", "--alias", "challenger", "--rank", "1"]) == 0
    assert (
        main(
            [
                "promote",
                "--alias",
                "challenger",
                "--derogation",
                "écart d'équité de deux millièmes face à 10 M€ de fraude évitée",
                "--responsable",
                "Direction des risques",
            ]
        )
        == 0
    )
    assert main(["rollback"]) in (0, 2)


def test_fairness_reweighting_only_corrects_a_real_gap():
    """La correction d'équité se déclenche sur un écart réel, et laisse le reste tranquille."""
    import numpy as np
    import pandas as pd
    from ondine_ml.evaluate import group_recall, rebalancing_weights

    groupes = pd.Series(["A"] * 100 + ["B"] * 100 + ["C"] * 5)
    labels = np.ones(205)
    # A détecté à 90 %, B à 50 %, C trop petit pour conclure.
    scores = np.concatenate(
        [np.r_[np.ones(90), np.zeros(10)], np.r_[np.ones(50), np.zeros(50)], np.ones(5)]
    )

    recalls = group_recall(groupes, labels, scores, threshold=0.5, min_frauds=50)
    assert set(recalls) == {"A", "B"}, "un groupe trop petit ne doit pas entrer dans la comparaison"
    assert recalls["A"] == pytest.approx(0.9) and recalls["B"] == pytest.approx(0.5)

    poids = rebalancing_weights(recalls, target_ratio=0.8)
    assert poids["B"] > poids["A"], "le groupe le moins bien détecté doit peser davantage"
    assert sum(poids.values()) / len(poids) == pytest.approx(1.0), "moyenne des poids ramenée à 1"

    # Écart déjà tenu : on ne touche pas à un modèle qui respecte la règle.
    assert rebalancing_weights({"A": 0.90, "B": 0.85}, target_ratio=0.8) == {}


def test_training_retrains_with_weights_when_the_gap_requires_it(lake, ml_settings):
    """Quand la règle des 4/5 n'est pas tenue, l'entraînement repart pondéré — et le dit.

    Vérifie la chaîne complète, pas seulement le calcul des poids : mesure en validation,
    décision, réentraînement, et trace du nombre de groupes repondérés.
    """
    from dataclasses import replace

    from ondine_ml.train import train

    # Marge au maximum et effectif minimal abaissé : le petit jeu de test compte trop peu de
    # fraudes par région pour le seuil de production.
    forcé = replace(ml_settings, fairness_margin=1.0, min_group_frauds=5, experiment="tests-equite")
    result = train(lake, forcé, register=False)
    assert result.metrics["fairness_reweighted_groups"] > 0
    assert "valid_fairness_region_recall_ratio" in result.metrics
    assert result.metrics["auc_pr"] > 0.5, "la correction ne doit pas casser le modèle"


def test_a_derogation_is_recorded_in_the_governance_register(trained, ml_settings, lake):
    """Mettre en service malgré un critère non tenu laisse une trace nominative et datée."""
    import psycopg
    from ondine_ml.promote import Derogation, record_exception

    record_exception(
        lake.pg_dsn,
        ml_settings.model_name,
        "42",
        ["équité insuffisante : fairness_region_recall_ratio 0.799 < 0.8 (règle des 4/5)"],
        Derogation("10 M€ de fraude évitée pour un écart de deux millièmes", "Dir. risques", 60),
    )
    with psycopg.connect(lake.pg_dsn) as connection:
        ligne = connection.execute(
            "SELECT criterion, justification, approved_by, status, is_overdue"
            " FROM governance.model_exceptions_due WHERE model_version = '42'"
        ).fetchone()
        tracee = connection.execute(
            "SELECT count(*) FROM audit.access_log WHERE action = 'model_promotion_derogation'"
        ).fetchone()[0]
    assert ligne is not None, "la dérogation doit figurer au registre"
    assert "4/5" in ligne[0] and ligne[2] == "Dir. risques" and ligne[3] == "ouverte"
    assert ligne[4] is False, "l'échéance de réexamen court encore"
    assert tracee >= 1, "la décision doit aussi apparaître au journal d'audit"
