"""Entraînement du modèle de détection de fraude et suivi de l'expérience.

Déroulé : chargement de l'entraînement et de la validation (découpage temporel), entraînement
LightGBM avec arrêt anticipé, choix du seuil de décision sur la validation complète relue par
lots, évaluation complète sur le test (jamais vu, lu lui aussi par lots), puis enregistrement
dans MLflow (paramètres, métriques, graphiques, modèle, empreinte des données).

Mémoire : à aucun moment un mois entier de transactions n'est chargé avec ses variables. Les
jeux d'apprentissage sont sous-échantillonnés et libérés dès l'entraînement terminé ; la
validation et le test sont ensuite parcourus par lots, dont seuls les scores et quelques
colonnes d'évaluation sont conservés. L'empreinte mémoire ne dépend donc pas du volume évalué.

Reproductibilité : graine fixe, sous-échantillonnage par hachage (et non par tirage),
versions des bibliothèques et empreinte des données enregistrées avec le modèle.
"""

from __future__ import annotations

import gc
import json
import logging
import platform
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import lightgbm as lgb
import matplotlib
import mlflow
import numpy as np
import pandas as pd
from ondine_pipelines.settings import Settings
from sklearn.metrics import precision_recall_curve

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import dataset as ds  # noqa: E402
from . import evaluate as ev  # noqa: E402
from . import scoring as sc  # noqa: E402
from .explain import global_importance  # noqa: E402
from .model import FraudModel, metadata, serving_frame  # noqa: E402
from .settings import MlSettings  # noqa: E402

logger = logging.getLogger(__name__)

MODEL_ARTIFACT = "model"

# Paramètres volontairement sobres : un modèle plus petit se réentraîne vite, s'explique
# mieux et sur-apprend moins sur un phénomène aussi rare que la fraude.
DEFAULT_PARAMS: dict[str, object] = {
    "objective": "binary",
    "metric": ["average_precision", "auc"],
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 5.0,
    "max_cat_threshold": 64,
    "num_threads": 4,
    "verbosity": -1,
}


@dataclass
class TrainingResult:
    """Résumé d'un entraînement, renvoyé à l'orchestrateur."""

    run_id: str
    model_version: str | None
    metrics: dict[str, float]
    threshold: float
    fingerprint: dict[str, object]
    duration_s: float


def _pr_curve_figure(
    y_true: np.ndarray, scores: np.ndarray, legacy: dict[str, float]
) -> plt.Figure:
    precision, recall, _ = precision_recall_curve(y_true, scores)
    figure, axes = plt.subplots(figsize=(6, 4.2))
    axes.plot(recall, precision, color="#1d4ed8", lw=2, label="Modèle")
    axes.scatter(
        [legacy["legacy_recall"]],
        [legacy["legacy_precision"]],
        color="#dc2626",
        zorder=5,
        label="Moteur de règles actuel",
    )
    axes.set_xlabel("Rappel (fraudes détectées)")
    axes.set_ylabel("Précision (alertes justifiées)")
    axes.set_title("Précision-rappel sur le jeu de test")
    axes.grid(alpha=0.3)
    axes.legend()
    figure.tight_layout()
    return figure


def _save_artifacts(directory: Path, artifacts: dict[str, object]) -> None:
    for name, value in artifacts.items():
        target = directory / name
        if isinstance(value, pd.DataFrame):
            value.to_csv(target, index=False, encoding="utf-8")
        elif isinstance(value, plt.Figure):
            value.savefig(target, dpi=140)
            plt.close(value)
        else:
            target.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def train(
    settings: Settings,
    ml: MlSettings | None = None,
    split: ds.Split | None = None,
    params: dict[str, object] | None = None,
    register: bool = True,
) -> TrainingResult:
    """Entraîne, évalue et enregistre un modèle. Retourne le résumé de l'expérience."""
    ml = ml or MlSettings.from_env()
    split = split or ml.split()
    started = time.perf_counter()
    mlflow.set_tracking_uri(ml.tracking_uri)
    mlflow.set_experiment(ml.experiment)

    # Entraînement et validation d'abord ; le test est lu ensuite, mois par mois.
    bundle = ds.load(settings, split, ml.negative_sampling_rate, ml.seed, parts=("train", "valid"))
    train_params = {**DEFAULT_PARAMS, **(params or {}), "seed": ml.seed, "deterministic": True}

    with mlflow.start_run(run_name=f"lightgbm-{time.strftime('%Y%m%d-%H%M%S')}") as run:
        train_set = lgb.Dataset(
            bundle.train.features, label=bundle.train.target, categorical_feature=bundle.categorical
        )
        valid_set = lgb.Dataset(
            bundle.valid.features, label=bundle.valid.target, reference=train_set
        )
        booster = lgb.train(
            train_params,
            train_set,
            num_boost_round=ml.num_boost_round,
            valid_sets=[valid_set],
            valid_names=["valid"],
            callbacks=[
                lgb.early_stopping(ml.early_stopping_rounds, verbose=False),
                lgb.log_evaluation(period=100),
            ],
        )
        model = FraudModel(
            booster=booster,
            feature_names=bundle.feature_names,
            categorical=bundle.categorical,
            categories={
                c: [str(v) for v in bundle.train.features[c].cat.categories]
                for c in bundle.categorical
            },
            negative_sampling_rate=bundle.train.negative_sampling_rate,
            version=run.info.run_id[:8],
        )

        train_summary = bundle.train.summary() | bundle.valid.summary()
        # Mémoire : le modèle est entraîné, les jeux d'apprentissage ne servent plus.
        del train_set, valid_set
        bundle.train = bundle.valid = None
        gc.collect()

        # Seuil choisi sur la validation complète, relue par lots : le test ne sert qu'à
        # mesurer, jamais à régler.
        y_valid, valid_scores, valid_amounts = sc.stream_scores(
            settings, split.valid, model, ml.seed
        )
        best_cost, budget = ev.choose_threshold(
            y_valid, valid_scores, valid_amounts.astype(float), ml.alert_budget
        )
        model.threshold = float(budget.threshold)
        train_summary["valid_full_rows"] = float(len(y_valid))
        train_summary["valid_full_frauds"] = float(y_valid.sum())
        del y_valid, valid_scores, valid_amounts
        gc.collect()

        test, test_sample = sc.score_range(
            settings, split.test, model, ml.seed, ml.robustness_sample
        )
        test_scores = test.context["score"].to_numpy()
        y_test = test.target.to_numpy()
        amounts = test.context["amount"].astype(float).to_numpy()
        metrics = {
            **ev.quality_metrics(y_test, test_scores),
            **ev.threshold_metrics(y_test, test_scores, amounts, model.threshold),
            **ev.legacy_rules_metrics(test),
        }
        # Le seuil est calé sur la validation ; on mesure ce que le même budget d'alertes
        # imposerait sur la période de test, plus récente. L'écart est une dérive.
        metrics.update(
            ev.threshold_drift(y_test, test_scores, amounts, model.threshold, ml.alert_budget)
        )
        fairness = ev.fairness_report(test, test_scores, model.threshold)
        metrics.update(ev.fairness_gaps(fairness))
        robustness = ev.robustness_report(
            test_sample, model.probabilities, model.threshold, ml.seed
        )
        monthly = ev.monthly_report(test, test_scores, model.threshold)
        importance = global_importance(model, test_sample, sample=ml.shap_sample, seed=ml.seed)
        metrics["cost_saving_vs_legacy_eur"] = (
            metrics["legacy_cost_total_eur"] - metrics["cost_total_eur"]
        )
        metrics["robustness_worst_recall"] = float(robustness["rappel"].min())
        for name, value in (train_summary | test.summary()).items():
            metrics[name] = value

        mlflow.log_params(
            {
                **{k: str(v) for k, v in train_params.items()},
                "best_iteration": booster.best_iteration,
                "features": len(model.feature_names),
                "negative_sampling_rate": ml.negative_sampling_rate,
                "alert_budget": ml.alert_budget,
                "threshold_strategy": budget.reason,
                "python": platform.python_version(),
                "lightgbm": lgb.__version__,
                **{f"split_{k}": v for k, v in split.as_dict().items()},
                **{f"data_{k}": v for k, v in bundle.fingerprint.items()},
            }
        )
        mlflow.log_metrics({k: float(v) for k, v in metrics.items() if np.isfinite(float(v))})
        mlflow.set_tags(
            {
                "solution": "ondine-fraude",
                "threshold_cost_optimal": f"{best_cost.threshold:.6f}",
                "data_sha256_16": bundle.fingerprint["sha256_16"],
            }
        )

        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            _save_artifacts(
                directory,
                {
                    "courbe_precision_rappel.png": _pr_curve_figure(y_test, test_scores, metrics),
                    "equite_par_groupe.csv": fairness,
                    "robustesse.csv": robustness,
                    "performance_mensuelle.csv": monthly,
                    "importance_variables.csv": importance,
                    "metriques.json": {k: float(v) for k, v in metrics.items()},
                    "modele.json": metadata(model),
                },
            )
            mlflow.log_artifacts(str(directory), artifact_path="rapport")

        example = serving_frame(test_sample.features.head(5))
        signature = mlflow.models.infer_signature(example, model.decide(example))
        mlflow.pyfunc.log_model(
            name=MODEL_ARTIFACT,
            python_model=model,
            signature=signature,
            input_example=example,
            registered_model_name=ml.model_name if register else None,
            pip_requirements=[f"lightgbm=={lgb.__version__}", "pandas", "scikit-learn"],
        )
        version = _latest_version(ml) if register else None
        if version:
            mlflow.MlflowClient(ml.tracking_uri).set_registered_model_alias(
                ml.model_name, "challenger", version
            )
        duration = time.perf_counter() - started
        mlflow.log_metric("training_duration_s", duration)
        logger.info(
            "Modèle entraîné en %.0f s : AUC-PR %.3f, rappel %.3f,"
            " alertes %.3f %%, économie %.0f €",
            duration,
            metrics["auc_pr"],
            metrics["recall"],
            100 * metrics["alert_rate"],
            metrics["cost_saving_vs_legacy_eur"],
        )
        return TrainingResult(
            run_id=run.info.run_id,
            model_version=version,
            metrics=metrics,
            threshold=model.threshold,
            fingerprint=bundle.fingerprint,
            duration_s=duration,
        )


def _latest_version(ml: MlSettings) -> str | None:
    client = mlflow.MlflowClient(ml.tracking_uri)
    versions = client.search_model_versions(f"name='{ml.model_name}'")
    return max(versions, key=lambda v: int(v.version)).version if versions else None


def training_summary(result: TrainingResult) -> dict[str, object]:
    """Résumé compact (journaux Airflow, retour de la ligne de commande)."""
    keep = (
        "auc_pr",
        "recall",
        "precision",
        "alert_rate",
        "cost_total_eur",
        "legacy_recall",
        "legacy_cost_total_eur",
        "cost_saving_vs_legacy_eur",
        # Dérive du seuil : sans ces trois chiffres, un volume d'alertes effondré ressemble
        # à un mauvais modèle alors que c'est le seuil qui a vieilli.
        "budget_threshold",
        "budget_recall",
        "threshold_drift_ratio",
    )
    summary = asdict(result)
    summary["metrics"] = {
        k: round(float(result.metrics[k]), 6) for k in keep if k in result.metrics
    }
    return summary
