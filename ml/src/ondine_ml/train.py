"""Entraînement du modèle de détection de fraude et suivi de l'expérience.

Déroulé : chargement des trois jeux (découpage temporel), entraînement LightGBM avec arrêt
anticipé sur la validation, choix du seuil de décision sur la validation, évaluation
complète sur le test (jamais vu), puis enregistrement dans MLflow (paramètres, métriques,
graphiques, modèle, empreinte des données).

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


def _score_test_by_month(
    settings: Settings,
    split: ds.Split,
    model: FraudModel,
    seed: int,
    sample_per_month: int,
) -> tuple[ds.Dataset, ds.Dataset]:
    """Score le jeu de test mois par mois et n'en garde que ce qui sert à l'évaluation.

    Retourne deux jeux : l'un complet mais sans variables (scores, labels, contexte), pour
    les métriques et l'équité ; l'autre échantillonné avec ses variables, pour la robustesse
    et l'explicabilité. Un mois est libéré dès qu'il est scoré : la mémoire reste constante,
    quel que soit le nombre de mois de test.
    """
    contexts, targets, samples, sample_targets, sample_contexts = [], [], [], [], []
    for period in ds.months_between(split.test):
        month = ds.load_month(settings, period, model.feature_names, model.categories, seed)
        if not len(month.target):
            continue
        context = month.context.copy()
        context["score"] = model.probabilities(month.features)
        contexts.append(context)
        targets.append(month.target)
        keep = min(sample_per_month, len(month.target))
        index = month.features.sample(keep, random_state=seed).index
        samples.append(month.features.loc[index])
        sample_targets.append(month.target.loc[index])
        sample_contexts.append(month.context.loc[index])
        del month
        gc.collect()

    full = ds.Dataset(
        name="test",
        features=pd.DataFrame(),
        target=pd.concat(targets, ignore_index=True),
        context=pd.concat(contexts, ignore_index=True),
    )
    sample = ds.Dataset(
        name="test_sample",
        features=pd.concat(samples, ignore_index=True),
        target=pd.concat(sample_targets, ignore_index=True),
        context=pd.concat(sample_contexts, ignore_index=True),
    )
    return full, sample


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

        # Seuil choisi sur la validation : le test ne sert qu'à mesurer, jamais à régler.
        valid_scores = model.probabilities(bundle.valid.features)
        best_cost, budget = ev.choose_threshold(
            bundle.valid.target.to_numpy(),
            valid_scores,
            bundle.valid.context["amount"].astype(float).to_numpy(),
            ml.alert_budget,
        )
        model.threshold = float(budget.threshold)

        train_summary = bundle.train.summary() | bundle.valid.summary()
        # Mémoire : le modèle est entraîné, les jeux d'entraînement ne servent plus.
        del train_set, valid_set, valid_scores
        bundle.train = bundle.valid = None  # les jeux d'entraînement sont libérés
        gc.collect()

        test, test_sample = _score_test_by_month(
            settings, split, model, ml.seed, ml.robustness_sample
        )
        test_scores = test.context["score"].to_numpy()
        y_test = test.target.to_numpy()
        amounts = test.context["amount"].astype(float).to_numpy()
        metrics = {
            **ev.quality_metrics(y_test, test_scores),
            **ev.threshold_metrics(y_test, test_scores, amounts, model.threshold),
            **ev.legacy_rules_metrics(test),
        }
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
    )
    summary = asdict(result)
    summary["metrics"] = {
        k: round(float(result.metrics[k]), 6) for k in keep if k in result.metrics
    }
    return summary
