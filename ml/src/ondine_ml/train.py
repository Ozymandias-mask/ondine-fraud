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
import re
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

# Attribut d'équité sur lequel porte la correction automatique (mesuré, jamais utilisé
# comme variable d'entrée du modèle).
FAIRNESS_ATTRIBUTE = "region"

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


def _fit(
    bundle: ds.DatasetBundle,
    params: dict[str, object],
    ml: MlSettings,
    version: str,
    weights: dict[str, float] | None = None,
) -> FraudModel:
    """Entraîne un LightGBM et l'emballe avec son prétraitement et ses catégories.

    `weights` pondère les exemples par groupe d'équité : un même entraînement, une seule
    définition, que la pondération soit active ou non.
    """
    sample_weight = None
    if weights:
        groups = bundle.train.context[FAIRNESS_ATTRIBUTE].astype("object")
        sample_weight = groups.map(weights).fillna(1.0).to_numpy(dtype="float64")
    train_set = lgb.Dataset(
        bundle.train.features,
        label=bundle.train.target,
        weight=sample_weight,
        categorical_feature=bundle.categorical,
    )
    valid_set = lgb.Dataset(bundle.valid.features, label=bundle.valid.target, reference=train_set)
    booster = lgb.train(
        params,
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
            c: [str(v) for v in bundle.train.features[c].cat.categories] for c in bundle.categorical
        },
        negative_sampling_rate=bundle.train.negative_sampling_rate,
        version=version,
    )
    del train_set, valid_set
    gc.collect()
    return model


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

    fairness_decision = pd.DataFrame()
    with mlflow.start_run(run_name=f"lightgbm-{time.strftime('%Y%m%d-%H%M%S')}") as run:
        model = _fit(bundle, train_params, ml, run.info.run_id[:8])

        # Équité : le rappel ne doit pas dépendre d'où vit la victime. On le mesure sur la
        # validation et, si l'écart entre régions dépasse la règle des 4/5, on **réentraîne**
        # en pondérant les exemples des régions moins bien détectées. La région n'entre jamais
        # dans les variables du modèle : elle ne sert qu'à doser l'attention à l'apprentissage.
        y_valid, valid_scores, valid_amounts, regions = sc.stream_scores(
            settings, split.valid, model, ml.seed, context_column=FAIRNESS_ATTRIBUTE
        )
        weights, valid_recalls = {}, {}
        if regions is None:
            # Un contrôle d'équité qui s'escamote sans bruit ne protège personne.
            logger.warning(
                "Attribut d'équité « %s » absent du contexte : aucune correction possible.",
                FAIRNESS_ATTRIBUTE,
            )
        else:
            valid_recalls = ev.group_recall(
                regions,
                y_valid,
                valid_scores,
                ev.budget_threshold(valid_scores, ml.alert_budget),
                min_frauds=ml.min_group_frauds,
            )
            weights = ev.rebalancing_weights(
                valid_recalls, ml.min_fairness_ratio + ml.fairness_margin
            )
        if weights:
            logger.info(
                "Écart de rappel entre régions : réentraînement pondéré (%s)",
                ", ".join(f"{g} ×{w:.2f}" for g, w in sorted(weights.items())),
            )
            model = _fit(bundle, train_params, ml, run.info.run_id[:8], weights)
            del y_valid, valid_scores, valid_amounts, regions
            gc.collect()
            y_valid, valid_scores, valid_amounts, _ = sc.stream_scores(
                settings, split.valid, model, ml.seed
            )

        # Seuil de repli, calé sur la validation complète : il sert au démarrage de l'API,
        # avant qu'elle n'ait vu assez de transactions pour se recaler elle-même.
        best_cost, budget = ev.choose_threshold(
            y_valid, valid_scores, valid_amounts.astype(float), ml.alert_budget
        )
        model.threshold = float(budget.threshold)
        model.alert_budget = ml.alert_budget

        train_summary = bundle.train.summary() | bundle.valid.summary()
        train_summary["valid_full_rows"] = float(len(y_valid))
        train_summary["valid_full_frauds"] = float(y_valid.sum())
        train_summary["fairness_reweighted_groups"] = float(len(weights))
        if valid_recalls:
            # Le détail par région va dans un rapport, pas dans les noms de métriques : ceux-ci
            # n'acceptent ni accent ni apostrophe, et « Provence-Alpes-Côte d'Azur » en a.
            train_summary["valid_fairness_region_recall_ratio"] = min(valid_recalls.values()) / max(
                valid_recalls.values()
            )
            fairness_decision = pd.DataFrame(
                [
                    {
                        "région": groupe,
                        "rappel_validation": rappel,
                        "poids_appliqué": weights.get(groupe, 1.0),
                    }
                    for groupe, rappel in sorted(valid_recalls.items())
                ]
            )

        # Mémoire : le modèle est entraîné, les jeux d'apprentissage ne servent plus.
        bundle.train = bundle.valid = None
        del y_valid, valid_scores, valid_amounts
        gc.collect()

        test, test_sample = sc.score_range(
            settings, split.test, model, ml.seed, ml.robustness_sample
        )
        test_scores = test.context["score"].to_numpy()
        y_test = test.target.to_numpy()
        amounts = test.context["amount"].astype(float).to_numpy()

        # Point de fonctionnement : le budget d'alertes est une contrainte d'organisation
        # (« l'équipe peut examiner une transaction sur mille »), donc un **quantile** de la
        # population décidée, pas une probabilité figée. Ici, 4 % d'écart sur le seuil change
        # le volume d'alertes d'un facteur cent : un seuil absolu serait ingérable. C'est ce
        # que l'API recale en continu sur les scores récents ; les chiffres annoncés sont donc
        # mesurés au même point, et non à un seuil qu'aucune production ne conserverait.
        operating = ev.budget_threshold(test_scores, ml.alert_budget)
        metrics = {
            **ev.quality_metrics(y_test, test_scores),
            **ev.threshold_metrics(y_test, test_scores, amounts, operating),
            **ev.legacy_rules_metrics(test),
        }
        # Pour comparaison : ce que donnerait le seuil de repli, jamais recalé.
        metrics.update(
            {
                f"fixed_{k}": v
                for k, v in ev.threshold_metrics(
                    y_test, test_scores, amounts, model.threshold
                ).items()
            }
        )
        metrics["threshold_drift_ratio"] = (
            model.threshold / operating if operating else float("inf")
        )
        fairness = ev.fairness_report(test, test_scores, operating)
        metrics.update(ev.fairness_gaps(fairness))
        robustness = ev.robustness_report(test_sample, model.probabilities, operating, ml.seed)
        monthly = ev.monthly_report(test, test_scores, operating)
        importance = global_importance(model, test_sample, sample=ml.shap_sample, seed=ml.seed)
        metrics["cost_saving_vs_legacy_eur"] = (
            metrics["legacy_cost_total_eur"] - metrics["cost_total_eur"]
        )
        metrics["fairness_reweighted_groups"] = float(len(weights))
        metrics["robustness_worst_recall"] = float(robustness["rappel"].min())
        for name, value in (train_summary | test.summary()).items():
            metrics[name] = value

        mlflow.log_params(
            {
                **{k: str(v) for k, v in train_params.items()},
                "best_iteration": model.booster.best_iteration,
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
        # Les noms de métriques n'acceptent qu'un alphabet restreint : on le garantit ici
        # plutôt que de faire échouer un entraînement de deux minutes sur un nom de région.
        mlflow.log_metrics(
            {
                re.sub(r"[^A-Za-z0-9_.:/ -]", "_", k): float(v)
                for k, v in metrics.items()
                if np.isfinite(float(v))
            }
        )
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
                    "equite_decision_validation.csv": fairness_decision,
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
        "threshold",
        "fixed_alert_rate",
        "fixed_recall",
        "threshold_drift_ratio",
    )
    summary = asdict(result)
    summary["metrics"] = {
        k: round(float(result.metrics[k]), 6) for k in keep if k in result.metrics
    }
    return summary
