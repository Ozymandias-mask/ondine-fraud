"""Configuration de la partie modèle, lue dans les variables d'environnement."""

from __future__ import annotations

import os
from dataclasses import dataclass

from .dataset import Split


@dataclass(frozen=True)
class MlSettings:
    """Paramètres d'entraînement, de suivi d'expérience et de mise en service."""

    tracking_uri: str = "http://mlflow:5000"
    experiment: str = "ondine-fraude"
    model_name: str = "ondine-fraud-detector"
    seed: int = 42
    negative_sampling_rate: float = 0.05
    num_boost_round: int = 1500
    early_stopping_rounds: int = 80
    alert_budget: float = 0.001
    shap_sample: int = 20_000
    # Lignes du jeu de test conservées avec leurs variables (robustesse, explicabilité) :
    # le reste n'est gardé que sous forme de score, quel que soit le volume évalué.
    robustness_sample: int = 200_000
    train_bounds: str = "2025-01..2025-08"
    valid_bounds: str = "2025-09..2025-09"
    test_bounds: str = "2025-10..2025-12"
    # Écarts tolérés avant de refuser une promotion automatique du modèle.
    min_auc_pr: float = 0.5
    min_fairness_ratio: float = 0.8
    # En deçà, un rappel par groupe ne veut rien dire : le groupe est écarté des comparaisons
    # comme de la correction.
    min_group_frauds: int = 200
    # Marge de sécurité sur la mesure d'équité faite en validation : ce mois sert aussi à
    # l'arrêt anticipé, le modèle y est donc flatteur et l'écart entre groupes y paraît plus
    # faible qu'il ne l'est sur des données jamais vues. On corrige donc avant la limite.
    fairness_margin: float = 0.1

    @classmethod
    def from_env(cls) -> MlSettings:
        """Lit les variables MLFLOW_* et ONDINE_ML_* (valeurs par défaut sinon)."""

        def value(name: str, default):
            raw = os.environ.get(name)
            return type(default)(raw) if raw not in (None, "") else default

        return cls(
            tracking_uri=os.environ.get("MLFLOW_TRACKING_URI", cls.tracking_uri),
            experiment=os.environ.get("ONDINE_ML_EXPERIMENT", cls.experiment),
            model_name=os.environ.get("ONDINE_ML_MODEL_NAME", cls.model_name),
            seed=value("ONDINE_ML_SEED", cls.seed),
            negative_sampling_rate=value("ONDINE_ML_NEGATIVE_RATE", cls.negative_sampling_rate),
            num_boost_round=value("ONDINE_ML_ROUNDS", cls.num_boost_round),
            early_stopping_rounds=value("ONDINE_ML_EARLY_STOPPING", cls.early_stopping_rounds),
            alert_budget=value("ONDINE_ML_ALERT_BUDGET", cls.alert_budget),
            shap_sample=value("ONDINE_ML_SHAP_SAMPLE", cls.shap_sample),
            robustness_sample=value("ONDINE_ML_ROBUSTNESS_SAMPLE", cls.robustness_sample),
            train_bounds=os.environ.get("ONDINE_ML_TRAIN", cls.train_bounds),
            valid_bounds=os.environ.get("ONDINE_ML_VALID", cls.valid_bounds),
            test_bounds=os.environ.get("ONDINE_ML_TEST", cls.test_bounds),
            min_auc_pr=value("ONDINE_ML_MIN_AUC_PR", cls.min_auc_pr),
            min_fairness_ratio=value("ONDINE_ML_MIN_FAIRNESS_RATIO", cls.min_fairness_ratio),
            min_group_frauds=value("ONDINE_ML_MIN_GROUP_FRAUDS", cls.min_group_frauds),
            fairness_margin=value("ONDINE_ML_FAIRNESS_MARGIN", cls.fairness_margin),
        )

    def rolling(self, latest: str, test_months: int = 3) -> MlSettings:
        """Mêmes réglages, mais avec un découpage temporel qui finit au mois donné.

        Un réentraînement automatique qui rejouerait les bornes gravées dans la configuration
        réapprendrait indéfiniment sur 2025 : il produirait un modèle identique et ne servirait
        à rien. Les bornes avancent donc avec les données — le test reste les derniers mois
        connus, la validation le mois qui les précède, l'entraînement tout ce qui est avant.
        L'ordre chronologique est conservé : aucun mois du test n'entre dans l'apprentissage.
        """
        from dataclasses import replace

        def month(offset: int) -> str:
            index = int(latest[:4]) * 12 + int(latest[5:7]) - 1 + offset
            return f"{index // 12}-{index % 12 + 1:02d}"

        first = self.train_bounds.split("..")[0].strip()
        test_start, valid = month(1 - test_months), month(-test_months)
        if valid <= first:
            raise ValueError(
                f"Historique insuffisant pour un découpage finissant en {latest} :"
                f" l'entraînement commencerait après sa validation ({first} > {valid})."
            )
        return replace(
            self,
            train_bounds=f"{first}..{month(-test_months - 1)}",
            valid_bounds=f"{valid}..{valid}",
            test_bounds=f"{test_start}..{latest}",
        )

    def split(self) -> Split:
        """Découpage temporel demandé (bornes « AAAA-MM..AAAA-MM »)."""

        def bounds(text: str) -> tuple[str, str]:
            first, _, last = text.partition("..")
            return first.strip(), (last or first).strip()

        return Split(bounds(self.train_bounds), bounds(self.valid_bounds), bounds(self.test_bounds))
