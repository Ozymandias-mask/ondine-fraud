"""Construction des jeux d'entraînement, de validation et de test à partir de la couche gold.

Trois principes, vérifiés par des tests :

- **Découpage temporel** : on entraîne sur le passé et on évalue sur le futur. Un découpage
  aléatoire donnerait des résultats trop optimistes (une même attaque se retrouverait des
  deux côtés) et ne dirait rien de la tenue du modèle dans le temps.
- **Aucune fuite** : les colonnes de labels, d'identifiants, la décision du moteur de règles
  actuel et les attributs d'équité sont retirés des variables d'entrée.
- **Historique complet** : les transactions du premier mois, dont les fenêtres de 30 jours
  sont incomplètes, sont écartées (colonne `history_complete`).

Le déséquilibre (environ 0,2 % de fraudes) est traité par sous-échantillonnage des
transactions légitimes ; les probabilités prédites sont ensuite corrigées pour retrouver la
proportion réelle (voir `calibration.py`).
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field

import duckdb
import pandas as pd
from ondine_pipelines.lake import connect
from ondine_pipelines.settings import Settings

logger = logging.getLogger(__name__)

TARGET = "is_fraud"

# Colonnes interdites en entrée du modèle, avec leur raison.
EXCLUDED: dict[str, str] = {
    "transaction_id": "identifiant",
    "client_id": "identifiant",
    "card_id": "identifiant",
    "merchant_id": "identifiant (trop fin, risque de sur-apprentissage)",
    "event_time": "horodatage brut (remplacé par heure, week-end, nuit)",
    "event_date": "horodatage brut",
    "is_fraud": "label",
    "label_reported_at": "label",
    "fraud_claim_category": "label",
    "has_rejected_claim": "label",
    "legacy_auth_status": "décision du moteur de règles actuel (référence de comparaison)",
    "legacy_decline_reason": "décision du moteur de règles actuel",
    "age_band": "attribut d'équité : mesuré, jamais utilisé pour décider",
    "region": "attribut d'équité",
    "segment": "attribut d'équité",
    "tenure_days": "attribut d'équité",
    "history_complete": "indicateur technique",
    "_source_file": "lignage",
    "month": "partition",
    "year": "partition",
}

# Colonnes conservées hors variables d'entrée : évaluation, équité, coût métier.
CONTEXT = (
    "transaction_id",
    "event_date",
    "amount",
    "legacy_auth_status",
    "age_band",
    "region",
    "segment",
    "channel",
    "fraud_claim_category",
)

CATEGORICAL = (
    "channel",
    "entry_mode",
    "merchant_category",
    "merchant_country",
    "wallet_type",
    "beneficiary_country",
    "beneficiary_bank_code",
)


@dataclass(frozen=True)
class Split:
    """Découpage temporel (bornes incluses, format AAAA-MM)."""

    train: tuple[str, str] = ("2025-01", "2025-08")
    valid: tuple[str, str] = ("2025-09", "2025-09")
    test: tuple[str, str] = ("2025-10", "2025-12")

    def as_dict(self) -> dict[str, str]:
        """Bornes du découpage, pour le suivi d'expérience."""
        return {
            "train": f"{self.train[0]}..{self.train[1]}",
            "valid": f"{self.valid[0]}..{self.valid[1]}",
            "test": f"{self.test[0]}..{self.test[1]}",
        }


@dataclass
class Dataset:
    """Un jeu (entraînement, validation ou test) : variables, label et contexte."""

    name: str
    features: pd.DataFrame
    target: pd.Series
    context: pd.DataFrame
    rows_before_sampling: int = 0
    negative_sampling_rate: float = 1.0

    @property
    def fraud_rate(self) -> float:
        """Proportion de fraudes dans le jeu tel qu'il est utilisé."""
        return float(self.target.mean()) if len(self.target) else 0.0

    def summary(self) -> dict[str, float]:
        """Volumétrie et taux de fraude, journalisés dans MLflow."""
        return {
            f"{self.name}_rows": float(len(self.target)),
            f"{self.name}_frauds": float(self.target.sum()),
            f"{self.name}_fraud_rate": self.fraud_rate,
            f"{self.name}_negative_sampling_rate": self.negative_sampling_rate,
        }


@dataclass
class DatasetBundle:
    """Les trois jeux et l'empreinte des données utilisées (reproductibilité)."""

    train: Dataset
    valid: Dataset
    test: Dataset
    feature_names: list[str]
    categorical: list[str]
    fingerprint: dict[str, object] = field(default_factory=dict)


def _months(con: duckdb.DuckDBPyConnection, pattern: str) -> list[str]:
    rows = con.execute(
        "SELECT DISTINCT strftime(event_date, '%Y-%m') AS m"
        f" FROM read_parquet('{pattern}') ORDER BY 1"
    ).fetchall()
    return [r[0] for r in rows]


def gold_pattern(settings: Settings) -> str:
    """Motif de lecture de la couche gold (tous les mois)."""
    return settings.uri("gold", "transaction_features", "**", "*.parquet")


def fingerprint(con: duckdb.DuckDBPyConnection, pattern: str, split: Split) -> dict[str, object]:
    """Empreinte des données : fichiers, lignes, bornes, somme de contrôle.

    Deux entraînements partant des mêmes données ont la même empreinte : c'est ce qui rend
    une expérience reproductible et auditable (quel modèle a vu quelles données).
    """
    files = [r[0] for r in con.execute("SELECT file FROM glob(?) ORDER BY 1", [pattern]).fetchall()]
    stats = con.execute(f"""
        SELECT count(*), min(event_date), max(event_date), sum(is_fraud::INT)
        FROM read_parquet('{pattern}')
    """).fetchone()
    digest = hashlib.sha256(
        json.dumps([files, [str(s) for s in stats]], sort_keys=True).encode()
    ).hexdigest()[:16]
    return {
        "files": len(files),
        "rows": int(stats[0]),
        "first_date": str(stats[1]),
        "last_date": str(stats[2]),
        "frauds": int(stats[3]),
        "months": ",".join(sorted({f.split("year=")[-1][:4] for f in files})) if files else "",
        "split": json.dumps(split.as_dict()),
        "sha256_16": digest,
    }


def _load_split(
    con: duckdb.DuckDBPyConnection,
    pattern: str,
    name: str,
    bounds: tuple[str, str],
    columns: list[str],
    negative_rate: float,
    seed: int,
) -> Dataset:
    where = (
        f"strftime(event_date, '%Y-%m') BETWEEN '{bounds[0]}' AND '{bounds[1]}'"
        " AND history_complete"
    )
    total = con.execute(f"SELECT count(*) FROM read_parquet('{pattern}') WHERE {where}").fetchone()[
        0
    ]
    # Sous-échantillonnage reproductible : hachage de l'identifiant, pas de tirage aléatoire.
    sampling = (
        f" AND ({TARGET} OR (hash(transaction_id || '{seed}') % 1000000)"
        f" < {negative_rate * 1e6:.0f})"
        if negative_rate < 1
        else ""
    )
    frame = con.execute(
        f"SELECT {', '.join(columns)} FROM read_parquet('{pattern}')"
        f" WHERE {where}{sampling} ORDER BY event_time"
    ).df()
    # Une colonne de contexte peut aussi être une variable d'entrée (le canal, le montant) :
    # seule la liste EXCLUDED décide de ce que le modèle voit.
    features = frame[[c for c in frame.columns if c not in EXCLUDED]].copy()
    for column in features.columns:
        if column in CATEGORICAL:
            features[column] = features[column].astype("category")
        elif features[column].dtype == "bool":
            # float32 partout : un booléen manquant reste représentable (schéma du modèle).
            features[column] = features[column].astype("float32")
        elif str(features[column].dtype).startswith(("object", "decimal")):
            features[column] = pd.to_numeric(features[column], errors="coerce").astype("float32")
        else:
            features[column] = features[column].astype("float32")
    context = frame[[c for c in CONTEXT if c in frame.columns]].copy()
    # Mémoire : sur 12 millions de lignes, ces colonnes répétitives coûtent plusieurs Go
    # en texte brut et presque rien en catégories.
    for column in (
        "legacy_auth_status",
        "age_band",
        "region",
        "segment",
        "channel",
        "fraud_claim_category",
    ):
        if column in context.columns:
            context[column] = context[column].astype("category")
    dataset = Dataset(
        name=name,
        features=features,
        target=frame[TARGET].astype("int8"),
        context=context,
        rows_before_sampling=int(total),
        negative_sampling_rate=negative_rate,
    )
    logger.info(
        "%s : %d lignes (%d avant échantillonnage), %d fraudes (%.3f %%)",
        name,
        len(dataset.target),
        dataset.rows_before_sampling,
        int(dataset.target.sum()),
        100 * dataset.fraud_rate,
    )
    return dataset


def load(
    settings: Settings,
    split: Split | None = None,
    negative_sampling_rate: float = 0.05,
    seed: int = 42,
) -> DatasetBundle:
    """Charge les trois jeux depuis la couche gold.

    `negative_sampling_rate` ne s'applique qu'à l'entraînement : la validation et le test
    gardent toutes les transactions, sinon les métriques ne refléteraient pas la réalité.
    """
    split = split or Split()
    con = connect(settings)
    try:
        pattern = gold_pattern(settings)
        available = _months(con, pattern)
        if not available:
            raise FileNotFoundError(f"Aucune donnée gold : {pattern}")
        all_columns = [
            r[0]
            for r in con.execute(f"DESCRIBE SELECT * FROM read_parquet('{pattern}')").fetchall()
        ]
        features = [c for c in all_columns if c not in EXCLUDED]
        columns = sorted(set(features) | set(CONTEXT) | {TARGET, "event_time"})
        columns = [c for c in columns if c in all_columns]
        bundle = DatasetBundle(
            train=_load_split(
                con, pattern, "train", split.train, columns, negative_sampling_rate, seed
            ),
            valid=_load_split(con, pattern, "valid", split.valid, columns, 1.0, seed),
            test=_load_split(con, pattern, "test", split.test, columns, 1.0, seed),
            feature_names=[],
            categorical=[c for c in CATEGORICAL if c in features],
            fingerprint=fingerprint(con, pattern, split),
        )
        bundle.feature_names = list(bundle.train.features.columns)
        # Mêmes colonnes et mêmes catégories partout : une catégorie vue à l'entraînement et
        # absente du test doit rester connue du modèle.
        for dataset in (bundle.valid, bundle.test):
            dataset.features = dataset.features[bundle.feature_names]
            for column in bundle.categorical:
                dataset.features[column] = dataset.features[column].cat.set_categories(
                    bundle.train.features[column].cat.categories
                )
        if bundle.train.target.sum() == 0:
            raise ValueError(f"Aucune fraude dans le jeu d'entraînement ({split.train})")
        return bundle
    finally:
        con.close()
