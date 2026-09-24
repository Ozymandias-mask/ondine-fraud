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
from collections.abc import Iterator
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

    train: Dataset | None
    valid: Dataset | None
    test: Dataset | None
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


CONTEXT_CATEGORICAL = (
    "legacy_auth_status",
    "age_band",
    "region",
    "segment",
    "channel",
    "fraud_claim_category",
)


def _connect_ordered(settings: Settings) -> duckdb.DuckDBPyConnection:
    """Connexion DuckDB qui préserve l'ordre des lignes.

    L'ordre des données d'entraînement influe sur le modèle : sans cette garantie, deux
    entraînements sur les mêmes données donneraient des modèles légèrement différents.
    """
    con = connect(settings)
    con.execute("SET preserve_insertion_order = true")
    return con


def category_values(
    con: duckdb.DuckDBPyConnection, pattern: str, where: str, columns: list[str]
) -> dict[str, list[str]]:
    """Valeurs distinctes des colonnes catégorielles, en une seule lecture.

    Fixer les catégories avant de lire les données permet de charger par lots sans que
    chaque lot ait ses propres catégories (ce qui ferait retomber les colonnes en texte,
    bien plus coûteux en mémoire).
    """
    if not columns:
        return {}
    selects = ", ".join(f'array_agg(DISTINCT "{c}")' for c in columns)
    row = con.execute(f"SELECT {selects} FROM read_parquet('{pattern}') WHERE {where}").fetchone()
    return {
        column: sorted(str(v) for v in (values or []) if v is not None)
        for column, values in zip(columns, row, strict=True)
    }


def _compact(
    frame: pd.DataFrame, categories: dict[str, list[str]]
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Convertit un lot brut en (variables, label, contexte) aux types les plus compacts."""
    features = frame[[c for c in frame.columns if c not in EXCLUDED]].copy()
    for column in features.columns:
        if column in categories:
            values = features[column].astype("object")
            values = values.where(values.isin(categories[column]), other=None)
            features[column] = pd.Categorical(values, categories=categories[column])
        elif features[column].dtype == "bool":
            # float32 partout : un booléen manquant reste représentable (schéma du modèle).
            features[column] = features[column].astype("float32")
        else:
            features[column] = pd.to_numeric(features[column], errors="coerce").astype("float32")
    context = frame[[c for c in CONTEXT if c in frame.columns]].copy()
    for column in CONTEXT_CATEGORICAL:
        if column in context.columns:
            context[column] = context[column].astype("category")
    if "amount" in context.columns:
        context["amount"] = pd.to_numeric(context["amount"], errors="coerce").astype("float32")
    return features, frame[TARGET].astype("int8"), context


def _where(bounds: tuple[str, str], negative_rate: float, seed: int) -> str:
    where = (
        f"strftime(event_date, '%Y-%m') BETWEEN '{bounds[0]}' AND '{bounds[1]}'"
        " AND history_complete"
    )
    if negative_rate < 1:
        # Sous-échantillonnage reproductible : hachage de l'identifiant, pas de tirage aléatoire.
        where += (
            f" AND ({TARGET} OR (hash(transaction_id || '{seed}') % 1000000)"
            f" < {negative_rate * 1e6:.0f})"
        )
    return where


def _fetch(
    con: duckdb.DuckDBPyConnection, columns: list[str], pattern: str, where: str
) -> pd.DataFrame:
    """Exécute une requête et rapatrie son résultat **en entier**.

    Volontairement **sans passer par Arrow** : sur de gros volumes, la conversion Arrow de
    DuckDB 1.5 échoue par intermittence avec une longueur négative (le lot demandé n'étant pas
    un multiple de la taille interne de ses blocs, le décompte restant passe sous zéro). Le
    contrôle d'intégrité du lac a confirmé que les données, elles, sont saines. La conversion
    native vers pandas suit un tout autre chemin dans DuckDB et n'a pas ce défaut.

    Le découpage est fait par nous, en requêtes complètes et indépendantes (un mois,
    éventuellement un quart de mois) : chacune tient en mémoire, et aucune ne dépend d'un
    mécanisme de lecture continue que nous ne maîtrisons pas.
    """
    query = f"SELECT {', '.join(columns)} FROM read_parquet('{pattern}') WHERE {where}"
    return con.execute(query).df()


def _bucket(index: int, count: int) -> str:
    """Condition isolant un sous-ensemble reproductible d'un mois (découpage par hachage)."""
    return "" if count <= 1 else f" AND (hash(transaction_id || 'lot') % {count}) = {index}"


def _selected_columns(con: duckdb.DuckDBPyConnection, pattern: str) -> list[str]:
    """Colonnes lues : variables du modèle, contexte d'évaluation et label."""
    available = [
        r[0] for r in con.execute(f"DESCRIBE SELECT * FROM read_parquet('{pattern}')").fetchall()
    ]
    wanted = set(c for c in available if c not in EXCLUDED) | set(CONTEXT) | {TARGET}
    return [c for c in sorted(wanted) if c in available]


def iter_chunks(
    settings: Settings,
    bounds: tuple[str, str],
    categories: dict[str, list[str]] | None = None,
    feature_names: list[str] | None = None,
    negative_rate: float = 1.0,
    seed: int = 42,
    parts_per_month: int = 4,
) -> Iterator[Dataset]:
    """Parcourt un intervalle de mois par lots, sans jamais tout charger en mémoire.

    C'est ce qui permet d'évaluer douze millions de transactions sur une machine de bureau :
    chaque lot est une requête indépendante, scorée puis libérée. Un mois complet est découpé
    en quatre par hachage ; un mois sous-échantillonné tient en une seule requête.
    """
    con = _connect_ordered(settings)
    try:
        pattern = gold_pattern(settings)
        columns = _selected_columns(con, pattern)
        if categories is None:
            categories = category_values(
                con,
                pattern,
                _where(bounds, negative_rate, seed),
                [c for c in CATEGORICAL if c in columns],
            )
        count = parts_per_month if negative_rate >= 1 else 1
        for month in months_between(bounds):
            for index in range(count):
                where = _where((month, month), negative_rate, seed) + _bucket(index, count)
                features, target, context = _compact(
                    _fetch(con, columns, pattern, where), categories
                )
                if feature_names is not None:
                    features = features[feature_names]
                yield Dataset(name="chunk", features=features, target=target, context=context)
    finally:
        con.close()


def _load_split(
    con: duckdb.DuckDBPyConnection,
    pattern: str,
    name: str,
    bounds: tuple[str, str],
    columns: list[str],
    negative_rate: float,
    seed: int,
    categories: dict[str, list[str]] | None = None,
) -> Dataset:
    """Charge un jeu complet, lot par lot, en types compacts."""
    where = _where(bounds, negative_rate, seed)
    total = con.execute(
        f"SELECT count(*) FROM read_parquet('{pattern}') WHERE"
        f" strftime(event_date, '%Y-%m') BETWEEN '{bounds[0]}' AND '{bounds[1]}'"
        " AND history_complete"
    ).fetchone()[0]
    if categories is None:
        categories = category_values(con, pattern, where, [c for c in CATEGORICAL if c in columns])
    # Un mois par requête : la mémoire est bornée par le plus gros mois, et l'ordre des lignes
    # (mois croissants, ordre d'insertion à l'intérieur) reste celui du fichier — condition de
    # la reproductibilité d'un entraînement.
    parts = [
        _compact(
            _fetch(con, columns, pattern, _where((month, month), negative_rate, seed)), categories
        )
        for month in months_between(bounds)
    ]
    parts = [p for p in parts if len(p[1])]
    if not parts:
        raise ValueError(f"Aucune donnée pour {name} ({bounds[0]}..{bounds[1]})")
    dataset = Dataset(
        name=name,
        features=pd.concat([p[0] for p in parts], ignore_index=True),
        target=pd.concat([p[1] for p in parts], ignore_index=True),
        context=pd.concat([p[2] for p in parts], ignore_index=True),
        rows_before_sampling=int(total),
        negative_sampling_rate=negative_rate,
    )
    del parts
    logger.info(
        "%s : %d lignes (%d avant échantillonnage), %d fraudes (%.3f %%)",
        name,
        len(dataset.target),
        dataset.rows_before_sampling,
        int(dataset.target.sum()),
        100 * dataset.fraud_rate,
    )
    return dataset


def months_between(bounds: tuple[str, str]) -> list[str]:
    """Liste des mois « AAAA-MM » couverts par des bornes incluses."""
    first, last = (
        int(bounds[0][:4]) * 12 + int(bounds[0][5:7]) - 1,
        int(bounds[1][:4]) * 12 + int(bounds[1][5:7]) - 1,
    )
    return [f"{i // 12}-{i % 12 + 1:02d}" for i in range(first, last + 1)]


def load(
    settings: Settings,
    split: Split | None = None,
    negative_sampling_rate: float = 0.05,
    seed: int = 42,
    parts: tuple[str, ...] = ("train", "valid", "test"),
) -> DatasetBundle:
    """Charge les jeux demandés depuis la couche gold.

    `negative_sampling_rate` s'applique aux jeux qui servent à **construire** le modèle
    (entraînement et validation : la validation ne sert qu'à l'arrêt anticipé, un mois entier
    en mémoire n'y apporterait rien). Le jeu de test, lui, garde toutes les transactions :
    c'est lui qui donne les chiffres annoncés. Le seuil de décision, qui dépend du taux réel
    d'alertes, est calculé à part sur la validation complète, lue par lots (voir `train.py`).

    `parts` permet de ne charger que ce dont on a besoin : sur 50 millions de lignes, garder
    les trois jeux en mémoire en même temps est inutile et coûteux.
    """
    split = split or Split()
    con = _connect_ordered(settings)
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
        loaded = {
            name: _load_split(
                con,
                pattern,
                name,
                getattr(split, name),
                columns,
                negative_sampling_rate if name in ("train", "valid") else 1.0,
                seed,
            )
            for name in parts
        }
        bundle = DatasetBundle(
            train=loaded.get("train"),
            valid=loaded.get("valid"),
            test=loaded.get("test"),
            feature_names=[],
            categorical=[c for c in CATEGORICAL if c in features],
            fingerprint=fingerprint(con, pattern, split),
        )
        reference = next(iter(loaded.values()))
        bundle.feature_names = list(reference.features.columns)
        # Mêmes colonnes et mêmes catégories partout : une catégorie vue à l'entraînement et
        # absente du test doit rester connue du modèle.
        others = [
            d for d in (bundle.train, bundle.valid, bundle.test) if d not in (None, reference)
        ]
        for dataset in others:
            dataset.features = dataset.features[bundle.feature_names]
            for column in bundle.categorical:
                dataset.features[column] = dataset.features[column].cat.set_categories(
                    reference.features[column].cat.categories
                )
        if bundle.train is not None and bundle.train.target.sum() == 0:
            raise ValueError(f"Aucune fraude dans le jeu d'entraînement ({split.train})")
        return bundle
    finally:
        con.close()
