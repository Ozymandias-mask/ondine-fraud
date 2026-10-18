"""Détection de dérive : le modèle en service vieillit-il, et depuis quand.

Un modèle de fraude se périme plus vite que la plupart : les fraudeurs changent de mode
opératoire précisément parce qu'ils sont détectés. La question n'est donc pas « le modèle
est-il bon ? » — il l'était au moment de sa mise en service — mais « l'est-il encore, sur
les transactions d'aujourd'hui ? ».

Trois signaux, du plus rapide au plus sûr :

1. **dérive des variables** — les transactions entrantes ne ressemblent plus à celles qui ont
   servi à l'apprentissage. Visible immédiatement, mais ne prouve rien : une population peut
   changer sans que le modèle en souffre ;
2. **dérive des scores** — la distribution des probabilités rendues s'est déplacée. C'est ce
   qui casse le seuil, donc le budget d'alertes, avant même que la qualité ne baisse ;
3. **dérive de la qualité** — le rappel et la précision mesurés sur les fraudes désormais
   confirmées. Le signal certain, mais le plus tardif : une déclaration de fraude arrive
   quelques semaines après l'opération.

Les trois sont mesurés, parce qu'aucun ne suffit seul : les deux premiers alertent tôt et
peuvent se tromper, le troisième tranche mais trop tard pour être le seul déclencheur.

L'indicateur retenu est l'**indice de stabilité de population** (PSI) : il compare deux
distributions bin par bin, ne suppose aucune forme particulière, et ses seuils d'usage sont
connus et comparables d'une variable à l'autre (0,10 : à surveiller ; 0,25 : dérive avérée).
Aucune bibliothèque de plus pour dix lignes d'arithmétique — une dépendance de moins à tenir
à jour dans une image qui tourne en production.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
import psycopg
from ondine_pipelines.settings import Settings
from psycopg.types.json import Jsonb

from . import evaluate as ev
from . import scoring as sc
from .dataset import months_between
from .serving import load_model
from .settings import MlSettings

logger = logging.getLogger(__name__)

# Seuils d'usage du PSI, repris tels quels de la pratique du risque de crédit.
PSI_WATCH = 0.10
PSI_DRIFT = 0.25

# Baisse relative d'AUC-PR au-delà de laquelle la qualité est jugée dégradée.
QUALITY_DROP = 0.10

# En deçà, la précision constatée par les analystes ne veut rien dire.
MIN_REVIEWS = 50

# Nombre de variables les plus dérivées conservées dans le rapport : la liste complète des
# 43 variables noierait le signal, et c'est la plus dérivée qui décide.
TOP_FEATURES = 10


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Indice de stabilité de population entre deux échantillons d'une même grandeur.

    Les bornes viennent des quantiles de la **référence** : chaque intervalle y pèse le même
    poids, ce qui évite qu'un intervalle vide de la référence ne domine l'indice. Un plancher
    remplace les proportions nulles, sans quoi l'indice partirait à l'infini dès qu'un
    intervalle se vide.
    """
    reference, current = _finite(reference), _finite(current)
    if len(reference) < bins or len(current) < bins:
        return 0.0
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # grandeur quasi constante : rien à comparer
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    expected = np.histogram(reference, bins=edges)[0] / len(reference)
    actual = np.histogram(current, bins=edges)[0] / len(current)
    floor = 1 / max(len(reference), len(current))
    expected, actual = np.maximum(expected, floor), np.maximum(actual, floor)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def _finite(values: np.ndarray) -> np.ndarray:
    """Valeurs exploitables : une variable d'historique est absente tant qu'il n'y en a pas."""
    array = np.asarray(values, dtype="float64")
    return array[np.isfinite(array)]


@dataclass
class DriftReport:
    """Ce qu'on sait de l'usure du modèle en service, et ce qu'on en conclut."""

    version: str
    reference: str
    current: str
    verdict: str = "stable"  # stable | surveiller | derive
    score_psi: float = 0.0
    feature_psi: dict[str, float] = field(default_factory=dict)
    quality: dict[str, float] = field(default_factory=dict)
    feedback: dict[str, float] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)

    @property
    def retrain(self) -> bool:
        """Vrai si la dérive justifie un réentraînement immédiat."""
        return self.verdict == "derive"

    def as_dict(self) -> dict[str, object]:
        """Forme lisible, renvoyée par la ligne de commande et par Airflow."""
        data = asdict(self)
        data["reentrainement"] = self.retrain
        return data

    def gauges(self) -> dict[str, float]:
        """Métriques publiées vers Prometheus (supervision et alerte)."""
        values = {
            "model_drift_timestamp_seconds": time.time(),
            "model_drift_score_psi": self.score_psi,
            "model_drift_feature_psi_max": max(self.feature_psi.values(), default=0.0),
            "model_drift_detected": float(self.retrain),
            "model_drift_watch": float(self.verdict != "stable"),
        }
        values.update({f"model_current_{k}": v for k, v in self.quality.items()})
        values.update({f"model_feedback_{k}": v for k, v in self.feedback.items()})
        return values


def analyst_feedback(dsn: str, days: int = 30) -> dict[str, float]:
    """Précision constatée par les analystes sur les alertes qu'ils ont tranchées.

    C'est la mesure de qualité la plus fraîche dont on dispose : elle arrive en heures, là
    où une déclaration de fraude arrive en semaines. Elle ne dit rien du rappel — on ne voit
    pas les fraudes non alertées — mais une précision qui s'effondre suffit à déclencher.
    """
    if not dsn:
        return {}
    with psycopg.connect(dsn, autocommit=True) as connection:
        row = connection.execute(
            """
            SELECT count(*) FILTER (WHERE status <> 'a_traiter'),
                   count(*) FILTER (WHERE status = 'fraude_confirmee'),
                   count(*)
              FROM ops.alerts
             WHERE scored_at >= now() - make_interval(days => %s)
            """,
            (days,),
        ).fetchone()
    reviewed, confirmed, total = (int(v or 0) for v in row)
    feedback = {"alertes": float(total), "tranchees": float(reviewed)}
    if reviewed >= MIN_REVIEWS:
        feedback["precision"] = confirmed / reviewed
    return feedback


# Part minimale du volume mensuel median en dessous de laquelle un mois est tenu pour
# incomplet. Le flux temps reel ecrit en continu : le mois courant n'a que quelques
# milliers de lignes la ou un mois clos en compte des millions.
MOIS_COMPLET = 0.5


def _dernier_mois_complet(volumes: list[tuple[str, int]]) -> str:
    """Dernier mois dont le volume est comparable aux autres.

    Prendre simplement le mois le plus recent reviendrait a caler le decoupage glissant
    sur un mois a peine entame : l'entrainement reclamerait alors une validation sur des
    mois qui n'existent pas encore. On ecarte donc les mois dont le volume est tres
    inferieur au volume median.
    """
    if not volumes:
        raise FileNotFoundError("Aucune donnee gold : la derive ne peut pas etre mesuree.")
    tries = sorted(n for _, n in volumes)
    median = tries[len(tries) // 2]
    complets = [mois for mois, n in volumes if n >= median * MOIS_COMPLET]
    if not complets:
        raise FileNotFoundError(
            "Aucun mois complet dans la couche gold : le lac n'a pas fini d'etre construit."
        )
    return max(complets)


def latest_gold_month(settings: Settings) -> str:
    """Dernier mois complet de la couche gold : la periode a confronter a l'apprentissage."""
    from ondine_pipelines.lake import connect

    from .dataset import gold_pattern

    con = connect(settings)
    try:
        lignes = con.execute(
            f"SELECT strftime(event_date, '%Y-%m') AS mois, count(*) AS n "  # noqa: S608
            f"FROM read_parquet('{gold_pattern(settings)}') GROUP BY 1 ORDER BY 1"  # noqa: E501
        ).fetchall()
    finally:
        con.close()
    return _dernier_mois_complet([(str(m), int(n)) for m, n in lignes])


def measure(
    settings: Settings,
    ml: MlSettings | None = None,
    alias: str = "champion",
    current: str = "",
    dsn: str = "",
) -> DriftReport:
    """Compare le modèle en service à la période sur laquelle il a été validé.

    La référence est le jeu de test du champion : c'est la période dont ses chiffres publiés
    proviennent, donc la seule à laquelle il soit honnête de comparer.
    """
    ml = ml or MlSettings.from_env()
    model, version = load_model(ml, alias)
    reference_bounds = ml.split().test
    current = current or latest_gold_month(settings)
    if current in months_between(reference_bounds):
        logger.warning(
            "Le mois %s fait partie de la référence : la dérive mesurée serait nulle par"
            " construction. Le découpage d'entraînement doit avancer avec les données.",
            current,
        )

    sample = ml.robustness_sample // 2
    reference, reference_rows = sc.score_range(settings, reference_bounds, model, ml.seed, sample)
    observed, observed_rows = sc.score_range(settings, (current, current), model, ml.seed, sample)
    report = DriftReport(
        version=version,
        reference=f"{reference_bounds[0]}..{reference_bounds[1]}",
        current=current,
    )
    # Les scores sont comparés sur la totalité des deux périodes ; les variables sur les
    # échantillons, seuls jeux qui les conservent (les garder toutes ne tiendrait pas en
    # mémoire et n'ajouterait rien à un indice de distribution).
    report.score_psi = psi(
        reference.context["score"].to_numpy(), observed.context["score"].to_numpy()
    )
    report.feature_psi = _feature_psi(reference_rows.features, observed_rows.features)
    report.quality = _quality(observed, model.threshold)
    report.feedback = analyst_feedback(dsn)
    _decide(report, model.threshold, ml)
    return report


def _feature_psi(reference: pd.DataFrame, observed: pd.DataFrame) -> dict[str, float]:
    """PSI de chaque variable numérique, les plus dérivées d'abord.

    Les variables catégorielles sont codées en entiers par le modèle : leur PSI se calcule de
    la même façon, les intervalles épousant alors les modalités.
    """
    scores = {}
    for name in reference.columns:
        if name not in observed.columns:
            continue
        try:
            reference_values = pd.to_numeric(reference[name], errors="coerce").to_numpy()
            observed_values = pd.to_numeric(observed[name], errors="coerce").to_numpy()
        except (TypeError, ValueError):
            continue
        scores[name] = psi(reference_values, observed_values)
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return {name: round(value, 4) for name, value in ranked[:TOP_FEATURES]}


def _quality(observed, threshold: float) -> dict[str, float]:
    """Qualité mesurée sur le mois courant, si ses fraudes sont déjà connues.

    Une déclaration de fraude arrive quelques semaines après l'opération : sur le mois le plus
    récent, les labels sont incomplets et la qualité y paraît toujours plus mauvaise qu'elle
    n'est. Le taux de fraude observé est donc publié à côté, pour que ce biais soit visible
    au lieu d'être pris pour une dérive.
    """
    y_true = observed.target.to_numpy()
    scores = observed.context["score"].to_numpy()
    amounts = observed.context["amount"].astype(float).to_numpy()
    if y_true.sum() == 0:
        return {"fraud_rate": 0.0, "rows": float(len(y_true))}
    return {
        **ev.quality_metrics(y_true, scores),
        **{
            k: v
            for k, v in ev.threshold_metrics(y_true, scores, amounts, threshold).items()
            if k in ("recall", "precision", "alert_rate")
        },
        "fraud_rate": float(y_true.mean()),
        "rows": float(len(y_true)),
    }


def _decide(report: DriftReport, threshold: float, ml: MlSettings) -> None:
    """Applique les règles de décision, en nommant chaque motif."""
    worst = max(report.feature_psi.items(), key=lambda item: item[1], default=("", 0.0))
    watch = []
    if report.score_psi >= PSI_DRIFT:
        report.reasons.append(
            f"distribution des scores déplacée (PSI {report.score_psi:.3f} ≥ {PSI_DRIFT}) :"
            " le seuil ne tient plus le budget d'alertes"
        )
    elif report.score_psi >= PSI_WATCH:
        watch.append(f"scores en mouvement (PSI {report.score_psi:.3f})")
    if worst[1] >= PSI_DRIFT:
        report.reasons.append(
            f"variable « {worst[0]} » dérivée (PSI {worst[1]:.3f} ≥ {PSI_DRIFT}) :"
            " les transactions entrantes ne ressemblent plus à l'apprentissage"
        )
    elif worst[1] >= PSI_WATCH:
        watch.append(f"variable « {worst[0]} » en mouvement (PSI {worst[1]:.3f})")

    alert_rate = report.quality.get("alert_rate")
    if alert_rate is not None and alert_rate > 2 * ml.alert_budget:
        report.reasons.append(
            f"volume d'alertes hors budget : {100 * alert_rate:.3f} %"
            f" au lieu de {100 * ml.alert_budget:.3f} % (seuil {threshold:.6f})"
        )

    precision = report.feedback.get("precision")
    if precision is not None and precision < 0.5:
        report.reasons.append(
            f"les analystes écartent plus d'une alerte sur deux (précision {precision:.1%}"
            f" sur {int(report.feedback['tranchees'])} alertes tranchées)"
        )

    report.verdict = "derive" if report.reasons else ("surveiller" if watch else "stable")
    report.reasons.extend(watch)
    if report.verdict == "stable":
        report.reasons.append("aucune dérive significative")


def record(dsn: str, report: DriftReport) -> None:
    """Conserve la mesure dans `ops.model_drift` : la dérive se lit dans le temps, pas à l'instant.

    Une mesure isolée ne dit pas si l'écart s'installe ou s'il s'agit d'un mois atypique
    (soldes, fêtes). L'historique est ce qui permet de distinguer les deux — et de justifier,
    après coup, pourquoi un réentraînement a été déclenché ce jour-là.
    """
    if not dsn:
        return
    with psycopg.connect(dsn, autocommit=True) as connection:
        connection.execute(
            """
            INSERT INTO ops.model_drift
                (model_version, reference_period, current_period, verdict, score_psi,
                 feature_psi_max, details)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (
                report.version,
                report.reference,
                report.current,
                report.verdict,
                report.score_psi,
                max(report.feature_psi.values(), default=0.0),
                Jsonb(report.as_dict()),
            ),
        )


def publish(settings: Settings, report: DriftReport) -> None:
    """Publie les indicateurs de dérive vers Prometheus (alerte Grafana)."""
    if not settings.pushgateway_url:
        return
    from prometheus_client import CollectorRegistry, Gauge, pushadd_to_gateway

    registry = CollectorRegistry()
    for name, value in report.gauges().items():
        Gauge(name, name, ["model_version"], registry=registry).labels(report.version).set(value)
    try:
        pushadd_to_gateway(
            settings.pushgateway_url, job="ondine_model_drift", registry=registry, timeout=5
        )
    except OSError as error:  # la supervision ne doit jamais interrompre la production
        logger.warning("Passerelle Prometheus injoignable : %s", error)
