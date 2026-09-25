"""Promotion contrôlée d'un modèle : du candidat (« challenger ») au modèle servi (« champion »).

Aucun modèle n'arrive en production parce qu'il vient d'être entraîné. Il est promu s'il
passe des **critères écrits à l'avance**, vérifiés sur le jeu de test :

- qualité minimale (AUC-PR) : un modèle qui ne détecte rien ne remplace personne ;
- équité : les rapports entre groupes d'âge et de région doivent respecter la règle des 4/5,
  sans quoi la solution introduirait une discrimination (AI Act, article 10) ;
- budget d'alertes : l'équipe fraude ne peut pas traiter plus d'alertes qu'annoncé ;
- gain métier : le candidat doit coûter moins cher que le champion en place.

Le refus est aussi important que l'acceptation : il est journalisé avec ses motifs, et la
décision est traçable dans MLflow (étiquettes sur la version du modèle). C'est cette fonction
que le réentraînement automatique appellera, sans intervention humaine.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import mlflow
import psycopg
from psycopg.types.json import Jsonb

from .settings import MlSettings

logger = logging.getLogger(__name__)

CANDIDATE_ALIAS = "challenger"
SERVING_ALIAS = "champion"


@dataclass(frozen=True)
class Derogation:
    """Décision assumée de mettre en service malgré un critère non tenu."""

    justification: str
    approved_by: str
    review_days: int = 90


@dataclass
class Decision:
    """Résultat d'une tentative de promotion, avec ses motifs."""

    promoted: bool
    version: str | None
    previous: str | None
    reasons: list[str]
    metrics: dict[str, float]

    def as_dict(self) -> dict[str, object]:
        """Forme lisible, renvoyée par la ligne de commande et par Airflow."""
        return {
            "promu": self.promoted,
            "version": self.version,
            "version_precedente": self.previous,
            "motifs": self.reasons,
            "metriques": {k: round(v, 6) for k, v in self.metrics.items()},
        }


def _metrics(client: mlflow.MlflowClient, version) -> dict[str, float]:
    """Métriques enregistrées lors de l'entraînement d'une version du modèle."""
    if not version or not version.run_id:
        return {}
    return dict(client.get_run(version.run_id).data.metrics)


def _version(client: mlflow.MlflowClient, name: str, alias: str):
    try:
        return client.get_model_version_by_alias(name, alias)
    except Exception:  # alias absent : aucun modèle sous ce nom  # noqa: BLE001
        return None


def check(candidate: dict[str, float], champion: dict[str, float], ml: MlSettings) -> list[str]:
    """Motifs de refus du candidat ; liste vide = il peut être promu."""
    reasons = []
    auc = candidate.get("auc_pr")
    if auc is None:
        reasons.append("métriques du candidat introuvables dans MLflow")
        return reasons
    if auc < ml.min_auc_pr:
        reasons.append(f"qualité insuffisante : AUC-PR {auc:.3f} < {ml.min_auc_pr}")
    alert_rate = candidate.get("alert_rate")
    if alert_rate is not None and alert_rate > 2 * ml.alert_budget:
        reasons.append(
            f"budget d'alertes dépassé : {100 * alert_rate:.3f} %"
            f" > {100 * 2 * ml.alert_budget:.3f} %"
        )
    # Un volume d'alertes effondré signale un seuil devenu inadapté (les scores ont baissé
    # depuis la période de calage) : le modèle laisserait passer presque toutes les fraudes.
    effondre = alert_rate is not None and alert_rate < ml.alert_budget / 10
    if effondre:
        ratio = candidate.get("threshold_drift_ratio")
        reasons.append(
            f"volume d'alertes effondré : {100 * alert_rate:.5f} %"
            f" au lieu de {100 * ml.alert_budget:.3f} % — seuil à recaler"
            + (f" (il est {ratio:.1f} fois trop haut)" if ratio else "")
        )
    # Les rapports d'équité ne veulent rien dire sur une poignée d'alertes : ils ne sont
    # examinés que si le modèle alerte à un volume réaliste.
    if not effondre:
        for name, value in candidate.items():
            if name.endswith("_recall_ratio") and value < ml.min_fairness_ratio:
                reasons.append(
                    f"équité insuffisante : {name} {value:.3f} < {ml.min_fairness_ratio}"
                    " (règle des 4/5)"
                )
    cost, previous_cost = candidate.get("cost_total_eur"), champion.get("cost_total_eur")
    if previous_cost is not None and cost is not None and cost > previous_cost:
        reasons.append(
            f"coût métier supérieur au champion en place : {cost:,.0f} € > {previous_cost:,.0f} €"
        )
    return reasons


def metrics_of(ml: MlSettings, alias: str = CANDIDATE_ALIAS) -> dict[str, object]:
    """Toutes les métriques enregistrées pour la version portant cet alias.

    Évite de réentraîner pour relire un chiffre : le suivi d'expérience est la source de
    vérité, la ligne de commande ne fait que l'interroger.
    """
    mlflow.set_tracking_uri(ml.tracking_uri)
    client = mlflow.MlflowClient(ml.tracking_uri)
    version = _version(client, ml.model_name, alias)
    if version is None:
        return {"alias": alias, "erreur": f"aucun modèle sous l'alias « {alias} »"}
    return {
        "alias": alias,
        "version": version.version,
        "run_id": version.run_id,
        "metriques": {k: round(v, 6) for k, v in sorted(_metrics(client, version).items())},
    }


def report_of(ml: MlSettings, alias: str = CANDIDATE_ALIAS, name: str = "") -> dict[str, object]:
    """Rapports détaillés joints à un entraînement : équité, robustesse, dérive mensuelle.

    Les métriques résument, les rapports expliquent. Quand un critère de promotion est refusé,
    c'est ici qu'on voit **quel groupe** est concerné et de combien.
    """
    mlflow.set_tracking_uri(ml.tracking_uri)
    client = mlflow.MlflowClient(ml.tracking_uri)
    version = _version(client, ml.model_name, alias)
    if version is None or not version.run_id:
        return {"alias": alias, "erreur": f"aucun modèle sous l'alias « {alias} »"}
    directory = Path(
        mlflow.artifacts.download_artifacts(run_id=version.run_id, artifact_path="rapport")
    )
    rapports = {}
    for fichier in sorted(directory.glob("*.csv")):
        if name and name not in fichier.name:
            continue
        rapports[fichier.stem] = fichier.read_text(encoding="utf-8").strip().splitlines()
    return {"alias": alias, "version": version.version, "rapports": rapports}


def record_exception(
    dsn: str, model: str, version: str, criteria: list[str], derogation: Derogation
) -> int:
    """Inscrit une dérogation au registre de gouvernance. Retourne le nombre de lignes créées.

    Mettre en service un modèle qui ne tient pas un critère est une décision qu'une banque
    peut légitimement prendre — un écart d'équité de quelques millièmes face à plusieurs
    millions d'euros de fraude évitée, par exemple. Ce qui ne serait pas légitime, c'est que
    cette décision soit invisible : elle est donc nommée, motivée, attribuée à une personne et
    assortie d'une échéance de réexamen.
    """
    with psycopg.connect(dsn, autocommit=True) as connection:
        for criterion in criteria:
            connection.execute(
                """
                INSERT INTO governance.model_exceptions
                    (model_name, model_version, criterion, justification, approved_by,
                     review_due_at)
                VALUES (%s, %s, %s, %s, %s, now() + make_interval(days => %s))
                """,
                (
                    model,
                    version,
                    criterion,
                    derogation.justification,
                    derogation.approved_by,
                    derogation.review_days,
                ),
            )
        connection.execute(
            """
            INSERT INTO audit.access_log (actor, action, resource, purpose, details)
            VALUES (%s, 'model_promotion_derogation', %s, 'PRC-002', %s)
            """,
            (derogation.approved_by, f"{model}:{version}", Jsonb({"criteres": criteria})),
        )
    return len(criteria)


def promote(
    ml: MlSettings,
    candidate_alias: str = CANDIDATE_ALIAS,
    derogation: Derogation | None = None,
    dsn: str = "",
    champion_metrics: dict[str, float] | None = None,
) -> Decision:
    """Promeut le candidat s'il passe les critères. Ne fait rien, et dit pourquoi, sinon.

    Une dérogation explicite permet de passer outre : elle est alors inscrite au registre de
    gouvernance, avec son motif, son responsable et son échéance de réexamen.
    """
    mlflow.set_tracking_uri(ml.tracking_uri)
    client = mlflow.MlflowClient(ml.tracking_uri)
    candidate = _version(client, ml.model_name, candidate_alias)
    if candidate is None:
        return Decision(False, None, None, [f"aucun modèle sous l'alias « {candidate_alias} »"], {})
    champion = _version(client, ml.model_name, SERVING_ALIAS)
    if champion is not None and champion.version == candidate.version:
        return Decision(True, candidate.version, champion.version, ["déjà en service"], {})

    candidate_metrics = _metrics(client, candidate)
    # Le champion a été évalué sur *sa* période de test. Quand le réentraînement automatique
    # fait avancer le découpage, comparer les deux coûts reviendrait à comparer deux mois
    # différents : l'appelant réévalue alors le champion sur la période du candidat et passe
    # ces chiffres-là.
    if champion_metrics is None:
        champion_metrics = _metrics(client, champion)
    reasons = check(candidate_metrics, champion_metrics, ml)
    if reasons and derogation is not None:
        if dsn:
            record_exception(dsn, ml.model_name, candidate.version, reasons, derogation)
        client.set_model_version_tag(
            ml.model_name, candidate.version, "derogation", derogation.justification[:480]
        )
        client.set_model_version_tag(
            ml.model_name, candidate.version, "derogation_responsable", derogation.approved_by
        )
        logger.warning(
            "Mise en service sous dérogation de %s : %s",
            derogation.approved_by,
            " ; ".join(reasons),
        )
        reasons = []
    elif reasons:
        # La décision de refus est tracée sur la version : elle reste consultable dans MLflow.
        client.set_model_version_tag(
            ml.model_name, candidate.version, "promotion_refusee", " ; ".join(reasons)[:480]
        )
        logger.warning("Promotion refusée (version %s) : %s", candidate.version, "; ".join(reasons))
        return Decision(
            False,
            candidate.version,
            champion.version if champion else None,
            reasons,
            candidate_metrics,
        )

    if champion is not None:
        # Le modèle sortant reste désigné : un retour arrière est immédiat (`ancien_champion`).
        client.set_registered_model_alias(ml.model_name, "ancien_champion", champion.version)
    client.set_registered_model_alias(ml.model_name, SERVING_ALIAS, candidate.version)
    client.set_model_version_tag(ml.model_name, candidate.version, "promotion", "automatique")
    logger.info(
        "Version %s promue champion (remplace %s).",
        candidate.version,
        champion.version if champion else "aucune",
    )
    return Decision(
        True,
        candidate.version,
        champion.version if champion else None,
        ["critères de qualité, d'équité, de budget d'alertes et de coût respectés"],
        candidate_metrics,
    )


def rollback(ml: MlSettings) -> Decision:
    """Remet en service le champion précédent (incident constaté après une promotion)."""
    mlflow.set_tracking_uri(ml.tracking_uri)
    client = mlflow.MlflowClient(ml.tracking_uri)
    previous = _version(client, ml.model_name, "ancien_champion")
    if previous is None:
        return Decision(False, None, None, ["aucun champion précédent enregistré"], {})
    current = _version(client, ml.model_name, SERVING_ALIAS)
    client.set_registered_model_alias(ml.model_name, SERVING_ALIAS, previous.version)
    logger.warning("Retour arrière : version %s remise en service.", previous.version)
    return Decision(
        True,
        previous.version,
        current.version if current else None,
        ["retour arrière demandé"],
        _metrics(client, previous),
    )
