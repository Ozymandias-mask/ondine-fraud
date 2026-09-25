"""Surveillance de la dérive et réentraînement automatique du modèle de fraude.

Un modèle de fraude se périme : les fraudeurs changent de mode opératoire *parce qu'*ils sont
détectés. Ce DAG est ce qui empêche la solution de se dégrader silencieusement entre deux
interventions humaines.

Chaque nuit :

1. **mesurer** la dérive du champion en service (variables, scores, qualité, retours des
   analystes) ; la mesure est conservée dans `ops.model_drift` et publiée vers Prometheus,
   qu'il y ait dérive ou non — c'est la série qui permet de distinguer un mois atypique d'une
   usure qui s'installe ;
2. **décider** : on réentraîne si la dérive est avérée, ou si le champion a plus d'un mois.
   Sinon le DAG s'arrête là, sans consommer une heure de calcul pour rien ;
3. **entraîner** un candidat sur un découpage temporel qui **avance avec les données** (le
   test reste les trois derniers mois connus) ; graine fixe, empreinte des données et version
   des bibliothèques enregistrées dans MLflow, comme pour tout entraînement manuel ;
4. **comparer** : le champion est réévalué sur la période du candidat. Sans cela on
   comparerait deux modèles sur deux mois différents, ce qui ne prouve rien ;
5. **promouvoir** le candidat s'il passe les critères écrits à l'avance (qualité, équité,
   budget d'alertes, coût métier). Un refus n'est pas une erreur : le candidat reste
   « challenger », le motif est tracé dans MLflow et une métrique le signale. Aucune
   dérogation n'est possible ici — passer outre un critère est une décision humaine, nommée.

Le réentraînement est lourd (plusieurs dizaines de minutes) : une seule exécution à la fois,
et jamais deux le même jour sans raison mesurée.
"""

from __future__ import annotations

from datetime import UTC, datetime

from airflow.sdk import dag, task
from airflow.timetables.trigger import CronTriggerTimetable
from ondine_pipelines.orchestration import DEFAULT_ARGS, LAKE_POOL

# Au-delà, le champion est réentraîné même sans dérive détectée : les indicateurs de dérive
# sont des indices, pas des preuves, et un modèle de fraude qui n'a pas vu les données du
# mois écoulé est déjà en retard sur les fraudeurs.
MAX_MODEL_AGE_DAYS = 31


def _settings():
    from ondine_pipelines.settings import Settings

    return Settings.from_env()


def _ml():
    from ondine_ml.settings import MlSettings

    return MlSettings.from_env()


@dag(
    dag_id="ondine_model_refresh",
    # 2 h du matin : la couche gold du jour est écrite, et l'entraînement ne dispute pas la
    # mémoire aux traitements de la journée.
    schedule=CronTriggerTimetable("0 2 * * *", timezone="UTC"),
    start_date=datetime(2026, 9, 1, tzinfo=UTC),
    catchup=False,
    max_active_runs=1,
    default_args=DEFAULT_ARGS,
    tags=["ondine", "bloc4", "modele", "derive"],
    doc_md=__doc__,
)
def ondine_model_refresh():
    """Dérive mesurée chaque nuit ; réentraînement et promotion seulement si nécessaire."""

    @task
    def mesurer_derive() -> dict:
        """Mesure la dérive du champion et la conserve, qu'elle déclenche ou non."""
        from ondine_ml.drift import measure, publish, record

        settings = _settings()
        report = measure(settings, _ml(), "champion", dsn=settings.pg_dsn)
        record(settings.pg_dsn, report)
        publish(settings, report)
        return report.as_dict()

    @task
    def decider(derive: dict) -> dict:
        """Réentraîner, ou non — et le dire. Une tâche sautée sans motif est indébogable."""
        from airflow.sdk.exceptions import AirflowSkipException

        motifs = list(derive["reasons"]) if derive["reentrainement"] else []
        age_days = _model_age_days()
        if age_days is None or age_days >= MAX_MODEL_AGE_DAYS:
            motifs.append(
                f"champion entraîné il y a {age_days:.0f} jours (limite {MAX_MODEL_AGE_DAYS})"
                if age_days is not None
                else "aucun champion en service"
            )
        if not motifs:
            raise AirflowSkipException(
                f"Aucune dérive et champion récent ({age_days:.0f} jours) :"
                " le réentraînement est inutile."
            )
        return {"motifs": motifs, "mois": derive["current"]}

    @task(pool=LAKE_POOL, execution_timeout=None)
    def entrainer(decision: dict) -> dict:
        """Entraîne un candidat sur les derniers mois connus, et l'enregistre comme challenger."""
        import mlflow
        from ondine_ml.train import train, training_summary

        ml = _ml().rolling(decision["mois"])
        result = train(_settings(), ml, register=True)
        if result.model_version:
            # Le motif du réentraînement reste attaché au modèle : dans six mois, la question
            # « pourquoi cette version existe-t-elle ? » a une réponse.
            mlflow.set_tracking_uri(ml.tracking_uri)
            mlflow.MlflowClient(ml.tracking_uri).set_model_version_tag(
                ml.model_name,
                result.model_version,
                "declencheur",
                " ; ".join(decision["motifs"])[:480],
            )
        return {"mois": decision["mois"], "resume": training_summary(result)}

    @task(pool=LAKE_POOL)
    def comparer(entrainement: dict) -> dict:
        """Réévalue le champion sur la période du candidat : sans cela la comparaison ment."""
        from ondine_ml.serving import evaluate_registered

        ml = _ml().rolling(entrainement["mois"])
        return evaluate_registered(_settings(), ml, "champion")

    @task
    def promouvoir(entrainement: dict, champion: dict) -> dict:
        """Promeut le candidat s'il passe les critères ; trace le refus sinon."""
        from ondine_ml.promote import promote

        settings = _settings()
        ml = _ml().rolling(entrainement["mois"])
        decision = promote(
            ml, "challenger", dsn=settings.pg_dsn, champion_metrics=champion.get("metrics")
        )
        _publish_promotion(settings, decision)
        return decision.as_dict()

    derive = mesurer_derive()
    decision = decider(derive)
    entrainement = entrainer(decision)
    promouvoir(entrainement, comparer(entrainement))


def _model_age_days() -> float | None:
    """Âge du champion en service, en jours (None s'il n'y en a pas)."""
    import mlflow
    from ondine_ml.promote import SERVING_ALIAS

    ml = _ml()
    mlflow.set_tracking_uri(ml.tracking_uri)
    client = mlflow.MlflowClient(ml.tracking_uri)
    try:
        version = client.get_model_version_by_alias(ml.model_name, SERVING_ALIAS)
    except Exception:  # aucun champion : il faut en produire un  # noqa: BLE001
        return None
    return (datetime.now(UTC).timestamp() - version.creation_timestamp / 1000) / 86400


def _publish_promotion(settings, decision) -> None:
    """Publie l'issue de la promotion : un refus doit être visible sans lire les journaux."""
    if not settings.pushgateway_url:
        return
    from prometheus_client import CollectorRegistry, Gauge, pushadd_to_gateway

    registry = CollectorRegistry()
    Gauge(
        "model_promotion_success",
        "1 si le dernier candidat a été promu champion, 0 s'il a été refusé",
        registry=registry,
    ).set(float(decision.promoted))
    Gauge(
        "model_promotion_timestamp_seconds",
        "Horodatage de la dernière tentative de promotion automatique",
        registry=registry,
    ).set(datetime.now(UTC).timestamp())
    try:
        pushadd_to_gateway(
            settings.pushgateway_url, job="ondine_model_refresh", registry=registry, timeout=5
        )
    except OSError:  # la supervision ne doit jamais faire échouer un réentraînement réussi
        pass


ondine_model_refresh()
