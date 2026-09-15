"""Ligne de commande des pipelines (utilisée en développement ; Airflow appelle les fonctions)."""

from __future__ import annotations

import argparse
import logging
import sys

from .settings import Settings


def _run_planned(settings: Settings, tasks: list, label: str) -> list[str]:
    """Exécute des tâches dans l'ordre ; une tâche en échec n'arrête pas les suivantes."""
    from .tasks import execute

    log = logging.getLogger("ondine_pipelines.cli")
    failures: list[str] = []
    if not tasks:
        log.info("%s : tout est à jour", label)
    for task in tasks:
        log.info("%s : %s", task.name, task.reason)
        try:
            print(task.name, execute(settings, task))
        except Exception:
            log.exception("%s : échec", task.name)
            failures.append(task.name)
    return failures


def run_silver(settings: Settings, dataset: str, month: str, resume: bool) -> list[str]:
    """Construit ce qui est à reconstruire en silver (voir planning). Retourne les échecs.

    Sans `resume`, tout est recalculé ; avec, seules les tâches jamais réussies, en échec ou
    dont les données bronze ont changé depuis leur dernier succès.
    """
    from .planning import plan_silver
    from .registry import Registry

    tasks = [
        t
        for t in plan_silver(Registry(settings), force=not resume)
        if dataset in ("all", t.dataset)
        and (month == "all" or t.year is None or t.name.endswith(month))
    ]
    return _run_planned(settings, tasks, "silver")


def run_gold(settings: Settings, month: str, resume: bool) -> list[str]:
    """Reconstruit les tables « première apparition » puis les mois gold à recalculer."""
    from .gold.features import build_first_seen
    from .lake import with_retries
    from .planning import plan_gold
    from .registry import Registry

    tasks = [
        t
        for t in plan_gold(Registry(settings), force=not resume)
        if month == "all" or t.name.endswith(month)
    ]
    if not tasks:
        return _run_planned(settings, tasks, "gold")
    try:
        # Recalculées dès qu'un mois change : elles dépendent de tout l'historique.
        print("first_seen", with_retries(lambda: build_first_seen(settings)))
    except Exception:
        logging.getLogger("ondine_pipelines.cli").exception("first_seen : échec")
        return ["first_seen"]
    return _run_planned(settings, tasks, "gold")


def run_kpis(settings: Settings, month: str, resume: bool) -> list[str]:
    """Rafraîchit les indicateurs métier des mois dont la couche gold a changé."""
    from .planning import plan_kpis
    from .registry import Registry

    tasks = [
        t
        for t in plan_kpis(Registry(settings), force=not resume)
        if month == "all" or t.name.endswith(month)
    ]
    return _run_planned(settings, tasks, "indicateurs")


def main(argv: list[str] | None = None) -> int:
    """Point d'entrée : `ondine-pipelines ingest --dataset transactions`."""
    parser = argparse.ArgumentParser(prog="ondine-pipelines")
    sub = parser.add_subparsers(dest="command", required=True)
    ingest = sub.add_parser("ingest", help="Zone d'arrivée -> bronze")
    ingest.add_argument("--dataset", default="all")
    ingest.add_argument("--limit", type=int, help="Nombre maximal de fichiers par jeu")
    silver = sub.add_parser("silver", help="Bronze -> silver")
    silver.add_argument(
        "--dataset",
        default="all",
        choices=[
            "all",
            "referentials",
            "fraud_labels",
            "transactions",
            "app_events",
            "support_messages",
        ],
    )
    silver.add_argument("--month", default="all", help="AAAA-MM ou all")
    silver.add_argument("--no-resume", action="store_true", help="Recalculer même ce qui a réussi")
    gold = sub.add_parser("gold", help="Silver -> gold (features et labels)")
    gold.add_argument("--month", default="all", help="AAAA-MM ou all")
    gold.add_argument("--no-resume", action="store_true", help="Recalculer même ce qui a réussi")
    kpis = sub.add_parser("kpis", help="Gold -> indicateurs métier (PostgreSQL)")
    kpis.add_argument("--month", default="all", help="AAAA-MM ou all")
    kpis.add_argument("--no-resume", action="store_true", help="Recalculer même ce qui est à jour")
    sub.add_parser("stream", help="Kafka -> bronze en continu (service stream-ingest)")
    sub.add_parser("rights", help="Traiter les demandes RGPD en attente")
    sub.add_parser("security-scan", help="Détecter les accès anormaux dans le journal d'audit")
    retention = sub.add_parser("retention", help="Purger la zone d'arrivée déjà ingérée")
    retention.add_argument("--days", type=int, default=7)
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s"
    )
    settings = Settings.from_env()
    try:
        if args.command == "ingest":
            from .ingest.landing import DATASETS, ingest_dataset

            datasets = DATASETS if args.dataset == "all" else (args.dataset,)
            for dataset in datasets:
                print(dataset, ingest_dataset(settings, dataset, args.limit))
        elif args.command == "silver":
            failures = run_silver(settings, args.dataset, args.month, not args.no_resume)
            if failures:
                logging.getLogger(__name__).error("Tâches en échec : %s", ", ".join(failures))
                return 1
        elif args.command == "kpis":
            failures = run_kpis(settings, args.month, not args.no_resume)
            if failures:
                logging.getLogger(__name__).error("Tâches en échec : %s", ", ".join(failures))
                return 1
        elif args.command == "rights":
            from .governance.monitoring import push_governance_metrics
            from .governance.rights import process_requests

            print(process_requests(settings))
            push_governance_metrics(settings)
        elif args.command == "security-scan":
            from .governance.monitoring import push_governance_metrics
            from .governance.security import detect_abnormal_access

            for incident in detect_abnormal_access(settings):
                print("Incident", incident)
            print(push_governance_metrics(settings))
        elif args.command == "retention":
            from .ingest.retention import purge_landing

            print(purge_landing(settings, args.days))
        elif args.command == "stream":
            from .stream.consumer import main as stream_main

            stream_main(settings)
        elif args.command == "gold":
            failures = run_gold(settings, args.month, not args.no_resume)
            if failures:
                logging.getLogger(__name__).error("Tâches en échec : %s", ", ".join(failures))
                return 1
    except Exception:
        logging.getLogger(__name__).exception("Échec du pipeline")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
