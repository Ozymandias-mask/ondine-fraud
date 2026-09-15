"""Ligne de commande des pipelines (utilisée en développement ; Airflow appelle les fonctions)."""

from __future__ import annotations

import argparse
import logging
import sys

from .settings import Settings


def run_silver(settings: Settings, dataset: str, month: str, resume: bool) -> list[str]:
    """Construit la couche silver en respectant les dépendances. Retourne les tâches en échec.

    Une tâche en échec n'arrête pas les autres (les mois sont indépendants) ; avec `resume`,
    les tâches dont la dernière exécution a réussi ne sont pas recalculées.
    """
    from .lake import with_retries
    from .registry import Registry
    from .silver.activity import build_month
    from .silver.labels import build_labels
    from .silver.referentials import build_referentials
    from .silver.support import build_support_month

    registry = Registry(settings)
    failures: list[str] = []
    log = logging.getLogger("ondine_pipelines.cli")

    def months(source: str) -> list[tuple[int, int]]:
        periods = registry.ingested_periods(source) if month == "all" else [month]
        return [(int(p[:4]), int(p[5:7])) for p in periods]

    tasks: list[tuple[str, object]] = []
    if dataset in ("all", "referentials"):
        tasks.append(("referentials", lambda: build_referentials(settings)))
    if dataset in ("all", "fraud_labels"):
        tasks.append(("fraud_labels", lambda: build_labels(settings)))
    for name in ("transactions", "app_events"):
        if dataset in ("all", name):
            for y, m in months(name):
                tasks.append(
                    (f"{name}_{y}-{m:02d}", lambda n=name, y=y, m=m: build_month(settings, n, y, m))
                )
    if dataset in ("all", "support_messages"):
        for y, m in months("support_emails"):
            tasks.append(
                (
                    f"support_messages_{y}-{m:02d}",
                    lambda y=y, m=m: build_support_month(settings, y, m),
                )
            )

    for task, action in tasks:
        if resume and registry.last_success("silver", task):
            log.info("%s : déjà construit, ignoré (--no-resume pour recalculer)", task)
            continue
        try:
            print(task, with_retries(action))
        except Exception:
            log.exception("%s : échec", task)
            failures.append(task)
    return failures


def run_gold(settings: Settings, month: str, resume: bool) -> list[str]:
    """Construit les tables « première apparition » puis les features mois par mois."""
    from .gold.features import build_features_month, build_first_seen
    from .lake import with_retries
    from .registry import Registry

    registry = Registry(settings)
    log = logging.getLogger("ondine_pipelines.cli")
    failures: list[str] = []
    # Toujours recalculées : elles dépendent de tout l'historique disponible.
    try:
        print("first_seen", with_retries(lambda: build_first_seen(settings)))
    except Exception:
        log.exception("first_seen : échec, features non calculables")
        return ["first_seen"]
    periods = registry.ingested_periods("transactions") if month == "all" else [month]
    for period in periods:
        y, m = int(period[:4]), int(period[5:7])
        task = f"transaction_features_{y}-{m:02d}"
        if resume and registry.last_success("gold", task):
            log.info("%s : déjà construit, ignoré", task)
            continue
        try:
            print(task, with_retries(lambda y=y, m=m: build_features_month(settings, y, m)))
        except Exception:
            log.exception("%s : échec", task)
            failures.append(task)
    return failures


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
