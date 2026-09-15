"""Ligne de commande des pipelines (utilisée en développement ; Airflow appelle les fonctions)."""

from __future__ import annotations

import argparse
import logging
import sys

from .settings import Settings


def run_silver(settings: Settings, dataset: str, month: str) -> None:
    """Construit la couche silver (tout ou partie) en respectant les dépendances."""
    from .registry import Registry
    from .silver.activity import build_month
    from .silver.labels import build_labels
    from .silver.referentials import build_referentials
    from .silver.support import build_support_month

    registry = Registry(settings)

    def months(source: str) -> list[tuple[int, int]]:
        periods = registry.ingested_periods(source) if month == "all" else [month]
        return [(int(p[:4]), int(p[5:7])) for p in periods]

    if dataset in ("all", "referentials"):
        print("referentials", build_referentials(settings))
    if dataset in ("all", "fraud_labels"):
        print("fraud_labels", build_labels(settings))
    for name in ("transactions", "app_events"):
        if dataset in ("all", name):
            for year, mon in months(name):
                print(name, f"{year}-{mon:02d}", build_month(settings, name, year, mon))
    if dataset in ("all", "support_messages"):
        for year, mon in months("support_emails"):
            print("support_messages", f"{year}-{mon:02d}", build_support_month(settings, year, mon))


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
            run_silver(settings, args.dataset, args.month)
    except Exception:
        logging.getLogger(__name__).exception("Échec du pipeline")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
