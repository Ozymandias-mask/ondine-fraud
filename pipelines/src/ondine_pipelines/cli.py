"""Ligne de commande des pipelines (utilisée en développement ; Airflow appelle les fonctions)."""

from __future__ import annotations

import argparse
import logging
import sys

from .settings import Settings


def main(argv: list[str] | None = None) -> int:
    """Point d'entrée : `ondine-pipelines ingest --dataset transactions`."""
    parser = argparse.ArgumentParser(prog="ondine-pipelines")
    sub = parser.add_subparsers(dest="command", required=True)
    ingest = sub.add_parser("ingest", help="Zone d'arrivée -> bronze")
    ingest.add_argument("--dataset", default="all")
    ingest.add_argument("--limit", type=int, help="Nombre maximal de fichiers par jeu")
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
    except Exception:
        logging.getLogger(__name__).exception("Échec du pipeline")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
