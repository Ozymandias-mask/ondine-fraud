"""Ligne de commande de l'API : servir, alimenter le magasin, charger le référentiel."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

from ondine_pipelines.lake import connect
from ondine_pipelines.settings import Settings

from .settings import ApiSettings
from .store import OnlineStore

logger = logging.getLogger(__name__)


def load_referentials(settings: Settings, store: OnlineStore) -> dict[str, int]:
    """Charge le référentiel marchands de la couche silver dans le magasin en ligne.

    Deux variables du modèle (secteur du marchand, marchand en ligne) viennent de ce
    référentiel : il est rechargé par une tâche planifiée, pas à chaque paiement.
    """
    con = connect(settings)
    try:
        uri = settings.uri("silver", "merchants", "merchants.parquet")
        # Le chemin vient de la configuration, pas d'une entrée utilisateur.
        query = f"SELECT merchant_id, category, is_online FROM read_parquet('{uri}')"  # noqa: S608
        rows = con.execute(query).fetchall()
    finally:
        con.close()
    loaded = store.load_merchants(
        [{"merchant_id": r[0], "category": r[1], "is_online": r[2]} for r in rows]
    )
    return {"marchands": loaded}


def main(argv: list[str] | None = None) -> int:
    """Point d'entrée `ondine-api`."""
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"), format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(prog="ondine-api", description=__doc__)
    commands = parser.add_subparsers(dest="commande", required=True)
    serve = commands.add_parser("serve", help="démarre l'API de scoring")
    serve.add_argument("--host", default="0.0.0.0")  # noqa: S104 (conteneur, exposé via compose)
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--workers", type=int, default=int(os.environ.get("API_WORKERS", "2")))
    commands.add_parser("feed", help="alimente le magasin en ligne depuis Kafka")
    commands.add_parser("referentials", help="charge le référentiel marchands dans le magasin")
    rejeu = commands.add_parser(
        "replay", help="rejoue de vraies transactions à travers l'API (démonstration, mesure)"
    )
    rejeu.add_argument("--month", required=True, help="mois à rejouer, au format AAAA-MM")
    rejeu.add_argument("--limit", type=int, default=5000, help="nombre de transactions")
    rejeu.add_argument("--url", default="http://scoring-api:8000", help="adresse de l'API")
    rejeu.add_argument(
        "--no-warmup",
        action="store_true",
        help="ne pas préchauffer le magasin (scores non représentatifs)",
    )
    args = parser.parse_args(argv)

    if args.commande == "serve":
        import uvicorn

        uvicorn.run(
            "ondine_api.app:build",
            factory=True,
            host=args.host,
            port=args.port,
            workers=args.workers,
            access_log=False,  # le journal d'accès est remplacé par les métriques Prometheus
        )
        return 0
    if args.commande == "replay":
        from .replay import read_transactions, replay, warm_store

        pipelines = Settings.from_env()
        transactions = read_transactions(pipelines, args.month, args.limit)
        if not args.no_warmup:
            api = ApiSettings.from_env()
            warm_store(
                pipelines,
                OnlineStore.from_url(api.redis_url, api.pseudonymisation_key),
                transactions,
            )
        logger.info("%d transactions lues, envoi à %s", len(transactions), args.url)
        report = replay(args.url, transactions)
        print(json.dumps(report.summary(), indent=2, ensure_ascii=False))
        return 0 if report.sent else 1
    if args.commande == "feed":
        from .stream import main as feed

        return feed()
    api = ApiSettings.from_env()
    store = OnlineStore.from_url(api.redis_url, api.pseudonymisation_key)
    print(json.dumps(load_referentials(Settings.from_env(), store), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
