"""Ligne de commande : `ondine-sim batch ...` et `ondine-sim stream ...`."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import PRESETS, get_config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ondine-sim", description="Simulateur de données Ondine Bank (fictives)."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    batch = sub.add_parser("batch", help="Générer l'historique dans la zone d'arrivée")
    batch.add_argument("--preset", choices=sorted(PRESETS), default="small")
    batch.add_argument("--output", type=Path, help="Dossier de sortie (défaut : data/landing)")
    batch.add_argument("--workers", type=int, help="Nombre de processus")
    batch.add_argument("--seed", type=int, help="Graine aléatoire")
    batch.add_argument("--months", type=int, help="Nombre de mois à générer")
    batch.add_argument(
        "--resume",
        action="store_true",
        help="Ne régénérer que les parts sans manifeste (reprise après échec)",
    )

    stream = sub.add_parser("stream", help="Envoyer des transactions en continu dans Kafka")
    stream.add_argument("--preset", choices=sorted(PRESETS), default="full")
    stream.add_argument(
        "--bootstrap", default="localhost:9092", help="Adresse(s) des brokers Kafka"
    )
    stream.add_argument("--rate", type=float, default=300.0, help="Transactions par seconde")
    stream.add_argument(
        "--burst-rate", type=float, default=2000.0, help="Débit pendant un pic de charge"
    )
    stream.add_argument(
        "--drift", action="store_true", help="Activer dès le départ le nouveau scénario de fraude"
    )
    stream.add_argument(
        "--time-scale",
        type=float,
        default=1 / 60,
        help="Compression de la durée des attaques (1/60 : 1 h devient 1 min)",
    )
    stream.add_argument(
        "--metrics-port",
        type=int,
        default=8000,
        help="Port des métriques Prometheus et de l'API de pilotage",
    )
    stream.add_argument(
        "--duration", type=float, default=0, help="Durée en secondes (0 : sans fin)"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Point d'entrée de la ligne de commande."""
    args = _parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s"
    )
    try:
        if args.command == "batch":
            from .batch import run_batch

            cfg = get_config(
                args.preset,
                output_dir=args.output,
                workers=args.workers,
                seed=args.seed,
                months=args.months,
            )
            run_batch(cfg, resume=args.resume)
        else:
            from .stream import StreamSettings, run_stream

            cfg = get_config(args.preset)
            run_stream(
                cfg,
                StreamSettings(
                    bootstrap=args.bootstrap,
                    rate=args.rate,
                    burst_rate=args.burst_rate,
                    drift=args.drift,
                    time_scale=args.time_scale,
                    metrics_port=args.metrics_port,
                    duration_s=args.duration,
                ),
            )
    except KeyboardInterrupt:
        logging.getLogger(__name__).warning("Interrompu par l'utilisateur")
        return 130
    except Exception:
        logging.getLogger(__name__).exception("Échec du simulateur")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
