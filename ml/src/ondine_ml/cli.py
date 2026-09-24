"""Ligne de commande du modèle : `train`, `evaluate`, `explain`, `promote`, `rollback`."""

from __future__ import annotations

import argparse
import json
import logging
import sys

from ondine_pipelines.settings import Settings

from .settings import MlSettings


def main(argv: list[str] | None = None) -> int:
    """Point d'entrée : entraînement, évaluation d'un modèle enregistré, explication."""
    parser = argparse.ArgumentParser(prog="ondine-ml")
    sub = parser.add_subparsers(dest="command", required=True)
    train_cmd = sub.add_parser("train", help="Entraîne un modèle et l'enregistre dans MLflow")
    train_cmd.add_argument(
        "--no-register", action="store_true", help="Ne pas enregistrer au registre"
    )
    train_cmd.add_argument("--rounds", type=int, help="Nombre maximal d'arbres")
    evaluate_cmd = sub.add_parser(
        "evaluate", help="Évalue un modèle du registre sur le jeu de test"
    )
    evaluate_cmd.add_argument(
        "--alias", default="champion", help="Alias du modèle (champion, challenger)"
    )
    explain_cmd = sub.add_parser("explain", help="Explique une transaction du jeu de test")
    explain_cmd.add_argument("--alias", default="champion")
    explain_cmd.add_argument(
        "--rank", type=int, default=1, help="n-ième transaction la plus risquée"
    )
    promote_cmd = sub.add_parser(
        "promote", help="Promeut le candidat en champion s'il passe les critères"
    )
    promote_cmd.add_argument("--alias", default="challenger", help="Alias du candidat")
    promote_cmd.add_argument(
        "--derogation",
        help="Motif d'une mise en service malgré un critère non tenu (inscrite au registre)",
    )
    promote_cmd.add_argument("--responsable", help="Personne qui engage sa responsabilité")
    promote_cmd.add_argument(
        "--reexamen-jours", type=int, default=90, help="Échéance de réexamen de la dérogation"
    )
    sub.add_parser("rollback", help="Remet en service le champion précédent")
    metrics_cmd = sub.add_parser("metrics", help="Affiche toutes les métriques d'un modèle")
    metrics_cmd.add_argument("--alias", default="challenger")
    report_cmd = sub.add_parser("report", help="Affiche les rapports détaillés d'un modèle")
    report_cmd.add_argument("--alias", default="challenger")
    report_cmd.add_argument("--name", default="", help="Filtre sur le nom du rapport")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s"
    )

    settings, ml = Settings.from_env(), MlSettings.from_env()
    try:
        if args.command == "train":
            from .train import train, training_summary

            overrides = {"num_boost_round": args.rounds} if args.rounds else {}
            result = train(
                settings,
                ml if not overrides else type(ml)(**{**ml.__dict__, **overrides}),
                register=not args.no_register,
            )
            print(json.dumps(training_summary(result), indent=2, ensure_ascii=False))
        elif args.command == "evaluate":
            from .serving import evaluate_registered

            print(
                json.dumps(
                    evaluate_registered(settings, ml, args.alias), indent=2, ensure_ascii=False
                )
            )
        elif args.command == "metrics":
            from .promote import metrics_of

            print(json.dumps(metrics_of(ml, args.alias), indent=2, ensure_ascii=False))
        elif args.command == "report":
            from .promote import report_of

            print(json.dumps(report_of(ml, args.alias, args.name), indent=2, ensure_ascii=False))
        elif args.command in ("promote", "rollback"):
            from .promote import Derogation, promote, rollback

            if args.command == "rollback":
                decision = rollback(ml)
            else:
                derogation = None
                if args.derogation:
                    if not args.responsable:
                        parser.error("--derogation exige --responsable : une décision s'assume.")
                    derogation = Derogation(args.derogation, args.responsable, args.reexamen_jours)
                decision = promote(ml, args.alias, derogation, settings.pg_dsn)
            print(json.dumps(decision.as_dict(), indent=2, ensure_ascii=False))
            return 0 if decision.promoted else 2
        elif args.command == "explain":
            from .serving import explain_registered

            print(
                json.dumps(
                    explain_registered(settings, ml, args.alias, args.rank),
                    indent=2,
                    ensure_ascii=False,
                )
            )
    except Exception:
        logging.getLogger(__name__).exception("Échec de la commande %s", args.command)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
