"""Génère le fichier .env avec des secrets aléatoires forts.

Usage (Windows, macOS ou Linux) :  python scripts/generate_env.py [--force]

Les secrets sont produits par le module `secrets` (générateur cryptographique du
système). Le fichier existant n'est jamais écrasé sans --force : régénérer les
mots de passe après le premier démarrage rendrait les volumes PostgreSQL et S3
inaccessibles.
"""

from __future__ import annotations

import argparse
import base64
import re
import secrets
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / ".env.example"
TARGET = ROOT / ".env"

# Variables dont la valeur n'est pas un secret : reprises telles quelles du modèle.
KEEP_TEMPLATE_VALUE = {"KAFKA_UI_USER", "POSTGRES_ADMIN_USER", "GRAFANA_ADMIN_USER"}


def secret_for(name: str) -> str:
    """Valeur adaptée à la variable : identifiant Kafka, clé d'accès ou mot de passe."""
    if name == "KAFKA_CLUSTER_ID":
        return base64.urlsafe_b64encode(uuid.uuid4().bytes).decode().rstrip("=")
    if name.endswith("_ACCESS_KEY"):
        return secrets.token_hex(10).upper()
    return secrets.token_urlsafe(24)


def main() -> int:
    """Crée .env à partir de .env.example."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="Écraser un .env existant")
    args = parser.parse_args()

    if TARGET.exists() and not args.force:
        print(f"{TARGET} existe déjà : rien à faire (utiliser --force pour le régénérer).")
        return 0

    lines = []
    for line in TEMPLATE.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^([A-Z0-9_]+)=(.*)$", line)
        if not match or match.group(1) in KEEP_TEMPLATE_VALUE:
            lines.append(line)
            continue
        lines.append(f"{match.group(1)}={secret_for(match.group(1))}")

    TARGET.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        TARGET.chmod(0o600)  # lisible par le seul propriétaire (sans effet sous Windows)
    except OSError:
        pass
    print(f"{TARGET} créé avec des secrets aléatoires. Ne jamais le versionner.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
