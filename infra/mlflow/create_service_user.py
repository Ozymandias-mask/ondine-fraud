"""Crée le compte de service MLflow utilisé par les traitements (moindre privilège).

Idempotent : si le compte existe déjà, son mot de passe est simplement remis à jour.
L'administrateur reste réservé aux personnes (interface web).
"""

from __future__ import annotations

import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request

BASE = os.environ["MLFLOW_URL"]
ADMIN = (os.environ["MLFLOW_ADMIN_USER"], os.environ["MLFLOW_ADMIN_PASSWORD"])
SERVICE = (os.environ["MLFLOW_SERVICE_USER"], os.environ["MLFLOW_SERVICE_PASSWORD"])


def call(path: str, payload: dict) -> int:
    """Appelle l'API d'authentification MLflow ; retourne le code HTTP."""
    token = base64.b64encode(f"{ADMIN[0]}:{ADMIN[1]}".encode()).decode()
    request = urllib.request.Request(  # noqa: S310 (URL interne fixée par l'environnement)
        f"{BASE}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Basic {token}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:  # noqa: S310
            return response.status
    except urllib.error.HTTPError as error:
        return error.code


def wait_for_server(attempts: int = 60) -> None:
    """Attend que le serveur réponde (démarrage de la base et des migrations)."""
    for _ in range(attempts):
        try:
            with urllib.request.urlopen(f"{BASE}/health", timeout=5) as response:  # noqa: S310
                if response.status == 200:
                    return
        except OSError:
            time.sleep(2)
    raise SystemExit("Serveur MLflow injoignable")


wait_for_server()
username, password = SERVICE
status = call("/api/2.0/mlflow/users/create", {"username": username, "password": password})
if status == 400:  # déjà présent : on remet le mot de passe du fichier .env
    status = call(
        "/api/2.0/mlflow/users/update-password", {"username": username, "password": password}
    )
print(f"Compte de service MLflow « {username} » prêt (HTTP {status}).")
sys.exit(0 if status < 400 else 1)
