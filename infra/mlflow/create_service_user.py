"""Crée le compte de service MLflow utilisé par les traitements (moindre privilège).

Idempotent : si le compte existe déjà, son mot de passe est remis à la valeur du fichier
`.env`. L'administrateur reste réservé aux personnes (interface web).

Le script est bavard en cas d'échec : il affiche le message du serveur et l'état du compte
administrateur, car un compte de service absent bloque tous les traitements du modèle.
"""

from __future__ import annotations

import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request

BASE = os.environ["MLFLOW_URL"].rstrip("/")
ADMIN_USER = os.environ["MLFLOW_ADMIN_USER"]
ADMIN_PASSWORD = os.environ["MLFLOW_ADMIN_PASSWORD"]
SERVICE_USER = os.environ["MLFLOW_SERVICE_USER"]
SERVICE_PASSWORD = os.environ["MLFLOW_SERVICE_PASSWORD"]


def call(
    path: str,
    payload: dict | None = None,
    user: str = ADMIN_USER,
    password: str = ADMIN_PASSWORD,
    method: str | None = None,
) -> tuple[int, str]:
    """Appelle l'API MLflow ; retourne (code HTTP, corps de la réponse tronqué)."""
    token = base64.b64encode(f"{user}:{password}".encode()).decode()
    request = urllib.request.Request(  # noqa: S310 (URL interne fixée par l'environnement)
        f"{BASE}{path}",
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={"Content-Type": "application/json", "Authorization": f"Basic {token}"},
        method=method or ("POST" if payload is not None else "GET"),
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:  # noqa: S310
            return response.status, response.read().decode("utf-8", "replace")[:300]
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode("utf-8", "replace")[:300]
    except OSError as error:  # serveur injoignable : message lisible plutôt qu'une trace
        return 0, str(error)


def wait_for_server(attempts: int = 60) -> None:
    """Attend que le serveur réponde (démarrage, migrations, création de l'administrateur)."""
    for _ in range(attempts):
        try:
            with urllib.request.urlopen(f"{BASE}/health", timeout=5) as response:  # noqa: S310
                if response.status == 200:
                    return
        except OSError:
            pass
        time.sleep(2)
    sys.exit("Serveur MLflow injoignable.")


def ensure_service_user() -> tuple[int, str]:
    """Crée le compte, ou remet son mot de passe s'il existe déjà."""
    payload = {"username": SERVICE_USER, "password": SERVICE_PASSWORD}
    status, message = call("/api/2.0/mlflow/users/create", payload)
    if status in (400, 409):  # déjà présent : le mot de passe est remis à jour (PATCH)
        status, message = call("/api/2.0/mlflow/users/update-password", payload, method="PATCH")
    return status, message


def main() -> int:
    """Attend le serveur, crée le compte de service et vérifie qu'il peut se connecter."""
    if len(SERVICE_PASSWORD) < 12:
        sys.exit("MLFLOW_SERVICE_PASSWORD doit faire plus de 12 caractères (voir .env).")
    wait_for_server()

    status, message = 0, ""
    for attempt in range(1, 6):
        status, message = ensure_service_user()
        if status < 400:
            break
        # L'administrateur peut ne pas encore être créé juste après le démarrage.
        print(f"Essai {attempt}/5 : le serveur a répondu HTTP {status} ({message.strip()})")
        time.sleep(5)

    if status >= 400 or status == 0:
        admin_status, admin_body = call(f"/api/2.0/mlflow/users/get?username={ADMIN_USER}")
        print(f"État du compte administrateur « {ADMIN_USER} » : HTTP {admin_status} {admin_body}")
        sys.exit(
            f"Compte de service « {SERVICE_USER} » non créé (HTTP {status}) : {message.strip()}"
        )

    # Vérification finale : le compte de service peut bien s'authentifier.
    check, body = call(
        "/api/2.0/mlflow/experiments/search",
        {"max_results": 1},
        user=SERVICE_USER,
        password=SERVICE_PASSWORD,
    )
    if check >= 400 or check == 0:
        sys.exit(f"Le compte « {SERVICE_USER} » ne peut pas se connecter (HTTP {check}) : {body}")
    print(f"Compte de service MLflow « {SERVICE_USER} » prêt et vérifié.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
