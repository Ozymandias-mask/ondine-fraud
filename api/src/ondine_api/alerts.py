"""File d'alertes soumise aux analystes : écriture par l'API, lecture par l'interface.

Une décision de blocage n'a de valeur que si quelqu'un peut l'examiner, la justifier au
client et, le cas échéant, la contredire. Chaque alerte conserve donc ce qui l'a produite —
probabilité, seuil appliqué, version du modèle, facteurs explicatifs — puis la suite qui lui
a été donnée par un analyste identifié.

Ces retours ne servent pas qu'à l'analyste : confirmations et rejets mesurent la qualité du
modèle en production, bien avant que les déclarations de fraude des clients n'arrivent.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

logger = logging.getLogger(__name__)

STATUSES = {
    "a_traiter": "À traiter",
    "fraude_confirmee": "Fraude confirmée",
    "ecartee": "Écartée (opération légitime)",
}


@dataclass
class Alert:
    """Une alerte telle que l'interface l'affiche."""

    transaction_id: str
    event_time: datetime
    client_id: str
    amount: float
    channel: str | None
    merchant_country: str | None
    probability: float
    threshold: float
    model_version: str
    explanation: str
    factors: list[dict]
    status: str
    reviewed_by: str | None = None
    reviewed_at: datetime | None = None
    review_note: str | None = None

    @property
    def status_label(self) -> str:
        """Libellé lisible de l'état, jamais un simple code."""
        return STATUSES.get(self.status, self.status)


class AlertQueue:
    """Accès à la file d'alertes. Une connexion par opération : le volume est faible."""

    def __init__(self, dsn: str) -> None:
        """File adossée à la base de gouvernance (compte de l'API, au moindre privilège)."""
        self.dsn = dsn

    def _connect(self):
        return psycopg.connect(self.dsn, autocommit=True, row_factory=dict_row)

    def record(self, alert: dict) -> None:
        """Enregistre une alerte. Rejouer la même transaction ne crée pas de doublon."""
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO ops.alerts (transaction_id, event_time, client_id, amount, channel,
                    merchant_country, probability, threshold, model_version, explanation, factors)
                VALUES (%(transaction_id)s, %(event_time)s, %(client_id)s, %(amount)s,
                    %(channel)s, %(merchant_country)s, %(probability)s, %(threshold)s,
                    %(model_version)s, %(explanation)s, %(factors)s)
                ON CONFLICT (transaction_id) DO NOTHING
                """,
                {**alert, "factors": Jsonb(alert["factors"])},
            )

    def pending(self, limit: int = 50, status: str = "a_traiter") -> list[Alert]:
        """Alertes d'un état donné, les plus risquées d'abord."""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM ops.alerts WHERE status = %s ORDER BY probability DESC LIMIT %s",
                (status, limit),
            ).fetchall()
        return [_to_alert(row) for row in rows]

    def get(self, transaction_id: str) -> Alert | None:
        """Une alerte précise, ou rien si elle n'existe pas."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM ops.alerts WHERE transaction_id = %s", (transaction_id,)
            ).fetchone()
        return _to_alert(row) if row else None

    def review(self, transaction_id: str, status: str, analyst: str, note: str = "") -> bool:
        """Enregistre la décision d'un analyste. Retourne faux si l'alerte est inconnue."""
        if status not in STATUSES:
            raise ValueError(f"État inconnu : {status}")
        with self._connect() as connection:
            updated = connection.execute(
                """
                UPDATE ops.alerts
                   SET status = %s, reviewed_by = %s, reviewed_at = now(), review_note = %s
                 WHERE transaction_id = %s
                """,
                (status, analyst, note or None, transaction_id),
            ).rowcount
            if updated:
                connection.execute(
                    """
                    INSERT INTO audit.access_log (actor, action, resource, purpose, details)
                    VALUES (%s, 'alert_review', %s, 'PRC-002', %s)
                    """,
                    (analyst, f"alert:{transaction_id}", Jsonb({"status": status})),
                )
        return bool(updated)

    def counts(self) -> dict[str, int]:
        """Nombre d'alertes par état : en-tête de la file de travail."""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT status, count(*) AS n FROM ops.alerts GROUP BY status"
            ).fetchall()
        return {row["status"]: int(row["n"]) for row in rows}


def _to_alert(row: dict) -> Alert:
    factors = row["factors"]
    return Alert(
        transaction_id=row["transaction_id"],
        event_time=row["event_time"],
        client_id=row["client_id"],
        amount=float(row["amount"]),
        channel=row["channel"],
        merchant_country=row["merchant_country"],
        probability=float(row["probability"]),
        threshold=float(row["threshold"]),
        model_version=row["model_version"],
        explanation=row["explanation"],
        factors=json.loads(factors) if isinstance(factors, str) else factors,
        status=row["status"],
        reviewed_by=row["reviewed_by"],
        reviewed_at=row["reviewed_at"],
        review_note=row["review_note"],
    )
