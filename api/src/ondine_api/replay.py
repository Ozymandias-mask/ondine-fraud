"""Rejeu de transactions réelles à travers l'API, pour démonstration et mesure.

Le meilleur moyen de vérifier une mise en service n'est pas de simuler des appels, mais de
rejouer de **vraies** transactions du data lake par le même chemin qu'en production : requête
HTTP, lecture du magasin en ligne, calcul des variables, modèle, décision, dépôt de l'alerte.

Ce que le rejeu mesure au passage — débit, latences observées, taux de blocage — est
exactement ce qu'il faut pour démontrer que le budget de 100 ms est tenu sur des données
réelles, et pour remplir la file des analystes avec des cas véritables.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from ondine_pipelines.lake import connect
from ondine_pipelines.settings import Settings

logger = logging.getLogger(__name__)

# Champs acceptés par l'API : le message d'autorisation, rien de plus.
FIELDS = (
    "transaction_id",
    "event_time",
    "client_id",
    "card_id",
    "merchant_id",
    "mcc",
    "merchant_country",
    "channel",
    "entry_mode",
    "amount",
    "currency",
    "three_ds_authenticated",
    "device_id",
    "beneficiary_iban",
    "latitude",
    "longitude",
    "wallet_type",
)


@dataclass
class ReplayReport:
    """Ce qu'a donné le rejeu : volume, décisions, latences."""

    sent: int = 0
    blocked: int = 0
    refused: int = 0
    latencies_ms: list[float] = field(default_factory=list)
    duration_s: float = 0.0

    def summary(self) -> dict[str, object]:
        """Résumé chiffré, affiché en fin de rejeu."""
        ordered = sorted(self.latencies_ms)
        if not ordered:
            return {"transactions": 0, "erreur": "aucune réponse exploitable"}
        return {
            "transactions": self.sent,
            "bloquees": self.blocked,
            "taux_blocage_%": round(100 * self.blocked / max(self.sent, 1), 4),
            "refusees_par_l_api": self.refused,
            "duree_s": round(self.duration_s, 1),
            "transactions_par_s": round(self.sent / max(self.duration_s, 1e-9), 1),
            "latence_mediane_ms": round(ordered[len(ordered) // 2], 2),
            "latence_p95_ms": round(ordered[int(len(ordered) * 0.95)], 2),
            "latence_p99_ms": round(ordered[int(len(ordered) * 0.99)], 2),
            "latence_max_ms": round(ordered[-1], 2),
        }


def read_transactions(settings: Settings, month: str, limit: int) -> list[dict]:
    """Lit des messages d'autorisation bruts (couche bronze), dans l'ordre chronologique."""
    con = connect(settings)
    try:
        pattern = settings.uri("bronze", "transactions", "**", "*.parquet")
        # Motif et colonnes viennent de la configuration et d'une liste fermée, jamais d'une
        # saisie ; le mois et la limite sont, eux, passés en paramètres.
        describe = f"DESCRIBE SELECT * FROM read_parquet('{pattern}')"  # noqa: S608
        columns = [r[0] for r in con.execute(describe).fetchall()]
        selected = [c for c in FIELDS if c in columns]
        colonnes = ", ".join(selected)
        query = f"SELECT {colonnes} FROM read_parquet('{pattern}') WHERE strftime(event_time, '%Y-%m') = ? ORDER BY event_time LIMIT ?"  # noqa: S608, E501
        rows = con.execute(query, [month, int(limit)]).fetchall()
    finally:
        con.close()
    return [
        {name: _plain(value) for name, value in zip(selected, row, strict=True)} for row in rows
    ]


def _plain(value):
    """Valeur transmissible en JSON (les décimales et horodatages deviennent du texte)."""
    if value is None or isinstance(value, bool | int | float | str):
        return value
    return str(value)


def replay(url: str, transactions: list[dict], timeout: float = 10.0) -> ReplayReport:
    """Envoie les transactions à l'API, une par une, et mesure ce qui revient."""
    report = ReplayReport()
    started = time.perf_counter()
    for payload in transactions:
        body = json.dumps(payload).encode()
        request = urllib.request.Request(  # noqa: S310 (URL interne fixée par la configuration)
            f"{url.rstrip('/')}/scorer",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        begin = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
                decision = json.loads(response.read())
        except urllib.error.HTTPError as error:
            report.refused += 1
            if report.refused <= 3:  # les premiers motifs suffisent à diagnostiquer
                logger.warning("Refus de l'API (%s) : %s", error.code, error.read()[:200])
            continue
        report.sent += 1
        report.blocked += decision["decision"] == "bloquer"
        # Latence vue par l'appelant : transport compris, comme pour le système d'autorisation.
        report.latencies_ms.append((time.perf_counter() - begin) * 1000)
    report.duration_s = time.perf_counter() - started
    return report
