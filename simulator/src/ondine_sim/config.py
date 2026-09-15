"""Paramètres du simulateur et préréglages de volumétrie."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date
from pathlib import Path


@dataclass(frozen=True)
class SimConfig:
    """Paramètres d'une génération.

    Une même configuration avec la même graine produit exactement les mêmes
    données, quel que soit le nombre de processus utilisés (reproductibilité).
    """

    seed: int = 42
    n_clients: int = 1_000_000
    n_merchants: int = 80_000
    n_mule_accounts: int = 2_000
    start_date: date = date(2025, 1, 1)
    months: int = 12

    # ~4,2 transactions par client et par mois => ~50 M de transactions sur 12 mois.
    avg_tx_per_client_month: float = 4.2
    # Événements applicatifs par transaction (connexions, 3DS, consultations...).
    events_per_tx: float = 1.0

    # Part des clients victimes d'une fraude chaque mois. Volontairement plus
    # élevée que la réalité pour disposer d'assez d'exemples positifs.
    fraud_victim_rate: float = 0.002
    # Mois (index à partir de 0) où apparaît le nouveau scénario de fraude
    # « enrôlement frauduleux de wallet » : sert à démontrer la dérive.
    drift_month: int = 9
    # Mois (index) où l'application passe en v5 : nouveaux champs dans les
    # événements et nouvelle colonne dans les transactions (évolution de schéma).
    schema_change_month: int = 6

    # Part des fraudes effectivement déclarées (le reste n'est jamais labellisé).
    fraud_report_rate: float = 0.9
    # Réclamations « non fraude » envoyées au support, par client et par mois.
    support_emails_per_client_month: float = 0.015

    rows_per_part: int = 1_000_000
    output_dir: Path = field(default_factory=lambda: Path("data/landing"))
    workers: int = 4


PRESETS: dict[str, SimConfig] = {
    # Tests unitaires et CI : quelques secondes.
    "small": SimConfig(
        n_clients=5_000,
        n_merchants=2_000,
        n_mule_accounts=50,
        months=2,
        drift_month=1,
        schema_change_month=1,
        rows_per_part=50_000,
    ),
    # Développement : ~5 M de transactions.
    "demo": SimConfig(n_clients=100_000, n_merchants=20_000, n_mule_accounts=500),
    # Volumétrie cible du projet : ~50 M de transactions, ~50 M d'événements.
    "full": SimConfig(),
}


def get_config(preset: str, **overrides: object) -> SimConfig:
    """Retourne un préréglage en appliquant les valeurs surchargées non nulles."""
    if preset not in PRESETS:
        raise ValueError(f"Préréglage inconnu : {preset!r}. Choix : {sorted(PRESETS)}")
    clean = {k: v for k, v in overrides.items() if v is not None}
    return replace(PRESETS[preset], **clean)
