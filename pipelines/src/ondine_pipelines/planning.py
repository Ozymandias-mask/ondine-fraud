"""Planification incrémentale : quelles tâches silver, gold et indicateurs recalculer.

Règle unique : une tâche est à (re)faire si elle n'a jamais réussi, si son dernier run a
échoué, ou si l'une de ses entrées a été écrite **après le début** de son dernier succès.
Les entrées sont lues dans le registre (fichiers ingérés, lots du flux Kafka, runs des
couches précédentes) : aucun parcours du data lake n'est nécessaire.

Conséquences : un nouveau fichier ne recalcule que son mois ; le flux temps réel ne
recalcule que le mois en cours ; une reprise après panne ne refait que ce qui a échoué.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime

from .registry import Registry

# Les déclarations de fraude arrivent jusqu'à ~2 mois après la transaction : une déclaration
# reçue au mois P peut modifier les labels gold des mois P-2 à P.
LABEL_SETTLEMENT_MONTHS = 3


@dataclass(frozen=True)
class PlannedTask:
    """Tâche à exécuter, sérialisable (transmise entre tâches Airflow)."""

    layer: str  # silver, gold, analytics
    name: str  # nom de la tâche dans ops.pipeline_runs
    dataset: str
    year: int | None = None
    month: int | None = None
    reason: str = ""

    def to_dict(self) -> dict[str, object]:
        """Représentation JSON (XCom Airflow)."""
        return asdict(self)


def _latest(*values: datetime | None) -> datetime | None:
    present = [v for v in values if v is not None]
    return max(present) if present else None


def _shift(period: str, months: int) -> str:
    index = int(period[:4]) * 12 + int(period[5:7]) - 1 + months
    return f"{index // 12}-{index % 12 + 1:02d}"


def _decide(
    registry: Registry, layer: str, name: str, inputs_updated_at: datetime | None, force: bool
) -> str | None:
    """Motif de recalcul, ou None si la tâche est à jour."""
    if force:
        return "recalcul forcé"
    window = registry.last_success_window(layer, name)
    if window is None:
        return "jamais réussie ou dernier run en échec"
    if inputs_updated_at is not None and inputs_updated_at > window[0]:
        return f"entrées mises à jour le {inputs_updated_at:%Y-%m-%d %H:%M}"
    return None


def plan_silver(registry: Registry, force: bool = False) -> list[PlannedTask]:
    """Tâches silver à exécuter : référentiels et labels d'abord, puis les mois."""
    referentials = _latest(
        *(registry.bronze_updates(d).get("*") for d in ("clients", "cards", "merchants"))
    )
    labels = _latest(*registry.bronze_updates("fraud_reports").values())
    tasks: list[PlannedTask] = []
    if referentials is not None:
        reason = _decide(registry, "silver", "referentials", referentials, force)
        if reason:
            tasks.append(PlannedTask("silver", "referentials", "referentials", reason=reason))
    if labels is not None:
        reason = _decide(registry, "silver", "fraud_labels", labels, force)
        if reason:
            tasks.append(PlannedTask("silver", "fraud_labels", "fraud_labels", reason=reason))

    emails = registry.bronze_updates("support_emails")
    tickets = registry.bronze_updates("support_tickets")
    monthly = [
        ("transactions", registry.bronze_updates("transactions")),
        ("app_events", registry.bronze_updates("app_events")),
        (
            "support_messages",
            {p: _latest(emails.get(p), tickets.get(p)) for p in emails},
        ),
    ]
    for dataset, updates in monthly:
        for period, updated_at in sorted(updates.items()):
            name = f"{dataset}_{period}"
            reason = _decide(registry, "silver", name, updated_at, force)
            if reason:
                tasks.append(
                    PlannedTask("silver", name, dataset, int(period[:4]), int(period[5:7]), reason)
                )
    return tasks


def _silver_done(registry: Registry, name: str) -> datetime | None:
    window = registry.last_success_window("silver", name)
    return window[1] if window else None


def plan_gold(registry: Registry, force: bool = False) -> list[PlannedTask]:
    """Mois gold à recalculer (à lancer après silver)."""
    periods = sorted(registry.bronze_updates("transactions"))
    referentials = _silver_done(registry, "referentials")
    reports = registry.bronze_updates("fraud_reports")
    tasks = []
    for period in periods:
        previous = _shift(period, -1)
        new_reports = _latest(
            *(reports.get(_shift(period, k)) for k in range(LABEL_SETTLEMENT_MONTHS))
        )
        updated_at = _latest(
            _silver_done(registry, f"transactions_{period}"),
            _silver_done(registry, f"app_events_{period}"),
            # Les fenêtres de 30 jours lisent la fin du mois précédent.
            _silver_done(registry, f"transactions_{previous}"),
            _silver_done(registry, f"app_events_{previous}"),
            referentials,
            new_reports,
        )
        name = f"transaction_features_{period}"
        reason = _decide(registry, "gold", name, updated_at, force)
        if reason:
            tasks.append(
                PlannedTask(
                    "gold", name, "transaction_features", int(period[:4]), int(period[5:7]), reason
                )
            )
    return tasks


def plan_kpis(registry: Registry, force: bool = False) -> list[PlannedTask]:
    """Mois dont les indicateurs métier doivent être rafraîchis (après gold)."""
    tasks = []
    for period in sorted(registry.bronze_updates("transactions")):
        gold = registry.last_success_window("gold", f"transaction_features_{period}")
        if gold is None:
            continue  # pas encore de gold pour ce mois
        name = f"daily_kpis_{period}"
        reason = _decide(registry, "analytics", name, gold[1], force)
        if reason:
            tasks.append(
                PlannedTask(
                    "analytics", name, "daily_kpis", int(period[:4]), int(period[5:7]), reason
                )
            )
    return tasks
