"""Indicateurs de gouvernance publiés vers Prometheus (alertes et tableau de bord).

- incidents de sécurité ouverts par gravité, et ceux dont l'échéance CNIL (72 h) approche ;
- demandes RGPD en cours, en retard (plus d'un mois) et en attente de validation du DPO.
"""

from __future__ import annotations

import logging
import time

import psycopg
from prometheus_client import CollectorRegistry, Gauge, pushadd_to_gateway

from ..settings import Settings

logger = logging.getLogger(__name__)


def collect(settings: Settings) -> dict[str, dict[str, float]]:
    """Lit les compteurs de gouvernance dans PostgreSQL."""
    with psycopg.connect(settings.pg_dsn, connect_timeout=10) as pg:
        incidents = dict(
            pg.execute(
                "SELECT s.severity, count(i.incident_id) FROM"
                " (VALUES ('low'), ('medium'), ('high')) AS s (severity)"
                " LEFT JOIN governance.security_incidents i ON i.severity = s.severity"
                " AND i.status IN ('open', 'investigating') GROUP BY 1"
            ).fetchall()
        )
        at_risk = pg.execute(
            "SELECT count(*) FROM governance.security_incidents_due WHERE deadline_at_risk"
        ).fetchone()[0]
        dsr = pg.execute(
            "SELECT count(*) FILTER (WHERE status NOT IN ('completed', 'rejected')),"
            " count(*) FILTER (WHERE is_overdue),"
            " count(*) FILTER (WHERE request_type = 'erasure' AND approved_by IS NULL"
            "   AND status NOT IN ('completed', 'rejected'))"
            " FROM governance.data_subject_requests_due"
        ).fetchone()
    return {
        "security_incidents_open": {k: float(v) for k, v in incidents.items()},
        "security_incidents_deadline_at_risk": {"": float(at_risk)},
        "dsr_open": {"": float(dsr[0])},
        "dsr_overdue": {"": float(dsr[1])},
        "dsr_awaiting_approval": {"": float(dsr[2])},
    }


def push_governance_metrics(settings: Settings) -> dict[str, dict[str, float]]:
    """Publie les compteurs vers la Pushgateway (sans interrompre en cas d'échec)."""
    values = collect(settings)
    if not settings.pushgateway_url:
        return values
    registry = CollectorRegistry()
    docs = {
        "security_incidents_open": "Incidents de sécurité ouverts, par gravité",
        "security_incidents_deadline_at_risk": "Incidents ouverts depuis plus de 48 h (CNIL 72 h)",
        "dsr_open": "Demandes RGPD non clôturées",
        "dsr_overdue": "Demandes RGPD au-delà du délai légal d'un mois",
        "dsr_awaiting_approval": "Demandes d'effacement en attente de validation du DPO",
    }
    for name, series in values.items():
        labelled = name == "security_incidents_open"
        gauge = Gauge(name, docs[name], ["severity"] if labelled else [], registry=registry)
        for label, value in series.items():
            (gauge.labels(severity=label) if labelled else gauge).set(value)
    Gauge(
        "governance_last_check_timestamp_seconds",
        "Dernier contrôle de gouvernance",
        registry=registry,
    ).set(time.time())
    try:
        pushadd_to_gateway(settings.pushgateway_url, job="governance", registry=registry, timeout=5)
    except OSError as exc:
        logger.warning("Métriques de gouvernance non publiées : %s", exc)
    return values
