"""Détection des accès anormaux aux données personnelles (violation de données potentielle).

Analyse le journal d'audit (`audit.access_log`, en ajout seul) avec des règles simples et
explicables, puis enregistre chaque écart dans `governance.security_incidents`. Les incidents
ouverts sont exposés à Prometheus (alerte) et à Grafana ; la procédure de traitement
(qualification, confinement, notification CNIL sous 72 h, information des personnes) est
décrite dans le plan de gouvernance.

| Règle | Gravité | Écart détecté |
|---|---|---|
| unknown_actor | high | compte absent de la liste des comptes autorisés |
| purpose_not_allowed | high | finalité non autorisée pour ce compte |
| mass_export | high | export de plus de 10 000 lignes par une personne |
| volume_spike | medium | volume horaire > 5 fois la moyenne des 7 jours précédents (min. 50) |
| off_hours_access | medium | accès par une personne la nuit (22 h - 6 h) ou le week-end |
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import psycopg

from ..settings import Settings

# Chaque règle renvoie : actor, window_start, window_end, evidence (JSON).
# Les fenêtres sont horaires : une même anomalie n'est signalée qu'une fois par heure.
RULES: dict[str, tuple[str, str]] = {
    "unknown_actor": (
        "high",
        """
        SELECT l.actor, date_trunc('hour', l.occurred_at), date_trunc('hour', l.occurred_at)
               + INTERVAL '1 hour',
               jsonb_build_object('accesses', count(*), 'actions', jsonb_agg(DISTINCT l.action),
                                  'resources', jsonb_agg(DISTINCT l.resource))
        FROM audit.access_log l
        LEFT JOIN governance.authorized_actors a ON a.actor = l.actor
        WHERE l.occurred_at >= %(since)s AND l.occurred_at < %(until)s AND a.actor IS NULL
        GROUP BY 1, 2, 3
        """,
    ),
    "purpose_not_allowed": (
        "high",
        """
        SELECT l.actor, date_trunc('hour', l.occurred_at), date_trunc('hour', l.occurred_at)
               + INTERVAL '1 hour',
               jsonb_build_object('accesses', count(*), 'purposes', jsonb_agg(DISTINCT l.purpose),
                                  'allowed', min(a.allowed_purposes))
        FROM audit.access_log l
        JOIN governance.authorized_actors a ON a.actor = l.actor
        WHERE l.occurred_at >= %(since)s AND l.occurred_at < %(until)s
          AND NOT l.purpose = ANY (a.allowed_purposes)
        GROUP BY 1, 2, 3
        """,
    ),
    "mass_export": (
        "high",
        """
        SELECT l.actor, date_trunc('hour', l.occurred_at), date_trunc('hour', l.occurred_at)
               + INTERVAL '1 hour',
               jsonb_build_object('exports', count(*),
                                  'rows', sum((l.details->>'rows')::bigint),
                                  'resources', jsonb_agg(DISTINCT l.resource))
        FROM audit.access_log l
        JOIN governance.authorized_actors a ON a.actor = l.actor AND a.kind = 'human'
        WHERE l.occurred_at >= %(since)s AND l.occurred_at < %(until)s AND l.action = 'export'
          AND coalesce(l.details->>'rows', '') ~ '^[0-9]+$'
        GROUP BY 1, 2, 3
        HAVING sum((l.details->>'rows')::bigint) > 10000
        """,
    ),
    "volume_spike": (
        "medium",
        """
        WITH hourly AS (
            SELECT actor, date_trunc('hour', occurred_at) AS hour, count(*) AS n
            FROM audit.access_log
            WHERE occurred_at >= %(since)s - INTERVAL '7 days' AND occurred_at < %(until)s
            GROUP BY 1, 2
        ), scored AS (
            SELECT h.actor, h.hour, h.n,
                   coalesce((SELECT sum(b.n) / (7 * 24.0) FROM hourly b
                             WHERE b.actor = h.actor AND b.hour < h.hour
                               AND b.hour >= h.hour - INTERVAL '7 days'), 0) AS baseline
            FROM hourly h
            WHERE h.hour >= date_trunc('hour', %(since)s::timestamptz)
        )
        SELECT actor, hour, hour + INTERVAL '1 hour',
               jsonb_build_object('accesses', n, 'hourly_baseline', round(baseline, 2))
        FROM scored
        WHERE n > greatest(50, 5 * baseline)
        """,
    ),
    "off_hours_access": (
        "medium",
        """
        SELECT l.actor, date_trunc('hour', l.occurred_at), date_trunc('hour', l.occurred_at)
               + INTERVAL '1 hour',
               jsonb_build_object('accesses', count(*),
                                  'local_time', min(to_char(l.occurred_at AT TIME ZONE
                                                    'Europe/Paris', 'Dy HH24:MI')),
                                  'resources', jsonb_agg(DISTINCT l.resource))
        FROM audit.access_log l
        JOIN governance.authorized_actors a ON a.actor = l.actor AND a.kind = 'human'
        WHERE l.occurred_at >= %(since)s AND l.occurred_at < %(until)s
          AND (extract(hour FROM l.occurred_at AT TIME ZONE 'Europe/Paris') NOT BETWEEN 6 AND 21
               OR extract(isodow FROM l.occurred_at AT TIME ZONE 'Europe/Paris') >= 6)
        GROUP BY 1, 2, 3
        """,
    ),
}


def detect_abnormal_access(
    settings: Settings, now: datetime | None = None, lookback_hours: int = 2
) -> list[dict[str, object]]:
    """Applique les règles sur la fenêtre récente. Retourne les **nouveaux** incidents.

    La fenêtre d'analyse chevauche l'exécution précédente (pas de trou si une exécution est
    retardée) ; la contrainte d'unicité évite les doublons.
    """
    until = now or datetime.now(UTC)
    params = {"since": until - timedelta(hours=lookback_hours), "until": until}
    created: list[dict[str, object]] = []
    with psycopg.connect(settings.pg_dsn, autocommit=True, connect_timeout=10) as pg:
        for rule, (severity, sql) in RULES.items():
            for actor, start, end, evidence in pg.execute(sql, params).fetchall():
                row = pg.execute(
                    "INSERT INTO governance.security_incidents"
                    " (rule, severity, actor, window_start, window_end, evidence)"
                    " VALUES (%s, %s, %s, %s, %s, %s)"
                    " ON CONFLICT (rule, actor, window_start) DO NOTHING RETURNING incident_id",
                    (rule, severity, actor, start, end, json.dumps(evidence)),
                ).fetchone()
                if row:
                    created.append(
                        {"incident_id": row[0], "rule": rule, "severity": severity, "actor": actor}
                    )
    return created
