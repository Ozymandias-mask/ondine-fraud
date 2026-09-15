"""Messages du support silver : texte libre dont les données personnelles sont masquées.

Le texte des e-mails sert à classer automatiquement les demandes (et à repérer les
signalements de fraude). Il contient des données personnelles que le modèle n'a pas à
voir : elles sont remplacées par des marqueurs ([IBAN], [NOM]...) avant publication, et un
contrôle bloquant vérifie qu'il n'en reste aucune détectable.
"""

from __future__ import annotations

import logging

from .. import quality
from ..lake import connect, month_glob
from ..registry import Registry
from ..settings import Settings
from .common import count, require_input, validate_and_publish

logger = logging.getLogger(__name__)

# Motifs de données personnelles (ordre important : l'IBAN avant les numéros génériques).
PII_PATTERNS: dict[str, tuple[str, str]] = {
    "iban": (r"\bFR\d{2}(?:\s?[0-9A-Z]){23}\b", "[IBAN]"),
    "email": (r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[EMAIL]"),
    "phone": (r"(?:\+33\s?|\b0)[1-9](?:[\s.-]?\d{2}){4}\b", "[TELEPHONE]"),
    "card": (r"((?:carte|par)\s)\d{4}\b", r"\1[CARTE]"),
    "address": (
        r"\b\d{1,3}\s(?:rue|avenue|boulevard|chemin|impasse|allée|place)\s[^,.\n]+",
        "[ADRESSE]",
    ),
}


def _mask_expression(column: str) -> str:
    expression = column
    for pattern, replacement in PII_PATTERNS.values():
        expression = f"regexp_replace({expression}, '{pattern}', '{replacement}', 'g')"
    # Prénom puis nom de l'expéditeur, où qu'ils apparaissent (signature, formule...).
    # « \b » ne reconnaît pas les lettres accentuées (Hervé) : les limites de mot sont donc
    # définies explicitement par « tout caractère qui n'est pas une lettre Unicode ».
    # Chaque nom est traité deux fois : une occurrence qui en suit immédiatement une autre
    # (« Hervé\nHervé ») partage son séparateur avec la précédente.
    for name in ("first_name", "last_name", "first_name", "last_name"):
        expression = (
            f"CASE WHEN {name} IS NULL THEN {expression} ELSE regexp_replace({expression}, "
            f"'(^|[^\\p{{L}}])' || regexp_escape({name}) || '([^\\p{{L}}]|$)',"
            " '\\1[NOM]\\2', 'gi') END"
        )
    return expression


def build_support_month(
    settings: Settings, year: int, month: int, orchestrator_id: str | None = None
) -> dict[str, int]:
    """Associe e-mails et tickets du mois, masque les données personnelles, publie."""
    registry = Registry(settings)
    con = connect(settings)
    emails = month_glob(settings, "bronze", "support_emails", year, month)
    tickets = month_glob(settings, "bronze", "support_tickets", year, month)
    try:
        with registry.run("silver", f"support_messages_{year}-{month:02d}", orchestrator_id) as run:
            require_input(con, emails)
            require_input(con, tickets)
            con.execute(f"""
                CREATE TEMP TABLE joined AS
                SELECT e.message_id, t.ticket_id, t.client_id, e.received_at, t.category,
                       t.priority, e.subject, e.body,
                       split_part(e.sender_name, ' ', 1) AS first_name,
                       nullif(trim(substr(e.sender_name, strpos(e.sender_name, ' ') + 1)), '')
                           AS last_name
                FROM read_parquet('{emails}') e
                LEFT JOIN read_parquet('{tickets}') t USING (message_id)
            """)
            run.rows_in = count(con, "joined")
            counts = ", ".join(
                f"len(regexp_extract_all(body, '{pattern}')) AS pii_{name}"
                for name, (pattern, _) in PII_PATTERNS.items()
            )
            con.execute(f"""
                CREATE TEMP TABLE silver_rows AS
                SELECT message_id, ticket_id, client_id, received_at, category, priority,
                       {_mask_expression("subject")} AS subject_masked,
                       {_mask_expression("body")} AS body_masked,
                       {counts}
                FROM joined
            """)
            residual = " + ".join(
                f"len(regexp_extract_all(body_masked, '{pattern}'))"
                for name, (pattern, _) in PII_PATTERNS.items()
                if name in ("iban", "email", "phone")
            )
            checks = [
                quality.Check(
                    "no_residual_pii",
                    f"SELECT coalesce(sum({residual}), 0) FROM silver_rows",
                    "==",
                    0,
                    description="Aucun IBAN, e-mail ou téléphone après masquage",
                ),
                quality.Check(
                    "no_residual_sender_name",
                    "SELECT count(*) FROM silver_rows s JOIN joined j USING (message_id)"
                    " WHERE j.last_name IS NOT NULL AND"
                    " contains(lower(s.body_masked), lower(j.last_name))",
                    "==",
                    0,
                    description="Nom de l'expéditeur absent du texte publié",
                ),
                quality.Check(
                    "ticket_matched",
                    "SELECT avg(CAST(ticket_id IS NOT NULL AS INTEGER)) FROM silver_rows",
                    ">=",
                    0.99,
                    severity="warning",
                    description="Chaque e-mail doit avoir été classé par un conseiller",
                ),
            ]
            validate_and_publish(
                con,
                settings,
                registry,
                run,
                "support_messages",
                "silver_rows",
                settings.uri(
                    "silver",
                    "support_messages",
                    f"year={year}",
                    f"month={month:02d}",
                    "support_messages.parquet",
                ),
                checks,
                order_by="received_at",
            )
            registry.audit(
                "svc-pipeline-silver", "mask_pii", emails, "PRC-002", {"messages": run.rows_in}
            )
    finally:
        con.close()
    return {"rows": run.rows_out}
