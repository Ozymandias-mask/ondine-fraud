"""Messages du support silver : texte libre dont les données personnelles sont masquées.

Le texte des e-mails sert à classer automatiquement les demandes (et à repérer les
signalements de fraude). Il contient des données personnelles que le modèle n'a pas à
voir : elles sont remplacées par des marqueurs ([IBAN], [NOM]...) avant publication, et un
contrôle bloquant vérifie qu'il n'en reste aucune détectable.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache

import pyarrow as pa

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
    """Expression SQL masquant les motifs de données personnelles (IBAN, e-mail...)."""
    expression = column
    for pattern, replacement in PII_PATTERNS.values():
        expression = f"regexp_replace({expression}, '{pattern}', '{replacement}', 'g')"
    return expression


def mask_names(text: str | None, first_name: str | None, last_name: str | None) -> tuple[str, bool]:
    """Remplace prénom et nom de l'expéditeur par [NOM]. Retourne (texte, nom résiduel ?).

    Les limites de mot sont vérifiées avec des assertions Unicode : « Hervé » est reconnu,
    mais « Adam » n'est pas remplacé à l'intérieur de « Madame » (faux positif évité).
    Le contrôle de nom résiduel utilise exactement la même définition du mot.
    """
    if not text:
        return text or "", False
    names = sorted({n for n in (first_name, last_name) if n and n.strip()}, key=len, reverse=True)
    if not names:
        return text, False
    pattern = _name_pattern(tuple(names))
    masked = pattern.sub("[NOM]", text)
    return masked, bool(pattern.search(masked))


@lru_cache(maxsize=50_000)
def _name_pattern(names: tuple[str, ...]) -> re.Pattern[str]:
    alternatives = "|".join(re.escape(n) for n in names)
    return re.compile(rf"(?<!\w)(?:{alternatives})(?!\w)", re.IGNORECASE)


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
            # 1. Motifs structurés masqués en SQL (vectorisé).
            staged = con.execute(f"""
                SELECT message_id, ticket_id, client_id, received_at, category, priority,
                       {_mask_expression("subject")} AS subject_masked,
                       {_mask_expression("body")} AS body_masked,
                       first_name, last_name, {counts}
                FROM joined
            """).to_arrow_table()
            # 2. Noms de l'expéditeur masqués en Python (motif propre à chaque message).
            first = staged.column("first_name").to_pylist()
            last = staged.column("last_name").to_pylist()
            subjects, bodies, residual_names = [], [], []
            for subject, body, f, name in zip(
                staged.column("subject_masked").to_pylist(),
                staged.column("body_masked").to_pylist(),
                first,
                last,
                strict=True,
            ):
                masked_subject, residual_subject = mask_names(subject, f, name)
                masked_body, residual_body = mask_names(body, f, name)
                subjects.append(masked_subject)
                bodies.append(masked_body)
                residual_names.append(residual_subject or residual_body)
            staged = staged.drop_columns(
                ["subject_masked", "body_masked", "first_name", "last_name"]
            )
            staged = staged.append_column("subject_masked", pa.array(subjects, pa.string()))
            staged = staged.append_column("body_masked", pa.array(bodies, pa.string()))
            staged = staged.append_column("residual_sender_name", pa.array(residual_names))
            con.register("staged", staged)
            con.execute("CREATE TEMP TABLE silver_rows AS SELECT * FROM staged")
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
                    "SELECT count(*) FROM silver_rows WHERE residual_sender_name",
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
