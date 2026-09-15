"""Exercice des droits RGPD : accès, portabilité, opposition et effacement.

Les demandes sont saisies dans `governance.data_subject_requests` (par l'API ou le DPO)
et traitées automatiquement par l'orchestrateur :

- accès (art. 15) et portabilité (art. 20) : export JSON des données brutes du client ;
- opposition (art. 21) : retrait du consentement à la prospection (PRC-003) ;
- effacement (art. 17) : suppression du client dans toutes les couches du data lake,
  **après validation du DPO** (absence d'obligation de conservation, LCB-FT : 5 ans) ;
- rectification et limitation : laissées au DPO (valeurs à saisir, décision au cas par cas).

Chaque action est inscrite dans `governance.data_subject_actions` et dans le journal d'audit.
Délai légal : un mois (vue `governance.data_subject_requests_due`).
"""

from __future__ import annotations

import json
import logging
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import psycopg

from ..lake import connect, copy_to_parquet, exists, sql_literal
from ..registry import Registry, Run
from ..settings import Settings

logger = logging.getLogger(__name__)

ACTOR = "svc-rgpd"
PURPOSE = "RGPD-DROITS"


@dataclass(frozen=True)
class SubjectSource:
    """Jeu de données du data lake contenant des données d'un client."""

    layer: str
    path: str
    exported: bool  # inclus dans l'export d'accès (données brutes détenues)
    predicate: str = "client_id = getvariable('subject_id')"

    @property
    def label(self) -> str:
        """Nom lisible, utilisé pour les fichiers d'export."""
        return f"{self.layer}_{self.path.replace('/', '_')}"


# Messages du client : les e-mails bruts ne portent pas l'identifiant, on passe par les tickets.
EMAIL_PREDICATE = "message_id IN (SELECT message_id FROM subject_messages)"

SOURCES: tuple[SubjectSource, ...] = (
    SubjectSource("bronze", "referentials/clients", exported=True),
    SubjectSource("bronze", "referentials/cards", exported=True),
    SubjectSource("bronze", "transactions", exported=True),
    SubjectSource("bronze", "app_events", exported=True),
    SubjectSource("bronze", "fraud_reports", exported=True),
    SubjectSource("bronze", "support_tickets", exported=True),
    SubjectSource("bronze", "support_emails", exported=True, predicate=EMAIL_PREDICATE),
    # Copies dérivées (nettoyées, pseudonymisées) : effacées, pas exportées en double.
    SubjectSource("silver", "clients", exported=False),
    SubjectSource("silver", "cards", exported=False),
    SubjectSource("silver", "transactions", exported=False),
    SubjectSource("silver", "app_events", exported=False),
    SubjectSource("silver", "fraud_labels", exported=False),
    SubjectSource("silver", "support_messages", exported=False),
    SubjectSource("gold", "transaction_features", exported=False),
    SubjectSource("gold", "first_seen", exported=False),
)

EXPORT_NOTICE = (
    "Export de vos données détenues par Ondine Bank (RGPD, articles 15 et 20). Les copies "
    "techniques pseudonymisées contiennent les mêmes informations et ne sont pas répétées. "
    "Les indicateurs internes de détection de la fraude ne sont pas communiqués : leur "
    "divulgation compromettrait la prévention de la fraude (limitation à confirmer par le DPO)."
)


class ErasureVerificationError(RuntimeError):
    """Des données du client subsistent après l'effacement."""


def _glob(settings: Settings, source: SubjectSource) -> str:
    return settings.uri(source.layer, source.path, "**", "*.parquet")


def _reader(pattern: str) -> str:
    return (
        f"read_parquet({sql_literal(pattern)}, filename = true, union_by_name = true, "
        "hive_partitioning = false)"
    )


def _prepare(con: duckdb.DuckDBPyConnection, settings: Settings, client_id: str) -> None:
    """Fixe le client concerné et retrouve ses messages avant toute modification."""
    con.execute("SET VARIABLE subject_id = ?", [client_id])
    tickets = settings.uri("bronze", "support_tickets", "**", "*.parquet")
    if exists(con, tickets):
        con.execute(
            "CREATE OR REPLACE TEMP TABLE subject_messages AS SELECT DISTINCT message_id "
            f"FROM {_reader(tickets)} WHERE client_id = getvariable('subject_id')"
        )
    else:
        con.execute("CREATE OR REPLACE TEMP TABLE subject_messages (message_id VARCHAR)")


def _files_with_subject(
    con: duckdb.DuckDBPyConnection, settings: Settings, source: SubjectSource
) -> dict[str, int]:
    pattern = _glob(settings, source)
    if not exists(con, pattern):
        return {}
    rows = con.execute(
        f"SELECT filename, count(*) FROM {_reader(pattern)} WHERE {source.predicate} GROUP BY 1"
    ).fetchall()
    return dict(rows)


def _quarantine_files_with_subject(
    con: duckdb.DuckDBPyConnection, settings: Settings
) -> dict[str, tuple[str, int]]:
    """Fichiers de quarantaine contenant le client, avec le filtre adapté à leur schéma.

    Les rejets d'ingestion n'ont pas toujours de colonne client_id (ligne brute illisible) :
    on cherche alors l'identifiant dans les colonnes texte.
    """
    found: dict[str, tuple[str, int]] = {}
    for (path,) in con.execute(
        "SELECT file FROM glob(?)", [settings.uri("quarantine", "**", "*.parquet")]
    ).fetchall():
        columns = con.execute(
            f"DESCRIBE SELECT * FROM read_parquet({sql_literal(path)}, hive_partitioning = false)"
        ).fetchall()
        names = {c[0]: c[1] for c in columns}
        if "client_id" in names:
            predicate = "client_id = getvariable('subject_id')"
        else:
            text = [f'"{n}"' for n, t in names.items() if t == "VARCHAR"]
            if not text:
                continue
            predicate = (
                f"concat_ws('|', {', '.join(text)}) LIKE '%' || getvariable('subject_id') || '%'"
            )
        n = con.execute(
            f"SELECT count(*) FROM read_parquet({sql_literal(path)}, hive_partitioning = false)"
            f" WHERE {predicate}"
        ).fetchone()[0]
        if n:
            found[path] = (predicate, int(n))
    return found


def _remaining(con: duckdb.DuckDBPyConnection, settings: Settings) -> int:
    total = sum(sum(_files_with_subject(con, settings, s).values()) for s in SOURCES)
    return total + sum(n for _, n in _quarantine_files_with_subject(con, settings).values())


def export_subject(settings: Settings, request_id: str, client_id: str) -> dict[str, object]:
    """Exporte en JSON les données brutes du client. Retourne le détail par jeu de données."""
    con = connect(settings)
    try:
        _prepare(con, settings, client_id)
        counts: dict[str, int] = {}
        base = settings.uri("exports", "dsr", request_id)
        for source in SOURCES:
            if not source.exported:
                continue
            pattern = _glob(settings, source)
            if not exists(con, pattern):
                continue
            target = f"{base}/{source.label}.json"
            n = con.execute(
                f"SELECT count(*) FROM {_reader(pattern)} WHERE {source.predicate}"
            ).fetchone()[0]
            if n:
                _copy_json(
                    con,
                    settings,
                    f"SELECT * EXCLUDE (filename) FROM {_reader(pattern)} WHERE {source.predicate}",
                    target,
                )
            counts[source.label] = int(n)
        with psycopg.connect(settings.pg_dsn, connect_timeout=10) as pg:
            consents = pg.execute(
                "SELECT purpose, granted, source, updated_at::text FROM governance.consents"
                " WHERE client_id = %s ORDER BY purpose",
                (client_id,),
            ).fetchall()
        manifest = {
            "request_id": request_id,
            "client_id": client_id,
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "notice": EXPORT_NOTICE,
            "rows_by_dataset": counts,
            "consents": [
                {"purpose": p, "granted": g, "source": s, "updated_at": u}
                for p, g, s, u in consents
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            local = Path(tmp) / "manifest.json"
            local.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
            _copy_json(
                con,
                settings,
                f"SELECT * FROM read_json({sql_literal(str(local))})",
                f"{base}/manifest.json",
                array=False,
            )
        return {"uri": base, "rows": sum(counts.values()), "datasets": counts}
    finally:
        con.close()


def _copy_json(
    con: duckdb.DuckDBPyConnection, settings: Settings, query: str, uri: str, array: bool = True
) -> None:
    if not settings.uses_s3:
        Path(uri).parent.mkdir(parents=True, exist_ok=True)
    con.execute(f"COPY ({query}) TO {sql_literal(uri)} (FORMAT json, ARRAY {str(array).lower()})")


def erase_subject(settings: Settings, client_id: str) -> dict[str, int]:
    """Supprime le client de toutes les couches, puis vérifie qu'il n'en reste rien.

    Chaque fichier concerné est réécrit sans les lignes du client (écriture atomique d'un
    objet S3 : un lecteur voit l'ancienne ou la nouvelle version, jamais un fichier partiel).
    Retourne le nombre de lignes supprimées par fichier.
    """
    con = connect(settings)
    try:
        _prepare(con, settings, client_id)
        plan: dict[str, tuple[str, int]] = {}
        for source in SOURCES:
            for path, n in _files_with_subject(con, settings, source).items():
                plan[path] = (source.predicate, n)
        plan.update(_quarantine_files_with_subject(con, settings))
        for path, (predicate, _) in plan.items():
            con.execute(
                "CREATE OR REPLACE TEMP TABLE kept AS SELECT * FROM read_parquet("
                f"{sql_literal(path)}, hive_partitioning = false) WHERE ({predicate}) IS NOT TRUE"
            )
            copy_to_parquet(con, settings, "SELECT * FROM kept", path)
        con.execute("DROP TABLE IF EXISTS kept")
        remaining = _remaining(con, settings)
        if remaining:
            raise ErasureVerificationError(f"{remaining} ligne(s) du client subsistent")
        return {path: n for path, (_, n) in plan.items()}
    finally:
        con.close()


def _complete(
    pg: psycopg.Connection, request_id: str, notes: str, actions: list[tuple[str, str, int]]
) -> None:
    with pg.transaction():
        pg.execute(
            "UPDATE governance.data_subject_requests SET status = 'completed',"
            " completed_at = now(), notes = %s WHERE request_id = %s",
            (notes, request_id),
        )
        with pg.cursor() as cur:
            cur.executemany(
                "INSERT INTO governance.data_subject_actions"
                " (request_id, action, target, rows_affected) VALUES (%s, %s, %s, %s)",
                [(request_id, a, t, n) for a, t, n in actions],
            )


def process_requests(settings: Settings, orchestrator_id: str | None = None) -> dict[str, int]:
    """Traite les demandes en attente. Une demande en échec n'empêche pas les autres."""
    registry = Registry(settings)
    summary = {"completed": 0, "awaiting_approval": 0, "manual": 0, "failed": 0}
    with psycopg.connect(settings.pg_dsn, autocommit=True, connect_timeout=10) as pg:
        requests = pg.execute(
            "SELECT request_id, client_id, request_type, approved_by"
            " FROM governance.data_subject_requests"
            " WHERE status IN ('received', 'in_progress') ORDER BY received_at"
        ).fetchall()
        for request_id, client_id, kind, approved_by in requests:
            if kind in ("rectification", "restriction"):
                summary["manual"] += 1
                continue
            if kind == "erasure" and not approved_by:
                summary["awaiting_approval"] += 1
                continue
            try:
                with registry.run("rgpd", f"{kind}_{request_id}", orchestrator_id) as run:
                    pg.execute(
                        "UPDATE governance.data_subject_requests SET status = 'in_progress'"
                        " WHERE request_id = %s",
                        (request_id,),
                    )
                    _execute(settings, registry, pg, run, request_id, client_id, kind, approved_by)
                summary["completed"] += 1
            except Exception:
                logger.exception("Demande %s (%s) en échec", request_id, kind)
                summary["failed"] += 1
    if summary["failed"]:
        raise RuntimeError(f"{summary['failed']} demande(s) RGPD en échec : {summary}")
    return summary


def _execute(
    settings: Settings,
    registry: Registry,
    pg: psycopg.Connection,
    run: Run,
    request_id: str,
    client_id: str,
    kind: str,
    approved_by: str | None,
) -> None:
    details = {"request_id": request_id, "client_id": client_id}
    if kind in ("access", "portability"):
        result = export_subject(settings, request_id, client_id)
        registry.audit(ACTOR, "export", result["uri"], PURPOSE, {**details, "rows": result["rows"]})
        run.rows_out = result["rows"]
        run.outputs.append({"uri": result["uri"], "rows": result["rows"]})
        _complete(
            pg,
            request_id,
            f"Export disponible : {result['uri']} ({result['rows']} lignes)",
            [("export", result["uri"], result["rows"])],
        )
    elif kind == "objection":
        pg.execute(
            "INSERT INTO governance.consents (client_id, purpose, granted, source)"
            " VALUES (%s, 'PRC-003', false, 'demande RGPD')"
            " ON CONFLICT (client_id, purpose) DO UPDATE SET granted = false,"
            " source = 'demande RGPD', updated_at = now()",
            (client_id,),
        )
        registry.audit(ACTOR, "consent_withdrawn", "governance.consents", PURPOSE, details)
        _complete(
            pg,
            request_id,
            "Opposition enregistrée : prospection désactivée",
            [("consent_withdrawn", "governance.consents", 1)],
        )
    elif kind == "erasure":
        erased = erase_subject(settings, client_id)
        consents = pg.execute(
            "DELETE FROM governance.consents WHERE client_id = %s", (client_id,)
        ).rowcount
        total = sum(erased.values())
        registry.audit(
            ACTOR,
            "erase",
            "data lake (bronze, silver, gold, quarantine)",
            PURPOSE,
            {**details, "rows": total, "files": len(erased), "approved_by": approved_by},
        )
        run.rows_out = total
        run.outputs.extend({"uri": p, "rows_removed": n} for p, n in erased.items())
        _complete(
            pg,
            request_id,
            f"Effacement validé par {approved_by} : {total} lignes dans {len(erased)} fichiers, "
            f"{consents} consentement(s). Copies résiduelles : flux Kafka (expiration sous 7 j), "
            "sauvegardes PostgreSQL (rotation 7 j).",
            [("erase", p, n) for p, n in erased.items()]
            + [("erase", "governance.consents", consents)],
        )
    else:  # pragma: no cover - filtré en amont
        raise ValueError(f"Type de demande non automatisé : {kind}")
