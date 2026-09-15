"""Ingestion de la zone d'arrivée vers la couche bronze.

Pour chaque fichier livré :
1. contrôle d'intégrité : présence, taille et empreinte SHA-256 conformes au manifeste ;
2. lecture typée (CSV, JSON lines ou mbox) ; les lignes illisibles vont en quarantaine ;
3. contrôle de complétude : lignes chargées + rejetées = lignes annoncées par la source ;
4. écriture Parquet en bronze, avec les colonnes techniques de lignage ;
5. enregistrement dans le registre (le fichier ne sera plus jamais ingéré deux fois).

Traitement incrémental et idempotent : relancer l'ingestion ne traite que les fichiers
nouveaux, et retraiter un fichier réécrit exactement le même objet bronze.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import duckdb

from ..lake import connect, copy_to_parquet, sql_literal
from ..registry import Registry, Run
from ..settings import Settings
from .emails import read_mbox_gz

logger = logging.getLogger(__name__)

TX_COLUMNS_V1 = {
    "transaction_id": "VARCHAR",
    "event_time": "TIMESTAMPTZ",
    "client_id": "VARCHAR",
    "card_id": "VARCHAR",
    "merchant_id": "VARCHAR",
    "mcc": "INTEGER",
    "merchant_country": "VARCHAR",
    "channel": "VARCHAR",
    "entry_mode": "VARCHAR",
    "amount": "DECIMAL(12,2)",
    "currency": "VARCHAR",
    "three_ds_authenticated": "BOOLEAN",
    "device_id": "VARCHAR",
    "ip_address": "VARCHAR",
    "latitude": "DOUBLE",
    "longitude": "DOUBLE",
    "beneficiary_iban": "VARCHAR",
    "auth_status": "VARCHAR",
    "decline_reason": "VARCHAR",
}
TX_COLUMNS_V2 = {**TX_COLUMNS_V1, "wallet_type": "VARCHAR"}

CSV_SCHEMAS: dict[str, dict[str, str]] = {
    "fraud_reports": {
        "report_id": "VARCHAR",
        "case_id": "VARCHAR",
        "transaction_id": "VARCHAR",
        "client_id": "VARCHAR",
        "reported_at": "TIMESTAMPTZ",
        "report_channel": "VARCHAR",
        "claim_category": "VARCHAR",
        "investigation_status": "VARCHAR",
    },
    "support_tickets": {
        "ticket_id": "VARCHAR",
        "message_id": "VARCHAR",
        "client_id": "VARCHAR",
        "received_at": "TIMESTAMPTZ",
        "channel": "VARCHAR",
        "category": "VARCHAR",
        "priority": "VARCHAR",
    },
    "clients": {
        "client_id": "VARCHAR",
        "first_name": "VARCHAR",
        "last_name": "VARCHAR",
        "birth_date": "DATE",
        "email": "VARCHAR",
        "phone": "VARCHAR",
        "street_address": "VARCHAR",
        "postal_code": "VARCHAR",
        "city": "VARCHAR",
        "region": "VARCHAR",
        "iban": "VARCHAR",
        "segment": "VARCHAR",
        "income_band": "VARCHAR",
        "kyc_level": "VARCHAR",
        "is_pep": "BOOLEAN",
        "created_at": "DATE",
        "consent_marketing": "BOOLEAN",
        "consent_marketing_at": "DATE",
    },
    "cards": {
        "card_id": "VARCHAR",
        "client_id": "VARCHAR",
        "pan_token": "VARCHAR",
        "last4": "VARCHAR",
        "card_type": "VARCHAR",
        "network": "VARCHAR",
        "issued_at": "DATE",
        "expiry": "VARCHAR",
        "contactless_enabled": "BOOLEAN",
        "status": "VARCHAR",
    },
    "merchants": {
        "merchant_id": "VARCHAR",
        "merchant_name": "VARCHAR",
        "category": "VARCHAR",
        "mcc": "INTEGER",
        "is_online": "BOOLEAN",
        "city": "VARCHAR",
        "postal_code": "VARCHAR",
        "country": "VARCHAR",
        "latitude": "DOUBLE",
        "longitude": "DOUBLE",
        "created_at": "DATE",
    },
}
EVENT_COLUMNS = {
    "event_id": "VARCHAR",
    "event_type": "VARCHAR",
    "event_time": "TIMESTAMPTZ",
    "client_id": "VARCHAR",
    "session": "JSON",
    "geo": "JSON",
    "payload": "JSON",
}
# Référentiels contenant des données personnelles : leur lecture est tracée dans l'audit.
PERSONAL_DATASETS = {"clients", "support_emails"}
DATASETS = (
    "clients",
    "cards",
    "merchants",
    "transactions",
    "app_events",
    "fraud_reports",
    "support_tickets",
    "support_emails",
)


class IntegrityError(RuntimeError):
    """Le fichier livré ne correspond pas à son manifeste (corrompu ou incomplet)."""


class CompletenessError(RuntimeError):
    """Des lignes ont été perdues entre la source et la couche bronze."""


@dataclass(frozen=True)
class SourceFile:
    """Fichier livré, tel que décrit par son manifeste."""

    dataset: str
    path: str  # relatif à la zone d'arrivée, séparateurs « / »
    rows: int
    bytes: int
    sha256: str
    period: str | None
    schema_version: str | None

    @property
    def bronze_key(self) -> str:
        """Clé de l'objet bronze : même partitionnement que la source, extension .parquet."""
        parts = PurePosixPath(self.path)
        name = parts.name.split(".")[0] + ".parquet"
        if self.dataset in ("clients", "cards", "merchants"):
            snapshot = parts.stem.split("_")[-1].split(".")[0]
            return f"referentials/{self.dataset}/snapshot_date={snapshot}/{name}"
        return str(parts.parent / name)


def discover(landing_root: Path) -> list[SourceFile]:
    """Liste les fichiers livrés d'après les manifestes (seules les livraisons complètes)."""
    files: list[SourceFile] = []
    for manifest_path in sorted((landing_root / "_manifests").rglob("*.json")):
        if "runs" in manifest_path.parts:
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for info in manifest.get("files", []):
            segments = info["path"].split("/")
            dataset = segments[1] if segments[0] == "referentials" else segments[0]
            files.append(
                SourceFile(
                    dataset=dataset,
                    path=info["path"],
                    rows=int(info["rows"]),
                    bytes=int(info["bytes"]),
                    sha256=info["sha256"],
                    period=manifest.get("dataset_period"),
                    schema_version=manifest.get("transaction_schema"),
                )
            )
    return files


def pending_files(settings: Settings, registry: Registry, dataset: str) -> list[SourceFile]:
    """Fichiers d'un jeu de données pas encore ingérés (comparaison chemin + empreinte)."""
    done = registry.ingested_files()
    return [
        f
        for f in discover(settings.landing_root)
        if f.dataset == dataset and (f.path, f.sha256) not in done
    ]


def verify_integrity(settings: Settings, source: SourceFile) -> Path:
    """Vérifie présence, taille et empreinte du fichier. Retourne son chemin local."""
    path = settings.landing_root / source.path
    if not path.exists():
        raise IntegrityError(f"Fichier annoncé mais absent : {source.path}")
    if path.stat().st_size != source.bytes:
        raise IntegrityError(
            f"Taille inattendue pour {source.path} : "
            f"{path.stat().st_size} au lieu de {source.bytes} octets"
        )
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 << 20), b""):
            digest.update(block)
    if digest.hexdigest() != source.sha256:
        raise IntegrityError(f"Empreinte SHA-256 invalide pour {source.path} (fichier altéré)")
    return path


def _columns_literal(columns: dict[str, str]) -> str:
    return "{" + ", ".join(f"{sql_literal(k)}: {sql_literal(v)}" for k, v in columns.items()) + "}"


def _lineage_columns(source: SourceFile, run: Run) -> str:
    return (
        f"{sql_literal(source.path)} AS _source_file, "
        f"{sql_literal(source.sha256)} AS _source_sha256, "
        f"{sql_literal(run.run_id)} AS _run_id, "
        "current_timestamp AS _ingested_at"
    )


def _load_csv(
    con: duckdb.DuckDBPyConnection,
    settings: Settings,
    source: SourceFile,
    path: Path,
    run: Run,
    target: str,
) -> tuple[int, int]:
    if source.dataset == "transactions":
        columns = TX_COLUMNS_V2 if source.schema_version == "v2" else TX_COLUMNS_V1
    else:
        columns = CSV_SCHEMAS[source.dataset]
    select_columns = ", ".join(f'"{c}"' for c in columns)
    if source.dataset == "transactions" and source.schema_version != "v2":
        # Schéma v1 (avant juillet 2025) : la colonne ajoutée ensuite vaut NULL.
        select_columns += ", NULL::VARCHAR AS wallet_type"
    con.execute("DROP TABLE IF EXISTS rejects; DROP TABLE IF EXISTS rejects_scan")
    reader = (
        f"read_csv({sql_literal(str(path))}, header = true, "
        f"columns = {_columns_literal(columns)}, "
        "store_rejects = true, rejects_table = 'rejects', rejects_scan = 'rejects_scan')"
    )
    query = f"SELECT {select_columns}, {_lineage_columns(source, run)} FROM {reader}"
    loaded = copy_to_parquet(con, settings, query, target)
    rejected = con.execute("SELECT count(DISTINCT line) FROM rejects").fetchone()[0]
    if rejected:
        quarantine = settings.uri(
            "quarantine", source.dataset, PurePosixPath(source.bronze_key).stem + "_rejects.parquet"
        )
        copy_to_parquet(
            con,
            settings,
            f"SELECT *, {sql_literal(source.path)} AS _source_file FROM rejects",
            quarantine,
        )
        logger.warning("%d ligne(s) rejetée(s) dans %s -> %s", rejected, source.path, quarantine)
    return loaded, rejected


def _load_json(
    con: duckdb.DuckDBPyConnection,
    settings: Settings,
    source: SourceFile,
    path: Path,
    run: Run,
    target: str,
) -> tuple[int, int]:
    reader = (
        f"read_json({sql_literal(str(path))}, format = 'newline_delimited', "
        f"columns = {_columns_literal(EVENT_COLUMNS)}, ignore_errors = true)"
    )
    query = f"SELECT *, {_lineage_columns(source, run)} FROM {reader}"
    loaded = copy_to_parquet(con, settings, query, target)
    return loaded, source.rows - loaded


def _load_mbox(
    con: duckdb.DuckDBPyConnection,
    settings: Settings,
    source: SourceFile,
    path: Path,
    run: Run,
    target: str,
) -> tuple[int, int]:
    emails = read_mbox_gz(path)
    con.register("emails", emails)
    lineage = _lineage_columns(source, run)
    loaded = copy_to_parquet(
        con,
        settings,
        f"SELECT * EXCLUDE (parse_error), {lineage} FROM emails WHERE parse_error IS NULL",
        target,
    )
    rejected = emails.num_rows - loaded
    if rejected:
        copy_to_parquet(
            con,
            settings,
            f"SELECT *, {lineage} FROM emails WHERE parse_error IS NOT NULL",
            settings.uri(
                "quarantine",
                "support_emails",
                PurePosixPath(source.bronze_key).stem + "_rejects.parquet",
            ),
        )
    con.unregister("emails")
    return loaded, rejected


LOADERS = {"app_events": _load_json, "support_emails": _load_mbox}


def ingest_file(
    con: duckdb.DuckDBPyConnection,
    settings: Settings,
    registry: Registry,
    run: Run,
    source: SourceFile,
) -> tuple[int, int]:
    """Ingère un fichier en bronze avec tous les contrôles. Retourne (chargées, rejetées)."""
    path = verify_integrity(settings, source)
    target = settings.uri("bronze", source.bronze_key)
    loader = LOADERS.get(source.dataset, _load_csv)
    loaded, rejected = loader(con, settings, source, path, run, target)

    if loaded + rejected != source.rows:
        raise CompletenessError(
            f"{source.path} : {loaded} chargées + {rejected} rejetées != {source.rows} annoncées"
        )
    # Au-delà de 1 % de rejets, la livraison est jugée anormale : on arrête et on alerte.
    if source.rows and rejected / source.rows > 0.01:
        raise CompletenessError(f"{source.path} : taux de rejet {rejected / source.rows:.2%} > 1 %")

    registry.record_file(
        run,
        source.path,
        source.sha256,
        source.dataset,
        source.period,
        source.rows,
        loaded,
        rejected,
        target,
    )
    if source.dataset in PERSONAL_DATASETS:
        registry.audit(
            "svc-ingestion", "ingest", target, "PRC-001", {"source": source.path, "rows": loaded}
        )
    run.inputs.append({"path": source.path, "sha256": source.sha256, "rows": source.rows})
    run.outputs.append({"uri": target, "rows": loaded})
    run.rows_in += source.rows
    run.rows_out += loaded
    run.rows_rejected += rejected
    if settings.purge_landing:
        path.unlink()
    return loaded, rejected


def ingest_dataset(
    settings: Settings, dataset: str, limit: int | None = None, orchestrator_id: str | None = None
) -> dict[str, int]:
    """Ingère les fichiers en attente d'un jeu de données (au plus `limit`)."""
    if dataset not in DATASETS:
        raise ValueError(f"Jeu de données inconnu : {dataset}")
    registry = Registry(settings)
    todo = pending_files(settings, registry, dataset)[:limit]
    if not todo:
        logger.info("%s : aucun fichier en attente", dataset)
        return {"files": 0, "rows": 0, "rejected": 0}
    con = connect(settings)
    try:
        with registry.run("ingestion", dataset, orchestrator_id) as run:
            for source in todo:
                loaded, rejected = ingest_file(con, settings, registry, run, source)
                logger.info(
                    "%s -> bronze : %s lignes (%d rejetées)", source.path, f"{loaded:,}", rejected
                )
    finally:
        con.close()
    return {"files": len(todo), "rows": run.rows_out, "rejected": run.rows_rejected}
