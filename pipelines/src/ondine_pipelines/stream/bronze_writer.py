"""Conversion d'un lot de messages Kafka en fichiers bronze.

Les fichiers produits ont **exactement le même schéma** que ceux de l'ingestion batch et
sont rangés dans les mêmes partitions mensuelles : les couches silver et gold traitent
indifféremment les données arrivées par fichier ou par flux.

Chaque message est validé : JSON lisible, identifiant présent, horodatage et types
conformes. Les messages invalides ne sont pas écrits en bronze ; ils sont retournés avec
leur motif pour être envoyés dans le topic d'erreurs (dead letter queue).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import duckdb
import pyarrow as pa

from ..ingest.landing import CSV_SCHEMAS, EVENT_COLUMNS, TX_COLUMNS_V2
from ..lake import copy_to_parquet, sql_literal
from ..settings import Settings


@dataclass(frozen=True)
class StreamDataset:
    """Correspondance entre un topic Kafka et un jeu de données bronze."""

    name: str
    columns: dict[str, str]
    key: str
    time_column: str


DATASETS_BY_TOPIC: dict[str, StreamDataset] = {
    "transactions.raw": StreamDataset(
        "transactions", TX_COLUMNS_V2, "transaction_id", "event_time"
    ),
    "app-events.raw": StreamDataset("app_events", EVENT_COLUMNS, "event_id", "event_time"),
    "fraud-reports.raw": StreamDataset(
        "fraud_reports", CSV_SCHEMAS["fraud_reports"], "report_id", "reported_at"
    ),
}


@dataclass
class Message:
    """Message Kafka réduit à ce dont l'ingestion a besoin."""

    topic: str
    partition: int
    offset: int
    value: bytes | None


@dataclass
class BatchResult:
    """Résultat de l'écriture d'un lot."""

    rows_valid: int = 0
    invalid: list[tuple[Message, str]] = field(default_factory=list)
    uris: list[str] = field(default_factory=list)
    min_event_time: object = None
    max_event_time: object = None
    duration_ms: int = 0


def _column_expression(name: str, sql_type: str) -> str:
    if sql_type == "JSON":
        return f"raw::JSON->'{name}' AS \"{name}\""
    if sql_type == "VARCHAR":
        return f"raw::JSON->>'{name}' AS \"{name}\""
    return f"TRY_CAST(raw::JSON->>'{name}' AS {sql_type}) AS \"{name}\""


def _type_errors(dataset: StreamDataset) -> str:
    """Branches CASE : valeur présente dans le message mais non convertible dans son type."""
    branches = []
    for name, sql_type in dataset.columns.items():
        if sql_type in ("JSON", "VARCHAR"):
            continue
        branches.append(
            f"WHEN raw::JSON->>'{name}' IS NOT NULL"
            f" AND TRY_CAST(raw::JSON->>'{name}' AS {sql_type}) IS NULL THEN 'invalid_type:{name}'"
        )
    return "\n".join(branches)


def batch_file_name(partition: int, first_offset: int, last_offset: int) -> str:
    """Nom déterministe : un même lot rejoué réécrit le même fichier (idempotence)."""
    return f"stream_p{partition:02d}_{first_offset:012d}-{last_offset:012d}.parquet"


def write_batch(
    con: duckdb.DuckDBPyConnection,
    settings: Settings,
    consumer_id: str,
    messages: list[Message],
) -> BatchResult:
    """Valide et écrit en bronze un lot de messages d'une même partition Kafka."""
    started = time.perf_counter()
    topic, partition = messages[0].topic, messages[0].partition
    dataset = DATASETS_BY_TOPIC[topic]
    first, last = messages[0].offset, messages[-1].offset
    raw = [(m.value or b"").decode("utf-8", errors="replace") for m in messages]
    con.register(
        "kafka_batch",
        pa.table({"raw": pa.array(raw, pa.string()), "kafka_offset": [m.offset for m in messages]}),
    )
    try:
        columns = ", ".join(_column_expression(n, t) for n, t in dataset.columns.items())
        type_errors = _type_errors(dataset)
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE parsed AS
            WITH checked AS (
                SELECT kafka_offset, raw, json_valid(raw) AS is_json FROM kafka_batch
            )
            SELECT kafka_offset, raw,
                CASE
                    WHEN NOT is_json THEN 'invalid_json'
                    WHEN raw::JSON->>'{dataset.key}' IS NULL THEN 'missing_{dataset.key}'
                    WHEN TRY_CAST(raw::JSON->>'{dataset.time_column}' AS TIMESTAMPTZ) IS NULL
                        THEN 'invalid_{dataset.time_column}'
                    {type_errors}
                END AS dq_reason
            FROM checked
        """)
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE valid_rows AS
            SELECT {columns} FROM parsed WHERE dq_reason IS NULL ORDER BY kafka_offset
        """)
        result = BatchResult()
        by_offset = {m.offset: m for m in messages}
        for offset, reason in con.execute(
            "SELECT kafka_offset, dq_reason FROM parsed WHERE dq_reason IS NOT NULL"
        ).fetchall():
            result.invalid.append((by_offset[offset], reason))

        stats = con.execute(f"""
            SELECT count(*), min("{dataset.time_column}"), max("{dataset.time_column}")
            FROM valid_rows
        """).fetchone()
        result.rows_valid, result.min_event_time, result.max_event_time = stats
        lineage = (
            f"{sql_literal(f'kafka://{topic}/{partition}/{first}-{last}')} AS _source_file, "
            f"NULL::VARCHAR AS _source_sha256, {sql_literal(consumer_id)} AS _run_id, "
            "current_timestamp AS _ingested_at"
        )
        months = con.execute(f"""
            SELECT DISTINCT year("{dataset.time_column}"), month("{dataset.time_column}")
            FROM valid_rows ORDER BY 1, 2
        """).fetchall()
        for year, month in months:
            uri = settings.uri(
                "bronze",
                dataset.name,
                f"year={year}",
                f"month={month:02d}",
                batch_file_name(partition, first, last),
            )
            copy_to_parquet(
                con,
                settings,
                f'SELECT *, {lineage} FROM valid_rows WHERE year("{dataset.time_column}") = {year}'
                f' AND month("{dataset.time_column}") = {month}',
                uri,
            )
            result.uris.append(uri)
        result.duration_ms = int((time.perf_counter() - started) * 1000)
        return result
    finally:
        con.unregister("kafka_batch")
