-- Lignage de l'ingestion en flux : chaque lot écrit en bronze est rattaché à sa plage
-- d'offsets Kafka. Permet de prouver qu'aucun message n'a été perdu (offsets contigus)
-- et de retrouver le fichier bronze d'un message donné.
CREATE TABLE ops.stream_batches (
    id              BIGSERIAL PRIMARY KEY,
    consumer_id     TEXT        NOT NULL,
    topic           TEXT        NOT NULL,
    kafka_partition INTEGER     NOT NULL,
    first_offset    BIGINT      NOT NULL,
    last_offset     BIGINT      NOT NULL,
    rows_valid      INTEGER     NOT NULL,
    rows_invalid    INTEGER     NOT NULL,
    target_uris     JSONB       NOT NULL DEFAULT '[]'::jsonb,
    min_event_time  TIMESTAMPTZ,
    max_event_time  TIMESTAMPTZ,
    duration_ms     INTEGER     NOT NULL,
    written_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (topic, kafka_partition, first_offset, last_offset)
);
CREATE INDEX stream_batches_written_idx ON ops.stream_batches (topic, written_at DESC);

GRANT SELECT, INSERT ON ops.stream_batches TO ingestion_app;
GRANT USAGE ON SEQUENCE ops.stream_batches_id_seq TO ingestion_app;
GRANT SELECT ON ops.stream_batches TO analyst_ro;
