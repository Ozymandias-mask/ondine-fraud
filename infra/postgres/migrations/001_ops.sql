-- Schéma « ops » : exécution des pipelines, fichiers ingérés, contrôles qualité.
-- C'est le registre de traçabilité (lignage) : pour chaque donnée du data lake, on sait
-- de quel fichier source elle vient, quel traitement l'a produite, quand et avec quel résultat.

CREATE SCHEMA IF NOT EXISTS ops;

CREATE TABLE ops.pipeline_runs (
    run_id          TEXT PRIMARY KEY,
    pipeline        TEXT        NOT NULL,           -- ingestion, silver, gold, rgpd...
    task            TEXT        NOT NULL,           -- ex. transactions
    orchestrator_id TEXT,                           -- identifiant du run Airflow
    started_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at     TIMESTAMPTZ,
    status          TEXT        NOT NULL DEFAULT 'running'
                    CHECK (status IN ('running', 'success', 'failed', 'skipped')),
    rows_in         BIGINT,
    rows_out        BIGINT,
    rows_rejected   BIGINT,
    duration_s      NUMERIC(12, 3),
    rows_per_s      NUMERIC(14, 1),
    inputs          JSONB NOT NULL DEFAULT '[]'::jsonb,   -- fichiers ou tables lus
    outputs         JSONB NOT NULL DEFAULT '[]'::jsonb,   -- fichiers ou tables écrits
    error           TEXT
);
CREATE INDEX pipeline_runs_started_idx ON ops.pipeline_runs (pipeline, task, started_at DESC);

CREATE TABLE ops.file_ingestions (
    source_path    TEXT        NOT NULL,
    sha256         TEXT        NOT NULL,
    dataset        TEXT        NOT NULL,
    period         TEXT,
    rows_expected  BIGINT      NOT NULL,
    rows_loaded    BIGINT      NOT NULL,
    rows_rejected  BIGINT      NOT NULL,
    target_uri     TEXT        NOT NULL,
    run_id         TEXT        NOT NULL REFERENCES ops.pipeline_runs (run_id),
    ingested_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (source_path, sha256)                -- un même fichier n'est jamais ingéré deux fois
);

CREATE TABLE ops.quality_results (
    id          BIGSERIAL PRIMARY KEY,
    run_id      TEXT        NOT NULL REFERENCES ops.pipeline_runs (run_id),
    layer       TEXT        NOT NULL,
    dataset     TEXT        NOT NULL,
    check_name  TEXT        NOT NULL,
    severity    TEXT        NOT NULL CHECK (severity IN ('warning', 'error')),
    passed      BOOLEAN     NOT NULL,
    observed    NUMERIC,
    threshold   NUMERIC,
    details     JSONB NOT NULL DEFAULT '{}'::jsonb,
    checked_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX quality_results_dataset_idx ON ops.quality_results (dataset, checked_at DESC);

GRANT USAGE ON SCHEMA ops TO ingestion_app, analyst_ro;
GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA ops TO ingestion_app;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA ops TO ingestion_app;
GRANT SELECT ON ALL TABLES IN SCHEMA ops TO analyst_ro;
-- Le pipeline trace ses accès aux données personnelles dans le journal d'audit.
GRANT INSERT ON audit.access_log TO ingestion_app;
