-- Alertes de fraude soumises à l'équipe d'analystes.
--
-- L'API décide en moins de 100 ms ; c'est ici que la décision devient un travail humain.
-- Chaque alerte conserve ce qui a conduit à la lever — probabilité, seuil appliqué, version
-- du modèle et facteurs explicatifs — puis la suite qui lui a été donnée. Deux raisons :
-- l'analyste doit pouvoir justifier sa décision au client, et les confirmations comme les
-- rejets alimentent la mesure de la qualité du modèle en production.
--
-- Aucune donnée directement identifiante : l'interface travaille sur des identifiants
-- techniques, comme le reste de la plateforme (minimisation).

CREATE TABLE IF NOT EXISTS ops.alerts (
    transaction_id   TEXT        PRIMARY KEY,
    scored_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    event_time       TIMESTAMPTZ NOT NULL,
    client_id        TEXT        NOT NULL,
    amount           NUMERIC(12, 2) NOT NULL,
    channel          TEXT,
    merchant_country TEXT,
    probability      DOUBLE PRECISION NOT NULL,
    threshold        DOUBLE PRECISION NOT NULL,
    model_version    TEXT        NOT NULL,
    explanation      TEXT        NOT NULL,
    factors          JSONB       NOT NULL,
    status           TEXT        NOT NULL DEFAULT 'a_traiter'
                     CHECK (status IN ('a_traiter', 'fraude_confirmee', 'ecartee')),
    reviewed_by      TEXT,
    reviewed_at      TIMESTAMPTZ,
    review_note      TEXT
);

-- File de travail : les alertes à traiter, les plus risquées d'abord.
CREATE INDEX IF NOT EXISTS alerts_queue_idx
    ON ops.alerts (probability DESC) WHERE status = 'a_traiter';
CREATE INDEX IF NOT EXISTS alerts_scored_at_idx ON ops.alerts (scored_at DESC);

GRANT SELECT, INSERT, UPDATE ON ops.alerts TO api_app;
GRANT SELECT ON ops.alerts TO analyst_ro;
GRANT USAGE ON SCHEMA ops TO api_app;
