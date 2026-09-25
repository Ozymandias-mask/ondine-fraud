-- Historique des mesures de dérive du modèle en service.
--
-- Une mesure isolée ne distingue pas un mois atypique (soldes, fêtes) d'une usure qui
-- s'installe. C'est la série qui le dit — et c'est elle qui justifie, après coup, pourquoi un
-- réentraînement a été déclenché ce jour-là plutôt qu'un autre.
--
-- Le détail complet de la mesure est conservé en JSON : les variables les plus dérivées, la
-- qualité constatée et les retours des analystes. Aucune donnée personnelle n'y figure.

CREATE TABLE IF NOT EXISTS ops.model_drift (
    drift_id         BIGINT      GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    measured_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    model_version    TEXT        NOT NULL,
    reference_period TEXT        NOT NULL,
    current_period   TEXT        NOT NULL,
    verdict          TEXT        NOT NULL CHECK (verdict IN ('stable', 'surveiller', 'derive')),
    score_psi        DOUBLE PRECISION NOT NULL,
    feature_psi_max  DOUBLE PRECISION NOT NULL,
    details          JSONB       NOT NULL
);

CREATE INDEX IF NOT EXISTS model_drift_measured_at_idx ON ops.model_drift (measured_at DESC);

-- Le réentraînement tourne sous le compte des traitements ; Grafana lit en seule lecture.
GRANT SELECT, INSERT ON ops.model_drift TO ingestion_app;
GRANT SELECT ON ops.model_drift TO analyst_ro;
