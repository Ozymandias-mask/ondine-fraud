-- Orchestration (étape 2e) : indicateurs métier, incidents de sécurité, exécution des droits.

-- ---------------------------------------------------------------------------------------------
-- Indicateurs métier quotidiens, alimentés automatiquement depuis la couche gold.
-- Lus par Grafana (compte analyst_ro, lecture seule).
-- ---------------------------------------------------------------------------------------------
CREATE SCHEMA IF NOT EXISTS analytics;

CREATE TABLE analytics.daily_kpis (
    day                   DATE        NOT NULL,
    channel               TEXT        NOT NULL,          -- canal de paiement, ou 'all'
    transactions          BIGINT      NOT NULL,
    amount_eur            NUMERIC(18, 2) NOT NULL,
    declined              BIGINT      NOT NULL,          -- refus du moteur de règles actuel
    frauds                BIGINT      NOT NULL,          -- fraudes confirmées à date
    fraud_amount_eur      NUMERIC(18, 2) NOT NULL,
    frauds_declined       BIGINT      NOT NULL,          -- fraudes arrêtées par les règles actuelles
    legit_declined        BIGINT      NOT NULL,          -- clients légitimes refusés (faux positifs)
    avg_report_delay_h    NUMERIC(10, 1),                -- délai moyen de déclaration par le client
    refreshed_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (day, channel)
);

GRANT USAGE ON SCHEMA analytics TO ingestion_app, analyst_ro;
GRANT SELECT, INSERT, DELETE ON analytics.daily_kpis TO ingestion_app;
GRANT SELECT ON analytics.daily_kpis TO analyst_ro;

-- ---------------------------------------------------------------------------------------------
-- Comptes autorisés à accéder aux données personnelles, avec les finalités permises.
-- Toute entrée du journal d'audit hors de cette liste est un incident potentiel.
-- ---------------------------------------------------------------------------------------------
CREATE TABLE governance.authorized_actors (
    actor             TEXT PRIMARY KEY,
    kind              TEXT   NOT NULL CHECK (kind IN ('service', 'human')),
    allowed_purposes  TEXT[] NOT NULL,
    description       TEXT   NOT NULL
);

INSERT INTO governance.authorized_actors VALUES
('svc-ingestion', 'service', ARRAY['PRC-001', 'PRC-002'], 'Ingestion des fichiers et du flux'),
('svc-pipeline-silver', 'service', ARRAY['PRC-001', 'PRC-002'], 'Nettoyage et pseudonymisation'),
('svc-rgpd', 'service', ARRAY['RGPD-DROITS'], 'Exécution des demandes d''exercice des droits'),
('svc-scoring-api', 'service', ARRAY['PRC-001'], 'API de scoring (étape 3)'),
('analyste.fraude', 'human', ARRAY['PRC-001'], 'Analyste de l''équipe fraude'),
('dpo', 'human', ARRAY['RGPD-DROITS', 'PRC-001', 'PRC-002', 'PRC-003'], 'Délégué à la protection des données');

-- ---------------------------------------------------------------------------------------------
-- Incidents de sécurité détectés automatiquement (violation de données potentielle).
-- Délai de notification à la CNIL : 72 h après la prise de connaissance (RGPD, article 33).
-- ---------------------------------------------------------------------------------------------
CREATE TABLE governance.security_incidents (
    incident_id    BIGSERIAL PRIMARY KEY,
    detected_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    rule           TEXT        NOT NULL,
    severity       TEXT        NOT NULL CHECK (severity IN ('low', 'medium', 'high')),
    actor          TEXT        NOT NULL,
    window_start   TIMESTAMPTZ NOT NULL,
    window_end     TIMESTAMPTZ NOT NULL,
    evidence       JSONB       NOT NULL DEFAULT '{}'::jsonb,
    status         TEXT        NOT NULL DEFAULT 'open' CHECK (status IN
                   ('open', 'investigating', 'false_positive', 'contained', 'notified', 'closed')),
    assigned_to    TEXT,
    resolution     TEXT,
    closed_at      TIMESTAMPTZ,
    UNIQUE (rule, actor, window_start)                   -- un même écart n'est signalé qu'une fois
);

CREATE VIEW governance.security_incidents_due AS
SELECT i.*, i.detected_at + INTERVAL '72 hours' AS cnil_deadline,
       (i.status IN ('open', 'investigating') AND now() > i.detected_at + INTERVAL '48 hours')
           AS deadline_at_risk
FROM governance.security_incidents i;

-- Effacement : exécuté seulement après validation du DPO, qui vérifie l'absence d'obligation
-- de conservation (LCB-FT : 5 ans pour les opérations et l'identité, RGPD art. 17.3.b).
ALTER TABLE governance.data_subject_requests
    ADD COLUMN approved_by TEXT,
    ADD COLUMN approved_at TIMESTAMPTZ;
-- La vue est recréée pour exposer les nouvelles colonnes (droits réaccordés plus bas).
DROP VIEW governance.data_subject_requests_due;
CREATE VIEW governance.data_subject_requests_due AS
SELECT r.*, r.received_at + INTERVAL '1 month' AS due_at,
       (r.status NOT IN ('completed', 'rejected') AND now() > r.received_at + INTERVAL '1 month')
           AS is_overdue
FROM governance.data_subject_requests r;

-- Journal d'exécution des demandes de droits : ce qui a été exporté ou effacé, où et quand.
CREATE TABLE governance.data_subject_actions (
    id            BIGSERIAL PRIMARY KEY,
    request_id    TEXT        NOT NULL REFERENCES governance.data_subject_requests (request_id),
    action        TEXT        NOT NULL,                   -- export, erase, consent_withdrawn
    target        TEXT        NOT NULL,                   -- objet du data lake ou table
    rows_affected BIGINT      NOT NULL,
    executed_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Détection : le compte de service lit le journal d'audit et écrit les incidents.
GRANT SELECT ON audit.access_log, governance.authorized_actors TO ingestion_app;
GRANT SELECT, INSERT ON governance.security_incidents TO ingestion_app;
GRANT USAGE ON SEQUENCE governance.security_incidents_incident_id_seq TO ingestion_app;
GRANT SELECT ON governance.security_incidents_due, governance.data_subject_requests_due
    TO ingestion_app, analyst_ro;
GRANT SELECT ON governance.data_subject_requests_due TO api_app;
GRANT SELECT ON governance.authorized_actors, governance.security_incidents,
    governance.data_subject_actions TO analyst_ro;
-- Suivi des incidents par l'équipe sécurité (via l'API ou le DPO).
GRANT UPDATE (status, assigned_to, resolution, closed_at) ON governance.security_incidents TO api_app;
GRANT SELECT ON governance.security_incidents TO api_app;
-- Droits : effacement des consentements, journal des actions.
GRANT UPDATE (approved_by, approved_at) ON governance.data_subject_requests TO api_app;
GRANT DELETE ON governance.consents TO ingestion_app;
GRANT SELECT, INSERT ON governance.data_subject_actions TO ingestion_app;
GRANT USAGE ON SEQUENCE governance.data_subject_actions_id_seq TO ingestion_app;
