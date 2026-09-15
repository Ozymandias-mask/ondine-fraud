-- Schémas de gouvernance et d'audit de la base « ondine ».
\connect ondine

CREATE SCHEMA IF NOT EXISTS audit;
CREATE SCHEMA IF NOT EXISTS governance;
REVOKE ALL ON SCHEMA public FROM PUBLIC;

-- ---------------------------------------------------------------------------------------------
-- Journal d'accès et de traitement : qui a fait quoi, sur quelle donnée, pour quelle finalité.
-- Table en ajout seul : toute modification ou suppression est refusée (preuve en cas d'audit).
-- ---------------------------------------------------------------------------------------------
CREATE TABLE audit.access_log (
    id           BIGSERIAL PRIMARY KEY,
    occurred_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    actor        TEXT        NOT NULL,              -- utilisateur ou compte de service
    action       TEXT        NOT NULL,              -- read, write, export, delete, score...
    resource     TEXT        NOT NULL,              -- ex. s3://silver/transactions/...
    purpose      TEXT        NOT NULL,              -- finalité déclarée au registre
    details      JSONB       NOT NULL DEFAULT '{}'::jsonb
);
CREATE INDEX access_log_occurred_at_idx ON audit.access_log (occurred_at);
CREATE INDEX access_log_actor_idx ON audit.access_log (actor);

CREATE FUNCTION audit.forbid_changes() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'Le journal d''audit est en ajout seul (% interdit)', TG_OP;
END;
$$;
CREATE TRIGGER access_log_append_only
    BEFORE UPDATE OR DELETE ON audit.access_log
    FOR EACH ROW EXECUTE FUNCTION audit.forbid_changes();
CREATE TRIGGER access_log_no_truncate
    BEFORE TRUNCATE ON audit.access_log
    FOR EACH STATEMENT EXECUTE FUNCTION audit.forbid_changes();

-- ---------------------------------------------------------------------------------------------
-- Registre des activités de traitement (RGPD, article 30).
-- ---------------------------------------------------------------------------------------------
CREATE TABLE governance.processing_register (
    processing_id   TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    purpose         TEXT NOT NULL,
    legal_basis     TEXT NOT NULL,
    data_categories TEXT[] NOT NULL,
    data_subjects   TEXT NOT NULL,
    recipients      TEXT NOT NULL,
    retention       TEXT NOT NULL,
    security        TEXT NOT NULL,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

INSERT INTO governance.processing_register VALUES
('PRC-001', 'Détection de la fraude aux paiements',
 'Bloquer ou signaler en temps réel les opérations frauduleuses',
 'Obligation légale (DSP2 : authentification et surveillance des opérations) et intérêt légitime',
 ARRAY['identification', 'données de transaction', 'données de connexion et appareil', 'géolocalisation approximative'],
 'Clients titulaires de comptes et de cartes', 'Équipe fraude, prestataire d''hébergement (UE)',
 '5 ans après la fin de la relation (LCB-FT), données brutes de flux : 24 h',
 'Pseudonymisation, chiffrement au repos, contrôle d''accès par rôle, journal d''audit', now()),
('PRC-002', 'Traitement des réclamations clients',
 'Répondre aux demandes et contestations reçues par le service client',
 'Exécution du contrat',
 ARRAY['identification', 'contenu des messages', 'coordonnées'],
 'Clients', 'Service client, équipe fraude',
 '5 ans après clôture de la réclamation',
 'Masquage des données personnelles dans les textes, accès restreint', now()),
('PRC-003', 'Prospection commerciale',
 'Informer les clients des nouvelles offres',
 'Consentement',
 ARRAY['identification', 'coordonnées'],
 'Clients ayant consenti', 'Équipe marketing',
 '3 ans après le dernier contact',
 'Retrait du consentement pris en compte sous 24 h', now());

-- ---------------------------------------------------------------------------------------------
-- Consentements (finalités qui en dépendent) et demandes d'exercice des droits.
-- ---------------------------------------------------------------------------------------------
CREATE TABLE governance.consents (
    client_id   TEXT        NOT NULL,
    purpose     TEXT        NOT NULL REFERENCES governance.processing_register (processing_id),
    granted     BOOLEAN     NOT NULL,
    source      TEXT        NOT NULL,                -- app, email, agence...
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (client_id, purpose)
);

CREATE TABLE governance.data_subject_requests (
    request_id    TEXT PRIMARY KEY,
    client_id     TEXT        NOT NULL,
    request_type  TEXT        NOT NULL CHECK (request_type IN
                  ('access', 'rectification', 'erasure', 'restriction', 'objection', 'portability')),
    received_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    status        TEXT        NOT NULL DEFAULT 'received' CHECK (status IN
                  ('received', 'in_progress', 'completed', 'rejected')),
    completed_at  TIMESTAMPTZ,
    notes         TEXT
);

-- Délai légal de réponse : un mois (article 12 du RGPD). Vue des demandes en retard ou à échéance.
CREATE VIEW governance.data_subject_requests_due AS
SELECT r.*, r.received_at + INTERVAL '1 month' AS due_at,
       (r.status NOT IN ('completed', 'rejected') AND now() > r.received_at + INTERVAL '1 month')
           AS is_overdue
FROM governance.data_subject_requests r;

-- ---------------------------------------------------------------------------------------------
-- Droits : ajout au journal pour les applications, lecture seule pour les analystes.
-- ---------------------------------------------------------------------------------------------
GRANT USAGE ON SCHEMA audit, governance TO ingestion_app, api_app, analyst_ro;
GRANT INSERT ON audit.access_log TO ingestion_app, api_app;
GRANT USAGE ON SEQUENCE audit.access_log_id_seq TO ingestion_app, api_app;
GRANT SELECT ON ALL TABLES IN SCHEMA governance TO analyst_ro, api_app;
GRANT SELECT ON audit.access_log TO analyst_ro;
GRANT INSERT, UPDATE ON governance.consents, governance.data_subject_requests TO api_app;
