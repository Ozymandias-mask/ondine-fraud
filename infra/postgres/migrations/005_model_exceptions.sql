-- Dérogations de mise en service d'un modèle.
--
-- Un contrôle automatique peut refuser une mise en production pour un critère qu'une banque
-- choisit malgré tout d'assumer — par exemple un écart d'équité de quelques millièmes face à
-- un gain de plusieurs millions. Cette décision est légitime ; ce qui ne l'est pas, c'est
-- qu'elle soit invisible. Elle est donc nommée, motivée, attribuée à une personne, assortie
-- d'une échéance de réexamen, et conservée.
--
-- C'est aussi ce qu'exige l'AI Act d'un système à haut risque : les écarts connus et les
-- mesures prises doivent être documentés (article 9, gestion des risques).


CREATE TABLE IF NOT EXISTS governance.model_exceptions (
    exception_id   BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    model_name     TEXT        NOT NULL,
    model_version  TEXT        NOT NULL,
    criterion      TEXT        NOT NULL,   -- critère non tenu, tel que formulé par le contrôle
    justification  TEXT        NOT NULL,   -- pourquoi la mise en service est tout de même décidée
    approved_by    TEXT        NOT NULL,   -- personne qui engage sa responsabilité
    approved_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    review_due_at  TIMESTAMPTZ NOT NULL,   -- date à laquelle l'écart doit être résorbé ou réexaminé
    status         TEXT        NOT NULL DEFAULT 'ouverte'
                   CHECK (status IN ('ouverte', 'resorbee', 'abandonnee')),
    closed_at      TIMESTAMPTZ,
    resolution     TEXT
);

CREATE INDEX IF NOT EXISTS model_exceptions_open_idx
    ON governance.model_exceptions (review_due_at) WHERE status = 'ouverte';

-- Dérogations à traiter : celles dont l'échéance de réexamen est passée.
CREATE OR REPLACE VIEW governance.model_exceptions_due AS
SELECT *, review_due_at < now() AS is_overdue
FROM governance.model_exceptions
WHERE status = 'ouverte';

GRANT SELECT ON governance.model_exceptions, governance.model_exceptions_due TO analyst_ro;
GRANT SELECT, INSERT, UPDATE ON governance.model_exceptions TO airflow;
GRANT USAGE, SELECT ON SEQUENCE governance.model_exceptions_exception_id_seq TO airflow;
