-- Droits d'écriture sur les dérogations de mise en service.
--
-- La commande de promotion s'exécute avec l'identité des traitements (`ingestion_app`), pas
-- avec celle de l'orchestrateur : c'est elle qui doit pouvoir inscrire une dérogation.
-- Fichier séparé et purement déclaratif, donc rejouable sans effet de bord.

GRANT SELECT, INSERT, UPDATE ON governance.model_exceptions TO ingestion_app;
GRANT USAGE, SELECT ON SEQUENCE governance.model_exceptions_exception_id_seq TO ingestion_app;
