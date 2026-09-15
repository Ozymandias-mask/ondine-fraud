-- Le pipeline alimente le registre des consentements à partir du référentiel clients
-- et enregistre l'exécution des demandes de droits (accès, effacement).
GRANT SELECT, INSERT, UPDATE ON governance.consents TO ingestion_app;
GRANT SELECT, UPDATE ON governance.data_subject_requests TO ingestion_app;
GRANT SELECT ON governance.processing_register TO ingestion_app;
