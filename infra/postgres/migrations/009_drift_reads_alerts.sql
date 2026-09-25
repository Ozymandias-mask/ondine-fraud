-- Lecture des retours d'analystes par la mesure de dérive.
--
-- La mesure de dérive s'exécute avec l'identité des traitements (`ingestion_app`) : c'est
-- elle qui a besoin de lire la file d'alertes pour en tirer la précision constatée par les
-- analystes. Elle n'y écrit rien — seule l'API dépose et met à jour des alertes.
--
-- Fichier séparé et purement déclaratif, donc rejouable sans effet de bord.

GRANT SELECT ON ops.alerts TO ingestion_app;
