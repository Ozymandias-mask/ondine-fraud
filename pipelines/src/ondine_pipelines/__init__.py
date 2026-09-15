"""Pipelines de données d'Ondine Bank : ingestion, qualité, pseudonymisation, features.

Couches du data lake (stockage S3, format Parquet) :
    bronze      données brutes typées, historisées, avec métadonnées de lignage
    silver      données nettoyées, contrôlées, pseudonymisées, données personnelles minimisées
    gold        tables prêtes pour le modèle (features, labels) et pour le pilotage
    quarantine  enregistrements rejetés, conservés pour analyse
"""

__version__ = "1.0.0"
