"""Couche silver : données nettoyées, contrôlées, pseudonymisées et minimisées.

Règles appliquées à tous les jeux :
- dédoublonnage sur l'identifiant métier ;
- lignes invalides isolées en quarantaine avec leur motif (jamais supprimées en silence) ;
- identifiants personnels directs retirés (nom, e-mail, téléphone, adresse, IBAN) ;
- identifiants indirects (appareil, IP, IBAN bénéficiaire) remplacés par un pseudonyme ;
- précision réduite quand la finalité ne l'exige pas (âge en tranche, géolocalisation à ~1 km) ;
- contrôles qualité enregistrés ; un contrôle bloquant en échec empêche la publication.
"""
