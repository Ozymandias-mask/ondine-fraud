"""Couche gold : tables prêtes pour le modèle de détection et pour le pilotage.

Principe central : **aucune fuite d'information du futur**. Chaque variable d'une
transaction est calculée uniquement à partir de ce qui était connu *avant* elle
(fenêtres temporelles qui excluent la ligne courante). Le label porte sa date de
connaissance (`label_reported_at`) pour que l'entraînement n'utilise que des labels
réellement disponibles à la date de coupure.
"""
