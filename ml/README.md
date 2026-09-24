# Modèle de détection de fraude (bloc 4)

Entraînement, évaluation, explicabilité et enregistrement du modèle qui décide, avant
l'autorisation d'un paiement, s'il doit être bloqué ou signalé.

> État : entraînement, suivi MLflow, explicabilité, tests de biais et de robustesse livrés.
> API de scoring, interface analyste, dérive et réentraînement automatique arrivent ensuite.

## Ce que fait le modèle

| Élément | Choix | Raison |
|---|---|---|
| Algorithme | LightGBM (arbres de décision boostés) | très efficace sur des données tabulaires déséquilibrées, rapide à réentraîner, explicable exactement (SHAP intégré) |
| Entrées | ~45 variables de la couche gold | calculées à date, sans fuite du futur |
| Sortie | probabilité de fraude, décision, facteurs explicatifs | l'analyste voit pourquoi, le client peut être informé |
| Découpage | 8 mois d'entraînement, 1 de validation, 3 de test | on entraîne sur le passé et on mesure sur le futur, comme en production |
| Déséquilibre | toutes les fraudes, 5 % des transactions légitimes | entraînement rapide ; les probabilités sont ensuite corrigées pour revenir au taux réel |
| Seuil | le meilleur coût métier **dans** le budget d'alertes (0,1 % des transactions) | l'équipe fraude ne peut pas traiter plus |

**Aucune fuite** : les identifiants, les labels, la décision du moteur de règles actuel et
les attributs d'équité (tranche d'âge, région, segment, ancienneté) sont exclus des entrées.
Ces attributs servent uniquement à mesurer d'éventuels écarts de traitement.

## Mesures produites à chaque entraînement

- **Qualité** : AUC-PR (la bonne métrique quand 0,2 % des lignes sont positives), AUC-ROC,
  score de Brier (calibration), rappel, précision, taux d'alerte.
- **Coût métier** : montant fraudé non détecté + coût des blocages à tort, comparé au moteur
  de règles actuel — l'écart est l'économie attendue.
- **Équité** : rappel et taux de fausses alertes par tranche d'âge et par région, avec les
  rapports entre groupes (règle des 4/5).
- **Robustesse** : 10 % puis 50 % de variables manquantes, montants bruités, catégorie
  inconnue — le modèle doit se dégrader progressivement, jamais s'effondrer.
- **Dérive** : performance mois par mois sur le jeu de test.

Tout est enregistré dans MLflow : paramètres, métriques, graphiques, modèle, **empreinte des
données** (fichiers, lignes, bornes, somme de contrôle) et graine aléatoire — un entraînement
est donc reproductible et auditable.

## Lancer un entraînement

```powershell
docker compose up -d                                   # dont MLflow (http://localhost:5001)
docker compose --profile ml build ml-cli
docker compose --profile ml run --rm ml-cli train
docker compose --profile ml run --rm ml-cli evaluate --alias challenger
docker compose --profile ml run --rm ml-cli explain --alias challenger --rank 1
```

Identifiants MLflow dans `.env` : `MLFLOW_ADMIN_USER` pour l'interface web, compte de service
`MLFLOW_SERVICE_USER` pour les traitements (aucun des deux ne détient de clé S3 : le serveur
relaie les artefacts).

Réglages utiles (variables d'environnement) : `ONDINE_ML_TRAIN`, `ONDINE_ML_VALID`,
`ONDINE_ML_TEST` (bornes « AAAA-MM..AAAA-MM »), `ONDINE_ML_NEGATIVE_RATE`,
`ONDINE_ML_ALERT_BUDGET`, `ONDINE_ML_SEED`.

## Organisation du code

```
ml/src/ondine_ml/
├── dataset.py    lecture de gold, découpage temporel, exclusion des fuites, échantillonnage
├── model.py      modèle servi : prétraitement + prédiction calibrée + seuil
├── train.py      entraînement, évaluation complète, enregistrement MLflow
├── evaluate.py   métriques, coût métier, seuils, équité, robustesse, dérive
├── explain.py    SHAP global et local, formulation lisible par un analyste
├── serving.py    chargement d'un modèle du registre (par alias), usages hors ligne
└── cli.py        ondine-ml train | evaluate | explain
```

## Tests

```powershell
$env:TEST_PG_ADMIN_DSN = "postgresql://<admin>:<mdp>@127.0.0.1:5432/postgres"
pytest ml/tests
```

Les tests construisent 5 mois de données avec le simulateur, les font passer par les vrais
traitements jusqu'à gold, puis vérifient : absence de fuite et découpage chronologique,
modèle enregistré avec son rapport, **reproductibilité** (même graine, mêmes résultats),
correction du sous-échantillonnage, rapports d'équité, de robustesse et de dérive,
explications non vides et hiérarchisées, tolérance aux entrées imparfaites (colonne absente,
catégorie inconnue, ordre des colonnes différent).
