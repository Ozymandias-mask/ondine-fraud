# Modèle de détection de fraude (bloc 4)

Entraînement, évaluation, explicabilité et enregistrement du modèle qui décide, avant
l'autorisation d'un paiement, s'il doit être bloqué ou signalé.

> État : entraînement, suivi MLflow, explicabilité, tests de biais et de robustesse livrés ;
> API de scoring temps réel livrée (voir `api/`). Interface analyste, détection de dérive et
> réentraînement automatique arrivent ensuite.

## Ce que fait le modèle

| Élément | Choix | Raison |
|---|---|---|
| Algorithme | LightGBM (arbres de décision boostés) | très efficace sur des données tabulaires déséquilibrées, rapide à réentraîner, explicable exactement (SHAP intégré) |
| Entrées | ~45 variables de la couche gold | calculées à date, sans fuite du futur |
| Sortie | probabilité de fraude, décision, facteurs explicatifs | l'analyste voit pourquoi, le client peut être informé |
| Découpage | 8 mois d'entraînement, 1 de validation, 3 de test | on entraîne sur le passé et on mesure sur le futur, comme en production |
| Déséquilibre | toutes les fraudes, 5 % des transactions légitimes (entraînement **et** validation) | entraînement rapide ; les probabilités sont ensuite corrigées pour revenir au taux réel |
| Seuil | le meilleur coût métier **dans** le budget d'alertes (0,1 % des transactions) | l'équipe fraude ne peut pas traiter plus |

Le jeu de **test n'est jamais échantillonné** : ce sont ses chiffres qui sont annoncés. Le
seuil, qui dépend du taux d'alertes réel, est calculé sur le mois de validation complet,
relu par lots après l'entraînement.

**Tenue à l'échelle** : 50 millions de transactions ne tiennent pas en mémoire. Les jeux
d'apprentissage sont sous-échantillonnés puis libérés dès le modèle entraîné ; la validation
et le test sont ensuite parcourus par lots de 400 000 lignes dont on ne garde que le score et
six colonnes d'évaluation, plus un échantillon borné (200 000 lignes) pour la robustesse et
l'explicabilité. L'empreinte mémoire ne dépend donc pas du volume évalué (mesurée à ~2,5 Gio
sur 50 M de transactions, conteneur limité à 10 Gio).

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
# Mise en service : le candidat ne devient « champion » que s'il passe les critères.
docker compose --profile ml run --rm ml-cli promote
docker compose --profile ml run --rm ml-cli rollback   # retour au champion précédent
```

## Mise en service d'un modèle

L'entraînement enregistre le modèle sous l'alias **`challenger`** ; l'API de scoring ne sert
que le **`champion`**. Le passage de l'un à l'autre n'est jamais automatique à l'aveugle : la
commande `promote` vérifie des critères écrits à l'avance sur le jeu de test et refuse, en
nommant les motifs, si l'un d'eux n'est pas tenu.

| Critère | Seuil | Pourquoi |
|---|---|---|
| Qualité (AUC-PR) | ≥ 0,50 | un modèle qui ne détecte rien ne remplace personne |
| Équité (règle des 4/5) | rapport de rappel ≥ 0,80 par tranche d'âge et région | ne pas introduire de discrimination (AI Act, art. 10) |
| Budget d'alertes | ≤ 2 × le budget visé | l'équipe fraude ne peut pas traiter davantage |
| Coût métier | inférieur au champion en place | une nouvelle version doit faire gagner de l'argent |

Un refus est journalisé **et** inscrit sur la version dans MLflow (`promotion_refusee`). Le
modèle sortant garde l'alias `ancien_champion` : le retour arrière est immédiat.

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
