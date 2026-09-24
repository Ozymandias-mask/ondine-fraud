# API de scoring temps réel (bloc 4)

Décide, avant l'autorisation d'un paiement, s'il doit être bloqué — avec l'explication de la
décision, en moins de 100 ms.

## Le problème à résoudre

Le modèle a appris sur des variables calculées en SQL sur des mois entiers de données
(couche gold) : plusieurs minutes de calcul. En production, la même transaction doit être
décidée avant que le client ne retire sa carte. Il faut donc recalculer **les mêmes
variables**, pour une seule transaction, en quelques millisecondes.

D'où un **magasin de variables en ligne** (Redis) alimenté en continu par le flux Kafka : il
conserve, par carte, par client, par appareil et par bénéficiaire, l'historique glissant
strictement nécessaire aux fenêtres du modèle (1 h, 24 h, 7 jours, 30 jours) — et rien d'autre.

```
  système d'autorisation ──► POST /scorer ──► magasin en ligne (Redis)
                                  │                    ▲
                                  ├── variables ───────┘
                                  ├── modèle « champion » (registre MLflow)
                                  └── décision + facteurs explicatifs
                                          ▲
  Kafka (transactions.raw, app-events.raw) ┴── service « online-features »
```

## Cohérence entraînement / production

C'est le risque majeur d'une mise en production : si une variable est calculée différemment
en ligne et hors ligne, le modèle voit des données qu'il n'a jamais apprises et ses scores
perdent leur sens, **sans qu'aucune erreur ne soit levée**.

Trois garde-fous :

1. les définitions en ligne sont écrites une seule fois (`features.py`), en miroir du SQL ;
2. un test rejoue de vraies données brutes dans le magasin, puis compare **les 43 variables,
   transaction par transaction**, à ce qu'a calculé la couche gold — l'égalité est exacte ;
3. les variables dont la valeur dépendrait de la profondeur d'historique chargée ont été
   bornées des deux côtés (30 jours) : une variable ne doit pas dépendre du lot dans lequel
   elle est calculée.

Le prétraitement de l'API est un chemin rapide (conversion directe en tableau de réels, sans
pandas) ; un test vérifie qu'il donne **exactement** la même prédiction que le chemin utilisé
à l'entraînement.

## Latence mesurée

| Étape | Médiane |
|---|---|
| lecture du magasin en ligne | 2,0 ms |
| calcul des 43 variables | 0,2 ms |
| prédiction LightGBM | 0,6 ms |
| explication (SHAP exacte) | 0,2 ms |
| **décision complète** | **3,0 ms** (95e centile 5,8 ms, 99e 12,8 ms) |

Le budget de 100 ms est vérifié par un test automatisé, et le dépassement est exporté comme
métrique (`scoring_over_budget_total`) pour être suivi en production.

## Disponibilité et erreurs

| Incident | Comportement |
|---|---|
| Magasin en ligne injoignable | décision rendue sans historique, `mode_degrade: true`, métrique et journal dédiés (configurable : `ONDINE_API_FAIL_OPEN=false` fait répondre 503) |
| Aucun modèle promu | démarrage quand même, `/pret` répond 503, `/scorer` répond 503 avec un message explicite |
| Demande malformée | 422 avant tout scoring — une entrée inattendue n'est jamais « scorée au mieux » |
| Écriture dans le magasin en échec | la réponse est déjà partie ; le flux Kafka rattrape l'écriture (elle est idempotente) |

## Sécurité et RGPD

- le contrat d'entrée **refuse les champs inconnus** et borne tous les formats ;
- les identifiants indirects (appareil, IBAN du bénéficiaire) sont pseudonymisés dès la
  réception, avec la même clé que le data lake ; ils ne sont jamais stockés ni renvoyés ;
- chaque clé du magasin porte une durée de vie égale à la fenêtre qu'elle sert : l'historique
  s'efface tout seul (minimisation) ;
- l'image de service ne contient ni Airflow ni outillage, et tourne sous un compte non
  administrateur ;
- aucune donnée personnelle ne figure dans les journaux ni dans les métriques.

## Points d'entrée

| Méthode | Chemin | Rôle |
|---|---|---|
| POST | `/scorer` | décision sur une demande d'autorisation |
| GET | `/sante` | vivacité (sans dépendance externe) |
| GET | `/pret` | mise en service : modèle chargé **et** magasin joignable |
| GET | `/metrics` | métriques Prometheus |
| GET | `/docs` | contrat OpenAPI interactif |

## Démarrer

```powershell
docker compose --profile api up -d --build
# Référentiel marchands (deux variables du modèle) : à recharger après chaque mise à jour.
docker compose --profile api run --rm scoring-api referentials
curl http://localhost:8010/pret
```

Réglages : `ONDINE_API_REDIS_URL`, `ONDINE_API_MODEL_ALIAS` (par défaut `champion`),
`ONDINE_API_LATENCY_BUDGET_MS`, `ONDINE_API_FAIL_OPEN`, `ONDINE_API_EXPLAIN_FACTORS`.

## Organisation du code

```
api/src/ondine_api/
├── schemas.py   contrats d'entrée et de sortie (mêmes champs que le message Kafka)
├── store.py     magasin en ligne : écriture depuis le flux, lecture pour la décision
├── features.py  calcul des 43 variables, miroir exact du SQL de la couche gold
├── app.py       application FastAPI : décision, sondes, métriques, mode dégradé
├── stream.py    consommateur Kafka qui alimente le magasin
└── cli.py       ondine-api serve | feed | referentials
```

## Tests

```powershell
$env:TEST_PG_ADMIN_DSN = "postgresql://<admin>:<mdp>@127.0.0.1:5432/postgres"
pytest api/tests
```

Les tests construisent quatre mois de données avec le simulateur, les font passer par les
vrais traitements jusqu'à la couche gold, entraînent un modèle, puis vérifient : égalité
exacte des variables en ligne et hors ligne, couverture de **toutes** les variables du
modèle, décision explicable, latence sous budget, historique visible par la transaction
suivante, mode dégradé, refus des entrées invalides, sondes et métriques.
