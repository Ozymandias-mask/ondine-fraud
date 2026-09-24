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

## Le seuil se recale tout seul

Le budget d'alertes est une contrainte d'organisation — une transaction sur mille — donc un
**quantile**, pas une probabilité figée. Sur ces données, 4 % d'écart de seuil change le
volume d'alertes d'un facteur cent : un seuil gravé à l'entraînement ne tiendrait pas la
promesse faite à l'équipe fraude.

L'API conserve donc, dans le magasin en ligne, une fenêtre glissante des scores rendus
(200 000 par défaut) et recalcule le seuil toutes les cinq minutes comme le quantile
correspondant au budget. Le seuil du modèle sert de repli au démarrage, avant que la fenêtre
ne soit assez remplie ; en deçà de 20 000 observations, rien n'est recalé — un quantile
extrême sur peu de données ne veut rien dire. Toutes les instances de l'API partagent la même
fenêtre : elles convergent sans avoir à se coordonner, et le seuil appliqué est exporté
(`scoring_model_threshold`, `scoring_threshold_recalibrations_total`).

## Latence mesurée

| Étape | Médiane |
|---|---|
| lecture du magasin en ligne | 2,0 ms |
| calcul des 43 variables | 0,2 ms |
| prédiction LightGBM | 0,6 ms |
| explication (SHAP exacte) | 0,2 ms |
| **décision complète** | **3,0 ms** (95e centile 5,8 ms, 99e 12,8 ms) |

Mesuré de bout en bout sur 5 000 transactions réelles rejouées par HTTP (`replay`), transport
compris : **médiane 2,2 ms, 95e centile 2,6 ms, 99e centile 3,0 ms, maximum 15,5 ms**, à
448 transactions par seconde. Le budget de 100 ms est tenu avec un facteur quarante de marge.

Le budget de 100 ms est vérifié par un test automatisé, et le dépassement est exporté comme
métrique (`scoring_over_budget_total`) pour être suivi en production.

## Dimensionnement du magasin en ligne

Le magasin conserve 30 jours d'historique par client et par carte, et 8 jours d'événements
applicatifs — c'est ce qu'exigent les fenêtres du modèle. À 4,6 millions de transactions par
mois, cela représente environ 9 millions d'entrées, soit **1,5 Gio** ; 3 Gio sont alloués
pour absorber les pics.

L'éviction est **désactivée** (`noeviction`) et c'est délibéré : perdre silencieusement de
l'historique fausserait les variables du modèle sans que rien ne le signale. Mieux vaut une
écriture en erreur, visible et corrigeable, que des décisions prises sur des données amputées.

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

## Interface analyste (accessible)

Une décision de blocage n'a de valeur que si quelqu'un peut l'examiner, la justifier au client
et, le cas échéant, la contredire. Chaque blocage dépose une alerte dans une file de travail
(`ops.alerts`) avec ce qui l'a produite — probabilité, seuil appliqué, version du modèle,
facteurs explicatifs — et l'analyste y répond : fraude confirmée, ou opération légitime. Sa
décision est nominative, datée et tracée au journal d'audit. Ces retours mesurent aussi la
qualité du modèle en production, bien avant que les déclarations des clients n'arrivent.

**Accessibilité (RGAA)** — pages produites par le serveur, sans cadriciel ni JavaScript
obligatoire : une interface bancaire doit rester utilisable au clavier, avec un lecteur
d'écran, et fonctionner si un script ne se charge pas.

| Exigence | Mise en œuvre |
|---|---|
| Langue, titre, structure | `lang="fr"`, titre unique par page, un seul `h1`, hiérarchie respectée |
| Navigation au clavier | lien d'évitement vers le contenu, focus visible (`:focus-visible`), aucun piège au clavier |
| Régions | `header`, `nav` nommées, `main`, `footer` |
| Tableaux de données | `caption` et `scope="col"` sur chaque en-tête |
| Formulaires | chaque champ associé à une étiquette **visible** |
| Couleur | aucune information portée par la seule couleur : chaque état porte son libellé, chaque facteur sa flèche ▲/▼ |
| Contrastes | supérieurs à 4,5:1 |
| Mouvement | `prefers-reduced-motion` respecté |
| Messages | confirmations annoncées (`role="status"`, `aria-live="polite"`) |
| Zoom | mise en page fluide, lisible à 200 % |

Ces points sont vérifiés par un test automatisé : ce sont ceux qui se dégradent le plus vite
au fil des modifications. Le reste (parcours réel au lecteur d'écran, audit RGAA complet) ne
se contrôle pas par un test et relève d'une recette dédiée.

Les données affichées viennent du flux de paiement et des analystes : elles sont échappées,
et un test vérifie qu'un balisage injecté s'affiche comme texte au lieu d'être interprété.

## Points d'entrée

| Méthode | Chemin | Rôle |
|---|---|---|
| POST | `/scorer` | décision sur une demande d'autorisation |
| GET | `/alertes` | file de travail des analystes |
| GET | `/alertes/{id}` | détail d'une alerte et ses facteurs |
| POST | `/alertes/{id}` | décision d'un analyste (tracée) |
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

### Remplir la file avec de vraies transactions

```powershell
docker compose --profile api run --rm scoring-api replay --month 2025-12 --limit 5000
```

Rejoue de **vraies** transactions du data lake par le chemin exact de la production —
requête HTTP, magasin en ligne, variables, modèle, décision, dépôt de l'alerte — et rapporte
débit, taux de blocage et latences observées (médiane, 95e, 99e centile, maximum). C'est à la
fois la démonstration de mise en production et la mesure du budget de 100 ms sur des données
réelles.

Le rejeu **préchauffe d'abord le magasin** avec les trente jours qui précèdent, pour les
clients concernés. Sans cela il partirait d'un magasin vide : tous les appareils y
paraîtraient neufs, toutes les vélocités vaudraient zéro, et le modèle verrait partout le
même profil de test de carte — des scores identiques, et un volume d'alertes dix fois
supérieur au budget. En production le magasin est alimenté en continu par le flux Kafka ; le
préchauffage reconstitue cette situation. `--no-warmup` permet d'observer la différence.

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
