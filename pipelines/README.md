# Pipelines de données Ondine Bank

Code des traitements du data lake (bloc 3). Moteur de calcul : **DuckDB** (SQL analytique
en colonnes, parallèle, capable de déborder sur disque) ; stockage : **Parquet sur S3**.

> État : bronze, silver et gold validées sur le PC (50 M de transactions). Ingestion en flux
> livrée ; orchestration Airflow et droits RGPD arrivent dans les prochaines livraisons.

## Ingestion : zone d'arrivée → bronze

Pour chaque fichier décrit par un manifeste :

| Étape | Contrôle | En cas d'échec |
|---|---|---|
| Intégrité | présence, taille, empreinte SHA-256 | arrêt, run « failed », fichier non ingéré |
| Lecture typée | types déclarés par colonne (CSV, JSON lines, mbox) | ligne envoyée en quarantaine |
| Complétude | chargées + rejetées = lignes annoncées | arrêt |
| Seuil de rejet | ≤ 1 % de lignes rejetées | arrêt (livraison jugée anormale) |
| Écriture | Parquet zstd, nom déterministe (idempotent) | – |
| Traçabilité | `ops.pipeline_runs`, `ops.file_ingestions`, `audit.access_log` | – |

- **Incrémental** : un fichier (chemin + empreinte) n'est jamais ingéré deux fois.
- **Évolution de schéma** : les transactions v1 et v2 sont unifiées (`wallet_type` vaut NULL avant juillet 2025).
- **Lignage** : chaque ligne bronze porte `_source_file`, `_source_sha256`, `_run_id`, `_ingested_at`.
- **Métriques** poussées vers Prometheus : lignes lues, écrites, rejetées, débit, durée, succès.

Débit mesuré en développement (2 threads) : ~80 000 transactions/s et ~110 000 événements/s,
contrôle SHA-256 compris.

## Ingestion en flux : Kafka → bronze

Service `stream-ingest` : **2 répliques** dans le groupe de consommateurs `bronze-ingestion`,
qui se répartissent les 15 partitions des topics `transactions.raw`, `app-events.raw` et
`fraud-reports.raw`. Les fichiers produits ont le même schéma et les mêmes partitions
mensuelles que l'ingestion batch : silver et gold ne font pas la différence.

Cycle d'un lot (toutes les 60 s ou 50 000 messages) :

| Étape | Rôle |
|---|---|
| 1. Validation | JSON lisible, identifiant présent, horodatage et types convertibles |
| 2. Écriture bronze | Parquet par partition Kafka et par mois, nom = plage d'offsets (`stream_p03_…-….parquet`) |
| 3. Topic d'erreurs | messages invalides republiés dans `<topic>.dlq`, motif en en-tête (`dq_reason`) |
| 4. Registre | `ops.stream_batches` : consommateur, partition, offsets, lignes, fichiers, période |
| 5. Commit | l'offset n'est validé qu'après 2, 3 et 4 réussis |

- **Aucune perte** : un plantage avant le commit fait relire le lot. Les coupures S3 passagères
  sont réessayées ; un échec persistant arrête le consommateur, Docker le redémarre.
- **Doublons** : un lot relu peut exister deux fois en bronze ; silver dédoublonne sur
  l'identifiant métier (testé).
- **Rééquilibrage** : stratégie `cooperative-sticky` ; une partition retirée est écrite et
  validée avant d'être rendue. Arrêt propre sur SIGTERM (90 s de grâce).
- **Supervision** (port 8001, collecte par découverte DNS des répliques) : messages lus,
  lignes écrites, invalides par motif, retard par partition, délai Kafka → bronze, durée des
  lots. Alertes : consommateur manquant, retard, blocage, messages invalides, délai p95.

```powershell
docker compose --profile stream up -d              # simulateur + 2 consommateurs
docker compose --profile stream ps stream-ingest
docker compose logs -f stream-ingest
```

## Couche silver : bronze → silver

| Jeu silver | Traitement | Protection des données | Contrôles bloquants |
|---|---|---|---|
| `clients` | dernière photo, dédoublonnage | nom, e-mail, téléphone, adresse, IBAN, date de naissance **retirés** ; âge en tranche | complétude, schéma sans identifiant direct |
| `cards`, `merchants` | dédoublonnage | jeton et 4 derniers chiffres retirés | unicité, cartes rattachées à un client, coordonnées et pays valides |
| `transactions` (par mois) | dédoublonnage, validation, horodatages dérivés | appareil, IP et IBAN bénéficiaire **pseudonymisés** ; pays et banque du bénéficiaire conservés | bronze = silver + quarantaine + doublons, taux d'invalides ≤ 0,1 %, schéma sans identifiant brut |
| `app_events` (par mois) | JSON aplati en colonnes typées, versions de l'application gérées | appareil et IP pseudonymisés, géolocalisation arrondie à ~1 km | complétude, invalides ≤ 0,1 %, appareil présent ≥ 99 %, précision géographique réduite |
| `fraud_labels` | dédoublonnage, statut d'enquête | – | unicité des déclarations |
| `support_messages` (par mois) | e-mail associé à son ticket | IBAN, e-mail, téléphone, carte, adresse et nom de l'expéditeur **masqués** | aucune donnée personnelle détectable après masquage |

- **Pseudonymisation** : `pseudo(v) = SHA-256(clé secrète | v)`. Pseudonymes stables (jointures possibles
  entre transactions et événements), non réversibles et non recalculables sans la clé.
- **Consentements** : les consentements marketing du référentiel sont synchronisés dans
  `governance.consents` (registre de gouvernance, source unique pour les autres applications).
- **Quarantaine** : chaque ligne invalide est écrite avec son motif (`dq_reason`) dans le bucket `quarantine`.
- **Contrôles non bloquants** (`warning`) : enregistrés et visibles, ils signalent sans arrêter la chaîne.
  Exemple réel : des clients de moins de 18 ans à la date de la photo du référentiel, anomalie
  de la source à transmettre à l'équipe KYC.
- Débit mesuré en développement (2 threads) : ~130 000 lignes/s (validation, pseudonymisation, écriture).

## Couche gold : features et labels pour le modèle

Table `gold/transaction_features` (une ligne par transaction, par mois) : 63 colonnes.

| Famille | Variables | Fraude visée |
|---|---|---|
| Transaction | montant (et log), canal, mode de saisie, MCC et catégorie, pays, heure, nuit, week-end, 3-D Secure, wallet | contexte |
| Vélocité carte | opérations sur 1 h et 24 h, montant 24 h, marchands distincts 1 h, petits achats en ligne 1 h, refus 1 h, délai depuis l'opération précédente | test de carte |
| Historique client | opérations, montant moyen et maximal sur 30 jours, ratio au montant habituel, pays distincts sur 7 jours, virements 24 h | montant inhabituel |
| Déplacement | distance et vitesse depuis le paiement physique précédent (haversine) | carte clonée |
| Appareil | ancienneté de l'appareil pour ce client | prise de contrôle |
| Bénéficiaire | premier virement, nouveaux payeurs du bénéficiaire sur 30 jours | comptes mules, faux conseiller |
| Session | échecs de connexion, réinitialisations, connexions depuis l'étranger ou un nouvel appareil (24 h), OTP par SMS (1 h), changement de méthode d'authentification et enrôlement wallet (7 jours) | prise de contrôle, SIM swap, wallet frauduleux |

**Pas de fuite du futur.** Toutes les fenêtres excluent la transaction courante et ne regardent
que le passé ; un test recalcule la vélocité à la main sur un échantillon. Le label porte
sa date de connaissance (`label_reported_at`) : l'entraînement n'utilisera que les labels
connus à la date de coupure. `history_complete` signale les 30 premiers jours (recul insuffisant).

**Équité.** `age_band`, `region`, `segment` et `tenure_days` sont présents pour les tests de
biais, pas comme variables d'entrée. La décision des règles actuelles (`legacy_auth_status`)
sert de point de comparaison. Aucun pseudonyme ne sort de la couche silver.

**Mémoire bornée.** Seules les colonnes et les événements utiles sont chargés, avec le recul
nécessaire (30 jours pour les transactions, 8 jours pour les événements) ; DuckDB déborde sur
disque au-delà de sa limite mémoire.

Contrôles bloquants : une ligne par transaction silver, identifiants uniques, cohérence
temporelle des fenêtres, label postérieur à la transaction, bornes du mois, absence de
pseudonyme. Avertissements : taux de fraude plausible (0,02 % à 2 %).

## Lancer les traitements (sans Airflow)

```powershell
docker compose up -d                                   # applique aussi les migrations (db-migrate)
docker compose --profile pipelines build pipelines-cli
docker compose --profile pipelines run --rm pipelines-cli ingest --dataset transactions --limit 1
docker compose --profile pipelines run --rm pipelines-cli ingest --dataset all
docker compose --profile pipelines run --rm pipelines-cli silver --dataset all
docker compose --profile pipelines run --rm pipelines-cli gold
```

Les commandes `silver` et `gold` reprennent là où elles s'étaient arrêtées : une tâche déjà
réussie n'est pas recalculée (`--no-resume` pour tout refaire).

## Tests

```powershell
$env:TEST_PG_ADMIN_DSN = "postgresql://<admin>:<mdp>@127.0.0.1:5432/postgres"
pytest pipelines/tests
```

Les tests créent une base temporaire à partir des vrais scripts SQL du dépôt et une petite
zone d'arrivée avec le simulateur : intégrité, complétude, quarantaine, incrémentalité, audit.
L'ingestion en flux est testée avec des messages produits par le simulateur et des doublures
des clients Kafka : schéma identique au batch, topic d'erreurs, commit après écriture, aucun
commit en cas d'échec, rééquilibrage, dédoublonnage silver des lots relus.
