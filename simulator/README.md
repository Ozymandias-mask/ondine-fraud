# Simulateur de données Ondine Bank

Génère les données **fictives** d'une néobanque : référentiels, transactions, événements de
l'application mobile, déclarations de fraude et e-mails du service client. Deux modes :

- **batch** : 12 mois d'historique déposés dans la zone d'arrivée (`data/landing`) ;
- **stream** : flux continu vers Kafka, pilotable pendant une démonstration.

Aucune personne réelle n'est concernée : noms, IBAN, téléphones et adresses sont inventés
(codes banque fictifs 99001 à 99021, domaine `ondine-bank.example`).

## Volumétrie

| Préréglage | Clients | Transactions / an | Événements / an | Usage |
|---|---|---|---|---|
| `small` | 5 000 | ~40 000 (2 mois) | ~40 000 | tests, CI |
| `demo` | 100 000 | ~5 M | ~5 M | développement |
| `full` | 1 000 000 | **~50 M** | **~50 M** | projet |

Mesure réelle (préréglage `full`, 4 processus, PC de développement, 15/09/2026) :

| Indicateur | Valeur |
|---|---|
| Transactions | 50 506 402 (dont 105 461 frauduleuses, 0,21 %) |
| Événements applicatifs | 50 504 302 |
| Déclarations de fraude | 99 226 |
| E-mails du support | 184 334 |
| Volume compressé | 6,4 Go |
| Durée | 296 s, soit **170 500 transactions/s** |
| Parts en échec | 0 |

Mémoire : ~2,5 Go par processus.

## Utilisation

```bash
# Installation locale (Python 3.11+)
pip install -r requirements-dev.txt && pip install -e .

# Historique complet (ou via Docker : docker compose --profile generate run --rm simulator-batch)
ondine-sim batch --preset full --output data/landing --workers 4

# Reprise après une interruption : seules les parts sans manifeste sont régénérées
ondine-sim batch --preset full --output data/landing --workers 4 --resume

# Flux temps réel (via Docker : docker compose --profile stream up -d simulator-stream)
ondine-sim stream --bootstrap localhost:9092 --rate 300
```

### Pilotage du flux pendant la démonstration

Le jeton est la valeur `SIM_CONTROL_TOKEN` du fichier `.env`.

```powershell
$h = @{ "X-Control-Token" = "<jeton>" }
Invoke-RestMethod -Method Post -Headers $h "http://localhost:8000/control/burst?enabled=1"   # pic à 2 000 tx/s
Invoke-RestMethod -Method Post -Headers $h "http://localhost:8000/control/drift?enabled=1"   # nouvelle fraude
Invoke-RestMethod -Method Post -Headers $h "http://localhost:8000/control/rate?value=500"
Invoke-RestMethod "http://localhost:8000/health"
```

Métriques Prometheus : `http://localhost:8000/metrics` (débit cible et réel, latence
d'acquittement Kafka, erreurs de livraison, incidents injectés par type).

## Données produites (zone d'arrivée)

Partitionnement `year=AAAA/month=MM`. Chaque livraison a un manifeste JSON dans `_manifests/`
(nombre de lignes, taille, empreinte SHA-256, version du générateur, graine) : il sert au
contrôle d'intégrité et à la traçabilité.

| Jeu | Format | Nature | Contenu |
|---|---|---|---|
| `referentials/clients` | CSV.gz | structuré | identité, coordonnées, IBAN, segment, consentement marketing |
| `referentials/cards` | CSV.gz | structuré | carte, jeton (jamais de numéro de carte), type, statut |
| `referentials/merchants` | CSV.gz | structuré | marchand, MCC, pays, coordonnées |
| `transactions` | CSV.gz | structuré | une ligne par autorisation de paiement ou virement |
| `app_events` | JSON lines.gz | semi-structuré | connexions, 3-D Secure, ajout de bénéficiaire, wallet… |
| `fraud_reports` | CSV.gz | structuré | déclarations de fraude (labels), avec délai et bruit |
| `support_emails` | mbox.gz | non structuré | e-mails bruts des clients, avec données personnelles |
| `support_tickets` | CSV.gz | structuré | classement des e-mails par les conseillers |

La **vérité terrain** (type réel de chaque fraude) est écrite à part, dans `data/ground_truth`,
et n'est jamais livrée avec les données brutes : comme dans la réalité, la banque ne connaît
la fraude qu'à travers les déclarations.

### Transactions

| Colonne | Description |
|---|---|
| `transaction_id` | identifiant unique (32 caractères hexadécimaux) |
| `event_time` | horodatage UTC ISO 8601 (ms) |
| `client_id`, `card_id`, `merchant_id` | références (carte et marchand vides pour un virement) |
| `mcc`, `merchant_country` | code catégorie et pays du marchand |
| `channel` | `pos_chip`, `pos_contactless`, `ecommerce`, `atm`, `sepa_instant` |
| `entry_mode` | `chip`, `contactless`, `magstripe`, `card_not_present`, `token`, `transfer` |
| `amount`, `currency` | montant en EUR |
| `three_ds_authenticated` | authentification forte en ligne (DSP2), vide hors e-commerce |
| `device_id`, `ip_address` | appareil et IP pour les opérations à distance |
| `latitude`, `longitude` | position du commerce pour les paiements physiques |
| `beneficiary_iban` | IBAN du bénéficiaire d'un virement |
| `auth_status`, `decline_reason` | décision des règles actuelles de la banque |
| `wallet_type` | **à partir de juillet 2025 uniquement** (schéma v2) : `none`, `apple_pay`, `google_pay` |

### Événements applicatifs (exemple)

```json
{"event_id": "746bc3ac…", "event_type": "wallet_provisioning", "event_time": "2025-02-12T03:19:09.174Z",
 "client_id": "C00004615",
 "session": {"device_id": "dev_1041…", "device_os": "android", "app_version": "5.0.3",
             "ip_address": "188.141.22.86", "ip_country": "FR", "biometric_enabled": true},
 "geo": {"lat": 48.3941, "lon": -4.4601, "accuracy_m": 42},
 "payload": {"wallet": "google_pay", "card_last4": "2443"}}
```

`geo` est absent quand le client refuse la géolocalisation ; `biometric_enabled` et
`accuracy_m` n'existent qu'à partir de l'application 5.0 (déployée progressivement dès juillet).

## Réalisme volontairement intégré

- **Règles métier** : plafond sans contact de 50 EUR (sauf wallet), 3-D Secure au-delà de 30 EUR,
  saisonnalité (soldes, Black Friday, Noël), profils horaires, segments de clientèle.
- **Six scénarios de fraude** : test de cartes, carte clonée, prise de contrôle de compte,
  SIM swap, faux conseiller (la victime agit elle-même : très peu de signaux) et, à partir
  d'octobre 2025, **enrôlement frauduleux de wallet** (nouveau mode opératoire : dérive).
- **Labels imparfaits** : 10 % des fraudes jamais déclarées, délais de 1 à 90 jours,
  contestations infondées rejetées après enquête.
- **Faux positifs** des règles actuelles (clients légitimes refusés) et e-mails de plainte associés.
- **Évolution de schéma** en cours d'année (nouvelle colonne, nouveaux champs JSON).
- **Données personnelles dans le texte libre** des e-mails, à détecter et masquer.

## Qualité du code

```bash
ruff check . && ruff format --check .   # PEP 8, imports, sécurité (Bandit), docstrings
pytest                                  # 35 tests : identifiants, référentiels, fraudes, batch, flux
```

La génération est **reproductible** : même graine, mêmes fichiers, quel que soit le nombre
de processus (vérifié par un test).
