# Ondine Bank – Détection de fraude en temps réel

Projet de fin d'études (RNCP38777, Architecte en intelligence artificielle).
**Ondine Bank** est une néobanque fictive (1 million de clients) qui veut détecter la fraude
aux paiements avant l'autorisation, en expliquant chaque décision et en respectant le RGPD,
la DSP2, la LCB-FT, DORA, l'AI Act et l'ISO 27001.

> Toutes les données sont synthétiques. Aucune personne ni aucun établissement réel n'est concerné.

## État d'avancement

| Étape | Contenu | État |
|---|---|---|
| 1. Fondations | Dépôt, infrastructure Docker (Kafka, stockage S3, PostgreSQL, supervision), simulateur de données | **fait, validé sur le PC** |
| 2. Pipelines de données | Ingestion, couches Bronze / Silver / Gold, qualité, pseudonymisation, traçabilité | **en cours** : bronze, silver, gold et flux validés ; orchestration et gouvernance livrées |
| 3. Modèle et déploiement | Entraînement, API de scoring, explicabilité, dérive, réentraînement, CI/CD | **en cours** : entraînement, suivi MLflow et explicabilité livrés |
| 4. Plans écrits | Gouvernance, infrastructure, pipeline | à venir |

## Architecture actuelle

```
 Simulateur ──batch──▶ data/landing (CSV, JSON, e-mails) ──▶ ingestion batch ──┐
     │                                                                           ├──▶ S3 : bronze ──▶ silver ──▶ gold ──▶ indicateurs
     └──stream──▶ Kafka (3 brokers, réplication 3) ──▶ stream-ingest ×2 ─────────┘
                        └──▶ topics d'erreurs *.dlq            [scoring temps réel, étape 3]

 PostgreSQL : audit (journal en ajout seul), gouvernance (registre RGPD, consentements, droits,
             incidents), ops (registre des traitements, lignage), analytics (indicateurs métier)
 Airflow : planifie et relance ingestion batch, silver, gold, indicateurs, droits RGPD, détection
          d'accès anormaux et rétention
 Prometheus + Grafana : métriques et alertes de tous les composants
```

| Service | Rôle | Adresse locale | Redondance / sécurité |
|---|---|---|---|
| `kafka-1..3` | flux temps réel | `localhost:9092-9094` | 3 brokers KRaft, réplication 3, `min.insync.replicas=2` |
| `kafka-ui` | exploration des topics | http://localhost:8080 | connexion par identifiant |
| `seaweed-*` | stockage objet S3 (data lake) | http://localhost:8333 | 2 serveurs de volumes (chaque objet en 2 copies), chiffrement AES-256-GCM, identités par usage |
| `postgres` | audit, gouvernance, bases Airflow / MLflow | `localhost:5432` | SCRAM-SHA-256, sommes de contrôle, rôles au moindre privilège |
| `postgres-backup` | sauvegarde quotidienne vérifiée | volume `postgres-backups` | rétention 7 jours |
| `prometheus` | métriques, règles d'alerte | http://localhost:9090 | rétention 15 jours |
| `grafana` | tableaux de bord | http://localhost:3000 | connexion obligatoire, pas d'inscription |
| `simulator-stream` | flux de transactions | http://localhost:8000/health | API de pilotage protégée par jeton |
| `stream-ingest` | Kafka → bronze (profil stream) | réseau interne, métriques :8001 | 2 répliques en groupe, commit après écriture |
| `pipelines-cli` | traitements batch (profil pipelines) | – | voir `pipelines/README.md` |
| `airflow-*` | orchestration (profil orchestration) | http://localhost:8081 | comptes nominatifs (admin, consultation), relances automatiques |
| `mlflow` | suivi des expériences et registre des modèles | http://localhost:5001 | authentification, compte de service distinct, seul détenteur d'une clé S3 pour les artefacts |
| `redis` | magasin de variables en ligne (profil api) | réseau interne | persistance activée, durées de vie alignées sur les fenêtres du modèle |
| `scoring-api` | décision de blocage d'un paiement (profil api) | http://localhost:8010/docs | image de service minimale, compte non administrateur, mode dégradé |
| `online-features` | Kafka → magasin en ligne (profil api) | réseau interne, métriques :9109 | groupe de consommateurs distinct de l'ingestion |

Tous les ports sont liés à `127.0.0.1` : rien n'est exposé sur le réseau local.

## Prérequis

- Windows 10/11 avec **Docker Desktop** (moteur WSL2) et **Git** ; Python 3.11+ pour les scripts.
- 32 Go de RAM recommandés : la plateforme utilise ~8 Go, Airflow jusqu'à ~10 Go pendant les recalculs, la génération complète ~10 Go pendant ~5 min (ne pas la lancer en même temps qu'un recalcul complet).
- Disque : ~8 Go d'images, 6,4 Go pour la zone d'arrivée brute, jusqu'à ~20 Go pour Kafka (rétention bornée), puis le data lake.
- Dans Docker Desktop > Settings > Resources : allouer au moins 20 Go de mémoire à WSL.

## Démarrage

```powershell
git clone <dépôt> ondine-fraud
cd ondine-fraud

# 1. Secrets : crée .env avec des mots de passe aléatoires (une seule fois)
python scripts/generate_env.py

# 2. Plateforme
docker compose up -d
docker compose ps          # kafka-init et seaweed-init doivent finir en « Exited (0) »

# 3. Historique de 12 mois (~5 min, 50 M de transactions)
docker compose --profile generate run --rm simulator-batch

# 4. Flux temps réel + ingestion en bronze (--build : reconstruit les images si le code a changé)
#    Arrêt quand il ne sert pas : docker compose --profile stream stop
docker compose --profile stream up -d --build

# 5. Orchestration : ingestion, silver, gold, indicateurs, RGPD et sécurité automatiques
docker compose --profile orchestration up -d --build

# 6. Modèle de fraude (entraînement ; suivi dans MLflow : http://localhost:5001)
# L'entraînement lit 50 millions de lignes : on arrête d'abord ce dont il n'a pas besoin
# (Kafka, Airflow, tableaux de bord), sans quoi la machine hôte peut manquer de mémoire.
docker compose stop kafka-1 kafka-2 kafka-3 kafka-ui grafana prometheus pushgateway `
  simulator-stream online-features scoring-api
docker compose --profile ml run --rm --build ml-cli train

# 7. API de scoring temps réel (contrat : http://localhost:8010/docs)
docker compose --profile api up -d --build
docker compose --profile api run --rm scoring-api referentials
```

### Mettre à jour le dépôt à partir d'un bundle livré

```powershell
git pull ..\ondine-fraud-historique.bundle main
```

Les identifiants de Kafka UI et Grafana sont dans `.env`. Tableau de bord Grafana :
**Ondine > Ondine - Flux temps réel et plateforme**.

## Arborescence

```
.
├── docker-compose.yml          plateforme complète (profils : stream, generate, pipelines, orchestration)
├── .env.example                modèle des secrets (le vrai .env n'est jamais versionné)
├── scripts/generate_env.py     génération de secrets forts
├── infra/
│   ├── kafka/                  création des topics
│   ├── seaweedfs/              démarrage S3 (identités, chiffrement), création des buckets
│   ├── postgres/               rôles, schémas audit et gouvernance, migrations, sauvegardes
│   ├── prometheus/             collecte et règles d'alerte
│   └── grafana/                source de données et tableau de bord provisionnés
├── simulator/                  générateur de données (voir simulator/README.md)
├── pipelines/                  traitements du data lake (voir pipelines/README.md)
├── ml/                         modèle de fraude (voir ml/README.md)
└── data/                       données générées (hors Git)
```

## Dépannage

| Symptôme | Cause probable | Solution |
|---|---|---|
| `/bin/sh^M: not found` | fins de ligne Windows sur les scripts | `git config core.autocrlf input`, puis recloner (le dépôt force LF via `.gitattributes`) |
| Kafka redémarre en boucle | mémoire WSL insuffisante | augmenter la mémoire allouée à WSL |
| `Variable manquante : S3_...` | `.env` absent | `python scripts/generate_env.py` |
| Mots de passe refusés après régénération de `.env` | les volumes gardent les anciens | `docker compose down -v` (efface les données) puis redémarrer |
| Génération interrompue | arrêt du PC, manque de mémoire | relancer avec `--resume` : seules les parts manquantes sont refaites |
| Disque qui se remplit | flux temps réel resté allumé | `docker compose --profile stream stop` ; la rétention Kafka est bornée à 6 h et 512 Mo par partition |
