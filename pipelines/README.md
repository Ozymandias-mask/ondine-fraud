# Pipelines de données Ondine Bank

Code des traitements du data lake (bloc 3). Moteur de calcul : **DuckDB** (SQL analytique
en colonnes, parallèle, capable de déborder sur disque) ; stockage : **Parquet sur S3**.

> État : ingestion zone d'arrivée → bronze livrée. Silver, gold, flux Kafka, orchestration
> Airflow et droits RGPD arrivent dans les prochaines livraisons.

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

## Lancer l'ingestion (sans Airflow)

```powershell
docker compose up -d                                   # applique aussi les migrations (db-migrate)
docker compose --profile pipelines build pipelines-cli
docker compose --profile pipelines run --rm pipelines-cli ingest --dataset transactions --limit 1
docker compose --profile pipelines run --rm pipelines-cli ingest --dataset all
```

## Tests

```powershell
$env:TEST_PG_ADMIN_DSN = "postgresql://<admin>:<mdp>@127.0.0.1:5432/postgres"
pytest pipelines/tests
```

Les tests créent une base temporaire à partir des vrais scripts SQL du dépôt et une petite
zone d'arrivée avec le simulateur : intégrité, complétude, quarantaine, incrémentalité, audit.
