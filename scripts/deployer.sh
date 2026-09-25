#!/usr/bin/env bash
# Mise en service de la plateforme, avec retour arrière automatique.
#
# Le flux GitHub Actions appelle ce script ; il est aussi exécutable à la main sur la machine
# de production. La logique de déploiement n'a rien à faire dans un fichier YAML : on doit
# pouvoir la lire, la relancer et la déboguer sans passer par l'orchestrateur.
#
# Principe : les images en service sont d'abord **marquées** `:precedent`. Si la plateforme
# ne répond pas après la bascule, on remet ces images-là et on redémarre. Un déploiement qui
# ne sait pas revenir en arrière n'est pas un déploiement, c'est un pari.
set -euo pipefail

BUILD="${ONDINE_BUILD:-$(git rev-parse HEAD)}"
COURT="${BUILD:0:7}"
IMAGES=(ondine/pipelines ondine/api ondine/simulator ondine/mlflow)
PROFILS=(--profile api --profile orchestration --profile flux)

echo "== Version déployée : ${COURT} ($(git describe --tags --always 2>/dev/null || echo 'sans étiquette'))"

marquer() {
    # $1 = source, $2 = destination. Une image absente n'est pas une erreur : au premier
    # déploiement, il n'y a pas de version précédente à conserver.
    for image in "${IMAGES[@]}"; do
        docker image inspect "${image}:$1" >/dev/null 2>&1 && docker tag "${image}:$1" "${image}:$2"
    done
    return 0
}

sonder() {
    # $1 = adresse, $2 = nom lisible. Vingt tentatives espacées de 5 s : un service qui
    # charge un modèle depuis le registre met plus longtemps qu'un simple démarrage.
    for _ in $(seq 1 20); do
        if curl -fsS --max-time 5 "$1" >/dev/null 2>&1; then
            echo "   $2 : en service"
            return 0
        fi
        sleep 5
    done
    echo "::error::$2 ne répond pas ($1)"
    return 1
}

echo "== 1/5 Conservation des images en service"
marquer local precedent

echo "== 2/5 Construction (${COURT})"
ONDINE_BUILD="${BUILD}" docker compose "${PROFILS[@]}" build
# Nom de version des images : l'empreinte du commit. « latest » ne dit pas quel code tourne.
for image in "${IMAGES[@]}"; do
    docker image inspect "${image}:local" >/dev/null 2>&1 && docker tag "${image}:local" "${image}:${COURT}"
done

echo "== 3/5 Migrations de base de données"
docker compose up --exit-code-from db-migrate db-migrate

echo "== 4/5 Bascule"
ONDINE_BUILD="${BUILD}" docker compose "${PROFILS[@]}" up -d --remove-orphans

echo "== 5/5 Vérification de la mise en service"
if sonder http://localhost:8010/pret "API de scoring" \
    && sonder http://localhost:5001/health "MLflow" \
    && sonder http://localhost:9090/-/healthy "Prometheus"; then
    echo "== Déploiement ${COURT} réussi"
    exit 0
fi

echo "::warning::Mise en service en échec : retour à la version précédente"
marquer precedent local
docker compose "${PROFILS[@]}" up -d --remove-orphans
sonder http://localhost:8010/pret "API de scoring (version précédente)"
echo "::error::Déploiement ${COURT} annulé, la version précédente est de nouveau en service"
exit 1
