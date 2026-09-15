#!/bin/sh
# Applique les migrations SQL non encore passées, dans l'ordre, chacune dans une transaction.
# Idempotent : lancé à chaque démarrage de la plateforme (service db-migrate).
set -eu

psql -v ON_ERROR_STOP=1 -d ondine -c \
  "CREATE TABLE IF NOT EXISTS public.schema_migrations (version TEXT PRIMARY KEY, applied_at TIMESTAMPTZ NOT NULL DEFAULT now());"

for file in $(ls /migrations/*.sql | sort); do
  version=$(basename "$file" .sql)
  applied=$(psql -tA -d ondine -c "SELECT 1 FROM public.schema_migrations WHERE version = '$version'")
  if [ "$applied" = "1" ]; then
    echo "déjà appliquée : $version"
    continue
  fi
  echo "application : $version"
  psql -v ON_ERROR_STOP=1 -1 -d ondine -f "$file" \
       -c "INSERT INTO public.schema_migrations (version) VALUES ('$version')"
done
echo "Migrations à jour."
