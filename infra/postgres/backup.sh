#!/bin/sh
# Sauvegarde logique complète de PostgreSQL à intervalle régulier, avec rétention.
# Chaque sauvegarde est vérifiée (archive gzip lisible) avant la purge des anciennes.
set -eu

mkdir -p /backups
while true; do
  stamp=$(date -u +%Y%m%dT%H%M%SZ)
  target="/backups/ondine_${stamp}.sql.gz"
  if pg_dumpall --clean --if-exists | gzip -6 > "${target}.tmp" && gzip -t "${target}.tmp"; then
    mv "${target}.tmp" "$target"
    echo "$(date -u +%FT%TZ) sauvegarde OK : $target ($(du -h "$target" | cut -f1))"
    find /backups -name 'ondine_*.sql.gz' -mtime +"${BACKUP_RETENTION_DAYS}" -delete
  else
    rm -f "${target}.tmp"
    echo "$(date -u +%FT%TZ) ÉCHEC de la sauvegarde" >&2
  fi
  sleep "${BACKUP_INTERVAL_S}"
done
