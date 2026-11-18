#!/bin/bash
# System packages and databases PV-Mapper needs; idempotent, run as root by pvmapper-update.
set -euo pipefail
PKGS=""
need() { dpkg -s "$1" >/dev/null 2>&1 || PKGS="$PKGS $1"; }
need postgresql
need postgresql-16-postgis-3
need osmium-tool
if [ -n "$PKGS" ]; then
  echo "Installiere:$PKGS"
  apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq $PKGS
fi
systemctl enable --now postgresql >/dev/null 2>&1 || true
psqlc() { sudo -u postgres psql -qtAX "$@"; }
# PostGIS for ground-mounted PV: owner = service user (peer login over the local socket, no password); claude may read
for r in pvmapper claude; do
  id "$r" >/dev/null 2>&1 || continue
  [ "$(psqlc -c "SELECT 1 FROM pg_roles WHERE rolname='$r'")" = 1 ] || psqlc -c "CREATE ROLE $r LOGIN"
done
[ "$(psqlc -c "SELECT 1 FROM pg_database WHERE datname='pvmapper_ground'")" = 1 ] || sudo -u postgres createdb -O pvmapper pvmapper_ground
psqlc -d pvmapper_ground -c "CREATE EXTENSION IF NOT EXISTS postgis"
id claude >/dev/null 2>&1 && psqlc -c "GRANT pg_read_all_data TO claude" || true
echo "System: PostGIS und osmium bereit."
