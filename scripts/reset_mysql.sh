#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if ! command -v docker &>/dev/null; then
  echo "[reset-mysql] docker is required but was not found in PATH" >&2
  exit 1
fi

COMPOSE_CMD=${COMPOSE_CMD:-"docker compose"}

# Shut everything down and remove dangling volumes so the mysql_data volume gets recreated
$COMPOSE_CMD down --volumes --remove-orphans

# Bring the MySQL service back up so it can recreate a clean data directory
$COMPOSE_CMD up -d mysql

cat <<'MSG'
MySQL has been restarted with a fresh data directory. The first startup can take a
few seconds while the schema and seed data are replayed from database/init.sql
and database/seed.sql. Once the mysql container reports "healthy" you can
restart the rest of the stack with:

    $COMPOSE_CMD up -d
MSG
