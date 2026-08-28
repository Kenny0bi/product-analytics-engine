#!/bin/sh
# Seed the shared data volume on first boot, then run the container command.
#
# The database is generated at runtime (not baked into the image) because
# the compose volume mounted at /app/data would shadow anything baked in,
# and a 50K-user build step makes `docker build` take 20+ minutes. Set
# PAE_SKIP_SEED=1 (the dagster service does) to wait for the analytics
# service to seed instead of racing it.
set -e

DB_PATH="${PAE_DB_PATH:-/app/data/analytics.duckdb}"
SEED_USERS="${PAE_SEED_USERS:-10000}"

if [ "${PAE_SKIP_SEED:-0}" != "1" ] && [ ! -f "$DB_PATH" ]; then
    echo "No database at $DB_PATH; generating ${SEED_USERS} users (one-time)..."
    python -m src.data.cli generate --num-users "$SEED_USERS" --seed 42
    python -m src.main analyze
fi

exec "$@"
