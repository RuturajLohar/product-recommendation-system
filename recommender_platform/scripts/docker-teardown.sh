#!/bin/sh
# Tear down EliteRec compose stack. If the Docker daemon returns HTTP 500,
# restart Docker Desktop — see docs/DOCKER_ENGINE_FIX.md
#
# Usage:
#   ./scripts/docker-teardown.sh           # compose down + force-remove rec_* containers
#   ./scripts/docker-teardown.sh -v        # also remove named volumes (postgres_data, qdrant_data)
set -u

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR" || exit 1

echo "==> Checking Docker daemon..."
if ! docker info >/dev/null 2>&1; then
  echo ""
  echo "Docker daemon is not responding (docker info failed)."
  echo "Fix: Quit Docker Desktop completely, reopen it, wait until 'Docker is running', then run this script again."
  echo "Details: $ROOT_DIR/docs/DOCKER_ENGINE_FIX.md"
  echo ""
  exit 1
fi

echo "==> docker compose down --timeout 15 $@"
docker compose down --timeout 15 "$@" || true

echo "==> Force-remove project containers (if any remain)"
for c in rec_api rec_frontend rec_postgres rec_qdrant rec_redis; do
  if docker inspect "$c" >/dev/null 2>&1; then
    docker rm -f "$c" && echo "   removed $c" || true
  fi
done

echo "==> Compose status"
docker compose ps -a 2>/dev/null || true
echo "Done."
