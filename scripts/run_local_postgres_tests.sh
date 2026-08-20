#!/usr/bin/env bash
set -e

# ==============================================================================
# Supabase Easy RAG: Local PostgreSQL + pgvector Live Integration Test Runner
# ==============================================================================

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "======================================================================"
echo "  🐘 Supabase Easy RAG — Local PostgreSQL Integration Test Runner"
echo "======================================================================"

# 1. Check if Docker is available
if ! command -v docker >/dev/null 2>&1; then
    echo "⚠️ Docker is not found. Please install Docker or start PostgreSQL manually."
    exit 1
fi

# 2. Start PostgreSQL container with pgvector
echo "\n[1/4] Starting PostgreSQL 16 container with pgvector..."
docker compose -f docker-compose.test.yml up -d

# 3. Wait for PostgreSQL to become healthy
echo "\n[2/4] Waiting for PostgreSQL to be ready..."
MAX_RETRIES=20
COUNT=0
until docker compose -f docker-compose.test.yml ps --status running | grep -q "easy_rag_test_postgres" && \
      docker compose -f docker-compose.test.yml exec -T postgres pg_isready -U postgres -d postgres >/dev/null 2>&1; do
    COUNT=$((COUNT + 1))
    if [ "$COUNT" -ge "$MAX_RETRIES" ]; then
        echo "❌ Timed out waiting for PostgreSQL container to start."
        exit 1
    fi
    sleep 1
    echo -n "."
done
echo " ✓ PostgreSQL is ready!"

# 4. Prepare local environment & apply standard Supabase Schema + Functions
echo "\n[3/4] Initializing local test scaffolding and applying standard Supabase migrations..."
docker compose -f docker-compose.test.yml exec -T postgres psql -U postgres -d postgres -f /docker-entrypoint-initdb.d/00_init_supabase_shim.sql
docker compose -f docker-compose.test.yml exec -T postgres psql -U postgres -d postgres -f /docker-entrypoint-initdb.d/01_schema.sql
docker compose -f docker-compose.test.yml exec -T postgres psql -U postgres -d postgres -f /docker-entrypoint-initdb.d/02_functions.sql
echo " ✓ Migrations applied successfully!"

# 5. Run Live Integration Tests
echo "\n[4/4] Running live test suite against local PostgreSQL instance..."
export POSTGRES_URL="postgresql://postgres:postgres@localhost:5432/postgres"

if [ -f ".venv/bin/python" ]; then
    .venv/bin/python -m unittest discover tests -v
else
    python3 -m unittest discover tests -v
fi

echo "\n======================================================================"
echo "  ✅ All live PostgreSQL integration & RAG tests passed successfully!"
echo "======================================================================"
