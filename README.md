# Product Recommendation System (EliteRec)

This repo contains **EliteRec**, a full-stack product recommendation platform:

- **Backend**: FastAPI + Postgres + Qdrant + Redis (`recommender_platform/app/`)
- **Frontend**: React (Vite + Tailwind) (`recommender_platform/frontend-react/`)
- **Catalog**: loaded from `amz_uk_processed_data.csv` (no synthetic seeding)

## Quickstart (Docker)

From the repo root:

```bash
cd recommender_platform
cp .env.example .env
docker compose down -v
docker compose up -d --build db qdrant redis api
docker compose run --rm api python scripts/ingest_data.py --csv /app/data/amz_uk_processed_data.csv
```

Start the frontend:

```bash
cd recommender_platform/frontend-react
npm i
npm run dev
```

Open:

- API docs: `http://127.0.0.1:8000/docs`
- Frontend: `http://127.0.0.1:5173`

## Professional Project Layout

```text
recommender_platform/
  app/                   # FastAPI backend (API, DB, ML, services)
  scripts/               # Operational scripts (ingestion/seeding)
  frontend-react/        # React frontend
  data/                  # Local mounted data files (gitignored)
  docker-compose.yml     # Multi-service local stack
  .env.example           # Backend/infra env template
  Makefile               # Common dev commands
```

## One-command local dev (watch mode)

```bash
cd recommender_platform
cp .env.example .env
docker compose up --build --watch
```

## Optional helper commands

```bash
cd recommender_platform
make up
make ingest
make logs
```

## Useful API endpoints

- `GET /health`
- `GET /api/v1/recommend/trending?limit=8`
- `GET /api/v1/recommend/user/{USER_ID}?limit=5`
- `GET /api/v1/items?q=echo&limit=8`
- `POST /api/v1/events` body: `{ "user_id": "USER_0", "asin": "B09B96TG33", "type": "click" }`

## Docker won’t stop / `compose down` errors

If commands hang or you see **HTTP 500** from Docker, the **Docker Desktop engine** needs a restart—not a repo change. See [recommender_platform/docs/DOCKER_ENGINE_FIX.md](recommender_platform/docs/DOCKER_ENGINE_FIX.md). After Docker is healthy:

```bash
cd recommender_platform
make teardown      # or: ./scripts/docker-teardown.sh
make teardown-v    # or: ./scripts/docker-teardown.sh -v (removes DB/Qdrant volumes)
```

## Notes

- **Dev:** With `npm run dev`, the frontend calls **`/api/v1`** on the Vite server; Vite **proxies** `/api` to `http://127.0.0.1:8000`, so you avoid cross-origin issues. Start the API on `:8000` (Docker or local `uvicorn`).
- **Override:** Set `VITE_API_BASE` to a full URL, or set `VITE_PROXY_API` if the API is not on `127.0.0.1:8000`.
- Ingestion is **idempotent** (upserts by `asin`) and uses deterministic Qdrant point IDs.
- Local secrets/config now come from `recommender_platform/.env` (copy from `.env.example`).
