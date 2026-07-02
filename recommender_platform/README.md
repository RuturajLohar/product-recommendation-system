# EliteRec Platform

This folder contains the deployable product recommendation platform.

## What This Service Does

```text
Product name search -> matched product -> content-based recommendations
```

The app uses:

- FastAPI backend
- React frontend
- PostgreSQL product catalog
- Qdrant vector database
- SentenceTransformer embeddings
- Optional Redis cache

## Local Docker Run

```bash
cp .env.example .env
docker compose up -d --build
```

Frontend:

```text
http://127.0.0.1:5173
```

Backend:

```text
http://127.0.0.1:8000/health/ready
```

## Import Products

```bash
docker compose cp data/enriched_products_50k.csv api:/tmp/enriched_products_50k.csv
docker compose exec api python scripts/import_enriched_data.py --csv /tmp/enriched_products_50k.csv
```

## GCP Beginner Deployment

The easiest deployment is one Compute Engine VM running Docker Compose.

```bash
sudo apt update
sudo apt install -y docker.io docker-compose-plugin git
git clone YOUR_REPO_URL
cd product-recommendation-system/recommender_platform
cp .env.example .env
docker compose up -d --build
```

Use this for the first live demo.

## GCP Cloud-Native Deployment

For a more advanced setup:

```text
Backend  -> Cloud Run
Frontend -> Cloud Run or Firebase Hosting
Postgres -> Cloud SQL PostgreSQL
Qdrant   -> Qdrant Cloud or Compute Engine
CSV      -> Cloud Storage
Import   -> Cloud Run Job or temporary VM
```

## Important Env Vars

```text
DATABASE_URL
QDRANT_HOST
QDRANT_PORT
QDRANT_URL
QDRANT_API_KEY
QDRANT_COLLECTION
REDIS_ENABLED
ALLOWED_ORIGINS
SBERT_MODEL
VITE_API_BASE
```

Do not commit real `.env` files.
