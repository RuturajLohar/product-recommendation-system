# EliteRec Product Recommendation System

EliteRec is a focused product recommendation project. The current goal is simple:

```text
Search product name -> find closest product -> show similar product recommendations
```

The project is intentionally optimized for a college/demo deployment, not a full Amazon/Flipkart clone.

## Current Focus

- Product-only recommendation system
- Content-based recommendations
- PostgreSQL product catalog
- Qdrant vector search
- SentenceTransformer embeddings
- Clean React search UI
- Docker local development
- GCP-friendly deployment configuration

Collaborative filtering, demo users, cart flows, and analytics dashboards are not the main goal.

## Main App

The deployable app lives here:

```text
recommender_platform/
```

Important files:

```text
recommender_platform/
├── app/                         # FastAPI backend
├── frontend-react/              # React frontend
├── scripts/import_enriched_data.py
├── Dockerfile                   # backend Docker image
├── docker-compose.yml           # local full stack
├── requirements.txt             # backend dependencies
├── .env.example                 # safe environment template
├── .dockerignore                # Docker/GCP build cleanup
└── .gcloudignore                # gcloud upload cleanup
```

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React + Vite + Tailwind |
| Backend | FastAPI |
| Database | PostgreSQL |
| Vector DB | Qdrant |
| Embeddings | SentenceTransformers |
| Cache | Redis, optional |
| Deployment | Docker, Docker Compose, GCP Compute Engine or Cloud Run |

## Recommendation Flow

```text
User searches product name
        ↓
Backend searches matching product in PostgreSQL
        ↓
Matched product text is embedded
        ↓
Qdrant finds semantically similar products
        ↓
Backend returns top recommendations
        ↓
Frontend displays product cards
```

## Dataset Strategy

For the demo, do not import the full 2.2M product CSV immediately.

Recommended path:

```text
Start with 50k-100k high-quality products
Then scale to 300k+
Only import 2.2M when the deployment is stable
```

Why:

- DB insert is manageable.
- Embedding generation is the slowest part.
- Qdrant indexing takes time.
- A strong 100k sample is enough for a polished demo.

## Local Setup

Go to the platform folder:

```bash
cd recommender_platform
```

Create local env:

```bash
cp .env.example .env
```

Start the stack:

```bash
docker compose up -d --build
```

Open:

```text
http://127.0.0.1:5173
```

Backend health:

```text
http://127.0.0.1:8000/health/ready
```

## Import Enriched Products

After generating or placing the enriched CSV:

```bash
docker compose cp data/enriched_products_50k.csv api:/tmp/enriched_products_50k.csv
docker compose exec api python scripts/import_enriched_data.py --csv /tmp/enriched_products_50k.csv
```

Expected successful verification:

```text
postgres_rows: 50,000
distinct_product_ids: 50,000
qdrant_points: 50,000
missing_enriched_values: 0
```

## Environment Variables

Use:

```text
recommender_platform/.env.example
```

Never commit real `.env` files.

Most important variables:

```text
DATABASE_URL
QDRANT_HOST
QDRANT_PORT
QDRANT_URL
QDRANT_API_KEY
QDRANT_COLLECTION
REDIS_ENABLED
SBERT_MODEL
ALLOWED_ORIGINS
VITE_API_BASE
```

## GCP Deployment Options

### Option A: Beginner Recommended

Use one Compute Engine VM and Docker Compose.

This is the simplest for a college/demo project.

```text
GCP Compute Engine VM
    ├── React frontend
    ├── FastAPI backend
    ├── PostgreSQL
    ├── Qdrant
    └── Redis
```

Good starting VM:

```text
e2-standard-4
Ubuntu 22.04 LTS
100-200 GB disk
HTTP/HTTPS enabled
```

Deployment:

```bash
sudo apt update
sudo apt install -y docker.io docker-compose-plugin git
sudo usermod -aG docker $USER
git clone YOUR_REPO_URL
cd product-recommendation-system/recommender_platform
cp .env.example .env
docker compose up -d --build
```

Then import the dataset.

### Option B: More Cloud-Native

Use separated managed services.

```text
Frontend -> Cloud Run or Firebase Hosting
Backend  -> Cloud Run
Postgres -> Cloud SQL PostgreSQL
Qdrant   -> Qdrant Cloud or Compute Engine
CSV      -> Cloud Storage
Import   -> Cloud Run Job or one-time VM script
```

This is cleaner, but harder for a beginner.

## GCP Cost Safety

Before deploying:

1. Create a billing budget.
2. Add alerts at 50%, 80%, and 100%.
3. Stop the VM when not using the demo.
4. Start with 50k or 100k products.
5. Keep `USE_LLM_RERANKING=false` unless you intentionally want API costs.

## API Endpoints

```text
GET /health
GET /health/ready
GET /api/v1/items/?q=baby camera&limit=1
GET /api/v1/recommend/item/{asin}?limit=12
```

## Clean Deployment Notes

- `.env` files are ignored.
- Large datasets are ignored.
- Generated embeddings and model artifacts are ignored.
- Docker builds do not upload local CSV/XLSX files.
- Backend Dockerfile uses `PORT`, so it is Cloud Run compatible.
- Frontend has a production Dockerfile under `frontend-react/`.

## Project Summary

EliteRec is now best understood as:

```text
A content-based product recommendation engine using enriched product metadata,
semantic embeddings, PostgreSQL, Qdrant, FastAPI, and React.
```

That is the right scope for a strong, understandable, demo-ready project.
