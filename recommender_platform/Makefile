SHELL := /bin/sh

.PHONY: up down watch ingest frontend logs

up:
	docker compose up -d --build db qdrant redis api

down:
	docker compose down

watch:
	docker compose up --build --watch

ingest:
	docker compose run --rm api python scripts/ingest_data.py --csv /app/data/amz_uk_processed_data.csv --sample-size $${SAMPLE_SIZE:-25000} --reset

frontend:
	cd frontend-react && npm install && npm run dev

logs:
	docker compose logs -f api db qdrant redis
