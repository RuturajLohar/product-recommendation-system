SHELL := /bin/sh

.PHONY: up down down-v teardown watch ingest ingest-100k ingest-all seed-demo frontend logs

up:
	docker compose up -d --build db qdrant redis api

down:
	docker compose down --timeout 15

down-v:
	docker compose down --timeout 15 -v

# Use when compose hangs: forces removal of named containers after compose down.
teardown:
	./scripts/docker-teardown.sh

teardown-v:
	./scripts/docker-teardown.sh --volumes

watch:
	docker compose up --build --watch

ingest:
	docker compose run --rm api python scripts/ingest_data.py --csv /app/data/amz_uk_processed_data.csv --sample-size $${SAMPLE_SIZE:-25000} --reset

ingest-100k:
	docker compose run --rm api python scripts/ingest_data.py --csv /app/data/amz_uk_processed_data.csv --sample-size 100000 --sampling stratified --random-seed 42 --reset

ingest-all:
	docker compose run --rm api python scripts/ingest_data.py --csv /app/data/amz_uk_processed_data.csv --sample-size 0 --reset

seed-demo:
	docker compose exec api python scripts/seed_demo_users.py

frontend:
	cd frontend-react && npm install && npm run dev

logs:
	docker compose logs -f api db qdrant redis
