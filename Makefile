SHELL := /bin/bash

.PHONY: up down rebuild logs ingest query

up:
	docker compose up -d --build

down:
	docker compose down

rebuild:
	docker compose build --no-cache api

logs:
	docker compose logs -f --tail=200 api

ingest:
	curl -s -X POST http://127.0.0.1:8000/ingest -H "Content-Type: application/json" -d '{"csv_paths": null, "overwrite": true}' | jq

query:
	python3 query_rag.py "$(Q)"


