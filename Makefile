.PHONY: help build up down logs demo ingest fetch splits stats shell clean

help:
	@echo "Targets:"
	@echo "  build       Build all images"
	@echo "  demo        Seed synthetic builds so the UI has data, then bring up api+web"
	@echo "  ingest      Load data/builds.csv into the DB (one-shot job)"
	@echo "  fetch       Fetch Dockerfiles from GitHub/GitLab (one-shot job)"
	@echo "  splits      Assign train/val/test/gold splits (one-shot job)"
	@echo "  stats       Print corpus stats"
	@echo "  up / down   Start / stop api+web"
	@echo "  logs        Tail api+web logs"
	@echo "  shell       sqlite3 shell into the DB volume"
	@echo "  clean       Stop everything and delete the DB volume"

build:
	docker compose build

up:
	docker compose up -d api web
	@echo "UI:  http://localhost:8501"
	@echo "API: http://localhost:8000/docs"

down:
	docker compose down

logs:
	docker compose logs -f api web

demo: build
	docker compose run --rm ingest demo-seed
	docker compose run --rm splits
	$(MAKE) up

ingest:
	docker compose run --rm ingest

fetch:
	docker compose run --rm fetch

splits:
	docker compose run --rm splits

stats:
	docker compose run --rm ingest stats

shell:
	docker compose run --rm --entrypoint python ingest -c "import sqlite3,code; con=sqlite3.connect('/data/ynobuild.db'); con.row_factory=sqlite3.Row; code.interact(local={'con':con})"

clean:
	docker compose down -v