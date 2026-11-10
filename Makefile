.PHONY: demo test bench corpus
demo:
	docker compose up --build -d --wait

test:
	uv run --locked pytest -q

corpus:
	./scripts/fetch-beir.sh

bench:
	uv run --locked python -m smart_search.benchmark
	./scripts/load.sh
