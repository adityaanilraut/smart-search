# Smart Search

A hybrid document search service that combines SQLite FTS5 BM25 and FAISS vector retrieval. It includes a FastAPI interface, Redis caching, and optional source-linked answers through Ollama.

## Quick start

Start the demo with Docker and Docker Compose running.

```sh
make demo
curl --get --data-urlencode 'q=Raft consensus' http://localhost:8000/search
```

Open [API documentation](http://localhost:8000/docs) to try all endpoints. The sample corpus is in [data/sample.jsonl](data/sample.jsonl). A JSONL input row contains an `id` (or BEIR `_id`), `title`, `text`, and an optional `url`. BEIR entries without URLs use a PubMed title lookup.

`GET /search` supports `q`, `k` (1–20), and `method` (`bm25`, `semantic`, or `hybrid`). For generated answers, run Ollama with `ollama pull qwen2.5:1.5b`, set `OLLAMA_URL` to the Ollama address reachable by the API, and recreate the API container. On Docker Desktop with Ollama running on the host, use `http://host.docker.internal:11434`.

```sh
curl -X POST http://localhost:8000/answer \
  -H 'Content-Type: application/json' \
  -d '{"query":"How does Raft replicate a log?"}'
```

## How it works

```text
Query → SQLite FTS5 BM25 ──────┐
      → MiniLM → FAISS HNSW ──┴→ reciprocal rank fusion → source excerpts
                                                     └─→ optional Ollama answer
```

MiniLM encodes each title and abstract up to 256 tokens. FAISS indexes the normalized 384-dimensional vectors, and hybrid retrieval combines the top 50 candidates from each method. Redis cache keys include the index generation. A cache outage falls back to retrieval.

The Compose worker rebuilds the index every night at 02:00 UTC. To use your own corpus, put it under `data/` and start with `SOURCE_FILE=/data/your-corpus.jsonl make demo`. The worker validates a new generation before atomically updating the active pointer. The API keeps serving the prior index during a rebuild. Earlier generations remain on disk for rollback. Stop the stack before pruning generations and keep the directory named in `CURRENT`.

## Verification

Use Python 3.12 and [uv](https://docs.astral.sh/uv/). Run `make test` for retrieval, cache isolation, failed reindex recovery, request validation, and mocked answer citation checks. See [results and reproduction commands](results/README.md) for the full-corpus benchmark.

Run `docker compose down` to stop the demo.
