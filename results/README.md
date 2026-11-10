# Results

The index contains all **171,332 documents** in BEIR TREC-COVID. Hybrid retrieval found a relevant result in the top five for all 50 judged queries.

| Ranking | Queries with a hit | Hit@5 | nDCG@10 |
| --- | ---: | ---: | ---: |
| BM25 | 49/50 | 98% | 0.604 |
| Semantic | 43/50 | 86% | 0.454 |
| Hybrid | 50/50 | 100% | 0.616 |

Both workloads offered 50 requests/sec for 30 seconds on an Apple Silicon Mac with 8 GB RAM. All requests succeeded. Both modes met the 200 ms p95 target.

| Mode | Achieved QPS | p95 latency | Successful requests |
| --- | ---: | ---: | ---: |
| Redis disabled | 49.87 | 96.9 ms | 1,500/1,500 |
| Redis warmed | 50.03 | 9.1 ms | 1,500/1,500 |

- [Ranking evaluation](quality.json)
- [Uncached load run](uncached.json)
- [Warmed Redis load run](warm.json)
- [Baseline before the concurrency and SQLite sorting changes](baseline/)

## Reproduce the evaluation

```sh
make corpus
uv run --locked python -m smart_search data/trec-covid/corpus.jsonl --index index/trec-covid
# Run Redis on localhost:6379, or set BENCH_REDIS_URL.
make bench
```

On Apple Silicon, the equivalent index build can use the GPU with `uv run --locked --extra apple python -m smart_search data/trec-covid/corpus.jsonl --index index/trec-covid --apple-gpu --batch-size 128`. Query serving uses ONNX CPU inference in both cases.

The BEIR TREC-COVID archive includes 50 judged test queries. Hit@5 counts queries with at least one document graded relevant among the first five. nDCG@10 uses gain $2^{relevance} - 1$ and treats unjudged documents as zero. Each HTTP workload sends 1,500 scheduled requests at 50 QPS through the same endpoint. The uncached run disables Redis and the warm run primes all 50 query keys. Latency is measured from the scheduled start to HTTP completion and includes queuing. Answer generation is separate from the search benchmark.
