# Results

## Reproduce the evaluation

```sh
make corpus
uv run --locked python -m smart_search data/trec-covid/corpus.jsonl --index index/trec-covid
# Run Redis on localhost:6379, or set BENCH_REDIS_URL.
make bench
```

On Apple Silicon, the equivalent index build can use the GPU with `uv run --locked --extra apple python -m smart_search data/trec-covid/corpus.jsonl --index index/trec-covid --apple-gpu --batch-size 128`. Query serving uses ONNX CPU inference in both cases.

The BEIR TREC-COVID archive includes 50 judged test queries. Hit@5 counts queries with at least one document graded relevant among the first five. nDCG@10 uses graded relevance and treats unjudged documents as zero. Each HTTP workload sends 1,500 scheduled requests at 50 QPS through the same endpoint. The uncached run disables Redis and the warm run primes all 50 query keys. Latency is measured from the scheduled start to HTTP completion and includes queuing. Answer generation is separate from the search benchmark.
