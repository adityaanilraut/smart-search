import argparse
import asyncio
import csv
from datetime import UTC, datetime
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
import time

import httpx

from .index import REVISION, Encoder, Index


def revision():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def quality(index, queries, qrels):
    rows = []
    for method in ("bm25", "semantic", "hybrid"):
        hits, gains = [], []
        for query in queries:
            relevant = qrels.get(query["_id"], {})
            if not relevant:
                continue
            results = index.search(query["text"], 10, method)
            grades = [relevant.get(item["id"], 0) for item in results]
            hits.append(int(any(grade > 0 for grade in grades[:5])))
            ideal = dcg(sorted(relevant.values(), reverse=True)[:10])
            gains.append(dcg(grades) / ideal if ideal else 0)
        rows.append({
            "method": method, "queries": len(hits), "hits_at_5": sum(hits),
            "hit_at_5": statistics.mean(hits), "ndcg_at_10": statistics.mean(gains),
        })
    return rows


def dcg(grades):
    return sum((2**grade - 1) / math.log2(i + 2) for i, grade in enumerate(grades))


async def load(base, queries, rate, seconds, warm):
    async with httpx.AsyncClient(base_url=base, timeout=5, limits=httpx.Limits(max_connections=80)) as http:
        health = (await http.get("/healthz")).raise_for_status().json()
        if warm:
            for query in queries:
                response = await http.get("/search", params={"q": query})
                if not response.is_success:
                    raise RuntimeError("Cache prewarm failed")
            check = (await http.get("/search", params={"q": queries[0]})).raise_for_status().json()
            if not check["cached"]:
                raise RuntimeError("Warm benchmark requires Redis")
        else:
            # The operator runs this mode against a service started without REDIS_URL.
            params = {"q": queries[0]}
            await http.get("/search", params=params)
            if (await http.get("/search", params=params)).raise_for_status().json()["cached"]:
                raise RuntimeError("Uncached benchmark requires the Redis cache to be disabled")

        async def request(query, scheduled):
            await asyncio.sleep(max(0, scheduled - time.monotonic()))
            try:
                response = await http.get("/search", params={"q": query})
                body = response.raise_for_status().json()
                return (time.monotonic() - scheduled) * 1000, body["cached"], None
            except httpx.HTTPError as exc:
                return None, False, type(exc).__name__

        start = time.monotonic()
        total = rate * seconds
        results = await asyncio.gather(*[
            request(queries[i % len(queries)], start + i / rate) for i in range(total)
        ])
        elapsed = time.monotonic() - start
        latencies = sorted(ms for ms, _, _ in results if ms is not None)
        errors = {}
        for _, _, error in results:
            if error:
                errors[error] = errors.get(error, 0) + 1
        return {
            "target_qps": rate, "seconds": seconds, "requests": total,
            "successes": len(latencies), "errors": errors, "achieved_qps": len(latencies) / elapsed,
            "p95_ms": latencies[math.ceil(0.95 * len(latencies)) - 1] if latencies else None,
            "cache_hits": sum(cached for _, cached, _ in results), **health,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=Path("data/trec-covid"))
    parser.add_argument("--index", type=Path, default=Path("index/trec-covid"))
    parser.add_argument("--mode", choices=["quality", "uncached", "warm"], default="quality")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--seconds", type=int, default=30)
    parser.add_argument("--rate", type=int, default=50)
    args = parser.parse_args()
    queries = [json.loads(line) for line in (args.corpus / "queries.jsonl").read_text().splitlines()]
    if args.mode == "quality":
        qrels = {}
        with (args.corpus / "qrels/test.tsv").open() as file:
            for row in csv.DictReader(file, delimiter="\t"):
                qrels.setdefault(row["query-id"], {})[row["corpus-id"]] = int(row["score"])
        generation = (args.index / "CURRENT").read_text().strip()
        index = Index(args.index, generation, Encoder())
        data = {"generation": generation, "documents": index.manifest["documents"],
                "ranking": quality(index, queries, qrels)}
    else:
        data = asyncio.run(load(
            args.url, [q["text"] for q in queries], args.rate, args.seconds, args.mode == "warm",
        ))
    report = {
        "tested_commit": revision(), "environment": platform.platform(),
        "python": platform.python_version(), "model_revision": REVISION,
        "corpus": args.corpus.name, "timestamp_utc": datetime.now(UTC).isoformat(),
        "mode": args.mode, **data,
    }
    target = Path("results") / f"{args.mode}.json"
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n")
    print(target.read_text(), end="")


if __name__ == "__main__":
    main()
