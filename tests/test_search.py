import hashlib
from pathlib import Path

from fastapi.testclient import TestClient
import httpx
import numpy as np
import pytest
import redis

from smart_search.api import SearchService, create_app
from smart_search.index import DIMENSIONS, build, fuse

SAMPLE = Path(__file__).parents[1] / "data/sample.jsonl"


class TestEncoder:
    __test__ = False

    def encode(self, texts):
        vectors = np.zeros((len(texts), DIMENSIONS), dtype=np.float32)
        for i, text in enumerate(texts):
            for word in text.lower().split():
                slot = int(hashlib.sha256(word.encode()).hexdigest(), 16) % DIMENSIONS
                vectors[i, slot] += 1
        norms = np.linalg.norm(vectors, axis=1)
        return vectors / np.maximum(norms, 1)[:, None]


class MemoryCache:
    def __init__(self):
        self.data = {}

    def get(self, key):
        return self.data.get(key)

    def setex(self, key, _expiry, value):
        self.data[key] = value


@pytest.fixture
def service(tmp_path):
    encoder = TestEncoder()
    build(SAMPLE, tmp_path, encoder)
    return SearchService(tmp_path, encoder, MemoryCache())


def test_fusion_rewards_overlap():
    assert fuse([4, 8, 9], [9, 12, 8])[:2] == [9, 8]


@pytest.mark.parametrize("method", ["bm25", "semantic", "hybrid"])
def test_retrieval_returns_ranked_sources(service, method):
    result = service.search("Raft consensus log replication", 1, method)
    assert result["results"][0]["id"] == "raft"
    assert result["results"][0]["url"] == "https://raft.github.io/raft.pdf"


def test_reindex_cache_is_generation_scoped(service):
    old = service.search("Raft", 5, "hybrid")
    assert not old["cached"]
    assert service.search("Raft", 5, "hybrid")["cached"]
    generation = build(SAMPLE, service.root, service.encoder)["generation"]
    new = service.search("Raft", 5, "hybrid")
    assert not new["cached"]
    assert new["generation"] == generation != old["generation"]


def test_failed_build_keeps_previous_generation(service):
    previous = service.index().generation
    bad = service.root / "broken.jsonl"
    bad.write_text('{"id":"duplicate","text":"first"}\n{"id":"duplicate","text":"second"}\n')
    with pytest.raises(Exception, match="UNIQUE constraint failed"):
        build(bad, service.root, service.encoder)
    assert (service.root / "CURRENT").read_text().strip() == previous
    assert service.search("Raft", 5, "hybrid")["results"]


def test_broken_pointer_preserves_loaded_index(service):
    previous = service.index().generation
    (service.root / "CURRENT").write_text("not a generation")
    assert service.index().generation == previous


def test_cache_failure_degrades_to_search(service, monkeypatch):
    def fail(*_args):
        raise redis.RedisError("offline")

    monkeypatch.setattr(service.cache, "get", fail)
    monkeypatch.setattr(service.cache, "setex", fail)
    assert service.search("Raft", 5, "hybrid")["results"]


def test_http_validation_and_opt_in(service):
    with TestClient(create_app(service)) as client:
        assert client.get("/healthz").json()["documents"] == 4
        assert client.get("/search", params={"q": "   "}).status_code == 422
        assert client.get("/search", params={"q": "raft", "k": 21}).status_code == 422
        assert client.get("/search", params={"q": "raft", "method": "unknown"}).status_code == 422
        assert client.post("/answer", json={"query": "What is Raft?"}).status_code == 503


def test_answer_returns_numbered_retrieval_links(service, monkeypatch):
    def response(url, **kwargs):
        assert url == "http://ollama:11434/api/chat"
        assert kwargs["json"]["messages"][1]["content"].find("Consensus Algorithm") >= 0
        return httpx.Response(
            200, json={"message": {"content": "Raft orders a replicated log [1]."}},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx, "post", response)
    with TestClient(create_app(service, "http://ollama:11434")) as client:
        result = client.post("/answer", json={"query": "Raft consensus replicated log"}).json()
        assert result["answer"].endswith("[1].")
        assert result["sources"][0]["number"] == 1
        assert result["sources"][0]["url"] == "https://raft.github.io/raft.pdf"
