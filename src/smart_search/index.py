"""Build and query immutable generations of the lexical and vector indexes."""

import fcntl
import json
import os
import re
import shutil
import sqlite3
import uuid
from contextlib import closing
from pathlib import Path
from urllib.parse import urlencode

import faiss
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
DIMENSIONS = 384
STOPWORDS = frozenset(
    "a an and are as at be been by did do does for from has have how in is it of on or "
    "the their this to was were what when where which who will with".split()
)


class Encoder:
    def __init__(self, threads: int = 2):
        def download(filename):
            return hf_hub_download(MODEL, filename, revision=REVISION)

        self.tokenizer = Tokenizer.from_file(download("tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=256)
        self.tokenizer.enable_padding()
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            download("onnx/model.onnx"), options, providers=["CPUExecutionProvider"]
        )

    def encode(self, texts):
        tokens = self.tokenizer.encode_batch(texts)
        mask = np.array([token.attention_mask for token in tokens], dtype=np.int64)
        vectors = self.session.run(
            None,
            {
                "input_ids": np.array([token.ids for token in tokens], dtype=np.int64),
                "attention_mask": mask,
                "token_type_ids": np.array([token.type_ids for token in tokens], dtype=np.int64),
            },
        )[0]
        pooled = (vectors * mask[:, :, None]).sum(axis=1) / mask.sum(axis=1)[:, None]
        pooled = np.ascontiguousarray(pooled, dtype=np.float32)
        faiss.normalize_L2(pooled)
        return pooled


def open_readonly(path):
    return sqlite3.connect(path.resolve().as_uri() + "?mode=ro&immutable=1", uri=True)


def build(source: Path, root: Path, encoder, batch_size=64):
    root.mkdir(parents=True, exist_ok=True)
    with (root / "build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        generation = uuid.uuid4().hex
        stage = root / generation
        stage.mkdir()
        try:
            count = build_generation(source, stage, encoder, batch_size)
        except BaseException:
            shutil.rmtree(stage)
            raise
        pointer = root / (generation + ".tmp")
        pointer.write_text(generation + "\n")
        os.replace(pointer, root / "CURRENT")
        return {"generation": generation, "documents": count}


def build_generation(source, stage, encoder, batch_size):
    faiss.omp_set_num_threads(min(4, os.cpu_count() or 1))
    vector = faiss.IndexHNSWFlat(DIMENSIONS, 24, faiss.METRIC_INNER_PRODUCT)
    vector.hnsw.efConstruction = 80
    vector.hnsw.efSearch = 80
    count = 0
    with closing(sqlite3.connect(stage / "docs.db")) as db, source.open(encoding="utf-8") as rows:
        db.executescript("""
            CREATE TABLE documents (
                id INTEGER PRIMARY KEY, external_id TEXT UNIQUE NOT NULL,
                title TEXT NOT NULL, body TEXT NOT NULL, url TEXT NOT NULL
            );
            CREATE VIRTUAL TABLE lexical USING fts5(
                title, body, content='documents', content_rowid='id', tokenize='porter unicode61'
            );
        """)
        batch = []
        for line in rows:
            doc = json.loads(line)
            doc_id = str(doc.get("_id") or doc["id"])
            title, body = doc.get("title", ""), doc["text"]
            url = doc.get("url") or "https://pubmed.ncbi.nlm.nih.gov/?" + urlencode({"term": title})
            if not url.startswith(("https://", "http://")):
                raise ValueError("Source URL must use HTTP or HTTPS")
            batch.append((doc_id, title, body, url))
            if len(batch) == batch_size:
                insert_batch(db, vector, encoder, batch)
                count += len(batch)
                if count % (batch_size * 100) == 0:
                    print(f"Indexed {count} documents", flush=True)
                batch.clear()
        if batch:
            insert_batch(db, vector, encoder, batch)
            count += len(batch)
        if count == 0:
            raise ValueError("Cannot publish an empty index")
        db.execute("INSERT INTO lexical(lexical) VALUES ('rebuild')")
        db.execute("INSERT INTO lexical(lexical, rank) VALUES ('integrity-check', 1)")
        db.commit()
    faiss.write_index(vector, str(stage / "vectors.faiss"))
    if faiss.read_index(str(stage / "vectors.faiss")).ntotal != count:
        raise ValueError("Vector index count differs from document count")
    (stage / "manifest.json").write_text(
        json.dumps({"documents": count, "model": MODEL, "revision": REVISION}) + "\n"
    )
    return count


def insert_batch(db, vector, encoder, batch):
    embeddings = encoder.encode([title + " " + body for _, title, body, _ in batch])
    if embeddings.shape != (len(batch), DIMENSIONS) or not np.isfinite(embeddings).all():
        raise ValueError("Invalid embeddings")
    db.executemany(
        "INSERT INTO documents(external_id, title, body, url) VALUES (?, ?, ?, ?)", batch
    )
    vector.add(embeddings)


def fuse(*rankings):
    scores = {}
    for ranking in rankings:
        for rank, row in enumerate(ranking, start=1):
            scores[row] = scores.get(row, 0.0) + 1 / (60 + rank)
    return sorted(scores, key=lambda row: (-scores[row], row))


class Index:
    def __init__(self, root, generation, encoder):
        self.generation = generation
        self.encoder = encoder
        stage = root / generation
        self.path = stage / "docs.db"
        self.manifest = json.loads((stage / "manifest.json").read_text())
        self.vector = faiss.read_index(str(stage / "vectors.faiss"))
        if self.vector.ntotal != self.manifest["documents"]:
            raise ValueError("Invalid index manifest")

    def rank(self, db, query, method, candidates):
        lexical = []
        semantic = []
        if method != "semantic":
            tokens = list(dict.fromkeys(re.findall(r"[^\W_]+", query.casefold())))
            words = ([token for token in tokens if token not in STOPWORDS] or tokens)[:24]
            if words:
                expression = " OR ".join(f'"{word}"' for word in words)
                lexical = [
                    row[0] for row in db.execute(
                        "SELECT rowid FROM lexical WHERE lexical MATCH ? "
                        "AND rank MATCH 'bm25(3,1)' ORDER BY rank LIMIT ?",
                        (expression, candidates),
                    )
                ]
        if method != "bm25":
            faiss.omp_set_num_threads(1)
            _, ids = self.vector.search(self.encoder.encode([query]), candidates)
            semantic = [int(i) + 1 for i in ids[0] if i >= 0]
        if method == "hybrid":
            return fuse(lexical, semantic)
        return lexical if method == "bm25" else semantic

    def search(self, query, k=5, method="hybrid"):
        with closing(open_readonly(self.path)) as db:
            ids = self.rank(db, query, method, max(50, k))[:k]
            if not ids:
                return []
            db.row_factory = sqlite3.Row
            placeholders = ",".join("?" for _ in ids)
            rows = db.execute(
                f"SELECT id, external_id, title, substr(body, 1, 600) AS excerpt, url "
                f"FROM documents WHERE id IN ({placeholders})", ids,
            )
            docs = {
                row["id"]: {
                    "id": row["external_id"], "title": row["title"],
                    "excerpt": row["excerpt"], "url": row["url"],
                }
                for row in rows
            }
            return [docs[row] for row in ids]
