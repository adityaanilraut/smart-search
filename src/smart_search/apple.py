"""Apple GPU inference for the pinned MiniLM BERT encoder."""

import faiss
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

from .index import MODEL, REVISION


class AppleEncoder:
    def __init__(self):
        def download(name):
            return hf_hub_download(MODEL, name, revision=REVISION)

        self.tokenizer = Tokenizer.from_file(download("tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=256)
        self.tokenizer.enable_padding(pad_to_multiple_of=16)
        weights = mx.load(download("model.safetensors"))
        self.weights = {name: value.astype(mx.float16) for name, value in weights.items()}
        mx.eval(*self.weights.values())

    def affine(self, x, name):
        return x @ self.weights[name + ".weight"].T + self.weights[name + ".bias"]

    def norm(self, x, name):
        return mx.fast.layer_norm(
            x, self.weights[name + ".weight"], self.weights[name + ".bias"], 1e-12,
        )

    def layer(self, hidden, mask, layer):
        base = f"encoder.layer.{layer}"
        batch, length, _ = hidden.shape

        def project(name):
            value = self.affine(hidden, base + ".attention.self." + name)
            return value.reshape(batch, length, 12, 32).transpose(0, 2, 1, 3)

        attention = mx.fast.scaled_dot_product_attention(
            project("query"), project("key"), project("value"), scale=32**-0.5, mask=mask,
        )
        attention = attention.transpose(0, 2, 1, 3).reshape(batch, length, 384)
        attention = self.affine(attention, base + ".attention.output.dense")
        hidden = self.norm(hidden + attention, base + ".attention.output.LayerNorm")
        intermediate = nn.gelu(self.affine(hidden, base + ".intermediate.dense"))
        return self.norm(hidden + self.affine(intermediate, base + ".output.dense"), base + ".output.LayerNorm")

    def encode(self, texts):
        tokens = self.tokenizer.encode_batch(texts)
        ids = mx.array([token.ids for token in tokens])
        mask = mx.array([token.attention_mask for token in tokens], dtype=mx.bool_)
        positions = mx.arange(ids.shape[1])
        weights = self.weights
        hidden = (
            weights["embeddings.word_embeddings.weight"][ids]
            + weights["embeddings.position_embeddings.weight"][positions]
            + weights["embeddings.token_type_embeddings.weight"][0]
        )
        hidden = self.norm(hidden, "embeddings.LayerNorm")
        for layer in range(6):
            hidden = self.layer(hidden, mask[:, None, None, :], layer)
        pooled = (hidden.astype(mx.float32) * mask[:, :, None]).sum(1) / mask.sum(1)[:, None]
        vectors = np.array(pooled, dtype=np.float32, order="C")
        faiss.normalize_L2(vectors)
        return vectors
