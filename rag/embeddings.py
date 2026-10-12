"""Embedding backends.

The embedder is hidden behind a small protocol so the vector store and
retriever never care which model produced the vectors. Two backends:

- ``sbert``: sentence-transformers model (default:
  ``bkai-foundation-models/vietnamese-bi-encoder``, trained for Vietnamese
  semantic similarity).
- ``hash``: deterministic hashing vector — no model download, used for tests
  and offline development. Retrieval quality is keyword-overlap based.
"""

from __future__ import annotations

import hashlib
import math
import re
from functools import lru_cache
from typing import Protocol

from rag.config import Settings, get_settings

_TOKEN_RE = re.compile(r"[\wÀ-ỹ]+", re.UNICODE)


class Embedder(Protocol):
    @property
    def dimension(self) -> int: ...

    def embed(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str, device: str = "cpu") -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name, device=device)

    @property
    def dimension(self) -> int:
        return self._model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]


class HashEmbedder:
    """Deterministic bag-of-words embedder for tests / offline runs."""

    def __init__(self, dimension: int = 384) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_one(text)

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self._dimension
        for token in _TOKEN_RE.findall(text.lower()):
            idx = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % self._dimension
            vec[idx] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        return [v / norm for v in vec] if norm else vec


def build_embedder(settings: Settings) -> Embedder:
    if settings.embedding_backend == "hash":
        return HashEmbedder()
    return SentenceTransformerEmbedder(
        settings.embedding_model, device=settings.embedding_device
    )


@lru_cache
def get_embedder() -> Embedder:
    return build_embedder(get_settings())
