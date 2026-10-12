"""Retrieval: embed the question, query the vector store, filter by threshold."""

from __future__ import annotations

from rag.embeddings import Embedder
from rag.vectorstore import SearchResult, VectorStore


class Retriever:
    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        top_k: int = 4,
        score_threshold: float = 0.0,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._top_k = top_k
        self._threshold = score_threshold

    def retrieve(self, question: str, top_k: int | None = None) -> list[SearchResult]:
        embedding = self._embedder.embed_query(question)
        results = self._store.query(embedding, top_k or self._top_k)
        return [r for r in results if r.score >= self._threshold]
