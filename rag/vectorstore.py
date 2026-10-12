"""Vector store abstraction + ChromaDB implementation.

``VectorStore`` is a protocol, so swapping ChromaDB for Qdrant/pgvector later
only means writing another implementation — retrieval code stays untouched.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Protocol

from rag.chunking import Chunk


@dataclass
class SearchResult:
    text: str
    metadata: dict
    score: float  # cosine similarity in [-1, 1]; higher is better


class VectorStore(Protocol):
    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> int: ...

    def query(self, embedding: list[float], top_k: int) -> list[SearchResult]: ...

    def count(self) -> int: ...

    def reset(self) -> None: ...


class ChromaStore:
    def __init__(self, persist_dir: str, collection_name: str) -> None:
        import chromadb

        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection_name = collection_name
        self._collection = self._client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

    @staticmethod
    def _chunk_id(chunk: Chunk) -> str:
        key = f"{chunk.metadata.get('source', '')}:{chunk.metadata.get('chunk', '')}:{chunk.text}"
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> int:
        if not chunks:
            return 0
        # Deterministic ids -> re-ingesting the same content upserts instead of duplicating.
        self._collection.upsert(
            ids=[self._chunk_id(c) for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[c.metadata for c in chunks],
            embeddings=embeddings,
        )
        return len(chunks)

    def query(self, embedding: list[float], top_k: int) -> list[SearchResult]:
        if self.count() == 0:
            return []
        res = self._collection.query(
            query_embeddings=[embedding],
            n_results=min(top_k, self.count()),
            include=["documents", "metadatas", "distances"],
        )
        results = []
        for doc, meta, dist in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0], strict=True
        ):
            # Chroma cosine distance = 1 - cosine similarity
            results.append(SearchResult(text=doc, metadata=meta, score=1.0 - dist))
        return results

    def count(self) -> int:
        return self._collection.count()

    def reset(self) -> None:
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name, metadata={"hnsw:space": "cosine"}
        )
