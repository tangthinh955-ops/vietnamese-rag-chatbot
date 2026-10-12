"""RAG pipeline: document ingestion and grounded question answering."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from rag.chunking import split_documents
from rag.config import Settings, get_settings
from rag.embeddings import build_embedder
from rag.llm import ChatLLM, build_llm
from rag.loaders import SUPPORTED_EXTENSIONS, load_file
from rag.prompts import NO_CONTEXT_ANSWER, build_messages
from rag.retriever import Retriever
from rag.sessions import SessionStore
from rag.vectorstore import ChromaStore, SearchResult

logger = logging.getLogger(__name__)


@dataclass
class IngestReport:
    files_seen: int = 0
    files_ingested: int = 0
    chunks_added: int = 0
    skipped: list[str] = field(default_factory=list)


@dataclass
class SourceItem:
    source: str
    score: float
    chunk: int | None
    text: str


@dataclass
class ChatResult:
    answer: str
    session_id: str
    sources: list[SourceItem]
    model: str
    is_mock: bool


class RAGPipeline:
    def __init__(
        self,
        settings: Settings,
        embedder=None,
        store=None,
        llm: ChatLLM | None = None,
    ) -> None:
        self._settings = settings
        self._embedder = embedder or build_embedder(settings)
        self._store = store or ChromaStore(
            settings.chroma_persist_dir, settings.collection_name
        )
        self._llm = llm or build_llm(settings)
        self._retriever = Retriever(
            self._embedder,
            self._store,
            top_k=settings.top_k,
            score_threshold=settings.score_threshold,
        )
        self.sessions = SessionStore(max_turns=settings.max_history_turns)

    # ---------- ingestion ----------

    def ingest_file(self, path: str | Path) -> int:
        documents = load_file(path)
        return self._ingest_documents(documents)

    def ingest_directory(self, directory: str | Path) -> IngestReport:
        directory = Path(directory)
        report = IngestReport()
        for path in sorted(directory.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                report.skipped.append(path.name)
                continue
            report.files_seen += 1
            try:
                report.chunks_added += self.ingest_file(path)
                report.files_ingested += 1
            except Exception as exc:  # noqa: BLE001 — report and continue
                logger.warning("Failed to ingest %s: %s", path.name, exc)
                report.skipped.append(f"{path.name} ({exc})")
        return report

    def _ingest_documents(self, documents) -> int:
        chunks = split_documents(
            documents, self._settings.chunk_size, self._settings.chunk_overlap
        )
        if not chunks:
            return 0
        embeddings = self._embedder.embed([c.text for c in chunks])
        return self._store.add(chunks, embeddings)

    # ---------- query ----------

    def _to_sources(self, results: list[SearchResult]) -> list[SourceItem]:
        return [
            SourceItem(
                source=r.metadata.get("source", "unknown"),
                score=round(r.score, 4),
                chunk=r.metadata.get("chunk"),
                text=r.text[:300],
            )
            for r in results
        ]

    def chat(
        self, question: str, session_id: str | None = None, top_k: int | None = None
    ) -> ChatResult:
        session_id = session_id or self.sessions.new_id()
        results = self._retriever.retrieve(question, top_k)

        if not results:
            answer = NO_CONTEXT_ANSWER
        else:
            messages = build_messages(
                question, results, history=self.sessions.get(session_id)
            )
            answer = self._llm.complete(messages)

        self.sessions.append(session_id, "user", question)
        self.sessions.append(session_id, "assistant", answer)
        return ChatResult(
            answer=answer,
            session_id=session_id,
            sources=self._to_sources(results),
            model=self._llm.name,
            is_mock=self._llm.is_mock,
        )

    def stream_chat(
        self, question: str, session_id: str | None = None, top_k: int | None = None
    ) -> Iterator[dict]:
        """Yield SSE-ready events: token* -> sources -> done."""
        session_id = session_id or self.sessions.new_id()
        results = self._retriever.retrieve(question, top_k)

        if not results:
            answer = NO_CONTEXT_ANSWER
            yield {"event": "token", "data": answer}
        else:
            messages = build_messages(
                question, results, history=self.sessions.get(session_id)
            )
            parts = []
            for token in self._llm.stream(messages):
                parts.append(token)
                yield {"event": "token", "data": token}
            answer = "".join(parts)

        self.sessions.append(session_id, "user", question)
        self.sessions.append(session_id, "assistant", answer)
        yield {
            "event": "sources",
            "data": [s.__dict__ for s in self._to_sources(results)],
        }
        yield {"event": "done", "data": {"session_id": session_id}}

    # ---------- stats ----------

    def reset(self) -> None:
        self._store.reset()

    def stats(self) -> dict:
        return {
            "chunks": self._store.count(),
            "collection": self._settings.collection_name,
            "embedding_model": (
                "hash"
                if self._settings.embedding_backend == "hash"
                else self._settings.embedding_model
            ),
            "llm_model": self._llm.name,
            "mock_llm": self._llm.is_mock,
            "top_k": self._settings.top_k,
        }


def build_pipeline(settings: Settings | None = None, **overrides) -> RAGPipeline:
    return RAGPipeline(settings or get_settings(), **overrides)
