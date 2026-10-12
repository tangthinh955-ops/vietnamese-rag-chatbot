"""FastAPI application exposing the RAG pipeline.

Run:
    uvicorn rag.api.main:app --reload
"""

from __future__ import annotations

import json
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from rag.api.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    SourceItemSchema,
    StatsResponse,
    UploadResponse,
)
from rag.loaders import SUPPORTED_EXTENSIONS
from rag.pipeline import IngestReport, RAGPipeline, build_pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    if getattr(app.state, "pipeline", None) is None:
        app.state.pipeline = build_pipeline()
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="Vietnamese RAG Chatbot",
        description=(
            "Hỏi–đáp tiếng Việt trên knowledge base nội bộ bằng RAG "
            "(retrieval + LLM, trả lời kèm trích nguồn)."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    def get_pipeline() -> RAGPipeline:
        return app.state.pipeline

    @app.get("/health", response_model=HealthResponse, tags=["ops"])
    def health(pipeline: RAGPipeline = Depends(get_pipeline)):
        return {"status": "ok", "chunks": pipeline.stats()["chunks"]}

    @app.get("/stats", response_model=StatsResponse, tags=["ops"])
    def stats(pipeline: RAGPipeline = Depends(get_pipeline)):
        return pipeline.stats()

    @app.post("/chat", response_model=ChatResponse, tags=["chat"])
    def chat(req: ChatRequest, pipeline: RAGPipeline = Depends(get_pipeline)):
        result = pipeline.chat(
            question=req.question, session_id=req.session_id, top_k=req.top_k
        )
        return ChatResponse(
            answer=result.answer,
            session_id=result.session_id,
            sources=[SourceItemSchema(**s.__dict__) for s in result.sources],
            model=result.model,
            is_mock=result.is_mock,
        )

    @app.get("/chat/stream", tags=["chat"])
    def chat_stream(
        question: str,
        session_id: str | None = None,
        top_k: int | None = None,
        pipeline: RAGPipeline = Depends(get_pipeline),
    ):
        """Server-Sent Events stream: `token` events, then `sources`, then `done`."""

        def events():
            for ev in pipeline.stream_chat(question, session_id, top_k):
                yield f"event: {ev['event']}\ndata: {json.dumps(ev['data'], ensure_ascii=False)}\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    @app.post("/ingest", response_model=IngestResponse, tags=["ingest"])
    def ingest(req: IngestRequest, pipeline: RAGPipeline = Depends(get_pipeline)):
        """Ingest server-side files/directories (use /ingest/files to upload)."""
        report = IngestReport()
        for raw in req.paths:
            path = Path(raw)
            if not path.exists():
                raise HTTPException(status_code=404, detail=f"Not found: {raw}")
            if path.is_dir():
                sub = pipeline.ingest_directory(path)
            else:
                sub = IngestReport(
                    files_seen=1,
                    files_ingested=1,
                    chunks_added=pipeline.ingest_file(path),
                )
            report.files_seen += sub.files_seen
            report.files_ingested += sub.files_ingested
            report.chunks_added += sub.chunks_added
            report.skipped.extend(sub.skipped)
        return IngestResponse(**report.__dict__)

    @app.post("/ingest/files", response_model=UploadResponse, tags=["ingest"])
    async def ingest_files(
        files: list[UploadFile], pipeline: RAGPipeline = Depends(get_pipeline)
    ):
        """Upload .txt/.md/.pdf files to ingest into the knowledge base."""
        ingested, chunks, names = 0, 0, []
        with tempfile.TemporaryDirectory() as tmp:
            for f in files:
                suffix = Path(f.filename or "").suffix.lower()
                if suffix not in SUPPORTED_EXTENSIONS:
                    raise HTTPException(
                        status_code=400, detail=f"Unsupported file type: {f.filename}"
                    )
                dest = Path(tmp) / Path(f.filename or "upload").name
                dest.write_bytes(await f.read())
                chunks += pipeline.ingest_file(dest)
                ingested += 1
                names.append(f.filename or dest.name)
        return UploadResponse(
            files_ingested=ingested, chunks_added=chunks, files=names
        )

    return app


app = create_app()
