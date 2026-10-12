from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, examples=["Chính sách đổi trả của shop?"])
    session_id: str | None = Field(
        default=None, description="Reuse to continue a conversation"
    )
    top_k: int | None = Field(default=None, ge=1, le=20)


class SourceItemSchema(BaseModel):
    source: str
    score: float
    chunk: int | None
    text: str


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    sources: list[SourceItemSchema]
    model: str
    is_mock: bool


class IngestRequest(BaseModel):
    paths: list[str] = Field(
        ..., description="Server-side files or directories to ingest"
    )


class IngestResponse(BaseModel):
    files_seen: int
    files_ingested: int
    chunks_added: int
    skipped: list[str]


class UploadResponse(BaseModel):
    files_ingested: int
    chunks_added: int
    files: list[str]


class StatsResponse(BaseModel):
    chunks: int
    collection: str
    embedding_model: str
    llm_model: str
    mock_llm: bool
    top_k: int


class HealthResponse(BaseModel):
    status: str
    chunks: int
