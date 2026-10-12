from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # LLM — any OpenAI-compatible endpoint (Groq by default)
    groq_api_key: str = ""
    llm_base_url: str = "https://api.groq.com/openai/v1"
    llm_model: str = "llama-3.3-70b-versatile"
    llm_temperature: float = 0.2
    llm_max_tokens: int = 1024

    # Embedding — "sbert" (real model) or "hash" (deterministic, offline/tests)
    embedding_backend: str = "sbert"
    embedding_model: str = "bkai-foundation-models/vietnamese-bi-encoder"
    embedding_device: str = "cpu"

    # Vector store
    chroma_persist_dir: str = str(BASE_DIR / "storage" / "chroma")
    collection_name: str = "vn_knowledge_base"

    # Chunking
    chunk_size: int = 800
    chunk_overlap: int = 120

    # Retrieval
    top_k: int = 4
    # vietnamese-bi-encoder returns modest cosine scores (~0.15-0.4 for relevant
    # pairs); 0.15 rejects off-topic queries without cutting good candidates
    score_threshold: float = 0.15

    # Chat memory (in-memory; swap for Redis in production)
    max_history_turns: int = 6

    # Default ingest directory
    data_dir: str = str(BASE_DIR / "data" / "sample_docs")


@lru_cache
def get_settings() -> Settings:
    return Settings()
