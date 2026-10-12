"""Test fixtures: deterministic HashEmbedder + MockLLM, temp vector store.

Runs fully offline — no model downloads, no API keys.
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["EMBEDDING_BACKEND"] = "hash"
os.environ["GROQ_API_KEY"] = ""
os.environ["SCORE_THRESHOLD"] = "0"

from rag.config import Settings  # noqa: E402
from rag.pipeline import RAGPipeline  # noqa: E402

MINI_DOCS = {
    "shipping.md": (
        "# Vận chuyển\n\nShop freeship cho đơn từ 299.000đ. "
        "Giao miền Bắc trong 3 đến 5 ngày."
    ),
    "return.md": (
        "# Đổi trả\n\nShop nhận đổi trả trong 7 ngày nếu sản phẩm lỗi. "
        "Quá 7 ngày không hỗ trợ."
    ),
    "product.md": (
        "# Son môi\n\nSon kem lì giá livestream 129.000đ, có 5 màu."
    ),
}


@pytest.fixture()
def settings(tmp_path, monkeypatch) -> Settings:
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
    return Settings()


@pytest.fixture()
def docs_dir(tmp_path) -> Path:
    d = tmp_path / "docs"
    d.mkdir()
    for name, text in MINI_DOCS.items():
        (d / name).write_text(text, encoding="utf-8")
    return d


@pytest.fixture()
def pipeline(settings, docs_dir) -> RAGPipeline:
    pipe = RAGPipeline(settings)
    pipe.ingest_directory(docs_dir)
    return pipe
