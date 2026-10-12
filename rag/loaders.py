"""Document loaders for .txt, .md and .pdf files."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


@dataclass
class Document:
    text: str
    metadata: dict = field(default_factory=dict)


def _clean(text: str) -> str:
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").split("\n")]
    return "\n".join(lines).strip()


def load_text_file(path: Path) -> list[Document]:
    text = _clean(path.read_text(encoding="utf-8"))
    if not text:
        return []
    return [Document(text=text, metadata={"source": path.name})]


def load_pdf_file(path: Path) -> list[Document]:
    from pypdf import PdfReader

    documents = []
    for page_num, page in enumerate(PdfReader(str(path)).pages, start=1):
        text = _clean(page.extract_text() or "")
        if text:
            documents.append(
                Document(
                    text=text,
                    metadata={"source": path.name, "page": page_num},
                )
            )
    return documents


def load_file(path: str | Path) -> list[Document]:
    path = Path(path)
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {path.suffix} ({path.name})")
    if path.suffix.lower() == ".pdf":
        return load_pdf_file(path)
    return load_text_file(path)


def load_directory(directory: str | Path) -> list[Document]:
    """Load every supported file under `directory` (recursive)."""
    directory = Path(directory)
    documents: list[Document] = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            documents.extend(load_file(path))
    return documents
