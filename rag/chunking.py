"""Recursive character text splitting with overlap.

Implemented from scratch (no framework dependency) so the chunking strategy is
explicit and easy to tune: split on the coarsest separator available
(paragraphs -> lines -> sentences -> words -> characters), then greedily merge
pieces into chunks of at most `chunk_size`, carrying a trailing overlap window
so context is not lost at chunk boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rag.loaders import Document

SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]


@dataclass
class Chunk:
    text: str
    index: int
    metadata: dict = field(default_factory=dict)


def _merge_splits(
    splits: list[str], separator: str, chunk_size: int, chunk_overlap: int
) -> list[str]:
    sep_len = len(separator)
    chunks: list[str] = []
    current: list[str] = []
    total = 0
    for piece in splits:
        piece_len = len(piece) + (sep_len if current else 0)
        if current and total + piece_len > chunk_size:
            chunks.append(separator.join(current))
            while current and (
                total > chunk_overlap or total + piece_len > chunk_size
            ):
                total -= len(current[0]) + (sep_len if len(current) > 1 else 0)
                current.pop(0)
        current.append(piece)
        total += piece_len
    if current:
        chunks.append(separator.join(current))
    return chunks


def recursive_split(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    separators: list[str] | None = None,
) -> list[str]:
    separators = separators or SEPARATORS
    separator = separators[-1]
    new_separators: list[str] = []
    for i, s in enumerate(separators):
        if s == "" or s in text:
            separator = s
            new_separators = separators[i + 1 :]
            break

    splits = list(text) if separator == "" else text.split(separator)

    final: list[str] = []
    good: list[str] = []
    for piece in splits:
        if len(piece) < chunk_size:
            good.append(piece)
            continue
        if good:
            final.extend(_merge_splits(good, separator, chunk_size, chunk_overlap))
            good = []
        if not new_separators:
            final.append(piece)
        else:
            final.extend(
                recursive_split(piece, chunk_size, chunk_overlap, new_separators)
            )
    if good:
        final.extend(_merge_splits(good, separator, chunk_size, chunk_overlap))

    return [c for c in (c.strip() for c in final) if c]


def split_documents(
    documents: list[Document], chunk_size: int, chunk_overlap: int
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in documents:
        for i, text in enumerate(
            recursive_split(doc.text, chunk_size, chunk_overlap)
        ):
            chunks.append(
                Chunk(text=text, index=len(chunks), metadata={**doc.metadata, "chunk": i})
            )
    return chunks
