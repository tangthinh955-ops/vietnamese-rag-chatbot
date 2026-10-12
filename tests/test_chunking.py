from rag.chunking import recursive_split, split_documents
from rag.loaders import Document


def test_short_text_single_chunk():
    assert recursive_split("Xin chào", chunk_size=100, chunk_overlap=10) == ["Xin chào"]


def test_chunks_respect_max_size():
    text = "\n\n".join(f"Đoạn văn số {i}. " * 20 for i in range(10))
    chunks = recursive_split(text, chunk_size=200, chunk_overlap=30)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)


def test_no_content_loss():
    text = " ".join(f"câu{i}" for i in range(300))
    chunks = recursive_split(text, chunk_size=100, chunk_overlap=10)
    reassembled = " ".join(chunks)
    for token in text.split():
        assert token in reassembled


def test_split_documents_keeps_metadata():
    docs = [
        Document(text="nội dung " * 200, metadata={"source": "a.md"}),
        Document(text="ngắn", metadata={"source": "b.md"}),
    ]
    chunks = split_documents(docs, chunk_size=100, chunk_overlap=10)
    assert all("source" in c.metadata and "chunk" in c.metadata for c in chunks)
    assert {c.metadata["source"] for c in chunks} == {"a.md", "b.md"}
    assert chunks[-1].metadata["source"] == "b.md"
