def test_ingest_counts_chunks(pipeline):
    stats = pipeline.stats()
    assert stats["chunks"] > 0
    assert stats["mock_llm"] is True


def test_chat_returns_grounded_answer(pipeline):
    result = pipeline.chat("Đổi trả trong bao nhiêu ngày?")
    assert "7 ngày" in result.answer
    assert any(s.source == "return.md" for s in result.sources)
    assert result.session_id


def test_chat_out_of_scope(settings, docs_dir, tmp_path):
    from rag.pipeline import RAGPipeline

    strict = RAGPipeline(
        settings.model_copy(
            update={
                "score_threshold": 0.95,
                "chroma_persist_dir": str(tmp_path / "chroma_strict"),
            }
        )
    )
    strict.ingest_directory(docs_dir)
    result = strict.chat("Giá cổ phiếu VNM hôm nay?")
    assert "không tìm thấy" in result.answer.lower()


def test_session_memory_persists(pipeline):
    r1 = pipeline.chat("Freeship từ bao nhiêu tiền?")
    r2 = pipeline.chat("Còn đổi trả thì sao?", session_id=r1.session_id)
    assert r2.session_id == r1.session_id


def test_stream_chat_emits_events(pipeline):
    events = list(pipeline.stream_chat("Son môi giá bao nhiêu?"))
    kinds = [e["event"] for e in events]
    assert kinds[0] == "token"
    assert "sources" in kinds
    assert kinds[-1] == "done"
    answer = "".join(e["data"] for e in events if e["event"] == "token")
    assert "129" in answer
