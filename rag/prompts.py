"""Prompt construction for the RAG chain."""

from __future__ import annotations

from rag.llm import Message
from rag.vectorstore import SearchResult

SYSTEM_PROMPT = """Bạn là trợ lý AI trả lời câu hỏi của khách hàng dựa trên tài liệu nội bộ của shop.

Quy tắc bắt buộc:
- Chỉ trả lời dựa trên NGỮ CẢNH được cung cấp bên dưới. Tuyệt đối không bịa thông tin.
- Nếu ngữ cảnh không chứa câu trả lời, nói rõ "Tôi không tìm thấy thông tin này trong tài liệu" và gợi ý khách liên hệ shop qua hotline hoặc inbox.
- Trả lời ngắn gọn, thân thiện, đúng văn phong chăm sóc khách hàng, bằng tiếng Việt.
- Khi dùng thông tin từ tài liệu, ghi nguồn ở cuối câu dạng [1], [2]... tương ứng thứ tự tài liệu trong ngữ cảnh."""

NO_CONTEXT_ANSWER = (
    "Xin lỗi, tôi không tìm thấy thông tin này trong tài liệu của shop. "
    "Bạn vui lòng inbox hoặc gọi hotline để được nhân viên hỗ trợ trực tiếp nhé."
)


def build_context_block(results: list[SearchResult]) -> str:
    blocks = []
    for i, r in enumerate(results, start=1):
        source = r.metadata.get("source", "unknown")
        blocks.append(f"[{i}] (nguồn: {source})\n{r.text}")
    return "\n\n".join(blocks)


def build_messages(
    question: str,
    results: list[SearchResult],
    history: list[Message] | None = None,
) -> list[Message]:
    context = build_context_block(results)
    user = (
        f"NGỮ CẢNH:\n{context}\n\n"
        f"CÂU HỎI: {question}\n\n"
        "Hãy trả lời câu hỏi dựa trên ngữ cảnh trên."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        *(history or []),
        {"role": "user", "content": user},
    ]
