# Vietnamese RAG Chatbot

> API hỏi–đáp tiếng Việt trên knowledge base nội bộ bằng **RAG** (Retrieval-Augmented Generation) — trả lời có trích nguồn, nhớ ngữ cảnh hội thoại, kèm bộ đánh giá chất lượng retrieval.
>
> *Vietnamese knowledge-base QA API built with RAG — grounded answers with citations, conversation memory and an eval harness.*

![CI](https://img.shields.io/badge/CI-pytest%20%2B%20ruff-blue)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688)
![License](https://img.shields.io/badge/license-MIT-green)

---

## 🇻🇳 Tiếng Việt

### Giới thiệu

Project mô phỏng một **AI trợ lý chăm sóc khách hàng** cho shop bán hàng livestream: nhân viên nạp tài liệu nội bộ (chính sách đổi trả, vận chuyển, bảo hành, thông tin sản phẩm...), khách hỏi bằng tiếng Việt tự nhiên, hệ thống truy xuất tài liệu liên quan rồi sinh câu trả lời **bám sát tài liệu, kèm trích nguồn** — không bịa thông tin.

Demo đi kèm: knowledge base mẫu của "Luna Store" (10 tài liệu tiếng Việt trong `data/sample_docs/`).

### Tính năng

- **Ingest tài liệu** `.txt` / `.md` / `.pdf` → làm sạch → chunking (recursive splitter, overlap) → embedding → ChromaDB
- **Hỏi đáp có trích nguồn**: mỗi câu trả lời kèm `sources[]` (file, chunk, điểm tương đồng)
- **Nhớ hội thoại** theo `session_id` (in-memory, dễ đổi sang Redis)
- **Streaming SSE** qua `GET /chat/stream`
- **Chống bịa**: hỏi ngoài tài liệu → trả lời "không tìm thấy" thay vì hallucinate
- **Evaluation**: script đo `hit_rate@k` + keyword coverage trên testset 20 câu tiếng Việt
- **Chạy được không cần API key**: `MockLLM` + `HashEmbedder` để dev/test/CI offline

### Kiến trúc

```
Ingest:  docs (.md/.txt/.pdf) → clean → chunk(800/120) → embed → ChromaDB (cosine)
Query:   câu hỏi → embed → top-k retrieve → prompt + context + history → LLM → answer + sources
```

Chi tiết quyết định kỹ thuật: [docs/DESIGN.md](docs/DESIGN.md)

### Tech stack

| Thành phần | Công nghệ |
|---|---|
| API | FastAPI, Pydantic v2, Uvicorn |
| Vector DB | ChromaDB (persistent, cosine) |
| Embedding | `bkai-foundation-models/vietnamese-bi-encoder` (sentence-transformers) |
| LLM | Groq API — `llama-3.3-70b-versatile` (OpenAI-compatible, đổi provider bằng env) |
| Eval | Custom script: hit@k, keyword coverage, out-of-scope abstention |
| DevOps | Docker, GitHub Actions CI, pytest, ruff |

### Cài đặt & chạy

```bash
git clone https://github.com/<you>/vietnamese-rag-chatbot.git
cd vietnamese-rag-chatbot
python -m venv .venv && .venv\Scripts\activate        # Windows
# source .venv/bin/activate                           # macOS/Linux
pip install -r requirements.txt
cp .env.example .env          # điền GROQ_API_KEY (free tại console.groq.com) — không bắt buộc
```

```bash
python scripts/ingest.py                              # nạp sample docs
uvicorn rag.api.main:app --reload                     # http://localhost:8000/docs
```

Docker (một lệnh, tự ingest sample docs khi boot):

```bash
docker build -t vn-rag-chatbot . && docker run -p 8000:8000 vn-rag-chatbot
```

> Không có `GROQ_API_KEY`? App vẫn chạy bình thường với `MockLLM` (trả lời trích xuất từ context). Thêm `EMBEDDING_BACKEND=hash` nếu muốn chạy hoàn toàn offline không cần tải model.

### Dùng thử API

```bash
# Hỏi đáp
curl -X POST localhost:8000/chat -H "Content-Type: application/json" \
  -d '{"question": "Đơn bao nhiêu tiền thì được freeship?"}'
# → {"answer": "...", "sources": [{"source": "chinh-sach-van-chuyen.md", "score": 0.81, ...}], ...}

# Tiếp tục hội thoại
curl -X POST localhost:8000/chat -H "Content-Type: application/json" \
  -d '{"question": "Còn đổi trả thì sao?", "session_id": "<id_từ_response_trên>"}'

# Streaming
curl -N "localhost:8000/chat/stream?question=Son%20m%C3%B4i%20gi%C3%A1%20bao%20nhi%C3%AAu%3F"

# Upload tài liệu
curl -X POST localhost:8000/ingest/files -F "files=@chinh-sach-moi.pdf"
```

| Endpoint | Mô tả |
|---|---|
| `POST /chat` | Hỏi đáp (body: `question`, `session_id?`, `top_k?`) |
| `GET /chat/stream` | SSE streaming: `token` → `sources` → `done` |
| `POST /ingest` | Nạp file/thư mục phía server |
| `POST /ingest/files` | Upload `.txt/.md/.pdf` |
| `GET /health`, `GET /stats` | Health check, thống kê vector store |

### Đánh giá chất lượng

```bash
python scripts/evaluate.py        # chạy trên eval/testset.json (20 câu), ghi report vào eval/reports/
```

Metrics: **hit_rate@k** (tài liệu đúng có trong top-k), **keyword coverage** (đáp án chứa ý bắt buộc), **abstention** (câu ngoài phạm vi phải trả lời "không tìm thấy").

Kết quả đo trên testset đi kèm (vietnamese-bi-encoder, `top_k=4`, MockLLM — chưa cấu hình Groq key):

| Metric | Kết quả |
|---|---|
| hit_rate@4 | **94.7%** (18/19 câu) |
| out-of-scope abstention | **100%** (câu lạc đề bị từ chối đúng) |
| keyword coverage | 66.7% — giới hạn bởi MockLLM (echo nguyên văn); tăng rõ khi bật LLM thật |

### Test & lint

```bash
pytest                            # offline, dùng HashEmbedder + MockLLM
ruff check rag tests scripts
```

---

## 🇬🇧 English

### Overview

A small but production-shaped **RAG service** that answers Vietnamese customer questions over an internal knowledge base. Documents (`.txt`/`.md`/`.pdf`) are cleaned, chunked, embedded and indexed in ChromaDB; queries retrieve top-k context, and the LLM answers **strictly grounded in the context with source citations** — or abstains when the docs don't cover the question.

Includes a 10-doc Vietnamese sample corpus (livestream-shop policies & products) and a 20-question eval testset.

### Highlights

- Grounded QA with per-answer `sources[]` (file, chunk index, similarity score)
- Conversation memory via `session_id`; SSE streaming endpoint
- Pluggable backends: any OpenAI-compatible LLM, swappable vector store/embedder behind protocols
- `MockLLM` + `HashEmbedder` → the whole app and test suite run with zero secrets and zero model downloads
- Eval harness reports `hit_rate@k`, keyword coverage and out-of-scope abstention
- CI (pytest + ruff) and a one-command Docker build

### Quickstart

```bash
pip install -r requirements.txt
cp .env.example .env            # set GROQ_API_KEY for real LLM answers (optional)
python scripts/ingest.py
uvicorn rag.api.main:app --reload   # Swagger UI: http://localhost:8000/docs
```

See the Vietnamese section above for API examples — the endpoints are language-agnostic. Design trade-offs live in [docs/DESIGN.md](docs/DESIGN.md).

### Repo layout

```
rag/            # package: config, chunking, loaders, embeddings, vectorstore,
│               # retriever, llm, prompts, sessions, pipeline
rag/api/        # FastAPI app (routes, schemas)
data/sample_docs/   # 10-doc Vietnamese knowledge base (Luna Store)
eval/testset.json   # 20 labeled QA pairs
scripts/            # ingest.py, evaluate.py
tests/              # offline pytest suite (HashEmbedder + MockLLM)
docs/DESIGN.md      # architecture & trade-offs
```
