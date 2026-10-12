# Build:  docker build -t vn-rag-chatbot .
# Run:    docker run -p 8000:8000 -e GROQ_API_KEY=gsk_... vn-rag-chatbot
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # CPU-only torch keeps the image ~1.5GB smaller
    PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000

# Ingest sample docs on first boot only if the store is empty, then serve.
CMD ["sh", "-c", "python scripts/ingest.py && uvicorn rag.api.main:app --host 0.0.0.0 --port 8000"]
