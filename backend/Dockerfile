# Backend image.
#
# Models are downloaded at build time, not first request: a cold start that
# pulls ~500MB of weights looks indistinguishable from a hang to someone
# visiting the deployed URL.

FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models \
    TRANSFORMERS_VERBOSITY=error

WORKDIR /app

# Build tools are needed for some wheels but not at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch: the CUDA build is ~2GB larger and useless on a CPU VM.
COPY requirements.txt .
RUN pip install --no-cache-dir \
      --extra-index-url https://download.pytorch.org/whl/cpu \
      -r requirements.txt

# Warm the model cache into the image layer.
RUN python -c "\
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification; \
m='intfloat/multilingual-e5-small'; \
AutoTokenizer.from_pretrained(m); AutoModel.from_pretrained(m); \
r='BAAI/bge-reranker-v2-m3'; \
AutoTokenizer.from_pretrained(r); AutoModelForSequenceClassification.from_pretrained(r)"

COPY app ./app
COPY scripts ./scripts
COPY pytest.ini .

# Run as non-root.
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app /models
USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
