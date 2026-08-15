# Backend image — builds for both amd64 and arm64 (Oracle Ampere).
#
# Models are baked in at build time rather than fetched on first request: a
# cold start that downloads ~500MB of weights is indistinguishable from a hang
# to whoever is visiting the deployed URL.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models \
    TRANSFORMERS_VERBOSITY=error \
    # FAISS and PyTorch each bundle an OpenMP runtime; loading both aborts
    # without this.
    KMP_DUPLICATE_LIB_OK=TRUE

WORKDIR /app

# build-essential is needed to compile wheels that have no arm64 build.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# The PyTorch CPU wheel index only publishes x86_64. On arm64 the default PyPI
# wheels are already CPU-only, so the extra index is applied conditionally
# rather than unconditionally — passing it on arm64 makes the install fail to
# resolve.
RUN ARCH="$(dpkg --print-architecture)" && \
    if [ "$ARCH" = "amd64" ]; then \
      pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.txt ; \
    else \
      pip install --no-cache-dir -r requirements.txt ; \
    fi

# Warm the model cache into an image layer.
RUN python -c "\
from transformers import AutoModel, AutoTokenizer; \
m='intfloat/multilingual-e5-small'; \
AutoTokenizer.from_pretrained(m); AutoModel.from_pretrained(m)"

COPY app ./app
COPY scripts ./scripts
COPY pytest.ini .

RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app /models
USER appuser

EXPOSE 8000

# One worker: the embedding model is loaded per process, so extra workers
# multiply memory for no throughput gain on a CPU-bound pipeline.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
