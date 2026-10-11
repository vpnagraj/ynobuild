# Multi-stage build. Compose selects a `target:` per service.
#   core -> the ynbtriage CLI (ingest, fetch, splits, seed) — used by batch jobs
#   api  -> FastAPI data-access service (the only online DB writer)
#   model -> model-serving FastAPI service (trained MLP; PyTorch lives only here)
#   web  -> Streamlit UI (thin HTTP client; does NOT install ynbtriage)

FROM python:3.11-slim AS base
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1
WORKDIR /app

# ---------------------------------------------------------------- core (CLI)
FROM base AS core
COPY pyproject.toml README.md ./
COPY src ./src
COPY taxonomy ./taxonomy
RUN pip install .
ENTRYPOINT ["ynbtriage"]

# ---------------------------------------------------------------- api
FROM base AS api
COPY pyproject.toml README.md ./
COPY src ./src
COPY taxonomy ./taxonomy
RUN pip install ".[api]"
EXPOSE 8000
CMD ["uvicorn", "ynbtriage.api:app", "--host", "0.0.0.0", "--port", "8000"]

# ---------------------------------------------------------------- model
# Serves the trained MLP (ynbtriage.serve). The only image with PyTorch.
FROM base AS model
# CPU-only torch, installed first in its own layer: much smaller than the default
# CUDA build, and cached across rebuilds when only source code changes.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml README.md ./
COPY src ./src
COPY taxonomy ./taxonomy
RUN pip install ".[api,model]"
# Hugging Face cache (only used by embedding-feature models); mounted as a volume.
ENV HF_HOME=/cache/hf
EXPOSE 8001
CMD ["uvicorn", "ynbtriage.serve:app", "--host", "0.0.0.0", "--port", "8001"]

# ---------------------------------------------------------------- web
FROM base AS web
COPY web/requirements.txt ./
RUN pip install -r requirements.txt
COPY web ./web
EXPOSE 8501
CMD ["streamlit", "run", "web/app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", \
     "--server.headless=true", "--browser.gatherUsageStats=false"]
