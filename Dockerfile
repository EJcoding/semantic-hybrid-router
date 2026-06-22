# ── Base ─────────────────────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ───────────────────────────────────────────────────────
# Install CPU-only PyTorch first as a separate layer.
# The full CUDA wheel is ~2GB; the CPU wheel is ~700MB.
# MPS is Apple-specific hardware — Docker containers always run on CPU.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application source ────────────────────────────────────────────────────────
COPY app/ ./app/

# ── Runtime artifacts (generated during development, not in git) ──────────────
# Trained sklearn models from Phase 3
COPY models/ ./models/

# Seeded SQLite database from Phase 4a
COPY data/orders.db ./data/orders.db

# HuggingFace transformer weights moved to project dir in Phase 2/5
# (~90MB; baked into image so container runs without network access)
COPY .cache/ ./.cache/

# ── Environment ───────────────────────────────────────────────────────────────
# Absolute path so HuggingFace resolves the cache correctly regardless of CWD
ENV HF_HOME=/app/.cache/huggingface
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ── Server ────────────────────────────────────────────────────────────────────
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]