# Phase 5 — API Wrapper & Local Containerisation

## Overview

Phase 5 wraps the complete routing pipeline in a production-grade REST API and
packages the entire system — application code, trained models, transformer
weights, and seeded database — into a Docker image that runs identically on any
machine with Docker installed.

By the end of this phase, the full system is accessible via a single HTTP
endpoint and deployable with a single command (`docker compose up`), with no
dependency on the host machine's Python version, installed packages, or
HuggingFace cache.

---

## Files Created

| File | Purpose |
|------|---------|
| `app/main.py` | FastAPI application — request/response schemas, lifespan, endpoints |
| `Dockerfile` | Container definition — runtime, dependencies, artifacts |
| `docker-compose.yml` | Local orchestration — port mapping, env var injection |
| `.dockerignore` | Build context filter — keeps the image lean |
| `scripts/test_api.py` | Integration test suite — 14 checks across all routing paths |

---

## API Design

### Endpoints

```
GET  /health
POST /api/v1/route_query
```

**`GET /health`** — a liveness check for container orchestration systems
(Kubernetes, Docker health checks, load balancers). Returns `{"status": "ok"}`
with no dependencies on the ML models or database.

**`POST /api/v1/route_query`** — the main endpoint. Accepts a raw customer
query and returns the full routing result.

### Request / Response Schemas

```python
# Request
{"query": "I want to cancel order ORD-1009"}

# Response
{
  "intent":     "cancel_order",
  "is_anomaly": false,
  "confidence": 0.5536,
  "response":   "Your order ORD-1009 has been cancelled..."
}
```

The `embedding` field from `RouterState` is intentionally excluded from the
response. It is a 384-element float array — an internal implementation detail
with no value to the API consumer, and not JSON-serializable without explicit
conversion. The response surface exposes only what a caller needs: the routing
decision and the agent's reply.

### Input Validation

Empty or whitespace-only queries return HTTP `422 Unprocessable Content` before
the pipeline is touched. This is implemented with a simple guard in the endpoint
rather than a Pydantic validator, since the error message needs to be explicit
about the reason.

### FastAPI Interactive Docs

FastAPI auto-generates a Swagger UI at `http://localhost:8000/docs`. Every
endpoint, schema, and example response is documented there without any additional
configuration — useful for demos and manual testing without curl.

---

## Lifespan: Model Pre-warming

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    _warm_embedder()
    _classifier._load_models()
    get_router_graph()
    yield
```

The `lifespan` context manager (the modern FastAPI pattern, replacing the
deprecated `@app.on_event("startup")`) pre-loads all local models before the
server accepts its first request:

1. **Transformer weights** (~90MB, ~2-3s on CPU) — the largest cold-start cost
2. **sklearn models** (kmeans, cluster_map, dbscan artifacts) — fast, ~50ms
3. **LangGraph state machine** — compiles the routing graph

Without pre-warming, the first request would pay all three loading costs, easily
exceeding several seconds. With pre-warming, the startup logs confirm readiness
before uvicorn reports `Application startup complete`.

The LangChain agents are **not** pre-warmed — they connect to OpenRouter lazily
on first use. Pre-warming them would make an outbound API call at container
startup, which is undesirable (it could fail, delay startup, or consume rate
limit quota before any real request arrives).

---

## Dockerfile

### Key decisions

**`python:3.11-slim` base image**

A slim image omits development tools and documentation, reducing the base size
significantly. Python 3.11 is used rather than the development environment's
3.13 for broader compatibility with the dependency ecosystem.

**CPU-only PyTorch installed as a separate layer**

```dockerfile
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
```

The full CUDA-enabled PyTorch wheel is ~2GB. The CPU-only wheel is ~700MB.
Docker containers on Mac run inside a Linux VM with no access to Apple Silicon
MPS hardware, and CUDA is NVIDIA-specific — containers always use CPU. Installing
PyTorch first as a dedicated layer means Docker can cache it independently from
the rest of the dependencies. If only `requirements.txt` changes, this layer is
not rebuilt.

**Runtime artifacts baked into the image**

```dockerfile
COPY models/ ./models/
COPY data/orders.db ./data/orders.db
COPY .cache/ ./.cache/
```

Three categories of generated artifacts are copied into the image:
- Trained sklearn models (Phase 3 output)
- The seeded SQLite database (Phase 4a output)
- HuggingFace transformer weights (relocated to project dir in Phase 2/5)

Baking these into the image means the container starts without any network calls
or generation scripts. A fresh `docker compose up` produces a fully functional
server with pre-populated data in under 30 seconds (after the initial build).

**`HF_HOME` set to an absolute path**

```dockerfile
ENV HF_HOME=/app/.cache/huggingface
```

In `.env` for local development, `HF_HOME=./.cache/huggingface` works because
the process CWD is the project root. Inside Docker with `WORKDIR /app`, using
an absolute path (`/app/.cache/huggingface`) ensures HuggingFace resolves the
cache correctly regardless of how the process is launched.

---

## docker-compose.yml

```yaml
services:
  semantic-hybrid-router:
    build: .
    ports:
      - "8000:8000"
    env_file:
      - .env
```

`env_file: .env` injects `OPENROUTER_API_KEY` and `OPENROUTER_MODEL` from the
local `.env` file as container environment variables at runtime. The API key is
never written into any Docker layer — it exists only in the running container's
environment. This is the correct secret management pattern for local development;
production systems would use a secrets manager (AWS Secrets Manager, HashiCorp
Vault, etc.) instead of a `.env` file.

---

## .dockerignore

The `.dockerignore` file controls what is sent to the Docker daemon as the build
context. Without it, the entire project directory — including `venv/` (~500MB),
`.git/`, and development data files — would be copied before the `Dockerfile`
runs, making builds slow.

Key exclusions:
- `venv/` — dependencies are installed fresh inside the container from `requirements.txt`
- `.env` — secrets are injected at runtime via `docker-compose.yml`, never baked in
- `data/*.csv`, `data/*.npz` — training data, not needed at serving time
- `scripts/` — development utilities, not needed at serving time

Key inclusions (explicitly not excluded):
- `models/` — trained sklearn artifacts required at runtime
- `.cache/` — transformer weights required at runtime
- `data/orders.db` — seeded database required at runtime

---

## Integration Test Suite

`scripts/test_api.py` runs 14 assertions against the live server (local or
Docker — same URL, same script):

| Test group | Checks |
|------------|--------|
| Health | HTTP 200, `{"status": "ok"}` |
| Cancel | Correct intent, not anomalous, confidence > 0, DB mutation confirmed |
| Track | Correct intent, not anomalous, confidence > 0, DB unchanged confirmed |
| Refund | Correct intent, not anomalous, confidence > 0 |
| Guardrail | `anomalous` intent, `is_anomaly=True`, `confidence=0.0` |
| Edge case | Empty query returns HTTP 422 |

The database check for the cancel test is the most meaningful assertion: it
verifies that `ORD-1009.status == "Cancelled"` in the actual database file after
the API call, not just that the response text claims the order was cancelled. An
LLM could hallucinate a plausible-sounding cancellation message without the
`cancel_order` tool ever executing its SQL `UPDATE`.

**All 14 checks passed in both environments:**
- Local: `uvicorn app.main:app --host 0.0.0.0 --port 8000`
- Docker: `docker compose up --build`

---

## Docker vs Local: What Changes

| Property | Local | Docker |
|----------|-------|--------|
| Python version | 3.13 (host) | 3.11 (container) |
| PyTorch device | MPS (Apple Silicon) | CPU (Linux VM) |
| Request source IP | `127.0.0.1` | `192.168.65.1` (bridge) |
| Database | `data/orders.db` on host | Internal copy baked in image |
| Behaviour | Identical | Identical |

The device change (MPS → CPU) is the most operationally significant difference.
The `_get_device()` function in `embedder.py` handles this transparently — the
embeddings produced on CPU are numerically identical to those produced on MPS
(same model weights, same operations, same float32 precision).

---

## Lessons

Docker's value for ML projects extends beyond dependency management. The
discipline of containerisation forced several good practices that might otherwise
have been deferred: relocating the HuggingFace model cache into the project
directory (making the runtime dependency explicit and portable), separating
training-time from serving-time dependencies in `.dockerignore`, and thinking
carefully about which artifacts belong in the image versus which should be
provided at runtime (model weights vs API keys).

The integration test suite (`test_api.py`) demonstrated the value of testing at
the HTTP boundary rather than the Python function boundary. It found no bugs —
but the DB mutation check in particular is a class of assertion that unit tests
cannot provide: confirmation that the full chain from HTTP request to SQL commit
to HTTP response is wired together correctly in both environments.