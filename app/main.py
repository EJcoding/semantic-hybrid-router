"""
app/main.py
============
Phase 5: FastAPI REST API

Endpoints:
  GET  /health               → liveness check
  POST /api/v1/route_query   → full pipeline: embed → classify → agent
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.pipeline.router_graph import get_router_graph, route_query
from app.pipeline.embedder import _load_model as _warm_embedder
import app.pipeline.classifier as _classifier


# =============================================================================
# REQUEST / RESPONSE SCHEMAS
# =============================================================================

class RouteRequest(BaseModel):
    query: str


class RouteResponse(BaseModel):
    intent:     str
    is_anomaly: bool
    confidence: float
    response:   str


# =============================================================================
# LIFESPAN — pre-warm local models at startup
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Pre-load the transformer weights and sklearn models at startup so the
    first request doesn't pay cold-start latency (~2-3s for the embedder).
    LangChain agents connect to OpenRouter lazily on first use.
    """
    _warm_embedder()
    _classifier._load_models()
    get_router_graph()
    yield


# =============================================================================
# APP
# =============================================================================

app = FastAPI(
    title="Semantic Hybrid Router",
    description=(
        "Zero-cost AI routing microservice. Classifies customer intent via "
        "local transformer embeddings + K-Means, then dispatches to a "
        "specialised LangChain agent."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/v1/route_query", response_model=RouteResponse)
def route_query_endpoint(request: RouteRequest):
    """
    Route a raw customer query through the full pipeline and return
    the agent's response alongside routing metadata.
    """
    if not request.query.strip():
        raise HTTPException(status_code=422, detail="query cannot be empty")

    result = route_query(request.query)

    return RouteResponse(
        intent=result["intent"],
        is_anomaly=result["is_anomaly"],
        confidence=result["confidence"],
        response=result["response"],
    )