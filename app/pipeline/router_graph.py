"""
app/pipeline/router_graph.py
=============================
Phase 4b: LangGraph State Machine — End-to-End Routing Pipeline

Wires together every previous phase:
  embed_node     -> embedder.get_embedding()   (Phase 2)
  classify_node  -> classifier.classify()      (Phase 3)
  route_decision -> dict lookup on intent (no LLM)
  cancel/track/refund_node -> Phase 4a agents
  guardrail_node -> safe fallback for anomalous queries, no agent call

State is a flat TypedDict (not MessagesState) since this graph routes
once and hands off to exactly one agent, each managing its own
MessagesState internally.

Each agent node wraps its run_*_agent() call in try/except — if the
entire OpenRouter fallback chain fails, the node returns a polite
fallback message instead of crashing the graph.
"""

import re
import numpy as np
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

from app.pipeline.embedder import get_embedding
from app.pipeline.classifier import classify
from app.agents.cancel_agent import run_cancel_agent
from app.agents.track_agent import run_track_agent
from app.agents.refund_agent import run_refund_agent


# =============================================================================
# STATE SCHEMA
# =============================================================================

class RouterState(TypedDict):
    """
    Shared state passed between every node in the graph.

    query      : raw customer input (set by the caller)
    embedding  : 384-dim L2-normalized vector (set by embed_node)
    intent     : "cancel_order" | "track_order" | "get_refund" | "anomalous"
    is_anomaly : True if DBSCAN flagged this query as out-of-distribution
    confidence : routing confidence in [0.0, 1.0]; 0.0 if anomalous
    response   : final natural-language reply (set by whichever node runs last)
    """
    query: str
    embedding: np.ndarray
    intent: str
    is_anomaly: bool
    confidence: float
    response: str


# =============================================================================
# QUERY NORMALIZATION
# =============================================================================

# Two vocabulary mismatches between real user queries and Bitext training data:
#
# 1. ORDER IDs: training data uses "{{Order Number}}" literal placeholder.
#    Real queries contain "ORD-1005". Fix: replace ORD-XXXX before embedding.
#
# 2. REFUND VOCABULARY + STRUCTURE: the get_refund training samples use
#    financial/legal language ("compensation of my money", "restitution",
#    "rebate") and — critically — have NO order numbers at all. Natural
#    queries like "I want a refund for ORD-1007" embed close to cancel_order
#    (which has order numbers) rather than get_refund. Fix: replace refund
#    vocabulary AND strip the order number context entirely for refund queries,
#    so the embedding matches the order-free training distribution.
#    Diagnosed via scripts/diagnose_refund.py.

_ORDER_ID_PATTERN      = re.compile(r'\bORD-\d+\b', re.IGNORECASE)
_REFUND_PATTERN        = re.compile(
    r'\b(refund|reimburs\w*|money back|paid back)\b', re.IGNORECASE
)
_ORDER_CONTEXT_PATTERN = re.compile(
    r'\s+(for\s+)?(my\s+)?(order\s+)?\{\{Order Number\}\}', re.IGNORECASE
)


def _normalize_query(query: str) -> str:
    """Normalize query vocabulary and structure to match training distribution."""
    # Step 1: normalize order IDs
    query = _ORDER_ID_PATTERN.sub("{{Order Number}}", query)

    # Step 2: detect refund intent before substitution
    has_refund = bool(_REFUND_PATTERN.search(query))

    # Step 3: normalize refund vocabulary
    query = _REFUND_PATTERN.sub("get my money back", query)

    # Step 4: for refund queries, strip order number context entirely.
    # get_refund training samples have no order numbers — their presence
    # pulls the embedding toward cancel_order/track_order clusters instead.
    if has_refund:
        query = _ORDER_CONTEXT_PATTERN.sub('', query)
        query = re.sub(r'\s+', ' ', query).strip()

    return query


# =============================================================================
# NODE: embed_node
# =============================================================================

def embed_node(state: RouterState) -> dict:
    """Normalize the query, then embed it. Pure math, zero LLM calls."""
    normalized = _normalize_query(state["query"])
    embedding = get_embedding(normalized)
    return {"embedding": embedding}


# =============================================================================
# NODE: classify_node
# =============================================================================

def classify_node(state: RouterState) -> dict:
    """
    Run the embedding through K-Means + DBSCAN (Phase 3) to determine
    intent, anomaly status, and routing confidence. Pure math, zero
    LLM calls.
    """
    result = classify(state["embedding"])
    return {
        "intent": result.intent,
        "is_anomaly": result.is_anomaly,
        "confidence": result.confidence,
    }


# =============================================================================
# CONDITIONAL EDGE: route_decision
# =============================================================================

# Maps classify_node's output intent directly to the next node's name.
# This dict is the ENTIRE routing logic — no LLM is involved in this decision.
_INTENT_TO_NODE = {
    "cancel_order": "cancel_node",
    "track_order":  "track_node",
    "get_refund":   "refund_node",
    "anomalous":    "guardrail_node",
}


def route_decision(state: RouterState) -> str:
    """
    Conditional edge: dict lookup on state["intent"] -> next node name.
    Falls back to "guardrail_node" for any unrecognized intent.
    """
    return _INTENT_TO_NODE.get(state["intent"], "guardrail_node")


# =============================================================================
# NODE: cancel_node / track_node / refund_node
# =============================================================================
# Each of these calls into a Phase 4a agent (its own internal LangGraph,
# built via create_agent). Wrapped in try/except per our error-handling
# decision: if OpenRouter's entire model fallback chain fails, the node
# degrades gracefully rather than crashing graph.invoke().

def cancel_node(state: RouterState) -> dict:
    """Run the cancellation agent (Phase 4a) on the customer's query."""
    try:
        response = run_cancel_agent(state["query"])
    except Exception:
        response = (
            "I'm sorry, I'm having trouble processing your cancellation "
            "request right now. Please try again in a moment, or contact "
            "support directly for assistance."
        )
    return {"response": response}


def track_node(state: RouterState) -> dict:
    """Run the tracking agent (Phase 4a) on the customer's query."""
    try:
        response = run_track_agent(state["query"])
    except Exception:
        response = (
            "I'm sorry, I'm having trouble looking up your order status "
            "right now. Please try again in a moment, or contact support "
            "directly for assistance."
        )
    return {"response": response}


def refund_node(state: RouterState) -> dict:
    """Run the refund agent (Phase 4a) on the customer's query."""
    try:
        response = run_refund_agent(state["query"])
    except Exception:
        response = (
            "I'm sorry, I'm having trouble calculating your refund right "
            "now. Please try again in a moment, or contact support "
            "directly for assistance."
        )
    return {"response": response}


# =============================================================================
# NODE: guardrail_node
# =============================================================================

def guardrail_node(state: RouterState) -> dict:
    """Fallback for anomalous queries — no agent call, zero token cost."""
    response = (
        "I'm not able to help with that request through this system. "
        "I can assist with order cancellations, order tracking, and "
        "refund questions. Could you rephrase your request, or contact "
        "our general support team for other inquiries?"
    )
    return {"response": response}


# =============================================================================
# GRAPH CONSTRUCTION
# =============================================================================

def build_router_graph():
    """
    START -> embed_node -> classify_node -> [conditional edge]
                                                   |
                      +-------------+-------------+--------------+
                      v             v             v              v
                cancel_node   track_node    refund_node    guardrail_node
                      |             |              |              |
                      +-------------+--------------+--------------+
                                          v
                                         END

    Returns a CompiledStateGraph, ready for .invoke({"query": "..."}).
    """
    builder = StateGraph(RouterState)

    # Register nodes
    builder.add_node("embed_node", embed_node)
    builder.add_node("classify_node", classify_node)
    builder.add_node("cancel_node", cancel_node)
    builder.add_node("track_node", track_node)
    builder.add_node("refund_node", refund_node)
    builder.add_node("guardrail_node", guardrail_node)

    # Sequential edges: embedding must complete before classification
    builder.add_edge(START, "embed_node")
    builder.add_edge("embed_node", "classify_node")

    # Conditional edge: classify_node's output determines the next node.
    # The dict maps route_decision's return value -> registered node name.
    builder.add_conditional_edges(
        "classify_node",
        route_decision,
        {
            "cancel_node":    "cancel_node",
            "track_node":     "track_node",
            "refund_node":    "refund_node",
            "guardrail_node": "guardrail_node",
        },
    )

    # All four terminal nodes lead to END
    builder.add_edge("cancel_node", END)
    builder.add_edge("track_node", END)
    builder.add_edge("refund_node", END)
    builder.add_edge("guardrail_node", END)

    return builder.compile()


# =============================================================================
# PUBLIC API
# =============================================================================

# Module-level singleton — building the graph is cheap (no I/O), but this
# keeps a single compiled graph instance shared across requests, consistent
# with the singleton pattern used in embedder.py and classifier.py.
_graph = None


def get_router_graph():
    """Return the compiled router graph, building it on first call."""
    global _graph
    if _graph is None:
        _graph = build_router_graph()
    return _graph


def route_query(query: str) -> dict:
    """
    Run a single customer query through the full pipeline.

    This is the function Phase 5's FastAPI endpoint will call directly.

    Args:
        query: Raw customer message, e.g. "I want to cancel order ORD-1001"

    Returns:
        dict with keys: query, embedding, intent, is_anomaly, confidence, response
        (the full final RouterState)

    Example:
        >>> result = route_query("Where is my order ORD-1002?")
        >>> result["intent"]
        'track_order'
        >>> result["response"]
        'Your order ORD-1002 ... is on its way! Tracking number: TRK-8842-ZX...'
    """
    graph = get_router_graph()
    initial_state: RouterState = {
        "query": query,
        "embedding": np.array([]),  # populated by embed_node
        "intent": "",
        "is_anomaly": False,
        "confidence": 0.0,
        "response": "",
    }
    return graph.invoke(initial_state)