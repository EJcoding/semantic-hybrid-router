# Phase 4 — Agentic Orchestration Layer

## Overview

Phase 4 builds the execution layer of the router — the part that actually does
something with a classified intent. It is split into two distinct sub-phases:

- **Phase 4a:** Three specialised LangChain agents, each owning one customer
  service capability backed by a real SQLite database
- **Phase 4b:** A LangGraph state machine that wires the entire pipeline together
  — from raw text to a final agent response — in a single `.invoke()` call

By the end of Phase 4, a single function call (`route_query("cancel my order ORD-1001")`)
triggers the full chain: embedding → classification → routing → tool execution →
natural language response.

---

---

# Phase 4a — LangChain Agents & SQLite Tools

## Overview

Three independent LangChain agents are built, one per intent. Each agent has
exactly one job, one tool, and a tightly scoped system prompt. The narrow scope
is intentional: it reduces the surface area for the LLM to go off-task and makes
each agent's behaviour predictable and testable in isolation.

---

## Files Created

| File | Purpose |
|------|---------|
| `app/db/mock_db.py` | SQLite schema, seed data (12 mock orders), `get_connection()` |
| `app/tools/sqlite_tools.py` | Three `@tool` decorated functions wrapping SQL operations |
| `app/agents/cancel_agent.py` | Cancellation agent — `run_cancel_agent(query)` |
| `app/agents/track_agent.py` | Tracking agent — `run_track_agent(query)` |
| `app/agents/refund_agent.py` | Refund agent — `run_refund_agent(query)` |
| `app/llm_config.py` | Shared LLM configuration — `get_llm()` |

---

## The Mock Database

The SQLite database (`data/orders.db`) contains 12 mock orders covering all
status branches the tool logic handles:

| Status | Orders | Tool behaviour |
|--------|--------|----------------|
| Processing | ORD-1001, 1004, 1006, 1009 | Cancellable; full refund eligible |
| Shipped | ORD-1002, 1005, 1008, 1010 | Not cancellable; 80% refund |
| Delivered | ORD-1003, 1007, 1012 | Not cancellable; 50% refund |
| Cancelled | ORD-1011 | Already cancelled |

This coverage ensures every code path in every tool is reachable during testing.

The `--reset` flag on `mock_db.py` (`python app/db/mock_db.py --reset`) drops
and re-seeds the table. Running without the flag is read-only — it prints the
current state without modifying anything. This distinction was added after a
test accidentally wiped evidence of a successful cancel by running the script
as a "verification" step, not realising it had `force=True` hardcoded.

---

## LangChain Tools

The `@tool` decorator converts a plain Python function into a LangChain `Tool`
object. The function's docstring becomes the LLM's instruction manual — it is
what the model reads to decide:
- Whether to call this tool for a given query
- What argument values to pass

Docstring quality directly determines tool-calling reliability. Vague docstrings
produce wrong calls. Each tool docstring specifies: what the tool does, when to
use it, the exact argument format expected (e.g., `ORD-XXXX`), and what the
return value contains.

**Tools created:**

| Tool | Operation | Mutates DB? |
|------|-----------|-------------|
| `cancel_order(order_id)` | `UPDATE orders SET status = 'Cancelled'` | Yes |
| `track_order(order_id)` | `SELECT status, tracking_number, estimated_delivery` | No |
| `calculate_refund(order_id)` | `SELECT status, order_total` + arithmetic | No |

All tools open a new SQLite connection on each call and close it in a
`try/finally` block. SQLite connections are not thread-safe and must never be
shared across threads or reused across calls.

---

## LangChain 1.0 API

The original implementation used `create_tool_calling_agent` +
`AgentExecutor` from `langchain.agents` — the standard pattern in LangChain
0.x documentation. These were removed in LangChain 1.0, replaced by a single
`create_agent()` function that builds a LangGraph state machine internally.

The new invocation pattern:
```python
agent = create_agent(model=get_llm(), tools=[cancel_order], system_prompt=SYSTEM_PROMPT)
result = agent.invoke({"messages": [{"role": "user", "content": query}]})
return result["messages"][-1].content
```

`result["messages"]` contains the full conversation: the human message, any
intermediate `AIMessage` objects containing tool calls, `ToolMessage` results,
and a final `AIMessage` with the natural-language response.
`result["messages"][-1].content` extracts that final response.

---

## LLM Configuration

All three agents share a single `ChatOpenRouter` instance via `get_llm()` in
`app/llm_config.py`. Key configuration decisions:

**`temperature=0`** — Deterministic tool-call decisions. The same customer
query produces the same order ID extraction on every run, which is essential
for a routing system where reproducibility is a requirement.

**`max_retries=4`** — Native to `ChatOpenRouter`, retry logic lives inside
the HTTP client layer. Importantly, `.with_retry()` (a generic LangChain
`Runnable` wrapper) was tried first but broke `create_agent` with
`AttributeError: 'RunnableRetry' object has no attribute 'bind_tools'` —
`create_agent` calls `model.bind_tools()` internally, a method specific to
`BaseChatModel` that `RunnableRetry` does not proxy.

**`openrouter_provider={"allow_fallbacks": True, "sort": "throughput"}`** —
If the preferred provider for the model is saturated (returning 429), OpenRouter
automatically retries on an alternate provider serving the same model.

**`model_kwargs={"models": [...3 models...]}`** — A model-level fallback list.
OpenRouter's free tier capacity is shared globally. During development,
`meta-llama/llama-3.3-70b-instruct:free` was repeatedly returning 429 with
`provider_name: "Venice"` — Venice was the only provider serving the free
variant and its capacity was exhausted. The `models` array instructs OpenRouter
to try alternate models if the primary is unavailable. OpenRouter caps this
list at 3 entries.

**Final fallback order:**
1. `meta-llama/llama-3.3-70b-instruct:free` — primary choice (strong tool calling, MoE fast)
2. `openai/gpt-oss-120b:free` — OpenAI Harmony format, native function calling
3. `google/gemma-4-31b-it:free` — native function calling, independent provider pool

---

## Phase 4a Verification

Each agent was tested in isolation before Phase 4b integration:

```bash
python3 -c "from app.agents.cancel_agent import run_cancel_agent; \
  print(run_cancel_agent('cancel order ORD-1001'))"
```

Successful cancellation was verified by checking the database state afterward
(`python3 app/db/mock_db.py`) — confirming the tool executed the SQL `UPDATE`
and committed it, rather than the LLM simply generating plausible-sounding
confirmation text without touching the database.

---

---

# Phase 4b — LangGraph State Machine

## Overview

Phase 4b builds the routing brain of the system: a LangGraph `StateGraph` that
sequences every previous phase into a single executable pipeline. The state
machine receives a raw customer query and returns a final natural-language
response, having internally run embedding, classification, conditional routing,
and agent execution — with zero LLM calls for the routing decision itself.

---

## Files Created

| File | Purpose |
|------|---------|
| `app/pipeline/router_graph.py` | The complete LangGraph state machine |

---

## State Schema

```python
class RouterState(TypedDict):
    query:     str           # raw customer input (set by caller)
    embedding: np.ndarray    # 384-dim vector (set by embed_node)
    intent:    str           # set by classify_node
    is_anomaly: bool         # set by classify_node
    confidence: float        # set by classify_node
    response:  str           # set by whichever terminal node runs
```

A flat `TypedDict` is used rather than LangGraph's `MessagesState` (a chat
history list). `MessagesState` is appropriate inside an individual agent's
tool-calling loop. This graph sits one level above the agents — it routes once
and hands off to exactly one agent, which manages its own `MessagesState`
internally via `create_agent`.

`np.ndarray` in state works for synchronous in-process invocation
(`graph.invoke()`). If LangGraph checkpointing or HTTP streaming of
intermediate state is added in the future, the embedding would need to be
converted to a list at the API boundary, since numpy arrays are not
JSON-serializable.

---

## Graph Structure

```
START
  ↓
embed_node       get_embedding(query)            Phase 2
  ↓
classify_node    classify(embedding)             Phase 3
  ↓
[conditional edge — dict lookup on intent]
  ↓              ↓              ↓              ↓
cancel_node  track_node  refund_node  guardrail_node
  ↓              ↓              ↓              ↓
                          END
```

The conditional edge function (`route_decision`) is a plain Python dict lookup:

```python
_INTENT_TO_NODE = {
    "cancel_order": "cancel_node",
    "track_order":  "track_node",
    "get_refund":   "refund_node",
    "anomalous":    "guardrail_node",
}

def route_decision(state: RouterState) -> str:
    return _INTENT_TO_NODE.get(state["intent"], "guardrail_node")
```

There is no LLM involved in this routing decision. This is the architectural
core of the project: a mathematical preprocessing layer (Phases 2–3) replaces
what would otherwise be an LLM supervisor making routing decisions at token cost.

---

## Query Normalisation

A significant debugging investigation in Phase 4b revealed two vocabulary
mismatches between production queries and the Bitext training distribution:

**1. Order IDs**
Training data uses `{{Order Number}}` as a literal placeholder. Production
queries contain `ORD-1005`. The token sequences are completely different,
shifting embeddings away from trained cluster boundaries.

**2. Refund vocabulary and structure**
The `get_refund` training samples use unusual financial/legal language
("compensation of my money", "restitution", "rebate") and — critically —
contain no order numbers at all. Natural English refund queries like
"I want a refund for ORD-1007" embedded ~0.72 from the nearest core sample,
while the DBSCAN boundary was at `eps=0.6185`. After raising eps to 0.7751
(99th percentile), queries passed DBSCAN but were routed to `cancel_order`
because the presence of `{{Order Number}}` in the normalised query pulled the
embedding toward the cancel cluster, which does contain order numbers.

**Solution: intent-aware normalisation**

```python
def _normalize_query(query: str) -> str:
    query = _ORDER_ID_PATTERN.sub("{{Order Number}}", query)
    has_refund = bool(_REFUND_PATTERN.search(query))
    query = _REFUND_PATTERN.sub("get my money back", query)
    if has_refund:
        # get_refund training data has no order numbers — strip them
        query = _ORDER_CONTEXT_PATTERN.sub('', query)
        query = re.sub(r'\s+', ' ', query).strip()
    return query
```

Normalisation applies only to the embedding input — `state["query"]` is never
modified, so agents always receive the original text with the real order ID.

---

## Error Handling

Each agent node wraps its `run_*_agent()` call in `try/except`:

```python
def cancel_node(state: RouterState) -> dict:
    try:
        response = run_cancel_agent(state["query"])
    except Exception:
        response = "I'm sorry, I'm having trouble processing your request..."
    return {"response": response}
```

If the entire OpenRouter fallback chain fails (all three models return errors),
the node degrades gracefully rather than propagating an exception that would
crash `graph.invoke()`. This was informed directly by the 429 rate-limit issues
encountered during Phase 4a testing — a real production failure mode, not a
hypothetical.

The guardrail node has no try/except because it makes no external calls — it
returns a hardcoded string with zero failure modes.

---

## Phase 4b Verification

End-to-end test across all four routing branches:

| Query | Intent | Anomaly | Confidence | DB mutated |
|-------|--------|---------|-----------|------------|
| "I want to cancel order ORD-1004" | cancel_order | False | 0.55 | ORD-1004 → Cancelled ✅ |
| "Where is my order ORD-1005?" | track_order | False | 0.33 | No (read-only) ✅ |
| "How much refund will I get for ORD-1007?" | get_refund | False | 0.32 | No (read-only) ✅ |
| "What is the capital of France?" | anomalous | True | 0.0 | No (guardrail) ✅ |

---

## Lessons

The vocabulary mismatch problem — where training data and production data speak
slightly different dialects — is one of the most common failure modes in ML
systems deployed against real user input. The Bitext dataset was written with
template placeholders (`{{Order Number}}`) and unusual financial vocabulary for
refund intents, neither of which matches natural user language. The solution
(query normalisation) is lightweight and effective, but the deeper lesson is
that **training-time metrics do not guarantee production-time performance**. The
97.59% routing accuracy measured in Phase 3 was measured against training data
— it was only by testing with realistic production-style queries in Phase 4b
that the gap became visible.

The distinction between `.with_retry()` and `max_retries` exposed an important
principle about LangChain's layered architecture: generic `Runnable` wrappers
(`.with_retry()`, `.with_fallbacks()`) intercept the object at the chain level,
hiding the underlying model's provider-specific methods. Code downstream of a
`Runnable` wrapper (`create_agent`, which calls `model.bind_tools()`) may
depend on those methods being present. Native constructor parameters
(`max_retries`) are always safer when available because they keep the object's
type and interface intact.