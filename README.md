# Semantic Hybrid Router

A zero-cost AI routing microservice that classifies customer intent using local
deep learning embeddings and traditional machine learning — eliminating the
latency, token cost, and hallucination risks of LLM-based supervisor agents.

---

## How It Works

Most multi-agent AI systems use an LLM to decide which agent should handle a
given query. This system eliminates that step entirely. Routing happens through
a local mathematical preprocessing layer before any LLM is ever called.

```
User query
    ↓
all-MiniLM-L6-v2      Local transformer — converts text to a 384-dim vector
    ↓                  (PyTorch, runs on-device, zero API calls)
K-Means + DBSCAN       Classifies intent / detects out-of-distribution queries
    ↓                  (scikit-learn, ~1-2ms)
LangGraph router       Conditionally dispatches to the correct agent
    ↓
LangChain agent        Executes the right tool against a SQLite database
    ↓                  (OpenRouter free-tier LLM)
JSON response          intent · confidence · is_anomaly · response
```

**The routing decision costs $0.00 and takes ~1-2ms.** The LLM is only invoked
after the correct agent has already been determined.

---

## Architecture

### Three execution layers

**1. Ingestion & Deep Learning Layer**
Raw text is tokenized and passed through `sentence-transformers/all-MiniLM-L6-v2`
locally via PyTorch. Mean pooling and L2 normalisation produce a 384-dimensional
embedding vector.

**2. Machine Learning Classification Layer**
The embedding is evaluated against a trained K-Means model (intent assignment)
and DBSCAN (anomaly detection). Out-of-distribution queries are rejected before
reaching any agent.

**3. Agentic Execution Layer**
A LangGraph state machine routes the classified intent to one of three
specialised LangChain agents, each equipped with a SQLite tool:

| Intent | Agent | Tool |
|--------|-------|------|
| `cancel_order` | Cancellation agent | `UPDATE orders SET status = 'Cancelled'` |
| `track_order` | Tracking agent | `SELECT` tracking number + delivery date |
| `get_refund` | Refund agent | `SELECT` order total + calculate refund |
| `anomalous` | Guardrail | Returns safe fallback — no LLM called |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.11+ |
| Deep learning | PyTorch, HuggingFace Transformers |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` (~90MB, local) |
| ML classification | scikit-learn (K-Means, DBSCAN), NumPy |
| Orchestration | LangGraph, LangChain 1.0 |
| LLM provider | OpenRouter (free-tier models) |
| Database | SQLite |
| API | FastAPI, Uvicorn |
| Containerisation | Docker, Docker Compose |

---

## Getting Started

### Prerequisites

- Python 3.11+
- Docker Desktop (for containerised deployment)
- A free [OpenRouter](https://openrouter.ai) API key

### 1. Clone and set up the environment

```bash
git clone https://github.com/your-username/semantic-hybrid-router.git
cd semantic-hybrid-router

python3 -m venv venv
source venv/bin/activate

pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and add your OpenRouter API key:

```
OPENROUTER_API_KEY=sk-or-v1-your-key-here
OPENROUTER_MODEL=meta-llama/llama-3.3-70b-instruct:free
HF_HOME=./.cache/huggingface
```

### 3. Prepare the data and train the models

```bash
# Download and filter the training dataset
python scripts/ingest_data.py

# Extract embeddings from the training data
python scripts/extract_embeddings.py

# Train K-Means + DBSCAN classification models
python scripts/train_models.py

# Initialise the mock SQLite database
python app/db/mock_db.py --reset
```

### 4. Run locally

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 5. Run in Docker

```bash
docker compose up --build
```

The Docker image includes the trained models, transformer weights, and seeded
database — no setup scripts needed inside the container.

---

## API Reference

### `POST /api/v1/route_query`

Route a customer query through the full pipeline.

**Request**
```json
{
  "query": "I want to cancel my order ORD-1001"
}
```

**Response**
```json
{
  "intent": "cancel_order",
  "is_anomaly": false,
  "confidence": 0.5536,
  "response": "Your order ORD-1001 has been successfully cancelled. A full refund of $79.99 will be issued within 3–5 business days."
}
```

**Response fields**

| Field | Type | Description |
|-------|------|-------------|
| `intent` | string | Classified intent: `cancel_order`, `track_order`, `get_refund`, or `anomalous` |
| `is_anomaly` | boolean | `true` if DBSCAN flagged the query as out-of-distribution |
| `confidence` | float | Routing confidence in [0.0, 1.0]. `0.0` for anomalous queries |
| `response` | string | Agent's natural-language reply, or guardrail message if anomalous |

### `GET /health`

```json
{"status": "ok"}
```

Interactive API documentation is available at `http://localhost:8000/docs` when
the server is running.

---

## Testing

Run the integration test suite against the live server:

```bash
python scripts/test_api.py
```

Tests all four routing paths, verifies database mutations, and checks edge cases.
Requires the server to be running on `http://localhost:8000`.

---

## Project Structure

```
semantic-hybrid-router/
├── app/
│   ├── main.py                  FastAPI application
│   ├── llm_config.py            Shared LLM configuration (OpenRouter)
│   ├── pipeline/
│   │   ├── embedder.py          Transformer embedding extraction
│   │   ├── classifier.py        K-Means + DBSCAN runtime inference
│   │   └── router_graph.py      LangGraph state machine
│   ├── agents/
│   │   ├── cancel_agent.py      Order cancellation agent
│   │   ├── track_agent.py       Order tracking agent
│   │   └── refund_agent.py      Refund calculation agent
│   ├── tools/
│   │   └── sqlite_tools.py      LangChain @tool functions (SQLite)
│   └── db/
│       └── mock_db.py           Database schema and seed data
├── scripts/
│   ├── ingest_data.py           Phase 1: dataset download and filtering
│   ├── extract_embeddings.py    Phase 2: batch embedding extraction
│   ├── train_models.py          Phase 3: K-Means + DBSCAN training
│   └── test_api.py              Phase 5: API integration tests
├── phases/
│   ├── PHASE_1.md               Data ingestion documentation
│   ├── PHASE_2.md               Embedding extraction documentation
│   ├── PHASE_3.md               ML classification documentation
│   ├── PHASE_4.md               Agents + LangGraph documentation
│   └── PHASE_5.md               API + Docker documentation
├── models/                      Trained sklearn artifacts (generated)
├── data/                        Dataset CSV, embeddings, SQLite DB (generated)
├── .cache/                      HuggingFace model weights (generated)
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Data

The system is trained on the
[Bitext Customer Support LLM Chatbot Training Dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset)
(Apache 2.0), filtered to three intents: `cancel_order`, `track_order`, and
`get_refund` (~2,990 samples, balanced at ~33% per class).

The dataset is not committed to this repository. Running `python scripts/ingest_data.py`
downloads it automatically from HuggingFace Hub.

---

## Definition of Done

From the original project specification:

- [x] A POST request routes raw text to the correct LangChain agent and returns a tool-assisted response
- [x] The architecture utilises $0.00 in embedding/classification costs
- [x] Embedding extraction runs entirely on local hardware without external API calls
- [x] No LLM-based reasoning is used for intent classification

---

## Phase Documentation

Detailed write-ups covering the implementation decisions, technical deep-dives,
and lessons learned for each phase are in the [`phases/`](phases/) directory.