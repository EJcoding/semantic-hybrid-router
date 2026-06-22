# Phase 2 — Deep Learning Layer (Embedding Extraction)

## Overview

Phase 2 solves the fundamental problem every ML routing system faces: computers
cannot reason about language directly. K-Means clustering (Phase 3) can only
measure distance between numbers — it has no concept of what "cancel my order"
means. Phase 2 is the bridge: a local transformer model that converts raw text
into 384-dimensional vectors where semantic similarity becomes geometric
proximity.

Two queries that mean the same thing — "I want to cancel my order" and "please
cancel my recent purchase" — land close together in this 384-dimensional space.
Two queries that mean different things — "cancel my order" and "where is my
package" — land far apart. K-Means exploits this geometry to classify intent
without any LLM involvement.

---

## Files Created

| File | Purpose |
|------|---------|
| `app/pipeline/embedder.py` | Runtime module — loads the model, runs the forward pass, exposes `get_embedding()` |
| `scripts/extract_embeddings.py` | One-off script — batch-embeds the training CSV and saves to `.npz` for Phase 3 |
| `data/training_data.npz` | Precomputed embeddings + labels (generated, not committed to git) |

---

## The Model: `all-MiniLM-L6-v2`

**Source:** `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace Hub)

| Property | Value |
|----------|-------|
| Architecture | Transformer (6 attention layers) |
| Hidden size | 384 dimensions |
| Parameters | ~22 million |
| Download size | ~90 MB |
| Training objective | Semantic similarity via contrastive learning |

The "L6" in the name means 6 attention layers. The "MiniLM" refers to a
knowledge distillation process where a larger model's reasoning was compressed
into this smaller one. The model was specifically fine-tuned for semantic
similarity tasks — which is exactly the use case here — making it a deliberate
choice rather than a default.

### Why not a larger model?

Larger models (BERT-base at 768 dims, 110M params) improve quality marginally for
this task but increase inference cost significantly. For a routing system where
every API request pays the embedding cost before any LLM is called, speed matters.
`all-MiniLM-L6-v2` was benchmarked to be ~5× faster than BERT-base while retaining
~90% of its semantic similarity accuracy.

---

## The Pipeline

```
raw text
   ↓
Tokenizer          "cancel my order" → [101, 17542, 2026, 2344, 102]
   ↓
Embedding matrix   token ID → 384-dim initial vector (lookup table)
   ↓
6× Attention       each token attends to every other token, updating its vector
   ↓
last_hidden_state  (seq_len, 384) — one contextual vector per token
   ↓
Mean pooling       (seq_len, 384) → (384,) — average across real tokens only
   ↓
L2 normalisation   scale to unit length (||v|| = 1.0)
   ↓
np.ndarray (384,)  → K-Means classifier
```

---

## Key Technical Decisions

### Why use the raw `transformers` API instead of `sentence-transformers`?

`sentence-transformers` wraps all of Phase 2 into a single line:
`model.encode("text")`. We deliberately used `AutoTokenizer` + `AutoModel`
directly to understand what is happening at each step — tokenization, forward
pass, pooling. This is the difference between using a tool and understanding it,
which matters when debugging embedding quality issues in production.

### Why mean pooling and not the `[CLS]` token?

The `[CLS]` token is a common shortcut for sentence-level classification (used
in BERT fine-tuning). However, `all-MiniLM-L6-v2` was trained using mean pooling
as its aggregation strategy. Using `[CLS]` here would underperform relative to
how the model was actually optimised.

Mean pooling averages the token vectors across the sequence, weighted by the
attention mask — padding tokens contribute zero to the average.

### Why L2 normalisation?

K-Means uses Euclidean distance. Without normalisation, longer sentences produce
larger-magnitude vectors simply because more token values were summed — not
because the meaning is more extreme. After L2 normalisation, every vector has
length exactly 1.0, placing it on the surface of a 384-dimensional unit sphere.
Distance now measures direction (semantic content) rather than magnitude (sentence
length).

Confirmed in the Phase 2 validation output:
```
L2 norms — min: 1.000000 | max: 1.000000 | mean: 1.000000
```

### Why precompute embeddings to disk?

K-Means is an iterative algorithm — it reads the training data multiple times
during convergence. Recomputing embeddings on each K-Means iteration would run
the transformer forward pass thousands of times. Precomputing once and saving to
`training_data.npz` makes Phase 3 training a pure math operation (no PyTorch)
that completes in seconds. This is the standard MLOps pattern called **feature
caching** or **offline feature extraction**.

### `torch.no_grad()` context manager

During training, PyTorch builds a computation graph to enable backpropagation.
During inference, gradients are unnecessary. Wrapping the forward pass in
`torch.no_grad()` disables this tracking, reducing memory usage by ~50% and
speeding up inference.

### `model.eval()` mode

Calling `.eval()` on the model disables dropout layers (which randomly zero out
neurons during training for regularisation). Without it, the same input would
produce slightly different embeddings on each call — an unacceptable property for
a deterministic routing system.

---

## Device Detection

```python
if torch.backends.mps.is_available():   # Apple Silicon GPU
    return torch.device("mps")
elif torch.cuda.is_available():          # NVIDIA GPU
    return torch.device("cuda")
else:
    return torch.device("cpu")           # Docker containers, fallback
```

Development ran on MPS (Apple M-series); Docker containers run on CPU. The
`_get_device()` function handles this automatically. Container startup logs
confirm: `Target device: cpu`.

---

## Model Storage: Why the Project Directory?

By default, HuggingFace caches models at `~/.cache/huggingface/`. This is
appropriate for the Bitext dataset (a training-time dependency), but the
transformer weights are a **runtime dependency** — every API request calls
`get_embedding()`.

The `HF_HOME=./.cache/huggingface` environment variable redirects the cache
into the project directory so:
1. The `Dockerfile` can `COPY .cache/ ./.cache/` to bake the weights into the
   image — the container never needs a network call at startup.
2. The weights are co-located with the other runtime artifacts (`models/*.pkl`).

The dataset cache stays global (`~/.cache/huggingface/`) because Docker never
needs it.

---

## Output Verification

A successful Phase 2 run produces:

```
Shape : (2990, 384)
dtype : float32
✅ No NaN or Inf values
✅ L2 norms — min: 1.000000 | max: 1.000000 | mean: 1.000000
✅ Mean feature variance: 0.001556  (non-zero = model is active)
```

The non-zero variance confirms that semantically different sentences are landing
in different regions of the 384-dimensional space — the essential precondition
for K-Means to form clean clusters in Phase 3.

---

## Lessons

The transformer forward pass is not a black box — it is a sequence of matrix
operations (attention, feed-forward, normalisation) that can be reasoned about
at each step. Understanding what `last_hidden_state`, mean pooling, and L2
normalisation each contribute makes it possible to debug subtle issues like
the vocabulary mismatch discovered in Phase 4b, where the training data's
`{{Order Number}}` placeholder tokens caused real order IDs in production
queries to shift embeddings outside the learned cluster boundaries.