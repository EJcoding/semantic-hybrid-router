# Phase 1 — Environment Setup & Data Ingestion

## Overview

Phase 1 establishes the data foundation the entire system depends on. The core
task is pulling a large customer support dataset from HuggingFace, filtering it
to exactly 3 intents, and saving a clean training set to disk.

The 3-intent constraint is not arbitrary. Every downstream decision — the number
of K-Means clusters in Phase 3, the number of LangChain agents in Phase 4, the
number of routing branches in the LangGraph state machine — is built on this
choice. Changing the intent count here would require rebuilding every subsequent
phase.

---

## Files Created

| File | Purpose |
|------|---------|
| `scripts/ingest_data.py` | Downloads the Bitext dataset, audits it, filters to 3 intents, saves to CSV |
| `data/filtered_intents.csv` | The training set — 2,990 rows × 5 columns (generated, not committed to git) |

---

## Dataset

**Source:** `bitext/Bitext-customer-support-llm-chatbot-training-dataset` (HuggingFace Hub)

The full dataset contains 26,872 rows across 27 customer service intent
categories. We filter to 3:

| Intent | Rows | Agent (Phase 4) |
|--------|------|-----------------|
| `cancel_order` | 998 | Cancellation agent — SQL UPDATE |
| `track_order` | 995 | Tracking agent — SQL SELECT |
| `get_refund` | 997 | Refund agent — SQL SELECT + calculation |

The near-perfect 33%/33%/33% class balance is important. K-Means is sensitive to
class imbalance — a heavily skewed distribution would bias the algorithm toward
the majority class, producing fuzzy cluster boundaries and unreliable routing.

---

## Key Decisions

### Why these 3 intents specifically?

They map cleanly to distinct database operations: one write (cancel), one read
(track), one read with computation (refund). This makes the Phase 4 tool
implementations concrete and testable rather than abstract.

### Why CSV over Parquet?

For ~3,000 rows there is no meaningful performance difference. CSV is
human-readable, requires no additional dependencies (`pyarrow`), and can be
opened directly in a text editor or spreadsheet for inspection. Parquet would be
appropriate at millions of rows.

### Why is the dataset not committed to git?

Three reasons: (1) Git stores file history permanently — committing a large file
and deleting it later still embeds it in the repository forever. (2) The Apache
2.0 dataset license permits use but redistributing third-party data directly is
different from referencing it. (3) `ingest_data.py` documents exactly what the
data is, where it comes from, and how it was shaped — one command fully
reproduces it. That is more transparent than a raw CSV.

---

## Technical Notes

### `load_dataset()` caching

HuggingFace's `datasets` library downloads Parquet files on first call and caches
them at `~/.cache/huggingface/datasets/`. Subsequent runs load from cache
instantly. The global cache location is appropriate here because the dataset is
only used during development (a training-time dependency), not at server runtime.
See Phase 2 for why the model weights are treated differently.

### `reset_index(drop=True)` after filtering

After filtering with `.isin()`, the DataFrame index has gaps (rows 0, 1, 5, 12,
...) because the original row positions are preserved. Resetting produces a
contiguous 0..N index, which matters in Phase 2 when positional indexing is used
during batch embedding extraction.

### Class balance check

`ingest_data.py` calculates the imbalance ratio (max class count / min class
count) and warns if it exceeds 2.0. On the Bitext dataset this ratio is 1.0x,
confirming the three intents are evenly distributed.

---

## Output Verification

A successful Phase 1 run produces:

```
Rows: 2,990
cancel_order    998 rows  (33.4%)
track_order     995 rows  (33.3%)
get_refund      997 rows  (33.3%)
✅ Class balance looks good (ratio: 1.0x)
```

---

## Lessons

Data quality is the foundation of the entire pipeline. A poorly filtered or
imbalanced training set would propagate errors forward into every subsequent
phase — fuzzy clusters in Phase 3, incorrect routing in Phase 4, wrong agents
called in production. The audit step in `ingest_data.py` (inspecting columns,
checking for nulls, printing the full intent distribution) is not boilerplate —
it is the engineering practice of verifying your assumptions before building on
them.