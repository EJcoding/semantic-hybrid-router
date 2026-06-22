# Phase 3 — Machine Learning Classification Layer

## Overview

Phase 3 takes the 384-dimensional embeddings produced in Phase 2 and trains two
scikit-learn models that together form the classification and safety layer of the
router. This is the last phase that runs before any LLM is ever involved — all
intent classification happens here, in pure mathematics.

**K-Means** answers: *which agent should handle this query?*
**DBSCAN** answers: *should we even try to route this query at all?*

Both models are trained once during development, saved to disk, and loaded at
server startup. At request time, classification takes ~1-2ms on CPU.

---

## Files Created

| File | Purpose |
|------|---------|
| `scripts/train_models.py` | One-off training script — trains K-Means + DBSCAN, evaluates quality, saves models |
| `app/pipeline/classifier.py` | Runtime module — loads saved models, exposes `classify()` |
| `models/kmeans.pkl` | Trained K-Means model (3 centroids, 384 dims each) |
| `models/cluster_map.pkl` | `{cluster_id → intent_name}` mapping dict |
| `models/dbscan_eps.pkl` | Calibrated eps threshold scalar |
| `models/dbscan_core_samples.npy` | Embeddings of DBSCAN core points (for runtime anomaly detection) |

---

## K-Means: Intent Classification

### How it works

K-Means finds 3 centroids in 384-dimensional space — one per intent — that
minimise the total sum of squared distances from each training point to its
nearest centroid (called **inertia**). At inference time, classifying a new query
is a single nearest-centroid lookup.

### The cluster ID mapping problem

K-Means cluster IDs (0, 1, 2) are assigned by the algorithm based on random
initialisation — they carry no semantic meaning. Cluster 0 is not guaranteed to
be `cancel_order` across different runs or machines. The cluster→intent mapping
is discovered empirically after training by asking: *for each cluster, what is
the majority intent label among its training members?*

This mapping is saved as `cluster_map.pkl` so the Phase 4 router never needs to
hardcode cluster IDs.

**Actual mapping produced on this machine:**
```
cluster_map[0] = 'get_refund'
cluster_map[1] = 'track_order'
cluster_map[2] = 'cancel_order'
```

### Training results

```
Silhouette score     : 0.2908
Overall accuracy     : 97.59%
cancel_order purity  : 99.9%
get_refund purity    : 99.9%
track_order purity   : 93.4%
```

The silhouette score of 0.29 looks moderate but is expected for 384-dimensional
data. The curse of dimensionality compresses silhouette scores toward zero as
dimensions increase — in high-dimensional space, pairwise distances between all
points tend to converge. A score of 0.29 at 384 dims corresponds to the clean
cluster separation visible in the 97.59% routing accuracy.

The `track_order` cluster has lower purity (93.4%) because ~70 `cancel_order`
training queries landed closer to the track centroid. These are semantically
adjacent — "cancel order #1234" and "track order #1234" share significant
vocabulary and structural similarity.

---

## DBSCAN: Anomaly Detection

### Why a second model?

K-Means always assigns a cluster. If a user asks "How do I reset my router?",
K-Means will route it to whichever agent happens to have the nearest centroid —
incorrectly. DBSCAN acts as a safety net: queries that are geometrically far from
all training data are rejected before any agent is called.

### How DBSCAN works here

DBSCAN identifies **core points** — training embeddings that have at least
`min_samples` neighbours within `eps` distance. The dense, in-distribution
regions of the embedding space are represented by these core points.

At inference time, DBSCAN's role is approximated with a simple check: if the
minimum Euclidean distance from the query embedding to any core point exceeds
`eps`, the query is flagged as anomalous.

### scikit-learn's DBSCAN has no `.predict()` method

This is a key implementation detail. Unlike K-Means, `DBSCAN.fit_predict()` only
works on the data it was fit on — there is no way to call it on a new individual
point. The solution: save the embeddings of all core points to
`dbscan_core_samples.npy` (shape: (2965, 384)) and perform the distance check
manually at runtime.

### Eps calibration via k-NN distances

Setting `eps` by hand is unreliable. The automated approach:
1. For each training point, find its k-th nearest neighbour (k = `min_samples`)
2. Sort these distances ascending
3. Use the `EPS_PERCENTILE`-th percentile as eps

This means `EPS_PERCENTILE`% of training data is considered in-distribution.
The percentile was raised from 95 to 99 after diagnosing that the `get_refund`
cluster's training vocabulary ("compensation of my money", "restitution",
"rebate") is structurally unusual — the 95th percentile boundary placed the
in-distribution threshold at `eps=0.6185`, while legitimate refund queries in
natural English landed at a minimum core sample distance of `0.7208`.

**Final calibration:**
```
EPS_PERCENTILE : 99
eps selected   : 0.7751
Core points    : 2,965
Outliers       : 10 (0.3%)
```

---

## Runtime Classification: `classifier.py`

The `classify(embedding)` function performs two sequential checks:

```
1. DBSCAN check  → is min(distance to core samples) > eps?
                   Yes → return ClassificationResult(intent="anomalous")

2. K-Means check → which centroid is nearest?
                   → lookup cluster_map[cluster_id]
                   → compute confidence score
                   → return ClassificationResult(intent=..., confidence=...)
```

### Confidence scoring

```
confidence = 1 - (d_nearest_centroid / d_second_nearest_centroid)
```

A query deep inside one cluster (d_nearest much smaller than d_second) scores
near 1.0. A query equidistant from two centroids scores near 0.0. This metric
reflects the routing certainty, not an absolute probability.

### Singleton pattern

All four model objects (`_kmeans`, `_cluster_map`, `_core_samples`,
`_dbscan_eps`) are module-level variables populated on first call to
`_load_models()`. Loading ~4.5MB of model files once at startup and caching
in memory means every subsequent `classify()` call is pure numpy operations
with no I/O.

---

## Key Decisions

### Why joblib over pickle for saving models?

`joblib.dump()` handles numpy arrays inside sklearn objects more efficiently
than Python's `pickle` — it uses memory-mapped file access and can compress
large arrays. For the core samples array (2965 × 384 float32 ≈ 4.4MB),
joblib is the standard and recommended serialisation tool for sklearn artifacts.

### Why separate files instead of one bundle?

Each file has a single clear responsibility. During debugging, any individual
model can be inspected or replaced without touching the others. `cluster_map.pkl`
is a plain Python dict that can be loaded and printed in one line without
reinstantiating the full sklearn models.

---

## Lessons

The relationship between silhouette score and routing accuracy demonstrated a
core principle: no single metric tells the whole story. A silhouette score that
triggers a warning flag (0.29, "moderate") still corresponds to 97.59% routing
accuracy — the metric that actually matters for the application. Understanding
*why* the silhouette score is compressed in high dimensions (curse of
dimensionality) prevents misinterpreting a valid result as a problem.

The DBSCAN calibration problem — where the 95th percentile eps was too tight
for the `get_refund` cluster — was not discoverable from the training metrics
alone. It surfaced in Phase 4b when real user queries were tested end-to-end.
This is a recurring theme in ML systems: training-time metrics are necessary
but not sufficient validation. The system must be tested with realistic inputs.