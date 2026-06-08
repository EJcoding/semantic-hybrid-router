"""
app/pipeline/classifier.py
===========================
Phase 3: Runtime Classification Module

WHAT THIS MODULE DOES:
  Loads the trained K-Means and DBSCAN models and exposes a single
  classify() function that takes a 384-dim embedding and returns a
  structured routing decision.

  This is the function the LangGraph state machine (Phase 4) calls
  on every incoming API request. It runs in roughly 1-2ms on CPU.

HOW IT FITS IN THE PIPELINE:
  384-dim vector
      ↓
  [DBSCAN outlier check]  ← if anomalous → return "anomalous" result
      ↓
  [K-Means cluster assign] ← assigns cluster_id 0, 1, or 2
      ↓
  [Cluster map lookup]     ← maps cluster_id → intent name
      ↓
  ClassificationResult → LangGraph router → correct LangChain agent
"""

import joblib
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict


# =============================================================================
# CONFIGURATION
# =============================================================================

# Navigate up from app/pipeline/ to project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR   = PROJECT_ROOT / "models"


# =============================================================================
# RESULT TYPE
# =============================================================================

@dataclass
class ClassificationResult:
    """
    Structured return type from classify().

    Using a dataclass rather than a plain dict provides:
      - IDE autocomplete on result.intent, result.is_anomaly
      - Readable __repr__ for debugging
      - .to_dict() for JSON serialization in the FastAPI response (Phase 5)

    Fields:
      cluster_id : K-Means cluster (0, 1, or 2). -1 if anomalous.
      intent     : "cancel_order", "track_order", "get_refund", or "anomalous"
      is_anomaly : True if DBSCAN flagged this query as out-of-distribution
      confidence : Routing confidence score in [0.0, 1.0]
                   0.0 = on the decision boundary between two clusters
                   1.0 = deep inside a single cluster, unambiguous
    """
    cluster_id : int
    intent     : str
    is_anomaly : bool
    confidence : float

    def to_dict(self) -> dict:
        return asdict(self)


# =============================================================================
# MODEL CACHE  (Singleton)
# =============================================================================
# All models are loaded once on first use and reused for every subsequent call.
# This avoids the I/O cost of reading from disk on each API request.

_kmeans        = None
_cluster_map   = None
_core_samples  = None  # np.ndarray: (n_core, 384) — DBSCAN core point embeddings
_dbscan_eps    = None  # float: calibrated eps threshold
_models_loaded = False


def _load_models() -> None:
    """
    Load all model artifacts into module-level cache.
    No-op after the first call.

    Raises FileNotFoundError with a helpful message if any model file is missing,
    rather than a confusing AttributeError later when the None is accessed.
    """
    global _kmeans, _cluster_map, _core_samples, _dbscan_eps, _models_loaded

    if _models_loaded:
        return

    required = {
        "kmeans.pkl":              "K-Means model",
        "cluster_map.pkl":         "cluster→intent mapping",
        "dbscan_eps.pkl":          "DBSCAN eps threshold",
        "dbscan_core_samples.npy": "DBSCAN core embeddings",
    }
    for filename, description in required.items():
        path = MODELS_DIR / filename
        if not path.exists():
            raise FileNotFoundError(
                f"\n[Classifier] ❌  {description} not found at: {path}"
                f"\n   Run Phase 3 training first: python scripts/train_models.py"
            )

    print("[Classifier] Loading models...")
    _kmeans       = joblib.load(MODELS_DIR / "kmeans.pkl")
    _cluster_map  = joblib.load(MODELS_DIR / "cluster_map.pkl")
    _dbscan_eps   = joblib.load(MODELS_DIR / "dbscan_eps.pkl")
    _core_samples = np.load(MODELS_DIR / "dbscan_core_samples.npy")

    _models_loaded = True
    print(f"[Classifier] ✅ Ready.")
    print(f"[Classifier]    Centroids    : {_kmeans.cluster_centers_.shape}")
    print(f"[Classifier]    Cluster map  : {_cluster_map}")
    print(f"[Classifier]    Core samples : {_core_samples.shape[0]:,} points")
    print(f"[Classifier]    DBSCAN eps   : {_dbscan_eps:.4f}\n")


# =============================================================================
# ANOMALY DETECTION
# =============================================================================

def _is_anomaly(embedding: np.ndarray) -> bool:
    """
    Determine if an embedding is out-of-distribution using DBSCAN core samples.

    APPROACH:
      During training, DBSCAN identified "core points" — embeddings with at least
      min_samples neighbours within eps distance. These represent the dense,
      in-distribution regions of the embedding space.

      For a new query, we ask: is this point within eps of ANY core point?
        Yes → in-distribution, proceed to K-Means routing
        No  → out-of-distribution, return safe fallback

    THE MATH:
      distances = ||core_samples - embedding||_2  for all core points
      is_anomaly = min(distances) > eps

      Broadcasting handles the shape arithmetic:
        _core_samples : (n_core, 384)
        embedding     : (384,)
        subtraction   : (n_core, 384)   ← broadcast embedding to every row
        norm(axis=1)  : (n_core,)
        min()         : scalar

      For ~2,000 core samples at 384 dims, this takes ~0.3-0.5ms on CPU.
      Fast enough for a sub-50ms request budget.
    """
    distances    = np.linalg.norm(_core_samples - embedding, axis=1)
    min_distance = float(np.min(distances))
    return min_distance > _dbscan_eps


# =============================================================================
# CONFIDENCE SCORING
# =============================================================================

def _compute_confidence(embedding: np.ndarray) -> float:
    """
    Score how confident the K-Means routing decision is.

    FORMULA:
      confidence = 1 - (d_nearest_centroid / d_second_nearest_centroid)

      d_nearest  ≈ 0, d_second  ≈ 1.0  →  confidence ≈ 1.0  (clear winner)
      d_nearest  = 0.9, d_second = 1.0  →  confidence ≈ 0.1  (borderline)
      d_nearest  = d_second              →  confidence = 0.0  (on the boundary)

    WHY THIS METRIC?
      The absolute distance to the centroid doesn't tell you much on its own.
      What matters is the *relative* distance: how much closer is the assigned
      centroid compared to the next candidate? A point equidistant from two
      centroids (confidence ≈ 0) might be misrouted; a point deep inside one
      cluster (confidence ≈ 1) almost certainly won't be.

    ARGS:
      embedding : np.ndarray shape (384,) — the query embedding

    RETURNS:
      float in [0.0, 1.0]
    """
    # Distance from query to each centroid
    distances    = np.linalg.norm(_kmeans.cluster_centers_ - embedding, axis=1)

    # Sort ascending: [d_min, d_2nd, d_3rd]
    sorted_dists = np.sort(distances)
    d_nearest    = sorted_dists[0]
    d_second     = sorted_dists[1]

    if d_second == 0.0:
        return 1.0  # degenerate case: query is exactly at a centroid

    confidence = 1.0 - (d_nearest / d_second)
    return round(float(np.clip(confidence, 0.0, 1.0)), 4)


# =============================================================================
# PUBLIC API
# =============================================================================

def classify(embedding: np.ndarray) -> ClassificationResult:
    """
    Classify a query embedding and return a structured routing decision.

    Called by the LangGraph state machine in Phase 4 on every request.

    PIPELINE INSIDE THIS FUNCTION:
      1. Load models (lazy — one-time I/O cost, ~50ms, amortised across all requests)
      2. DBSCAN anomaly check (reject out-of-domain queries before agents are called)
      3. K-Means cluster assignment (nearest centroid in 384-dim space)
      4. Cluster map lookup (arbitrary cluster_id → semantic intent name)
      5. Confidence scoring (how unambiguous is the routing decision?)

    ARGS:
      embedding : np.ndarray
        Shape (384,) or (1, 384). L2-normalised output of embedder.get_embedding().

    RETURNS:
      ClassificationResult dataclass with:
        cluster_id = 0, 1, 2  (or -1 if anomalous)
        intent     = "cancel_order" | "track_order" | "get_refund" | "anomalous"
        is_anomaly = True if DBSCAN flagged as out-of-distribution
        confidence = 0.0–1.0  (routing confidence; 0.0 for anomalous)

    EXAMPLE:
      >>> from app.pipeline.embedder import get_embedding
      >>> from app.pipeline.classifier import classify
      >>> emb    = get_embedding("I want to cancel my order")
      >>> result = classify(emb)
      >>> print(result)
      ClassificationResult(cluster_id=1, intent='cancel_order', is_anomaly=False, confidence=0.7841)
      >>> result.to_dict()
      {'cluster_id': 1, 'intent': 'cancel_order', 'is_anomaly': False, 'confidence': 0.7841}
    """
    _load_models()

    # Ensure shape is (384,) regardless of what the caller passes
    embedding = embedding.flatten()

    # ── Stage 1: Anomaly detection ────────────────────────────────────────────
    # Ask DBSCAN: is this query far from all known in-distribution data?
    # If yes, we return immediately — no agent is ever invoked.
    if _is_anomaly(embedding):
        return ClassificationResult(
            cluster_id=-1,
            intent="anomalous",
            is_anomaly=True,
            confidence=0.0
        )

    # ── Stage 2: K-Means cluster assignment ───────────────────────────────────
    # predict() requires shape (1, 384) and returns shape (1,)
    cluster_id = int(_kmeans.predict(embedding.reshape(1, -1))[0])

    # ── Stage 3: Intent lookup ────────────────────────────────────────────────
    # cluster_map is a plain Python dict: {0: "track_order", 1: "cancel_order", ...}
    intent = _cluster_map[cluster_id]

    # ── Stage 4: Confidence scoring ───────────────────────────────────────────
    confidence = _compute_confidence(embedding)

    return ClassificationResult(
        cluster_id=cluster_id,
        intent=intent,
        is_anomaly=False,
        confidence=confidence
    )