"""
scripts/train_models.py
=======================
Phase 3: Machine Learning Classification Layer — Training

WHAT THIS SCRIPT DOES:
  1. Loads pre-computed embeddings from data/training_data.npz (Phase 2).
  2. Trains a K-Means model (n_clusters=3) to learn the 3 intent cluster boundaries.
  3. Discovers the cluster→intent mapping empirically (cluster IDs are arbitrary).
  4. Evaluates K-Means quality with silhouette score, per-cluster purity, and a
     full classification report.
  5. Calibrates DBSCAN's eps threshold automatically using k-NN distance analysis.
  6. Trains DBSCAN and extracts its core-point embeddings for runtime inference.
  7. Saves all models to models/ for Phase 4 to load.

WHY TWO MODELS?
  K-Means: always assigns a cluster. Fast, deterministic. Works for in-domain queries.
  DBSCAN:  the safety net. If a new query is geometrically far from all training
           data (e.g., "How do I reset my router?"), K-Means would still route it —
           just incorrectly. DBSCAN catches those before any agent is ever called.

RUN:
  $ python scripts/train_models.py
"""

import sys
import joblib
import numpy as np
from pathlib import Path
from collections import Counter
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, classification_report
from sklearn.neighbors import NearestNeighbors

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR   = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
NPZ_PATH   = DATA_DIR / "training_data.npz"

# DBSCAN calibration parameters — adjust these if outlier rate is off
MIN_SAMPLES    = 5   # minimum neighborhood density for a core point
EPS_PERCENTILE = 99  # raised from 95 after diagnosing that the get_refund
                     # cluster's training vocabulary ("compensation of my money",
                     # "restitution") is structurally different from natural user
                     # phrasing — the 95th percentile boundary was too tight to
                     # cover legitimate refund queries (min core sample distance
                     # 0.7208 vs eps 0.6185 at 95th percentile).


# =============================================================================
# STEP 1: LOAD EMBEDDINGS
# =============================================================================

def load_training_data() -> tuple[np.ndarray, np.ndarray]:
    """
    Load pre-computed embeddings and labels from Phase 2's npz archive.

    np.load with allow_pickle=True is required because the labels array
    stores Python strings (str dtype requires pickle for serialization).
    """
    print("[1/6] Loading training data...")
    if not NPZ_PATH.exists():
        print("  ❌  training_data.npz not found.")
        print("     Run Phase 2 first: python scripts/extract_embeddings.py")
        sys.exit(1)

    data       = np.load(NPZ_PATH, allow_pickle=True)
    embeddings = data["embeddings"]  # (n, 384) float32
    labels     = data["labels"]      # (n,) str

    print(f"  Loaded {len(embeddings):,} embeddings  shape: {embeddings.shape}")
    return embeddings, labels


# =============================================================================
# STEP 2: TRAIN K-MEANS
# =============================================================================

def train_kmeans(embeddings: np.ndarray) -> KMeans:
    """
    Fit a K-Means model with n_clusters=3 on the 384-dim embeddings.

    KEY PARAMETERS EXPLAINED:

    n_clusters=3
      Must match the number of intents. K-Means finds 3 centroids that
      minimise the total sum of squared distances from each point to its
      nearest centroid (called inertia).

    n_init=20
      K-Means is sensitive to initial centroid placement. It can converge
      to a poor local minimum if the random seed is unlucky. Running 20
      independent trials and keeping the best result (lowest inertia) makes
      the output deterministic in practice, even though each trial is random.
      sklearn's default is 10; we use 20 for extra robustness.

    max_iter=500
      Upper limit on iterations per trial. K-Means stops early if centroids
      stop moving. 500 is generous — well-separated semantic clusters typically
      converge in under 100 iterations.

    random_state=42
      Seeds the random number generator for reproducible centroid initialisation
      across runs. Remove this to explore different random starts.

    algorithm='lloyd'
      The classic K-Means update rule: assign → recalculate → repeat.
      'elkan' is faster for low-dimensional data via the triangle inequality,
      but 'lloyd' is more numerically stable for the high-dimensional (384-dim)
      L2-normalised vectors we're using.
    """
    print("\n[2/6] Training K-Means (n_clusters=3, n_init=20)...")

    kmeans = KMeans(
        n_clusters=3,
        n_init=20,
        max_iter=500,
        random_state=42,
        algorithm='lloyd',
        verbose=0
    )
    kmeans.fit(embeddings)

    print(f"  Converged in {kmeans.n_iter_} iterations")
    print(f"  Final inertia : {kmeans.inertia_:,.1f}")
    print(f"  Centroid shape: {kmeans.cluster_centers_.shape}")

    return kmeans


# =============================================================================
# STEP 3: BUILD CLUSTER → INTENT MAP
# =============================================================================

def build_cluster_map(
    kmeans:     KMeans,
    embeddings: np.ndarray,
    labels:     np.ndarray
) -> dict[int, str]:
    """
    Discover which intent each K-Means cluster corresponds to.

    WHY THIS IS NECESSARY:
      K-Means cluster IDs (0, 1, 2) are assigned by the algorithm based on
      centroid initialisation — they carry no semantic meaning. Cluster 0 will
      not reliably be 'cancel_order' across different runs or machines.

      We solve this by predicting on the labelled training set and asking:
      "For each cluster, what is the majority intent label?"

      This mapping is saved to disk so Phase 4's LangGraph router can perform
      intent-based routing without ever hardcoding a cluster ID.

    COLLISION GUARD:
      If two clusters map to the same intent (shouldn't happen with clean data
      but possible with poor hyper-parameters), we warn loudly.
    """
    print("\n[3/6] Building cluster → intent map...")

    predictions = kmeans.predict(embeddings)
    cluster_map  = {}

    for cid in range(kmeans.n_clusters):
        mask           = predictions == cid
        cluster_labels = labels[mask]
        counts         = Counter(cluster_labels.tolist())
        majority_intent, majority_count = counts.most_common(1)[0]
        cluster_map[cid] = majority_intent
        pct = majority_count / mask.sum() * 100
        print(f"  Cluster {cid} → {majority_intent:20s} ({majority_count:,}/{mask.sum():,} = {pct:.1f}% majority)")

    # Sanity check: all 3 clusters should map to different intents
    if len(set(cluster_map.values())) < kmeans.n_clusters:
        print("  ⚠️  Two clusters mapped to the same intent.")
        print("     Try rerunning with a different random_state or increasing n_init.")

    return cluster_map


# =============================================================================
# STEP 4: EVALUATE K-MEANS QUALITY
# =============================================================================

def evaluate_kmeans(
    kmeans:      KMeans,
    embeddings:  np.ndarray,
    labels:      np.ndarray,
    cluster_map: dict[int, str]
) -> None:
    """
    Run a full quality audit on the K-Means clustering result.

    THREE LENSES:

    1. Silhouette Score (-1 to +1):
       Measures how well-separated clusters are, globally.
       For each point p, it computes:
         a(p) = mean distance from p to all other points in its cluster (cohesion)
         b(p) = mean distance from p to all points in the nearest other cluster (separation)
         s(p) = (b(p) - a(p)) / max(a(p), b(p))
       Score > 0.5 = excellent   ✅
       Score 0.35-0.5 = good     ✅
       Score 0.2-0.35 = moderate ⚠️
       Score < 0.2   = poor      ❌

    2. Per-cluster purity:
       For each cluster, what % of its members share the dominant intent?
       100% = perfectly pure cluster. Our target is >90%.

    3. Classification report (precision, recall, F1):
       After applying the cluster_map, treat this as a standard classification
       problem. Shows exactly which intents are being confused with each other.
       This is the most actionable output for debugging.
    """
    print("\n[4/6] Evaluating K-Means quality...")

    predictions = kmeans.predict(embeddings)

    # ── Silhouette Score (sample for speed) ──────────────────────────────────
    n_sample = min(1500, len(embeddings))
    idx = np.random.RandomState(42).choice(len(embeddings), n_sample, replace=False)
    sil = silhouette_score(embeddings[idx], predictions[idx])

    if sil >= 0.5:
        sil_label = "✅  excellent separation"
    elif sil >= 0.35:
        sil_label = "✅  good separation"
    elif sil >= 0.2:
        sil_label = "⚠️   moderate — may cause some routing errors"
    else:
        sil_label = "❌  poor — clustering may not be reliable"

    print(f"\n  Silhouette score : {sil:.4f}  {sil_label}")

    # ── Per-cluster breakdown ─────────────────────────────────────────────────
    print(f"\n  {'Cluster':<10} {'Intent':<22} {'Size':>6}  {'Purity':>8}")
    print(f"  {'-'*54}")

    for cid in range(kmeans.n_clusters):
        mask           = predictions == cid
        cluster_labels = labels[mask]
        counts         = Counter(cluster_labels.tolist())
        total          = int(mask.sum())
        top_intent, top_count = counts.most_common(1)[0]
        purity         = top_count / total * 100
        flag           = "✅" if purity >= 90 else ("⚠️" if purity >= 75 else "❌")
        print(f"  {cid:<10} {cluster_map[cid]:<22} {total:>6,}  {purity:>7.1f}% {flag}")

    # ── Classification report ─────────────────────────────────────────────────
    mapped_preds = np.array([cluster_map[c] for c in predictions])
    overall_acc  = (mapped_preds == labels).mean() * 100

    flag = "✅" if overall_acc >= 90 else ("⚠️" if overall_acc >= 75 else "❌")
    print(f"\n  Overall routing accuracy: {overall_acc:.2f}%  {flag}")
    print(f"\n  Full classification report (after cluster→intent mapping):")
    print(classification_report(labels, mapped_preds, zero_division=0))


# =============================================================================
# STEP 5: CALIBRATE AND TRAIN DBSCAN
# =============================================================================

def calibrate_and_train_dbscan(
    embeddings: np.ndarray
) -> tuple[DBSCAN, np.ndarray, float]:
    """
    Select eps automatically via k-NN distance analysis, then fit DBSCAN.

    THE EPS SELECTION PROBLEM:
      DBSCAN's eps is the maximum distance between two points to be considered
      "neighbours". Set it too tight and even real customer-support queries get
      flagged as outliers. Set it too loose and off-topic queries slip through.

      The k-NN distance heuristic solves this:
        1. For each training point, find its k-th nearest neighbour (k = min_samples).
        2. Sort those k-NN distances ascending. The resulting curve starts flat
           (dense, in-distribution points) then rises sharply (sparse, outlier points).
        3. The inflection point ("elbow") is the natural eps boundary.
        4. We approximate the elbow with the EPS_PERCENTILE-th percentile, which
           means EPS_PERCENTILE% of training data is considered in-distribution.

    INFERENCE-TIME ANOMALY DETECTION:
      scikit-learn's DBSCAN has no .predict() method — it only works on the
      data it was fit on. To detect outliers at runtime on new individual queries,
      we save the embeddings of DBSCAN's core points.

      A new query is in-distribution if its minimum Euclidean distance to any
      core point is ≤ eps. This closely approximates what DBSCAN would have
      decided if it had seen the new point during training.
    """
    print(f"\n[5/6] Calibrating DBSCAN (min_samples={MIN_SAMPLES}, target percentile={EPS_PERCENTILE})...")

    # ── Step 5a: Compute k-NN distances ──────────────────────────────────────
    print(f"  Computing {MIN_SAMPLES}-NN distances for {len(embeddings):,} points...")

    nbrs = NearestNeighbors(
        n_neighbors=MIN_SAMPLES + 1,  # +1 because a point is its own 0-distance neighbour
        metric='euclidean',
        n_jobs=-1
    ).fit(embeddings)

    distances, _ = nbrs.kneighbors(embeddings)

    # distances[:, 0]          = 0.0     (distance to self — always excluded)
    # distances[:, MIN_SAMPLES] = distance to the MIN_SAMPLES-th neighbour
    knn_distances = np.sort(distances[:, MIN_SAMPLES])  # ascending for the elbow curve

    eps = float(np.percentile(knn_distances, EPS_PERCENTILE))

    print(f"  k-NN distance distribution:")
    print(f"    Min      : {knn_distances.min():.4f}")
    print(f"    25th pct : {np.percentile(knn_distances, 25):.4f}")
    print(f"    Median   : {np.median(knn_distances):.4f}")
    print(f"    75th pct : {np.percentile(knn_distances, 75):.4f}")
    print(f"    {EPS_PERCENTILE}th pct  : {eps:.4f}  ← selected as eps")
    print(f"    Max      : {knn_distances.max():.4f}")

    # ── Step 5b: Train DBSCAN with calibrated eps ─────────────────────────────
    print(f"\n  Training DBSCAN (eps={eps:.4f}, min_samples={MIN_SAMPLES})...")

    dbscan       = DBSCAN(eps=eps, min_samples=MIN_SAMPLES, metric='euclidean', n_jobs=-1)
    dbscan_labels = dbscan.fit_predict(embeddings)

    n_total   = len(embeddings)
    n_outlier = int((dbscan_labels == -1).sum())
    n_core    = int(len(dbscan.core_sample_indices_))
    out_pct   = n_outlier / n_total * 100

    print(f"\n  DBSCAN result:")
    print(f"    Core points      : {n_core:,}")
    print(f"    Border points    : {n_total - n_core - n_outlier:,}")
    print(f"    Outliers flagged : {n_outlier:,} ({out_pct:.1f}%)")

    if out_pct < 0.5:
        print(f"    ⚠️  Very few outliers — eps may be too loose.")
        print(f"       Try lowering EPS_PERCENTILE (e.g., 90).")
    elif out_pct > 15:
        print(f"    ⚠️  High outlier rate — eps may be too tight.")
        print(f"       Try raising EPS_PERCENTILE (e.g., 97).")
    else:
        print(f"    ✅ Outlier rate is well-calibrated.")

    # ── Step 5c: Extract core-point embeddings for runtime use ────────────────
    core_sample_embeddings = embeddings[dbscan.core_sample_indices_]
    print(f"\n  Core sample embeddings saved: {core_sample_embeddings.shape}")

    return dbscan, core_sample_embeddings, eps


# =============================================================================
# STEP 6: SAVE MODELS
# =============================================================================

def save_models(
    kmeans:       KMeans,
    cluster_map:  dict[int, str],
    core_samples: np.ndarray,
    dbscan_eps:   float
) -> None:
    """
    Persist all trained artifacts to models/.

    FILES:
      kmeans.pkl              — The K-Means model (centroids + metadata).
      cluster_map.pkl         — {cluster_id → intent_name} mapping dict.
      dbscan_eps.pkl          — The calibrated eps scalar.
      dbscan_core_samples.npy — Embeddings of all DBSCAN core points.
                                Used at runtime to detect outliers.

    WHY SEPARATE FILES INSTEAD OF ONE BUNDLE?
      Each file has one clear responsibility. During Phase 4 debugging it's much
      easier to inspect or swap one model without touching the others.
      joblib is preferred over pickle for sklearn objects because it handles
      numpy arrays more efficiently (memory-mapped file access).
    """
    print("\n[6/6] Saving models to models/...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(kmeans,       MODELS_DIR / "kmeans.pkl")
    joblib.dump(cluster_map,  MODELS_DIR / "cluster_map.pkl")
    joblib.dump(dbscan_eps,   MODELS_DIR / "dbscan_eps.pkl")
    np.save(                  MODELS_DIR / "dbscan_core_samples.npy", core_samples)

    print(f"  ✅ kmeans.pkl              ({(MODELS_DIR/'kmeans.pkl').stat().st_size / 1024:.1f} KB)")
    print(f"  ✅ cluster_map.pkl         ({(MODELS_DIR/'cluster_map.pkl').stat().st_size / 1024:.1f} KB)")
    print(f"  ✅ dbscan_eps.pkl          ({(MODELS_DIR/'dbscan_eps.pkl').stat().st_size / 1024:.1f} KB)")
    print(f"  ✅ dbscan_core_samples.npy ({(MODELS_DIR/'dbscan_core_samples.npy').stat().st_size / 1024:.1f} KB)")

    print(f"\n  ── Cluster map (needed by Phase 4 router) ──────────────")
    for cid, intent in sorted(cluster_map.items()):
        print(f"  cluster_map[{cid}] = '{intent}'")
    print(f"  dbscan_eps = {dbscan_eps:.4f}")


# =============================================================================
# ENTRYPOINT
# =============================================================================

def main():
    print("=" * 60)
    print("PHASE 3: ML Classification — Training")
    print("=" * 60 + "\n")

    embeddings, labels = load_training_data()
    kmeans             = train_kmeans(embeddings)
    cluster_map        = build_cluster_map(kmeans, embeddings, labels)
    evaluate_kmeans(kmeans, embeddings, labels, cluster_map)
    _, core_samples, eps = calibrate_and_train_dbscan(embeddings)
    save_models(kmeans, cluster_map, core_samples, eps)

    print("\n" + "=" * 60)
    print("✅ Phase 3 Complete.")
    print("   Next: Phase 4 — LangGraph + LangChain Agents")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()