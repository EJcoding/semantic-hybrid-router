"""
scripts/extract_embeddings.py
==============================
Phase 2: Batch Training Data Embedding

WHAT THIS SCRIPT DOES:
  Loads data/filtered_intents.csv (produced by Phase 1), runs all rows
  through the local embedding model, and saves the resulting matrix to
  data/training_data.npz for Phase 3 K-Means training.

WHY PRECOMPUTE AND SAVE?
  K-Means is an iterative algorithm — it repeatedly recalculates cluster
  centroids until convergence, meaning it reads the training data many times.
  If we called the transformer on each iteration, we'd run thousands of
  forward passes. By precomputing once and saving to disk, Phase 3 training
  becomes a pure math operation (no PyTorch involved) and completes in seconds.

  This is a standard MLOps pattern called "feature caching" or "offline
  feature extraction" — compute expensive features once, store them cheaply.

OUTPUT FILE: data/training_data.npz
  A NumPy archive containing two arrays:
    - "embeddings" : float32 array of shape (n_samples, 384)
    - "labels"     : string array of shape (n_samples,)
  They are index-aligned: embeddings[i] corresponds to labels[i].

RUN:
  $ python scripts/extract_embeddings.py
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# ── Path Setup ────────────────────────────────────────────────────────────────
# We need to import from app/pipeline/embedder.py, which lives in the project
# root — not inside the scripts/ directory. sys.path.insert() adds the project
# root to Python's module search path so the import resolves correctly.
#
# WHY NOT use a relative import (from ..app.pipeline import ...)?
#   Relative imports only work inside packages (directories with __init__.py).
#   scripts/ is a loose directory of runnable scripts, not a package.
#   sys.path manipulation is the correct approach for this pattern.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.pipeline.embedder import get_embeddings_batch

# ── File Paths ────────────────────────────────────────────────────────────────
DATA_DIR   = PROJECT_ROOT / "data"
INPUT_CSV  = DATA_DIR / "filtered_intents.csv"
OUTPUT_NPZ = DATA_DIR / "training_data.npz"


# =============================================================================
# STEP 1: LOAD
# =============================================================================

def load_training_data() -> tuple[list[str], list[str]]:
    """
    Load the filtered CSV from Phase 1.

    Returns two aligned lists:
      texts  : the raw user query strings (the model's input)
      labels : the intent strings (used to verify cluster quality in Phase 3)
    """
    print(f"[1/4] Loading: {INPUT_CSV.name}")

    if not INPUT_CSV.exists():
        print(f"\n❌ Error: {INPUT_CSV} not found.")
        print("   Run Phase 1 first: python scripts/ingest_data.py")
        sys.exit(1)

    df = pd.read_csv(INPUT_CSV)

    # Validate expected columns exist
    required_cols = {"instruction", "intent"}
    if not required_cols.issubset(df.columns):
        print(f"\n❌ Error: CSV missing expected columns.")
        print(f"   Expected: {required_cols}")
        print(f"   Found:    {set(df.columns)}")
        sys.exit(1)

    texts  = df["instruction"].tolist()
    labels = df["intent"].tolist()

    print(f"      Loaded {len(texts):,} rows")
    print(f"      Intent distribution:")
    for intent, count in df["intent"].value_counts().items():
        print(f"        {intent:<22} {count:>5,} rows")

    return texts, labels


# =============================================================================
# STEP 2: EMBED
# =============================================================================

def extract_embeddings(texts: list[str]) -> np.ndarray:
    """
    Run all texts through the local transformer model in batches.
    See app/pipeline/embedder.py for the full technical explanation.
    """
    print(f"\n[2/4] Extracting embeddings...")
    print(f"      {len(texts):,} texts | batch_size=64")
    print(f"      Estimated time: 1–3 min on CPU, 20–40s on MPS\n")

    embeddings = get_embeddings_batch(texts, batch_size=64)

    print(f"\n      Shape : {embeddings.shape}")   # (n, 384)
    print(f"      dtype : {embeddings.dtype}")     # float32
    return embeddings


# =============================================================================
# STEP 3: VALIDATE
# =============================================================================

def validate_embeddings(embeddings: np.ndarray) -> None:
    """
    Run sanity checks on the extracted embeddings before saving.

    CHECKS:
      1. Shape:         Should be (n_samples, 384)
      2. No NaN/Inf:    Corrupted embeddings would silently break K-Means
      3. L2 norms:      All vectors should have length ≈ 1.0 (unit-normalized)
      4. Variance:      Non-zero variance confirms the model actually ran
                        (all-zero or all-same embeddings = bug)
    """
    print(f"\n[3/4] Validating embeddings...")

    # Check 1: Shape
    assert embeddings.ndim == 2,    f"Expected 2D array, got {embeddings.ndim}D"
    assert embeddings.shape[1] == 384, f"Expected 384 dims, got {embeddings.shape[1]}"
    print(f"      ✅ Shape check passed: {embeddings.shape}")

    # Check 2: NaN / Inf
    has_nan = np.isnan(embeddings).any()
    has_inf = np.isinf(embeddings).any()
    if has_nan or has_inf:
        print(f"      ❌ Found NaN: {has_nan} | Inf: {has_inf}")
        sys.exit(1)
    print(f"      ✅ No NaN or Inf values")

    # Check 3: L2 norms (all should be ~1.0 after normalization)
    norms = np.linalg.norm(embeddings, axis=1)
    print(f"      ✅ L2 norms — min: {norms.min():.6f} | max: {norms.max():.6f} | mean: {norms.mean():.6f}")
    if not np.allclose(norms, 1.0, atol=1e-5):
        print(f"      ⚠️  Warning: Some vectors are not unit-normalized (unexpected)")

    # Check 4: Variance across the dataset
    variance = embeddings.var(axis=0).mean()
    print(f"      ✅ Mean feature variance: {variance:.6f} (non-zero = model is active)")


# =============================================================================
# STEP 4: SAVE
# =============================================================================

def save_training_data(embeddings: np.ndarray, labels: list[str]) -> None:
    """
    Save embeddings and labels together in a single .npz file.

    WHY .npz (not .npy or .csv)?
      .npy     : single array only — we have two aligned arrays to save
      .csv     : can't natively store float32 arrays efficiently
      .npz     : NumPy's native archive format. Stores multiple named arrays
                 in one file, compressed. np.load() retrieves them by name.

    LOADING IN PHASE 3:
      data = np.load("data/training_data.npz", allow_pickle=True)
      embeddings = data["embeddings"]   # (n, 384) float32
      labels     = data["labels"]       # (n,) str

    INDEX ALIGNMENT:
      embeddings[i] and labels[i] always correspond to the same training row.
      This alignment is critical for Phase 3's cluster label mapping step.
    """
    print(f"\n[4/4] Saving → {OUTPUT_NPZ.name}")

    np.savez(
        OUTPUT_NPZ,
        embeddings=embeddings,
        labels=np.array(labels)  # convert list[str] to np.ndarray for archiving
    )

    file_size_mb = OUTPUT_NPZ.stat().st_size / (1024 * 1024)
    print(f"      ✅ Saved  ({file_size_mb:.2f} MB)")
    print(f"         'embeddings' : {embeddings.shape} float32")
    print(f"         'labels'     : ({len(labels)},) str")


# =============================================================================
# ENTRYPOINT
# =============================================================================

def main():
    print("=" * 60)
    print("PHASE 2: Embedding Extraction")
    print("=" * 60 + "\n")

    texts, labels      = load_training_data()
    embeddings         = extract_embeddings(texts)
    validate_embeddings(embeddings)
    save_training_data(embeddings, labels)

    print("\n" + "=" * 60)
    print("✅ Phase 2 Complete.")
    print("   data/training_data.npz is ready for K-Means training.")
    print("   Next: python scripts/train_models.py  (Phase 3)")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()