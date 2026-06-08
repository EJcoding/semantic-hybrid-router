"""
scripts/ingest_data.py
======================
Phase 1: Data Ingestion & Filtering

WHAT THIS SCRIPT DOES:
  1. Downloads the Bitext customer-support dataset from HuggingFace Hub.
  2. Audits the raw data (columns, intent distribution, sample rows).
  3. Filters down to exactly 3 intents that will map to our 3 LangChain agents.
  4. Saves the clean dataset to data/filtered_intents.csv.

WHY THIS MATTERS FOR THE OVERALL SYSTEM:
  In Phase 3 we train a K-Means model with n_clusters=3. K-Means is an
  unsupervised algorithm — it groups data by geometric proximity in
  embedding space. If we train it on clean, well-separated intent clusters,
  the cluster boundaries become crisp. That crispness is what makes the
  routing in Phase 4 deterministic rather than probabilistic.

  Think of it this way:
    Garbage data → fuzzy clusters → ambiguous routing → wrong agent called
    Clean data   → tight clusters → confident routing → correct agent called

RUN THIS SCRIPT:
  From your project root with venv activated:
  $ python scripts/ingest_data.py
"""

import os
import sys
import pandas as pd
from pathlib import Path
from datasets import load_dataset


# =============================================================================
# CONFIGURATION
# =============================================================================

# The 3 intents we care about. Each maps 1:1 to a LangChain agent in Phase 4:
#   cancel_order  → Agent 1 (Cancellations): updates SQLite order status
#   track_order   → Agent 2 (Tracking):      reads tracking number from SQLite
#   get_refund    → Agent 3 (Refunds):        calculates refund amount
TARGET_INTENTS = ["cancel_order", "track_order", "get_refund"]

# Resolve paths relative to this script's location, not the working directory.
# Path(__file__) = absolute path to THIS script file
# .parent        = scripts/
# .parent.parent = project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
OUTPUT_PATH  = DATA_DIR / "filtered_intents.csv"

# The HuggingFace dataset identifier
DATASET_NAME = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"


# =============================================================================
# STEP 1: LOAD RAW DATASET
# =============================================================================

def load_raw_dataset() -> pd.DataFrame:
    """
    Stream and cache the Bitext dataset from HuggingFace Hub.

    ABOUT load_dataset():
      - On first run: downloads Parquet files and caches them in
        ~/.cache/huggingface/datasets/
      - On subsequent runs: loads from cache instantly (no re-download)
      - split="train" is required because this dataset only has a train split
        (no validation or test splits)

    ABOUT .to_pandas():
      HuggingFace returns a Dataset object (their custom format). We convert
      it immediately to a Pandas DataFrame because:
        1. Pandas is more familiar and widely understood
        2. Scikit-learn and our downstream pipeline expect NumPy arrays,
           which Pandas converts to trivially with .values or .to_numpy()
    """
    print("=" * 60)
    print("PHASE 1: Data Ingestion & Filtering")
    print("=" * 60)
    print(f"\n[1/4] Downloading dataset: {DATASET_NAME}")
    print("      (This may take 30–60s on first run; cached afterwards)\n")

    dataset = load_dataset(DATASET_NAME, split="train")
    df = dataset.to_pandas()

    print(f"      ✅ Loaded {len(df):,} rows across {df['intent'].nunique()} intents")
    return df


# =============================================================================
# STEP 2: AUDIT THE RAW DATA
# =============================================================================

def audit_dataset(df: pd.DataFrame) -> None:
    """
    Print a structured audit of the raw dataset before we touch it.

    WHY AUDIT FIRST?
      Before any transformation in an ML pipeline, you want to verify:
        - The columns are what you expect (no schema drift)
        - There are no NaN values in critical columns
        - The class distribution is reasonable (no extreme imbalance)
        - Sample rows look correct

      Skipping this step is how silent data bugs slip into model training.
    """
    print("\n[2/4] Auditing raw dataset...")
    print("-" * 60)

    # Column overview
    print(f"Columns : {list(df.columns)}")
    print(f"Shape   : {df.shape[0]:,} rows × {df.shape[1]} columns")

    # Check for NaN in the columns we rely on
    nulls_instruction = df['instruction'].isna().sum()
    nulls_intent      = df['intent'].isna().sum()
    print(f"\nNull check:")
    print(f"  instruction column nulls : {nulls_instruction}")
    print(f"  intent column nulls      : {nulls_intent}")

    # Full intent distribution (see all 27 intents)
    print(f"\nFull intent distribution ({df['intent'].nunique()} intents):")
    print(df['intent'].value_counts().to_string())

    # Show 3 sample rows for our target intents
    print("\nSample rows for target intents:")
    for intent in TARGET_INTENTS:
        sample = df[df['intent'] == intent]['instruction'].iloc[0]
        print(f"\n  [{intent}]")
        print(f"  → \"{sample[:100]}\"")

    print("-" * 60)


# =============================================================================
# STEP 3: FILTER TO TARGET INTENTS
# =============================================================================

def filter_to_target_intents(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only rows whose 'intent' column is in our TARGET_INTENTS list.

    KEY OPERATIONS EXPLAINED:

    df['intent'].isin(TARGET_INTENTS)
      Returns a boolean Series (True/False mask) the same length as df.
      Using isin() is vectorized — it runs at C speed internally, much
      faster than a Python-level loop or chained .str.contains() calls.

    .copy()
      Creates a deep copy of the filtered DataFrame. Without this, Pandas
      may return a "view" of the original — any mutations we make later
      (e.g., adding columns in Phase 2) would trigger a SettingWithCopyWarning.

    .reset_index(drop=True)
      After filtering, the integer index will have gaps (e.g., 0, 1, 5, 12...).
      reset_index() makes it contiguous again (0, 1, 2, 3...).
      drop=True discards the old index rather than moving it to a column.
      This matters in Phase 2 when we use positional indexing.
    """
    print("\n[3/4] Filtering to target intents...")

    mask     = df['intent'].isin(TARGET_INTENTS)
    filtered = df[mask].copy()
    filtered = filtered.reset_index(drop=True)

    print(f"      Rows before filter : {len(df):,}")
    print(f"      Rows after filter  : {len(filtered):,}")
    print(f"\n      Intent distribution in filtered set:")

    dist = filtered['intent'].value_counts()
    for intent, count in dist.items():
        pct = count / len(filtered) * 100
        print(f"        {intent:<20} {count:>5,} rows  ({pct:.1f}%)")

    # IMPORTANT: Check class balance.
    # Severely imbalanced classes can cause K-Means to weight one cluster
    # more heavily. Ideally each intent has a similar count.
    max_count = dist.max()
    min_count = dist.min()
    imbalance_ratio = max_count / min_count
    if imbalance_ratio > 2.0:
        print(f"\n  ⚠️  Warning: Class imbalance ratio is {imbalance_ratio:.1f}x.")
        print("     Consider downsampling the larger classes in a future iteration.")
    else:
        print(f"\n      ✅ Class balance looks good (ratio: {imbalance_ratio:.1f}x)")

    return filtered


# =============================================================================
# STEP 4: SAVE TO DISK
# =============================================================================

def save_filtered_dataset(df: pd.DataFrame) -> None:
    """
    Persist the filtered DataFrame to a CSV file.

    WHY CSV (not Parquet or JSON)?
      - CSV is human-readable: you can open it in VS Code or Excel
      - For ~3,000 rows, there's no performance difference that matters
      - Zero extra dependencies (Parquet needs pyarrow)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
      Creates the data/ directory if it doesn't exist yet.
      parents=True  → creates intermediate directories too (like mkdir -p)
      exist_ok=True → doesn't raise an error if directory already exists

    index=False
      Prevents Pandas from writing the 0,1,2... row numbers as the first
      column of the CSV. That column would confuse later code that expects
      only real data columns.
    """
    print(f"\n[4/4] Saving filtered dataset...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    file_size_kb = OUTPUT_PATH.stat().st_size / 1024
    print(f"      ✅ Saved → {OUTPUT_PATH}")
    print(f"         Shape     : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"         File size : {file_size_kb:.1f} KB")


# =============================================================================
# ENTRYPOINT
# =============================================================================

def main():
    df_raw = load_raw_dataset()
    audit_dataset(df_raw)
    df_filtered = filter_to_target_intents(df_raw)
    save_filtered_dataset(df_filtered)

    print("\n" + "=" * 60)
    print("✅ Phase 1 Complete.")
    print("   Next step: Phase 2 — Embedding Extraction")
    print(f"   Run: python scripts/train_embeddings.py")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    # Guard clause: if someone accidentally imports this script as a module,
    # main() won't execute. It only runs when called directly with `python`.
    main()