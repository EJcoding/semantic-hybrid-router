"""
scripts/diagnose_refund.py
Diagnose why all get_refund queries are being flagged as anomalous.
"""
import sys
import re
import numpy as np
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.pipeline.embedder import get_embedding
import app.pipeline.classifier as classifier_module

# Trigger model loading so all module-level variables are populated
classifier_module._load_models()

_ORDER_ID_PATTERN = re.compile(r'\bORD-\d+\b', re.IGNORECASE)

def embed_normalized(text):
    return get_embedding(_ORDER_ID_PATTERN.sub('{{Order Number}}', text))

# Access internals through module reference (not direct import)
# so we get the post-load values, not the None values at import time
def get_internals():
    return (
        classifier_module._kmeans,
        classifier_module._cluster_map,
        classifier_module._core_samples,
        classifier_module._dbscan_eps,
    )

print("\n=== 1. CLUSTER MAP ===")
_kmeans, _cluster_map, _core_samples, _dbscan_eps = get_internals()
print(_cluster_map)

print("\n=== 2. CENTROID DISTANCES for a raw refund query ===")
q = "I want a refund for {{Order Number}}"
emb = embed_normalized(q)
dists = np.linalg.norm(_kmeans.cluster_centers_ - emb, axis=1)
for cid, dist in enumerate(dists):
    print(f"  Cluster {cid} ({_cluster_map[cid]}): distance = {dist:.4f}")

print(f"\n=== 3. DBSCAN eps threshold: {_dbscan_eps:.4f} ===")
min_core_dist = np.min(np.linalg.norm(_core_samples - emb, axis=1))
print(f"  Min distance to any core sample: {min_core_dist:.4f}")
print(f"  Anomalous? {min_core_dist > _dbscan_eps}")

print("\n=== 4. get_refund cluster geometry from training data ===")
data_path = Path(__file__).resolve().parent.parent / "data" / "training_data.npz"
data = np.load(data_path, allow_pickle=True)
embeddings = data["embeddings"]
labels = data["labels"]

refund_mask = labels == "get_refund"
refund_embs = embeddings[refund_mask]
print(f"  get_refund training samples: {len(refund_embs)}")

dists_to_refund = np.linalg.norm(refund_embs - emb, axis=1)
print(f"  Distance to nearest get_refund training sample : {dists_to_refund.min():.4f}")
print(f"  Distance to farthest get_refund training sample: {dists_to_refund.max():.4f}")
print(f"  Mean distance to get_refund training samples   : {dists_to_refund.mean():.4f}")

print("\n=== 5. SAMPLE get_refund training queries ===")
csv_path = Path(__file__).resolve().parent.parent / "data" / "filtered_intents.csv"
df = pd.read_csv(csv_path)
refund_samples = df[df['intent'] == 'get_refund']['instruction'].head(10).tolist()
for i, s in enumerate(refund_samples):
    print(f"  {i+1:2}. {s}")

print("\n=== 6. track_order query for comparison ===")
q2 = "Where is my order {{Order Number}}?"
emb2 = embed_normalized(q2)
dists2 = np.linalg.norm(_kmeans.cluster_centers_ - emb2, axis=1)
for cid, dist in enumerate(dists2):
    print(f"  Cluster {cid} ({_cluster_map[cid]}): distance = {dist:.4f}")
min_core_dist2 = np.min(np.linalg.norm(_core_samples - emb2, axis=1))
print(f"  Min distance to core sample: {min_core_dist2:.4f}")
print(f"  Anomalous? {min_core_dist2 > _dbscan_eps}")