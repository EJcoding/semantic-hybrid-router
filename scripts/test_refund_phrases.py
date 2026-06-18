import sys
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.pipeline.embedder import get_embedding
from app.pipeline.classifier import classify
from app.pipeline.router_graph import _normalize_query

queries = [
    'How much refund will I get for ORD-1007?',
    'I want a refund for ORD-1007',
    'get refund for order ORD-1007',
    'I need a refund for my order ORD-1007',
    'request refund ORD-1007',
]

for q in queries:
    normalized = _normalize_query(q)
    print(f'normalized: {normalized}')
    emb = get_embedding(normalized)
    result = classify(emb)
    print(f'intent={result.intent} anomaly={result.is_anomaly} conf={result.confidence:.4f} | {q}')
    print()