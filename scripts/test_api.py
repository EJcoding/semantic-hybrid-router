"""
scripts/test_api.py
====================
Integration tests for the Phase 5 FastAPI endpoint.

Requires the server to be running:
  uvicorn app.main:app --host 0.0.0.0 --port 8000

Run:
  python3 scripts/test_api.py
"""

import sys
import json
import sqlite3
from pathlib import Path

try:
    import requests
except ImportError:
    print("requests not installed. Run: pip install requests")
    sys.exit(1)

BASE_URL   = "http://localhost:8000"
DB_PATH    = Path(__file__).resolve().parent.parent / "data" / "orders.db"
PASS, FAIL = "✅ PASS", "❌ FAIL"


def post(query: str) -> dict:
    r = requests.post(
        f"{BASE_URL}/api/v1/route_query",
        json={"query": query},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()


def db_status(order_id: str) -> str:
    conn   = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT status FROM orders WHERE order_id = ?", (order_id,))
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else "NOT FOUND"


def check(label: str, condition: bool, detail: str = ""):
    status = PASS if condition else FAIL
    print(f"  {status}  {label}")
    if detail:
        print(f"         {detail}")
    return condition


def run_tests():
    all_passed = True

    # ── Health check ──────────────────────────────────────────────────────────
    print("\n── Health ───────────────────────────────────────────────────────")
    r = requests.get(f"{BASE_URL}/health", timeout=5)
    all_passed &= check("GET /health returns 200", r.status_code == 200)
    all_passed &= check("status is ok", r.json().get("status") == "ok")

    # ── Cancel ────────────────────────────────────────────────────────────────
    print("\n── Cancel order ─────────────────────────────────────────────────")
    result = post("I want to cancel order ORD-1009")
    print(f"  intent={result['intent']}  conf={result['confidence']}  anomaly={result['is_anomaly']}")
    print(f"  response: {result['response']}")
    all_passed &= check("intent = cancel_order",  result["intent"]     == "cancel_order")
    all_passed &= check("not anomalous",           result["is_anomaly"] == False)
    all_passed &= check("confidence > 0",          result["confidence"] >  0)
    db_stat = db_status("ORD-1009")
    all_passed &= check(f"ORD-1009 status = Cancelled (was: {db_stat})",
                        db_stat == "Cancelled")

    # ── Track ─────────────────────────────────────────────────────────────────
    print("\n── Track order ──────────────────────────────────────────────────")
    result = post("Where is my order ORD-1010?")
    print(f"  intent={result['intent']}  conf={result['confidence']}  anomaly={result['is_anomaly']}")
    print(f"  response: {result['response']}")
    all_passed &= check("intent = track_order",    result["intent"]     == "track_order")
    all_passed &= check("not anomalous",           result["is_anomaly"] == False)
    all_passed &= check("confidence > 0",          result["confidence"] >  0)
    db_stat = db_status("ORD-1010")
    all_passed &= check(f"ORD-1010 status unchanged (Shipped, was: {db_stat})",
                        db_stat == "Shipped")

    # ── Refund ────────────────────────────────────────────────────────────────
    print("\n── Refund request ───────────────────────────────────────────────")
    result = post("I want a refund for ORD-1003")
    print(f"  intent={result['intent']}  conf={result['confidence']}  anomaly={result['is_anomaly']}")
    print(f"  response: {result['response']}")
    all_passed &= check("intent = get_refund",     result["intent"]     == "get_refund")
    all_passed &= check("not anomalous",           result["is_anomaly"] == False)
    all_passed &= check("confidence > 0",          result["confidence"] >  0)

    # ── Guardrail ─────────────────────────────────────────────────────────────
    print("\n── Guardrail ────────────────────────────────────────────────────")
    result = post("What is the weather in San Francisco?")
    print(f"  intent={result['intent']}  conf={result['confidence']}  anomaly={result['is_anomaly']}")
    print(f"  response: {result['response']}")
    all_passed &= check("intent = anomalous",      result["intent"]     == "anomalous")
    all_passed &= check("is_anomaly = True",        result["is_anomaly"] == True)
    all_passed &= check("confidence = 0.0",         result["confidence"] == 0.0)

    # ── Edge case: empty query ────────────────────────────────────────────────
    print("\n── Edge case: empty query ───────────────────────────────────────")
    r = requests.post(f"{BASE_URL}/api/v1/route_query", json={"query": "  "}, timeout=10)
    all_passed &= check("empty query returns 422",  r.status_code == 422)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "─" * 68)
    if all_passed:
        print("✅  All tests passed. Phase 5 complete.")
    else:
        print("❌  Some tests failed — check output above.")
    print("─" * 68 + "\n")
    return all_passed


if __name__ == "__main__":
    print("Semantic Hybrid Router — API Integration Tests")
    print(f"Target: {BASE_URL}")
    try:
        run_tests()
    except requests.exceptions.ConnectionError:
        print(f"\n❌  Could not connect to {BASE_URL}")
        print("   Is the server running?")
        print("   uvicorn app.main:app --host 0.0.0.0 --port 8000\n")
        sys.exit(1)