"""
app/db/mock_db.py
==================
Mock SQLite database for Phase 4 LangChain tool testing.

SCHEMA: one table — orders
  order_id          TEXT PK   — ORD-XXXX format
  customer_name     TEXT      — full name
  item_name         TEXT      — product description
  item_id           TEXT      — SKU
  quantity          INTEGER   — units ordered
  status            TEXT      — Processing | Shipped | Delivered | Cancelled
  tracking_number   TEXT      — TRK-XXXX-XX (null if not yet shipped)
  estimated_delivery TEXT     — YYYY-MM-DD  (null if not yet shipped)
  order_total       REAL      — USD
  created_at        TEXT      — ISO 8601

DATA COVERS ALL STATUS BRANCHES so every tool path can be tested:
  Processing  → cancellable, full refund eligible
  Shipped     → not cancellable, 80% refund eligible
  Delivered   → not cancellable, 50% refund eligible
  Cancelled   → already cancelled

RUN STANDALONE to create/inspect the database:
  $ python app/db/mock_db.py
"""

import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DB_PATH      = PROJECT_ROOT / "data" / "orders.db"

# fmt: off
MOCK_ORDERS = [
    # (order_id, customer_name, item_name, item_id, qty, status, tracking, delivery, total, created_at)
    ("ORD-1001", "Alice Johnson",  "Wireless Headphones",  "ITEM-WH-001", 1, "Processing", None,           None,         79.99,  "2025-06-01T09:00:00"),
    ("ORD-1002", "Bob Smith",      "Running Shoes",        "ITEM-RS-002", 1, "Shipped",    "TRK-8842-ZX",  "2025-06-18", 124.99, "2025-06-02T11:30:00"),
    ("ORD-1003", "Carol White",    "Yoga Mat",             "ITEM-YM-003", 1, "Delivered",  "TRK-5521-AB",  "2025-06-10", 34.99,  "2025-05-28T14:00:00"),
    ("ORD-1004", "Dave Brown",     "Laptop Stand",         "ITEM-LS-004", 2, "Processing", None,           None,         49.99,  "2025-06-03T08:15:00"),
    ("ORD-1005", "Eve Davis",      "Coffee Maker",         "ITEM-CM-005", 1, "Shipped",    "TRK-7731-CD",  "2025-06-20", 89.99,  "2025-06-04T16:45:00"),
    ("ORD-1006", "Frank Miller",   "Desk Lamp",            "ITEM-DL-006", 1, "Processing", None,           None,         27.99,  "2025-06-05T10:00:00"),
    ("ORD-1007", "Grace Wilson",   "Backpack",             "ITEM-BP-007", 1, "Delivered",  "TRK-3310-EF",  "2025-06-08", 65.99,  "2025-05-30T09:30:00"),
    ("ORD-1008", "Henry Moore",    "Phone Case",           "ITEM-PC-008", 3, "Shipped",    "TRK-9944-GH",  "2025-06-19", 19.99,  "2025-06-05T13:00:00"),
    ("ORD-1009", "Iris Taylor",    "Mechanical Keyboard",  "ITEM-KB-009", 1, "Processing", None,           None,         149.99, "2025-06-06T11:00:00"),
    ("ORD-1010", "Jack Anderson",  "Monitor",              "ITEM-MN-010", 1, "Shipped",    "TRK-2267-IJ",  "2025-06-22", 329.99, "2025-06-06T15:30:00"),
    ("ORD-1011", "Karen Thomas",   "Webcam",               "ITEM-WC-011", 1, "Cancelled",  None,           None,         54.99,  "2025-06-07T09:00:00"),
    ("ORD-1012", "Leo Martinez",   "Gaming Mouse",         "ITEM-GM-012", 1, "Delivered",  "TRK-1145-KL",  "2025-06-05", 44.99,  "2025-06-01T10:00:00"),
]
# fmt: on


def init_db(force: bool = False) -> None:
    """
    Create and seed the orders table.

    Args:
        force: Drop and recreate the table from scratch.
               Useful during development to reset test data.
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn   = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    if force:
        cursor.execute("DROP TABLE IF EXISTS orders")

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS orders (
            order_id           TEXT PRIMARY KEY,
            customer_name      TEXT NOT NULL,
            item_name          TEXT NOT NULL,
            item_id            TEXT NOT NULL,
            quantity           INTEGER NOT NULL DEFAULT 1,
            status             TEXT NOT NULL,
            tracking_number    TEXT,
            estimated_delivery TEXT,
            order_total        REAL NOT NULL,
            created_at         TEXT NOT NULL
        )
    """)

    # Only seed when empty — prevents duplicate rows on re-runs
    cursor.execute("SELECT COUNT(*) FROM orders")
    if cursor.fetchone()[0] == 0:
        cursor.executemany("""
            INSERT INTO orders
                (order_id, customer_name, item_name, item_id, quantity,
                 status, tracking_number, estimated_delivery, order_total, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, MOCK_ORDERS)
        print(f"  Seeded {len(MOCK_ORDERS)} mock orders → {DB_PATH}")

    conn.commit()
    conn.close()


def get_connection() -> sqlite3.Connection:
    """
    Return a fresh SQLite connection, auto-initialising the DB if needed.

    THREAD SAFETY:
      SQLite connections are NOT thread-safe and must never be shared across
      threads. This function creates a new connection on every call.
      Callers must close the connection when done (use try/finally).
    """
    if not DB_PATH.exists():
        init_db()
    return sqlite3.connect(DB_PATH)


def print_orders() -> None:
    """
    Print the current contents of the orders table — read-only, no side effects.

    Safe to run after agent tests to check whether a tool (cancel_order,
    track_order, calculate_refund) actually mutated the database.
    """
    conn   = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT order_id, customer_name, item_name, status, order_total FROM orders"
    )
    rows = cursor.fetchall()
    conn.close()

    print(f"\norders ({len(rows)} rows):")
    print(f"  {'ID':<10} {'Customer':<18} {'Item':<24} {'Status':<12} {'Total':>8}")
    print(f"  {'-'*76}")
    for r in rows:
        print(f"  {r[0]:<10} {r[1]:<18} {r[2]:<24} {r[3]:<12} ${r[4]:>7.2f}")


if __name__ == "__main__":
    import sys

    if "--reset" in sys.argv:
        # DESTRUCTIVE: drops the table and re-seeds all 12 mock orders to
        # their original state. Use this when you want a clean slate after
        # a batch of testing has left orders in cancelled/modified states.
        print("Resetting database to seed data...")
        init_db(force=True)
    else:
        # READ-ONLY: creates the DB only if it doesn't exist yet; otherwise
        # leaves existing data (including any agent-made changes) untouched.
        init_db(force=False)

    print_orders()