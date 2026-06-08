"""
app/tools/sqlite_tools.py
==========================
LangChain-wrapped SQLite tool functions for the three customer-service agents.

THE ROLE OF THE @tool DECORATOR:
  The decorator converts a plain Python function into a LangChain Tool object.
  The most important effect: the function's docstring becomes the LLM's
  "instruction manual" for that tool. The LLM reads the docstring to decide:
    (a) Should I call this tool for this query?
    (b) What value should I pass as each argument?

  A vague docstring = the LLM guesses wrong.
  A precise docstring = the LLM calls the tool correctly on the first attempt.

TOOL → AGENT ASSIGNMENT:
  cancel_order     → CancellationAgent  (app/agents/cancel_agent.py)
  track_order      → TrackingAgent      (app/agents/track_agent.py)
  calculate_refund → RefundAgent        (app/agents/refund_agent.py)

Each tool is intentionally narrow — one agent, one responsibility.
"""

import sqlite3
from langchain_core.tools import tool
from app.db.mock_db import get_connection


# =============================================================================
# TOOL 1: cancel_order  →  CancellationAgent
# =============================================================================

@tool
def cancel_order(order_id: str) -> str:
    """
    Cancel a customer's order by setting its status to 'Cancelled' in the database.

    Call this tool when the customer wants to cancel an order.
    Only orders with status 'Processing' can be cancelled.
    Orders with status 'Shipped' or 'Delivered' cannot be cancelled —
    advise those customers to request a return instead.

    Args:
        order_id: The customer's order ID. Always in ORD-XXXX format
                  (for example: ORD-1001, ORD-1009). Extract this value
                  directly from the customer's message.

    Returns:
        A success confirmation with refund details, or an explanation of
        why the order could not be cancelled.
    """
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """SELECT status, item_name, customer_name, order_total
               FROM orders WHERE order_id = ?""",
            (order_id.strip().upper(),)
        )
        row = cursor.fetchone()

        if not row:
            return (
                f"No order found with ID '{order_id}'. "
                "Please check the order ID — it should look like ORD-1001."
            )

        status, item_name, customer_name, order_total = row

        if status == "Cancelled":
            return f"Order {order_id} ({item_name}) has already been cancelled. No further action needed."

        if status in ("Shipped", "Delivered"):
            return (
                f"Order {order_id} ({item_name}) cannot be cancelled — it has already been "
                f"{status.lower()}. Please advise the customer to initiate a return for a refund."
            )

        # status == "Processing" → safe to cancel
        cursor.execute(
            "UPDATE orders SET status = 'Cancelled' WHERE order_id = ?",
            (order_id.strip().upper(),)
        )
        conn.commit()

        return (
            f"Order {order_id} has been successfully cancelled. "
            f"Item: {item_name}. Customer: {customer_name}. "
            f"A full refund of ${order_total:.2f} will be issued within 3–5 business days."
        )

    except sqlite3.Error as e:
        return f"Database error while cancelling order {order_id}: {e}"
    finally:
        conn.close()


# =============================================================================
# TOOL 2: track_order  →  TrackingAgent
# =============================================================================

@tool
def track_order(order_id: str) -> str:
    """
    Look up the current shipping status and delivery information for an order.

    Call this tool when the customer wants to know where their order is,
    when it will arrive, or what the tracking number is.

    Args:
        order_id: The customer's order ID. Always in ORD-XXXX format
                  (for example: ORD-1002, ORD-1005). Extract this value
                  directly from the customer's message.

    Returns:
        Current order status, tracking number (if shipped), and estimated
        delivery date. If not yet shipped, explains that tracking is unavailable.
    """
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """SELECT status, item_name, tracking_number, estimated_delivery, customer_name
               FROM orders WHERE order_id = ?""",
            (order_id.strip().upper(),)
        )
        row = cursor.fetchone()

        if not row:
            return (
                f"No order found with ID '{order_id}'. "
                "Please verify the order ID — it should look like ORD-1001."
            )

        status, item_name, tracking_number, estimated_delivery, customer_name = row

        if status == "Processing":
            return (
                f"Order {order_id} ({item_name}) for {customer_name} is currently being processed "
                f"and has not shipped yet. A tracking number will be emailed once it dispatches."
            )

        if status == "Shipped":
            return (
                f"Order {order_id} ({item_name}) is on its way! "
                f"Tracking number: {tracking_number}. "
                f"Estimated delivery: {estimated_delivery}."
            )

        if status == "Delivered":
            return (
                f"Order {order_id} ({item_name}) was delivered on {estimated_delivery}. "
                f"Tracking number was {tracking_number}. "
                f"If you did not receive the package, please contact support."
            )

        if status == "Cancelled":
            return f"Order {order_id} ({item_name}) was cancelled and will not be delivered."

        return f"Order {order_id} — current status: {status}."

    except sqlite3.Error as e:
        return f"Database error while tracking order {order_id}: {e}"
    finally:
        conn.close()


# =============================================================================
# TOOL 3: calculate_refund  →  RefundAgent
# =============================================================================

@tool
def calculate_refund(order_id: str) -> str:
    """
    Calculate the refund amount a customer is eligible for based on their order status.

    Call this tool when the customer asks for a refund, wants to know how much
    money they will get back, or asks about the refund policy for their order.

    Refund policy applied by this tool:
      Processing → 100% refund   (order not yet shipped; cancel first for best result)
      Shipped    → 80%  refund   (20% deducted for shipping and handling)
      Delivered  → 50%  refund   (50% restocking fee; return must be initiated within 30 days)
      Cancelled  → 100% refund already in progress

    Args:
        order_id: The customer's order ID. Always in ORD-XXXX format
                  (for example: ORD-1003, ORD-1007). Extract this value
                  directly from the customer's message.

    Returns:
        The eligible refund amount in USD with the applicable policy explanation.
    """
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """SELECT status, item_name, order_total, customer_name
               FROM orders WHERE order_id = ?""",
            (order_id.strip().upper(),)
        )
        row = cursor.fetchone()

        if not row:
            return (
                f"No order found with ID '{order_id}'. "
                "Please verify the order ID — it should look like ORD-1001."
            )

        status, item_name, order_total, customer_name = row

        if status == "Cancelled":
            return (
                f"Order {order_id} ({item_name}) is already cancelled. "
                f"A full refund of ${order_total:.2f} has been initiated for {customer_name}."
            )

        if status == "Processing":
            return (
                f"Order {order_id} ({item_name}): eligible for a full refund of ${order_total:.2f} "
                f"since it has not shipped yet. Refund will be issued within 3–5 business days "
                f"after the cancellation is confirmed."
            )

        if status == "Shipped":
            refund   = round(order_total * 0.80, 2)
            deducted = round(order_total * 0.20, 2)
            return (
                f"Order {order_id} ({item_name}): eligible for a refund of ${refund:.2f} "
                f"(${deducted:.2f} deducted for shipping and handling). "
                f"Original total: ${order_total:.2f}. "
                f"Please allow 5–7 business days after we receive the return."
            )

        if status == "Delivered":
            refund   = round(order_total * 0.50, 2)
            deducted = round(order_total * 0.50, 2)
            return (
                f"Order {order_id} ({item_name}): eligible for a refund of ${refund:.2f} "
                f"(50% restocking fee — ${deducted:.2f} deducted from ${order_total:.2f} total). "
                f"Return must be initiated within 30 days of delivery."
            )

        return (
            f"Unable to calculate refund for order {order_id} (status: {status}). "
            "Please contact support."
        )

    except sqlite3.Error as e:
        return f"Database error while calculating refund for {order_id}: {e}"
    finally:
        conn.close()