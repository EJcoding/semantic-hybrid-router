"""
app/agents/track_agent.py
==========================
Phase 4a: Tracking LangChain Agent

RESPONSIBILITY: Handle all order tracking and delivery status requests.
TOOL:           track_order  (app/tools/sqlite_tools.py)
LLM:            Shared instance from app.llm_config

CALLED BY: LangGraph state machine (Phase 4b) when result.intent == "track_order"
"""

from dotenv import load_dotenv
from langchain.agents import create_agent
from app.llm_config import get_llm
from app.tools.sqlite_tools import track_order

load_dotenv()

SYSTEM_PROMPT = """You are a customer service specialist handling order tracking and delivery status.

Your only job is to provide shipping and delivery information using the track_order tool.

Instructions:
- Extract the order ID from the customer's message (format: ORD-XXXX, e.g. ORD-1002).
- Call track_order with that order ID.
- If no order ID is in the message, politely ask the customer to provide it.
- Be informative, friendly, and concise.
- Respond in plain conversational text — no markdown (no bold, bullet points, or headers).
- Do not handle cancellations, refunds, or any other topics."""

_agent = None


def _get_agent():
    global _agent
    if _agent is not None:
        return _agent

    _agent = create_agent(
        model=get_llm(),
        tools=[track_order],
        system_prompt=SYSTEM_PROMPT,
    )
    return _agent


def run_track_agent(query: str) -> str:
    """
    Run the tracking agent on a raw customer query.

    Args:
        query: Customer message, e.g. "Where is my order ORD-1002?"

    Returns:
        The agent's final response string.
    """
    result = _get_agent().invoke({
        "messages": [{"role": "user", "content": query}]
    })
    return result["messages"][-1].content