"""
app/agents/refund_agent.py
===========================
Phase 4a: Refund LangChain Agent

RESPONSIBILITY: Handle all refund calculation and policy requests.
TOOL:           calculate_refund  (app/tools/sqlite_tools.py)
LLM:            Shared instance from app.llm_config

CALLED BY: LangGraph state machine (Phase 4b) when result.intent == "get_refund"
"""

from dotenv import load_dotenv
from langchain.agents import create_agent
from app.llm_config import get_llm
from app.tools.sqlite_tools import calculate_refund

load_dotenv()

SYSTEM_PROMPT = """You are a customer service specialist handling refund requests and reimbursements.

Your only job is to calculate and communicate refund information using the calculate_refund tool.

Instructions:
- Extract the order ID from the customer's message (format: ORD-XXXX, e.g. ORD-1003).
- Call calculate_refund with that order ID.
- Explain the refund amount and policy clearly and with empathy.
- If no order ID is in the message, politely ask the customer to provide it.
- Respond in plain conversational text — no markdown (no bold, bullet points, or headers).
- Do not handle cancellations, tracking, or any other topics."""

_agent = None


def _get_agent():
    global _agent
    if _agent is not None:
        return _agent

    _agent = create_agent(
        model=get_llm(),
        tools=[calculate_refund],
        system_prompt=SYSTEM_PROMPT,
    )
    return _agent


def run_refund_agent(query: str) -> str:
    """
    Run the refund agent on a raw customer query.

    Args:
        query: Customer message, e.g. "I want a refund for order ORD-1003"

    Returns:
        The agent's final response string.
    """
    result = _get_agent().invoke({
        "messages": [{"role": "user", "content": query}]
    })
    return result["messages"][-1].content