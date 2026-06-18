"""
app/agents/cancel_agent.py
===========================
Phase 4a: Cancellation LangChain Agent

RESPONSIBILITY: Handle all order cancellation requests.
TOOL:           cancel_order  (app/tools/sqlite_tools.py)
LLM:            Shared instance from app.llm_config

CALLED BY: LangGraph state machine (Phase 4b) when result.intent == "cancel_order"
"""

from dotenv import load_dotenv
from langchain.agents import create_agent
from app.llm_config import get_llm
from app.tools.sqlite_tools import cancel_order

load_dotenv()

SYSTEM_PROMPT = """You are a customer service specialist handling order cancellations.

Your only job is to cancel orders using the cancel_order tool.

Instructions:
- Extract the order ID from the customer's message (format: ORD-XXXX, e.g. ORD-1001).
- Call cancel_order with that order ID.
- If no order ID is in the message, politely ask the customer to provide it.
- Be brief, professional, and empathetic.
- Respond in plain conversational text — no markdown (no bold, bullet points, or headers).
- Do not discuss topics outside of order cancellations."""

_agent = None


def _get_agent():
    """
    Build and cache the compiled agent graph on first call.

    create_agent returns a CompiledStateGraph (LangGraph), not an
    AgentExecutor. It is invoked the same way any LangGraph graph is:
    agent.invoke({"messages": [...]}).
    """
    global _agent
    if _agent is not None:
        return _agent

    _agent = create_agent(
        model=get_llm(),
        tools=[cancel_order],
        system_prompt=SYSTEM_PROMPT,
    )
    return _agent


def run_cancel_agent(query: str) -> str:
    """
    Run the cancellation agent on a raw customer query.

    Args:
        query: Customer message, e.g. "Please cancel my order ORD-1001"

    Returns:
        The agent's final response string.

    HOW THE RESPONSE IS EXTRACTED:
      result["messages"] is the full conversation: the human message,
      any AIMessage(s) containing tool calls, ToolMessage(s) with tool
      results, and a final AIMessage with the natural-language reply.
      messages[-1] is always that final AIMessage; .content is its text.
    """
    result = _get_agent().invoke({
        "messages": [{"role": "user", "content": query}]
    })
    return result["messages"][-1].content