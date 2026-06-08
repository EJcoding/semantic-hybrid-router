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
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor
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
- Do not handle cancellations, tracking, or any other topics."""

_executor: AgentExecutor | None = None


def _get_executor() -> AgentExecutor:
    global _executor
    if _executor is not None:
        return _executor

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    tools = [calculate_refund]
    agent = create_tool_calling_agent(get_llm(), tools, prompt)

    _executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        handle_parsing_errors=True,
        max_iterations=5,
        return_intermediate_steps=False,
    )
    return _executor


def run_refund_agent(query: str) -> str:
    """
    Run the refund agent on a raw customer query.

    Args:
        query: Customer message, e.g. "I want a refund for order ORD-1003"

    Returns:
        The agent's final response string.
    """
    result = _get_executor().invoke({"input": query})
    return result["output"]