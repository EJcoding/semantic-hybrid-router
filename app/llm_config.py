"""
app/llm_config.py
==================
Centralised LLM configuration for all LangChain agents.

ChatOpenRouter is the purpose-built LangChain ntegration for OpenRouter. It handles:
    - The API base URL internally (no manual override needed)
    - The OPENROUTER_API_KEY env var automatically
    - OpenRouter-specific request headers
    - Tool calling support for compatible models
  The result is cleaner code that accurately reflects what we're using.

WHY ONE FIXED MODEL INSTEAD OF THE FREE ROUTER (openrouter/free)?
  We use meta-llama/llama-3.3-70b-instruct:free explicitly rather than
  "openrouter/free" or "auto" because:
    - The free router can randomly select models ranging from 1.2B to 405B
    - Smaller selections will fail reliably at structured tool calling
    - We cannot debug prompt issues if the model changes between runs
    - Consistent model = consistent behavior = trustworthy system

USAGE IN AGENTS:
  from app.llm_config import get_llm
  llm = get_llm()  # drop-in for create_tool_calling_agent
"""

import os
from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter

load_dotenv()

# Quick check to see if OpenRouter API key exists in .env
_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not _API_KEY:
    raise EnvironmentError(
        "OPENROUTER_API_KEY is not set. "
        "Add it to your .env file: OPENROUTER_API_KEY=sk-or-v1-..."
    )

_MODEL = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free")

# Module-level singleton — all three agents share one LLM instance.
# ChatOpenRouter is stateless between calls (no conversation history stored
# internally), so sharing is safe and avoids creating three separate HTTP clients.
_llm: ChatOpenRouter | None = None


def get_llm() -> ChatOpenRouter:
    """
    Return the shared ChatOpenRouter instance, creating it on first call.

    PARAMETERS EXPLAINED:

    temperature=0
      Makes the model's tool-call decisions deterministic. Given the same
      customer query, the model will always extract the same order ID and
      call the same tool. Critical for a routing system — non-zero temperature
      would mean identical queries could produce different tool calls.

    max_tokens=512
      Generous enough for a complete, well-formatted customer service response.
      Caps token usage on the free tier to prevent runaway generation if the
      model gets verbose. Adjust upward if responses are being cut off.

    Returns:
        A configured ChatOpenRouter instance ready for use with
        create_tool_calling_agent and AgentExecutor.
    """
    global _llm
    if _llm is None:
        _llm = ChatOpenRouter(
            model=_MODEL,
            temperature=0,
            max_tokens=512,
        )
    return _llm