"""
app/llm_config.py
==================
Centralised LLM configuration shared by all agents.

KEY DECISIONS:
  - ChatOpenRouter (not ChatOpenAI): purpose-built OpenRouter integration —
    handles base URL, API key, headers, and tool-calling natively.

  - One fixed model (not "openrouter/free" or "auto"): the free router can
    randomly select anything from 1.2B to 405B params. Smaller models fail
    at structured tool calling, and a changing model makes prompts
    impossible to debug. meta-llama/llama-3.3-70b-instruct:free is large
    enough for reliable tool calls and small enough to be fast.

  - max_retries (not .with_retry()): .with_retry() returns a generic
    RunnableRetry with no bind_tools method, which crashes create_agent
    (`AttributeError: 'RunnableRetry' object has no attribute 'bind_tools'`).
    max_retries is a native ChatOpenRouter field — retry happens inside the
    chat model at the HTTP layer, so the object stays a real ChatOpenRouter.

  - provider + models fallback: free models share infra with all OpenRouter
    users and can 429 when a specific provider (e.g. Venice) is saturated.
    `provider.allow_fallbacks` retries the SAME model on a different
    provider; `models` is a second layer — an ordered list of DIFFERENT
    models to try if the primary is down entirely (OpenRouter caps this list
    at 3 entries). Fallbacks (gpt-oss-120b, gemma-4-31b) are free,
    tool-call-capable, and from different labs/provider pools than Llama
    3.3 70B. Verified working via curl: when Llama 3.3 70B was saturated,
    gpt-oss-120b answered correctly via a different provider (OpenInference)
    at $0 cost.

USAGE:
  from app.llm_config import get_llm
  llm = get_llm()  # pass directly to create_agent(model=llm, ...)
"""

import os
from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter

load_dotenv()

_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not _API_KEY:
    raise EnvironmentError(
        "OPENROUTER_API_KEY is not set. "
        "Add it to your .env file: OPENROUTER_API_KEY=sk-or-v1-..."
    )

_MODEL = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free")

_FALLBACK_MODELS = [
    _MODEL,
    "openai/gpt-oss-120b:free",
    "google/gemma-4-31b-it:free",
]

# Module-level singleton — shared across all agents (stateless, one HTTP client).
_llm: ChatOpenRouter | None = None


def get_llm() -> ChatOpenRouter:
    """
    Return the shared ChatOpenRouter instance, creating it on first call.

    temperature=0   -> deterministic tool-call decisions (routing reliability)
    max_tokens=512  -> caps generation length on free tier
    max_retries=4   -> retries transient errors at the HTTP layer
    provider        -> allow_fallbacks + sort=throughput: route around a
                       saturated provider for the same model
    models          -> ordered fallback list across different models if the
                       primary model is down entirely
    """
    global _llm
    if _llm is None:
        _llm = ChatOpenRouter(
            model=_MODEL,
            temperature=0,
            max_tokens=512,
            max_retries=4,
            openrouter_provider={
                "allow_fallbacks": True,
                "sort": "throughput",
            },
            model_kwargs={"models": _FALLBACK_MODELS},
        )
    return _llm