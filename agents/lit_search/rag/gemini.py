"""Small Gemini helper shared by lit_search's routing + judging (SPEC Step 10).

Both ``router.py`` and ``judge.py`` need the same thing critic's
``langgraph_verifier.py`` needs: a ``ChatGoogleGenerativeAI`` built from
``GEMINI_API_KEY`` / ``GEMINI_MODEL``, a way to pull plain text out of a
(possibly block-structured) response, and one runtime-error type for a
call that blows up or comes back empty. This is the rag package's local
copy of that pattern — deliberately not imported from critic, so the two
agents stay independently swappable.
"""

from __future__ import annotations

import os
from typing import Any

# Same default as synthesis / critic — verified reachable in this repo's
# sandbox. Overridden by GEMINI_MODEL (the .env in this folder pins
# gemini-flash-lite-latest, which all three agents already share).
DEFAULT_GEMINI_MODEL = "gemini-3.6-flash"


class GeminiConfigError(RuntimeError):
    """Raised at startup when agentic retrieval is selected but
    GEMINI_API_KEY is missing — lit_search should fail loudly here, the
    same pattern as critic's CriticProviderConfigError and synthesis's
    SynthesisProviderConfigError, rather than on the first sub-query."""


class GeminiError(RuntimeError):
    """Raised when a live Gemini call fails outright: an API error, rate
    limit, timeout, network failure, or an empty response. The router
    and judge map this to a safe fallback rather than crashing the whole
    A2A task (see their call sites)."""


def resolve_model(model: str | None = None) -> str:
    return model or os.environ.get("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL


def build_chat_model(api_key: str | None = None, model: str | None = None):
    """Builds a ``ChatGoogleGenerativeAI`` — mirrors LangGraphVerifier.__init__.

    Raises GeminiConfigError if no key is available (env or argument).
    """
    from langchain_google_genai import ChatGoogleGenerativeAI

    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise GeminiConfigError(
            "agentic retrieval needs GEMINI_API_KEY (same key synthesis and "
            "critic use). Set it, or set LIT_SEARCH_MODE=provider to keep the "
            "Tavily-only flow."
        )
    return ChatGoogleGenerativeAI(model=resolve_model(model), google_api_key=key)


def extract_text(content: Any) -> str:
    """Pulls plain text out of a LangChain message's ``.content``.

    Newer Gemini models return a list of content blocks — ``{'type':
    'text', 'text': ..., 'extras': {'signature': ...}}`` among others —
    rather than a plain string. Only the text blocks matter here; the
    thinking-signature blobs must not leak into a parsed decision. Same
    shape-handling as critic's ``_extract_text``.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return ""


async def ainvoke_text(llm, prompt: str) -> str:
    """Invokes ``llm`` with ``prompt`` and returns its text.

    Collapses any SDK/network/HTTP error into GeminiError, and treats an
    empty response as a failure too — the caller decides what fallback
    that maps to.
    """
    try:
        response = await llm.ainvoke(prompt)
    except Exception as exc:  # noqa: BLE001 - collapse SDK/network/HTTP errors
        raise GeminiError(f"Gemini call failed: {exc}") from exc

    text = extract_text(response.content)
    if not text or not text.strip():
        raise GeminiError("Gemini returned an empty response")
    return text
