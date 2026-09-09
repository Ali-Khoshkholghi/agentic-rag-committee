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


# One process-wide client-side limiter, shared by every ChatGoogleGenerativeAI
# built here so the *combined* call rate (router + judge + reformulate, plus
# whatever the caller layers on top) stays under a quota. Off unless
# GEMINI_REQUESTS_PER_MINUTE is set — the live agents never set it; the Step 11
# eval does, because it fans out enough calls to trip the Gemini free tier
# (15 req/min) and spend the run in 429 backoff.
_RATE_LIMITER = None
_RATE_LIMITER_RPM: float | None = None


def _shared_rate_limiter():
    global _RATE_LIMITER, _RATE_LIMITER_RPM

    raw = os.environ.get("GEMINI_REQUESTS_PER_MINUTE", "").strip()
    try:
        rpm = float(raw) if raw else 0.0
    except ValueError:
        rpm = 0.0
    if rpm <= 0:
        return None

    if _RATE_LIMITER is None or _RATE_LIMITER_RPM != rpm:
        from langchain_core.rate_limiters import InMemoryRateLimiter

        _RATE_LIMITER = InMemoryRateLimiter(
            requests_per_second=rpm / 60.0,
            check_every_n_seconds=0.5,
            max_bucket_size=max(1, int(rpm // 6)),  # small burst allowance
        )
        _RATE_LIMITER_RPM = rpm
    return _RATE_LIMITER


def build_chat_model(api_key: str | None = None, model: str | None = None):
    """Builds a ``ChatGoogleGenerativeAI`` — mirrors LangGraphVerifier.__init__.

    Raises GeminiConfigError if no key is available (env or argument). If
    ``GEMINI_REQUESTS_PER_MINUTE`` is set, every model built here shares one
    client-side rate limiter (see ``_shared_rate_limiter``).
    """
    from langchain_google_genai import ChatGoogleGenerativeAI

    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise GeminiConfigError(
            "agentic retrieval needs GEMINI_API_KEY (same key synthesis and "
            "critic use). Set it, or set LIT_SEARCH_MODE=provider to keep the "
            "Tavily-only flow."
        )
    kwargs: dict = {"model": resolve_model(model), "google_api_key": key}
    limiter = _shared_rate_limiter()
    if limiter is not None:
        kwargs["rate_limiter"] = limiter
    # Opt-in hardening for batch use (the Step 11 eval): a per-request
    # timeout so a stalled socket fails and retries instead of hanging the
    # run forever, and extra retries to ride out free-tier 429 bursts. The
    # live agents leave these unset and keep the library defaults.
    timeout = os.environ.get("GEMINI_TIMEOUT", "").strip()
    if timeout:
        kwargs["timeout"] = float(timeout)
    retries = os.environ.get("GEMINI_MAX_RETRIES", "").strip()
    if retries.isdigit():
        kwargs["max_retries"] = int(retries)
    return ChatGoogleGenerativeAI(**kwargs)


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
