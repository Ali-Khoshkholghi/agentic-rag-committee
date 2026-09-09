"""Selects lit_search's evidence backend: Tavily-only, or agentic RAG.

Step 10 of the SPEC. ``LIT_SEARCH_MODE`` picks how lit_search gathers
evidence for a sub-query:

- unset (default): if ``GEMINI_API_KEY`` is set, ``agentic``; otherwise
  ``provider`` — same zero-key-required principle as ``CRITIC_MODE`` /
  ``SYNTHESIS_PROVIDER``.
- ``provider`` (or ``web`` / ``tavily``): the Step 5 flow, unchanged —
  ``build_provider()`` per ``LIT_SEARCH_PROVIDER`` (fake or Tavily).
- ``agentic``: router -> vector store / web / both -> merge -> judge /
  reformulate loop (``rag/pipeline.py``). ``LIT_SEARCH_PROVIDER`` still
  selects the *web arm* of that pipeline.

``AgenticSearchProvider`` implements the same ``SearchProvider.search()``
the executor already calls, so ``agent_executor.py`` and the A2A contract
(Agent Card, artifact shape, event sequencing) don't change at all — the
sub-query goes in as text, ``{title, url, snippet}`` dicts stream back.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from agents.lit_search.search_provider import (
    SearchProvider,
    SearchProviderConfigError,
    SearchProviderError,
    build_provider,
)

logger = logging.getLogger(__name__)


class AgenticSearchProvider(SearchProvider):
    """Adapts the agentic RAG pipeline to the SearchProvider interface.

    ``search()`` runs one sub-query through router -> retrieval -> merge ->
    judge/reformulate and returns the judged evidence. The route decision
    and any reformulate/retry cycles are logged (INFO) as ``agentic-rag:``
    lines — visible in lit_search's server log during a committee run, and
    printed directly by ``python -m agents.lit_search.rag.route_demo``.
    """

    def __init__(self, pipeline) -> None:
        self._pipeline = pipeline

    async def search(self, query: str, max_results: int) -> list[dict[str, Any]]:
        result = await self._pipeline.gather(query)
        sources = result.sources[:max_results] if max_results else result.sources
        if not sources:
            raise SearchProviderError(
                f"agentic retrieval found no evidence for query: {query!r}"
            )
        return sources


def build_search_provider() -> SearchProvider:
    """Selects the evidence backend from ``LIT_SEARCH_MODE``.

    Fails loudly (SearchProviderConfigError) if ``agentic`` is selected —
    explicitly or by the key-present default — but its Gemini key is
    missing, same fail-at-startup contract as the Step 5 provider check.
    """
    mode = os.environ.get("LIT_SEARCH_MODE", "").strip().lower()
    if not mode:
        mode = "agentic" if os.environ.get("GEMINI_API_KEY") else "provider"

    if mode in ("provider", "web", "tavily", "fake"):
        return build_provider()

    if mode == "agentic":
        from agents.lit_search.rag.gemini import GeminiConfigError
        from agents.lit_search.rag.pipeline import build_pipeline_from_env
        from agents.lit_search.rag.router import RouterConfigError

        try:
            pipeline = build_pipeline_from_env()
        except (RouterConfigError, GeminiConfigError) as exc:
            raise SearchProviderConfigError(
                f"LIT_SEARCH_MODE=agentic but agentic retrieval is "
                f"misconfigured: {exc}"
            ) from exc
        return AgenticSearchProvider(pipeline)

    raise SearchProviderConfigError(
        f"Unknown LIT_SEARCH_MODE={mode!r} (expected 'provider' or 'agentic')"
    )
