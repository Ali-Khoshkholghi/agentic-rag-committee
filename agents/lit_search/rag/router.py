"""Per-sub-query source routing for lit_search (SPEC Step 10).

A single Gemini call: given a sub-query and short descriptions of the two
evidence sources lit_search can draw on, return a structured choice —
``vector_store``, ``web``, or ``both``.

Follows the adapter shape used elsewhere in the repo
(``agents/critic/langgraph_verifier.py``, ``agents/synthesis/llm_provider.py``):
a config error at startup if ``GEMINI_API_KEY`` is missing, a runtime
error if the live call fails, and — because a wrong route only costs a
little extra retrieval, never a crash — a safe fallback to ``both`` when
the model's answer can't be parsed into a clean choice.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from .gemini import GeminiError, ainvoke_text, build_chat_model

logger = logging.getLogger(__name__)

VECTOR_STORE = "vector_store"
WEB = "web"
BOTH = "both"
VALID_ROUTES = (VECTOR_STORE, WEB, BOTH)

FALLBACK_ROUTE = BOTH
FALLBACK_REASON = "router could not parse a clear choice — querying both sources"

# Short, stable descriptions of what each source is good for. Kept here
# (not in the prompt inline) so the standalone route demo can print the
# exact text the model was shown.
VECTOR_STORE_DESCRIPTION = (
    "Local corpus: five foundational retrieval-augmented-generation papers — "
    "RAG (Lewis et al. 2020), Self-RAG, CRAG, REALM, and DPR. Hybrid "
    "dense+BM25 retrieval over their full text. Best for questions about the "
    "methods, architectures, training details, ablations, and results in "
    "those specific papers."
)
WEB_DESCRIPTION = (
    "Tavily live web search. Best for anything current or outside those five "
    "papers: recent papers and systems, released tools and libraries, "
    "benchmark numbers, news, or general background the corpus does not cover."
)

_ROUTE_RE = re.compile(r"route\s*:\s*(vector_store|web|both)", re.IGNORECASE)
_REASON_RE = re.compile(r"reason\s*:\s*(.+)", re.IGNORECASE | re.DOTALL)


class RouterConfigError(RuntimeError):
    """Raised at startup when routing is selected but GEMINI_API_KEY is
    missing — same fail-loudly contract as critic / synthesis."""


@dataclass
class RouteDecision:
    """Which source(s) to query for one sub-query, and why."""

    route: str
    reasoning: str
    query: str

    @property
    def use_vector_store(self) -> bool:
        return self.route in (VECTOR_STORE, BOTH)

    @property
    def use_web(self) -> bool:
        return self.route in (WEB, BOTH)


def _build_prompt(sub_query: str) -> str:
    return (
        "You route a research sub-query to the right evidence source(s). "
        "You are given the sub-query and a description of each source. Pick "
        "the source that can actually answer it — use the local corpus for "
        "questions about those five papers' contents, the web for anything "
        "current or outside them, and both only when the sub-query genuinely "
        "spans corpus material and outside/current information.\n\n"
        f"Sub-query: {sub_query}\n\n"
        f"Source '{VECTOR_STORE}': {VECTOR_STORE_DESCRIPTION}\n\n"
        f"Source '{WEB}': {WEB_DESCRIPTION}\n\n"
        "Respond with EXACTLY two lines, no other text:\n"
        "Route: VECTOR_STORE or WEB or BOTH\n"
        "Reason: one sentence explaining why."
    )


def parse_route(text: str, sub_query: str) -> RouteDecision:
    """Parses a 'Route: ...' + 'Reason: ...' response into a RouteDecision.

    Falls back to ``both`` (never raises) when no clear route is found —
    a bad parse should cost one extra retrieval, not the whole task.
    """
    route_match = _ROUTE_RE.search(text)
    if route_match is None:
        logger.warning("router: could not parse a route from Gemini response: %r", text)
        return RouteDecision(FALLBACK_ROUTE, FALLBACK_REASON, sub_query)

    route = route_match.group(1).strip().lower()
    reason_match = _REASON_RE.search(text)
    reason = (
        reason_match.group(1).strip().splitlines()[0].strip()
        if reason_match
        else text.strip()
    )
    return RouteDecision(route, reason or "(no reason given)", sub_query)


class SourceRouter:
    """One Gemini call per sub-query, returning a RouteDecision."""

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        self._llm = build_chat_model(api_key, model)

    async def route(self, sub_query: str) -> RouteDecision:
        prompt = _build_prompt(sub_query)
        try:
            text = await ainvoke_text(self._llm, prompt)
        except GeminiError as exc:
            # A dead router shouldn't kill the sub-query — degrade to
            # querying both sources, same spirit as the parse fallback.
            logger.warning("router: Gemini call failed (%s) — falling back to %r", exc, FALLBACK_ROUTE)
            return RouteDecision(FALLBACK_ROUTE, f"router error: {exc}", sub_query)
        return parse_route(text, sub_query)


def build_router_from_env() -> SourceRouter:
    """Builds a SourceRouter from GEMINI_API_KEY / GEMINI_MODEL.

    Raises RouterConfigError if GEMINI_API_KEY isn't set — fail loudly at
    startup, same as critic's build_verifier_from_env().
    """
    try:
        return SourceRouter()
    except Exception as exc:  # GeminiConfigError from build_chat_model
        raise RouterConfigError(str(exc)) from exc
