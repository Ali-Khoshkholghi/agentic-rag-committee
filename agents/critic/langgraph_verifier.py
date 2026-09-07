"""LangGraph + Gemini claim-verification for the critic agent.

Step 7 of the SPEC: rebuild critic's internal verification logic on top
of LangGraph, backed by the Gemini API (same GEMINI_API_KEY already used
by synthesis, Step 6) — without touching the A2A server layer, Agent
Card, skill, port, or DataPart shape. This module is a drop-in
alternative to agents/critic/matching.py's verify_claims(): same input
(draft, sources), same output ([{claim, supported, note}, ...]).

Mirrors the provider-adapter pattern from agents/synthesis/llm_provider.py
(Step 6) and agents/lit_search/search_provider.py (Step 5): a config
error at startup if misconfigured, a runtime error mapped to a failed
task if a live call blows up.

The graph itself is small and is run once per claim (not once per
draft): each claim gets its own retrieve -> judge -> assemble walk, since
every claim independently needs its own candidate source and its own
LLM judgement. State only ever holds one claim at a time.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, TypedDict

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph

from agents.critic.matching import best_matching_source, split_claims

logger = logging.getLogger(__name__)

# Same model default as synthesis (agents/synthesis/llm_provider.py) —
# verified reachable in this repo's sandbox as of Step 6.
DEFAULT_GEMINI_MODEL = "gemini-3.6-flash"

INCONCLUSIVE_NOTE = "Verification inconclusive — could not parse a clear verdict from the model"


class CriticProviderConfigError(RuntimeError):
    """Raised at startup when CRITIC_MODE=langgraph but GEMINI_API_KEY is
    missing — critic should fail loudly here, the same pattern as
    synthesis's SynthesisProviderConfigError, rather than discovering it
    on the first verify-claims request."""


class CriticProviderError(RuntimeError):
    """Raised when a live judge_claim call fails outright: an API error,
    rate limit, timeout, network failure, or an empty response. Caught by
    the executor and mapped to a failed task state — a parse failure on
    an otherwise-successful response is NOT this; that falls back to a
    per-claim 'inconclusive' verdict instead (see judge_claim)."""


class ClaimState(TypedDict):
    claim: str
    sources: list[dict[str, Any]]
    candidate_source: dict[str, Any] | None
    supported: bool
    note: str


def _build_prompt(claim: str, source: dict[str, Any]) -> str:
    title = source.get("title", "")
    snippet = source.get("snippet", "")
    return (
        "You are a strict fact-checker. You are given one claim sentence "
        "and one candidate source. Decide whether the source genuinely "
        "supports the claim — not just topically related, but actually "
        "asserts or entails what the claim states. Do not use any "
        "outside knowledge; judge only from the source text given.\n\n"
        f"Claim: {claim}\n\n"
        f"Source title: {title}\n"
        f"Source text: {snippet}\n\n"
        "Respond with EXACTLY two lines, no other text:\n"
        "Verdict: YES or NO\n"
        "Reason: one sentence explaining why."
    )


_VERDICT_RE = re.compile(r"verdict\s*:\s*(yes|no)", re.IGNORECASE)
_REASON_RE = re.compile(r"reason\s*:\s*(.+)", re.IGNORECASE | re.DOTALL)


def _parse_verdict(text: str) -> tuple[bool, str] | None:
    """Parses a 'Verdict: YES/NO' + 'Reason: ...' response.

    Returns None if no clear yes/no verdict could be found — the caller
    falls back to an inconclusive verdict rather than crashing.
    """
    verdict_match = _VERDICT_RE.search(text)
    if verdict_match is None:
        return None

    supported = verdict_match.group(1).strip().lower() == "yes"
    reason_match = _REASON_RE.search(text)
    reason = reason_match.group(1).strip().splitlines()[0].strip() if reason_match else text.strip()
    if not reason:
        return None
    return supported, reason


def _extract_text(content: Any) -> str:
    """Pulls plain text out of a LangChain message's `.content`.

    Newer Gemini models (e.g. gemini-3.x with "thinking" enabled) return
    a list of content blocks — {'type': 'text', 'text': ..., 'extras': {
    'signature': ...}} among possibly other block types — rather than a
    plain string. Only the text blocks' text matters here; the rest
    (thinking-signature blobs, etc.) must not leak into the parsed note.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return ""


def _make_retrieve_candidate_source():
    def retrieve_candidate_source(state: ClaimState) -> dict[str, Any]:
        source, _ratio = best_matching_source(state["claim"], state["sources"])
        return {"candidate_source": source}

    return retrieve_candidate_source


def _make_judge_claim(llm: ChatGoogleGenerativeAI):
    async def judge_claim(state: ClaimState) -> dict[str, Any]:
        source = state["candidate_source"]
        if source is None:
            return {
                "supported": False,
                "note": "Unable to verify — insufficient data",
            }

        prompt = _build_prompt(state["claim"], source)
        try:
            response = await llm.ainvoke(prompt)
        except Exception as exc:  # noqa: BLE001 - collapse SDK/network/HTTP errors
            raise CriticProviderError(f"Gemini call failed: {exc}") from exc

        text = _extract_text(response.content)
        if not text or not text.strip():
            raise CriticProviderError("Gemini returned an empty response")

        parsed = _parse_verdict(text)
        if parsed is None:
            logger.warning("critic: could not parse a verdict from Gemini response: %r", text)
            return {"supported": False, "note": INCONCLUSIVE_NOTE}

        supported, reason = parsed
        return {"supported": supported, "note": reason}

    return judge_claim


def _assemble(state: ClaimState) -> dict[str, Any]:
    return {}


def _build_claim_graph(llm: ChatGoogleGenerativeAI):
    graph = StateGraph(ClaimState)
    graph.add_node("retrieve_candidate_source", _make_retrieve_candidate_source())
    graph.add_node("judge_claim", _make_judge_claim(llm))
    graph.add_node("assemble", _assemble)

    graph.set_entry_point("retrieve_candidate_source")
    graph.add_edge("retrieve_candidate_source", "judge_claim")
    graph.add_edge("judge_claim", "assemble")
    graph.add_edge("assemble", END)

    return graph.compile()


class LangGraphVerifier:
    """Verifies claims one at a time by walking a small LangGraph graph
    (retrieve_candidate_source -> judge_claim -> assemble) per claim."""

    def __init__(self, api_key: str, model: str = DEFAULT_GEMINI_MODEL) -> None:
        llm = ChatGoogleGenerativeAI(model=model, google_api_key=api_key)
        self._app = _build_claim_graph(llm)

    async def verify_claims(self, draft: str, sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
        verdicts = []
        for claim in split_claims(draft):
            result = await self._app.ainvoke(
                {
                    "claim": claim,
                    "sources": sources,
                    "candidate_source": None,
                    "supported": False,
                    "note": "",
                }
            )
            verdicts.append(
                {
                    "claim": claim,
                    "supported": result["supported"],
                    "note": result["note"],
                }
            )
        return verdicts


def build_verifier_from_env() -> LangGraphVerifier:
    """Builds a LangGraphVerifier from GEMINI_API_KEY / GEMINI_MODEL.

    Raises CriticProviderConfigError if GEMINI_API_KEY isn't set — same
    fail-loudly-at-startup pattern as synthesis's build_provider().
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise CriticProviderConfigError(
            "CRITIC_MODE=langgraph but GEMINI_API_KEY is not set. Set "
            "GEMINI_API_KEY, or set CRITIC_MODE=naive (or leave it unset "
            "with no key) to use the built-in keyword-overlap checker instead."
        )
    model = os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
    return LangGraphVerifier(api_key=api_key, model=model)
