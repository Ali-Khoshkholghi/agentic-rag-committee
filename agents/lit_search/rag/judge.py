"""LangGraph self-correction loop for lit_search evidence (SPEC Step 10).

A two-node graph, backed by Gemini, that decides whether the merged
evidence for a sub-query is good enough and — if not — reformulates the
query and retrieves again, up to ``max_retries`` times:

    judge ──sufficient────────────────────────────► END
      ▲     └─insufficient, retries exhausted─────► END  (best-effort)
      │
      └────────── reformulate_and_retrieve ◄──insufficient, retries left

Same LangGraph + Gemini construction critic uses in
``agents/critic/langgraph_verifier.py``: a ``StateGraph`` over a
``TypedDict`` state, async nodes calling ``llm.ainvoke``, conditional
edges, compiled once and ``ainvoke``-d per sub-query. Retrieval itself is
injected as ``retrieve_fn`` so this module stays independent of the
Chroma / Tavily wiring (that lives in ``pipeline.py``).
"""

from __future__ import annotations

import logging
import re
from typing import Any, Awaitable, Callable, TypedDict

from langgraph.graph import END, StateGraph

from .gemini import GeminiError, ainvoke_text, build_chat_model
from .router import RouteDecision

logger = logging.getLogger(__name__)

DEFAULT_MAX_RETRIES = 2

# retrieve_fn(query) -> (route decision for that query, merged evidence).
RetrieveFn = Callable[[str], Awaitable[tuple[RouteDecision, list[dict[str, Any]]]]]

_SUFFICIENT_RE = re.compile(r"sufficient\s*:\s*(yes|no)", re.IGNORECASE)
_REASON_RE = re.compile(r"reason\s*:\s*(.+)", re.IGNORECASE | re.DOTALL)
_QUERY_RE = re.compile(r"query\s*:\s*(.+)", re.IGNORECASE)

_EVIDENCE_CAP = 8


class JudgeState(TypedDict):
    sub_query: str            # the original sub-query, fixed for the whole run
    current_query: str        # query behind the evidence currently in hand
    evidence: list[dict[str, Any]]
    retry_count: int
    max_retries: int
    sufficient: bool
    reasoning: str            # judge's rationale for the latest verdict
    history: list[dict[str, Any]]  # one entry per judge pass, for observability


def _format_evidence(evidence: list[dict[str, Any]]) -> str:
    if not evidence:
        return "(no evidence retrieved)"
    lines = []
    for i, item in enumerate(evidence):
        origin = item.get("origin", "?")
        title = item.get("title", "(untitled)")
        snippet = " ".join(str(item.get("snippet", "")).split())
        if len(snippet) > 500:
            snippet = snippet[:499] + "…"
        lines.append(f"[{i}] ({origin}) {title}\n    {snippet}")
    return "\n".join(lines)


def _build_judge_prompt(sub_query: str, evidence: list[dict[str, Any]]) -> str:
    return (
        "You are a research editor deciding whether the evidence below is "
        "enough to answer a sub-query, under its most reasonable "
        "interpretation. Answer YES if the evidence contains specific "
        "statements that directly and substantially answer it — do not "
        "fail it merely because the original phrasing was informal or "
        "used a vague pronoun, as long as the evidence gives a clear, "
        "concrete answer. Answer NO if the sub-query is too "
        "underspecified to interpret at all, or if the evidence is only "
        "tangentially related, missing the key detail, or contradictory "
        "— and say what is missing or how the sub-query should be "
        "narrowed.\n\n"
        f"Sub-query: {sub_query}\n\n"
        f"Evidence:\n{_format_evidence(evidence)}\n\n"
        "Respond with EXACTLY two lines, no other text:\n"
        "Sufficient: YES or NO\n"
        "Reason: one sentence — if NO, name the missing detail or the "
        "narrower question to ask."
    )


def _build_reformulate_prompt(sub_query: str, reasoning: str) -> str:
    return (
        "A retrieval attempt for the sub-query below returned evidence that "
        "was judged insufficient. Using the judge's reasoning, write ONE "
        "improved search query that is more specific and more likely to "
        "retrieve a direct answer. Keep it concise (a search query, not a "
        "sentence).\n\n"
        f"Original sub-query: {sub_query}\n"
        f"Why the evidence was insufficient: {reasoning}\n\n"
        "Respond with EXACTLY one line, no other text:\n"
        "Query: the improved search query"
    )


def _parse_sufficiency(text: str) -> tuple[bool, str] | None:
    verdict = _SUFFICIENT_RE.search(text)
    if verdict is None:
        return None
    sufficient = verdict.group(1).strip().lower() == "yes"
    reason_match = _REASON_RE.search(text)
    reason = (
        reason_match.group(1).strip().splitlines()[0].strip()
        if reason_match
        else text.strip()
    )
    return sufficient, reason or "(no reason given)"


def _parse_query(text: str, fallback: str) -> str:
    match = _QUERY_RE.search(text)
    if match:
        line = match.group(1).strip().splitlines()[0].strip().strip('"')
        if line:
            return line
    # No "Query:" prefix — take the first non-empty line as-is.
    for line in text.splitlines():
        if line.strip():
            return line.strip().strip('"')
    return fallback


def merge_evidence(
    existing: list[dict[str, Any]],
    fresh: list[dict[str, Any]],
    cap: int = _EVIDENCE_CAP,
) -> list[dict[str, Any]]:
    """Union of prior and freshly retrieved evidence, deduped, newest first.

    A reformulate cycle should only ever *add* coverage — so the best
    hits from a later, sharper query lead, and earlier ones are kept
    behind them as best-effort padding up to ``cap``.
    """
    seen: set[tuple[str, str]] = set()
    merged: list[dict[str, Any]] = []
    for item in list(fresh) + list(existing):
        key = (
            str(item.get("title", "")).strip().lower(),
            str(item.get("snippet", "")).strip().lower()[:200],
        )
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)
        if len(merged) >= cap:
            break
    return merged


def _make_judge_node(llm):
    async def judge(state: JudgeState) -> dict[str, Any]:
        sub_query = state["sub_query"]
        evidence = state["evidence"]

        if not evidence:
            entry = {
                "attempt": state["retry_count"],
                "query": state["current_query"],
                "evidence_count": 0,
                "sufficient": False,
                "reasoning": "no evidence retrieved for this query",
            }
            return {
                "sufficient": False,
                "reasoning": entry["reasoning"],
                "history": state["history"] + [entry],
            }

        prompt = _build_judge_prompt(sub_query, evidence)
        try:
            text = await ainvoke_text(llm, prompt)
            parsed = _parse_sufficiency(text)
        except GeminiError as exc:
            # A dead judge shouldn't spin the loop — accept what we have.
            logger.warning("judge: Gemini call failed (%s) — accepting current evidence", exc)
            parsed = (True, f"judge error: {exc} — returning best-effort evidence")

        if parsed is None:
            logger.warning("judge: could not parse a verdict from Gemini response: %r", text)
            parsed = (True, "judge inconclusive — returning best-effort evidence")

        sufficient, reasoning = parsed
        entry = {
            "attempt": state["retry_count"],
            "query": state["current_query"],
            "evidence_count": len(evidence),
            "sufficient": sufficient,
            "reasoning": reasoning,
        }
        return {
            "sufficient": sufficient,
            "reasoning": reasoning,
            "history": state["history"] + [entry],
        }

    return judge


def _make_reformulate_node(llm, retrieve_fn: RetrieveFn):
    async def reformulate_and_retrieve(state: JudgeState) -> dict[str, Any]:
        sub_query = state["sub_query"]
        prompt = _build_reformulate_prompt(sub_query, state["reasoning"])
        try:
            text = await ainvoke_text(llm, prompt)
            new_query = _parse_query(text, fallback=sub_query)
        except GeminiError as exc:
            logger.warning("judge: reformulate call failed (%s) — reusing original sub-query", exc)
            new_query = sub_query

        logger.info(
            "judge: retry %d/%d — reformulated %r -> %r",
            state["retry_count"] + 1,
            state["max_retries"],
            sub_query,
            new_query,
        )

        try:
            decision, fresh = await retrieve_fn(new_query)
        except Exception as exc:  # noqa: BLE001 - retrieval failure is best-effort
            logger.warning("judge: re-retrieval failed (%s) — keeping prior evidence", exc)
            decision, fresh = None, []

        merged = merge_evidence(state["evidence"], fresh)
        history_note = {
            "attempt": state["retry_count"] + 1,
            "reformulated_query": new_query,
            "route": decision.route if decision is not None else "(retrieval failed)",
            "route_reason": decision.reasoning if decision is not None else "",
            "fresh_count": len(fresh),
            "merged_count": len(merged),
        }
        return {
            "current_query": new_query,
            "evidence": merged,
            "retry_count": state["retry_count"] + 1,
            "history": state["history"] + [history_note],
        }

    return reformulate_and_retrieve


def _decide_after_judge(state: JudgeState) -> str:
    if state["sufficient"]:
        return END
    if state["retry_count"] >= state["max_retries"]:
        return END
    return "reformulate_and_retrieve"


def _build_graph(llm, retrieve_fn: RetrieveFn):
    graph = StateGraph(JudgeState)
    graph.add_node("judge", _make_judge_node(llm))
    graph.add_node("reformulate_and_retrieve", _make_reformulate_node(llm, retrieve_fn))

    graph.set_entry_point("judge")
    graph.add_conditional_edges(
        "judge",
        _decide_after_judge,
        {END: END, "reformulate_and_retrieve": "reformulate_and_retrieve"},
    )
    graph.add_edge("reformulate_and_retrieve", "judge")
    return graph.compile()


class JudgeOutcome(TypedDict):
    evidence: list[dict[str, Any]]
    sufficient: bool
    retry_count: int
    reasoning: str
    final_query: str
    history: list[dict[str, Any]]


class EvidenceJudge:
    """Runs the judge / reformulate-and-retrieve graph for one sub-query."""

    def __init__(
        self,
        retrieve_fn: RetrieveFn,
        api_key: str | None = None,
        model: str | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> None:
        self._llm = build_chat_model(api_key, model)
        self._max_retries = max_retries
        self._app = _build_graph(self._llm, retrieve_fn)

    async def run(
        self,
        sub_query: str,
        evidence: list[dict[str, Any]],
        initial_query: str | None = None,
    ) -> JudgeOutcome:
        final = await self._app.ainvoke(
            {
                "sub_query": sub_query,
                "current_query": initial_query or sub_query,
                "evidence": list(evidence),
                "retry_count": 0,
                "max_retries": self._max_retries,
                "sufficient": False,
                "reasoning": "",
                "history": [],
            }
        )
        return JudgeOutcome(
            evidence=final["evidence"],
            sufficient=final["sufficient"],
            retry_count=final["retry_count"],
            reasoning=final["reasoning"],
            final_query=final["current_query"],
            history=final["history"],
        )
