"""Agentic retrieval pipeline for lit_search (SPEC Step 10).

Ties the Step 10 pieces together for one sub-query:

    sub-query
       │
       ▼
    router (router.py)  ──►  vector store (retriever.py, Step 9)
       │                     and/or web search (search_provider.py, Step 5)
       ▼
    merge  ──►  judge / reformulate-and-retrieve graph (judge.py)
       │
       ▼
    evidence  (list of {title, url, snippet, origin} dicts)

The output list is the same ``{title, url, snippet}`` shape lit_search
already streams to the chair — with one extra ``origin`` key noting which
source each item came from. ``agents/lit_search/agentic_search.py`` adapts
this to the ``SearchProvider`` interface the executor calls, so the A2A
layer is untouched.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from typing import Any

from agents.lit_search.search_provider import (
    SearchProvider,
    SearchProviderError,
    build_provider,
)

from .judge import DEFAULT_MAX_RETRIES, EvidenceJudge, JudgeOutcome, merge_evidence
from .router import RouteDecision, SourceRouter, build_router_from_env

logger = logging.getLogger(__name__)

DEFAULT_VECTOR_TOP_K = 5
DEFAULT_WEB_MAX_RESULTS = 4


@dataclass
class GatherResult:
    """Everything one ``gather()`` produced — evidence plus the trace of
    how the router and judge got there (for logs and the route demo)."""

    sub_query: str
    sources: list[dict[str, Any]]
    initial_decision: RouteDecision
    outcome: JudgeOutcome
    per_query_routes: list[RouteDecision] = field(default_factory=list)

    @property
    def retried(self) -> bool:
        return self.outcome["retry_count"] > 0


def _chunk_to_source(chunk: Any) -> dict[str, Any]:
    """Maps a Step 9 ``RetrievedChunk`` to lit_search's source dict shape."""
    return {
        "title": chunk.title or chunk.source,
        "url": f"local-corpus://{chunk.source}#chunk-{chunk.chunk_index}",
        "snippet": chunk.text,
        "origin": "vector_store",
        "score": round(float(chunk.rerank_score), 4),
    }


class AgenticRetrievalPipeline:
    """Router + hybrid retriever + web search + self-correction judge."""

    def __init__(
        self,
        router: SourceRouter,
        web_provider: SearchProvider,
        retriever_factory,
        api_key: str | None = None,
        model: str | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        vector_top_k: int = DEFAULT_VECTOR_TOP_K,
        web_max_results: int = DEFAULT_WEB_MAX_RESULTS,
    ) -> None:
        self._router = router
        self._web_provider = web_provider
        # Deferred: building a HybridRetriever loads torch + two MiniLM
        # models. Do it on the first vector-store route, not at startup,
        # and only once.
        self._retriever_factory = retriever_factory
        self._retriever = None
        self._api_key = api_key
        self._model = model
        self._max_retries = max_retries
        self._vector_top_k = vector_top_k
        self._web_max_results = web_max_results

    # -- individual sources ------------------------------------------------

    def _vector_search_sync(self, query: str) -> list[dict[str, Any]]:
        if self._retriever is None:
            logger.info("agentic-rag: loading hybrid retriever (first vector-store route)")
            self._retriever = self._retriever_factory()
        chunks = self._retriever.retrieve(query, top_k=self._vector_top_k)
        return [_chunk_to_source(c) for c in chunks]

    async def _vector_search(self, query: str) -> list[dict[str, Any]]:
        try:
            # Retrieval is sync + CPU-bound (embeddings, BM25, rerank) —
            # keep it off the event loop so A2A streaming stays responsive.
            return await asyncio.to_thread(self._vector_search_sync, query)
        except Exception as exc:  # noqa: BLE001 - degrade, don't crash the task
            logger.warning("agentic-rag: vector search failed (%s)", exc)
            return []

    async def _web_search(self, query: str) -> list[dict[str, Any]]:
        try:
            results = await self._web_provider.search(query, self._web_max_results)
        except SearchProviderError as exc:
            # "no results" is a SearchProviderError too — for the agentic
            # flow that's just an empty web arm, not a failed task.
            logger.warning("agentic-rag: web search failed (%s)", exc)
            return []
        return [{**r, "origin": "web"} for r in results]

    # -- one retrieval round (router -> sources -> merge) ----------------

    async def retrieve_once(
        self, query: str
    ) -> tuple[RouteDecision, list[dict[str, Any]]]:
        decision = await self._router.route(query)
        logger.info(
            "agentic-rag: route %r -> %s (%s)",
            query, decision.route, decision.reasoning,
        )

        evidence: list[dict[str, Any]] = []
        if decision.use_vector_store:
            evidence += await self._vector_search(query)
        if decision.use_web:
            evidence += await self._web_search(query)

        merged = merge_evidence([], evidence)
        return decision, merged

    # -- full pipeline for one sub-query --------------------------------

    async def gather(self, sub_query: str) -> GatherResult:
        routes: list[RouteDecision] = []

        async def tracked_retrieve(query: str):
            decision, evidence = await self.retrieve_once(query)
            routes.append(decision)
            return decision, evidence

        initial_decision, evidence = await tracked_retrieve(sub_query)

        judge = EvidenceJudge(
            retrieve_fn=tracked_retrieve,
            api_key=self._api_key,
            model=self._model,
            max_retries=self._max_retries,
        )
        outcome = await judge.run(sub_query, evidence, initial_query=sub_query)

        logger.info(
            "agentic-rag: sub-query %r settled after %d retr%s — sufficient=%s, %d source(s)",
            sub_query,
            outcome["retry_count"],
            "y" if outcome["retry_count"] == 1 else "ies",
            outcome["sufficient"],
            len(outcome["evidence"]),
        )
        return GatherResult(
            sub_query=sub_query,
            sources=outcome["evidence"],
            initial_decision=initial_decision,
            outcome=outcome,
            per_query_routes=routes,
        )


def build_pipeline_from_env(
    web_provider: SearchProvider | None = None,
    max_retries: int | None = None,
) -> AgenticRetrievalPipeline:
    """Builds the pipeline from the shared GEMINI_* / LIT_SEARCH_* env.

    - Router + judge: ``GEMINI_API_KEY`` / ``GEMINI_MODEL`` (same key
      synthesis and critic use). Raises if the key is missing.
    - Web arm: the existing ``build_provider()`` (``LIT_SEARCH_PROVIDER``),
      unless one is passed in.
    - Vector arm: a ``HybridRetriever`` over ``data/chroma_db/`` — built
      lazily on the first vector-store route, so a missing DB only errors
      if/when a sub-query actually needs the corpus.
    """
    from .retriever import RetrieverError, load_retriever

    router = build_router_from_env()

    if web_provider is None:
        web_provider = build_provider()

    retries = max_retries
    if retries is None:
        raw = os.environ.get("LIT_SEARCH_MAX_RETRIES", "").strip()
        retries = int(raw) if raw.isdigit() else DEFAULT_MAX_RETRIES

    def retriever_factory():
        try:
            return load_retriever()
        except RetrieverError as exc:
            raise RetrieverError(
                f"{exc}\nA vector-store route needs the Chroma DB — run "
                f"`python -m agents.lit_search.rag.ingest` first."
            ) from exc

    return AgenticRetrievalPipeline(
        router=router,
        web_provider=web_provider,
        retriever_factory=retriever_factory,
        max_retries=retries,
    )
