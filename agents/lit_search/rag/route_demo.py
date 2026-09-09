"""Standalone acceptance check for SPEC Step 10 (routing + self-correction).

Runs one or more sub-queries through the full agentic pipeline
(``rag/pipeline.py``) directly — no A2A server — and prints, for each:

- the router's decision (vector_store / web / both) and its reasoning,
- every judge pass (sufficient? why?) and every reformulate-and-retry
  cycle (the new query, its route, how many sources it added),
- the final evidence list with each item's origin.

    python -m agents.lit_search.rag.route_demo
    python -m agents.lit_search.rag.route_demo --query "your sub-query here"
    python -m agents.lit_search.rag.route_demo --max-retries 3

Needs the same env the committee uses (``GEMINI_API_KEY``, and
``TAVILY_API_KEY`` if ``LIT_SEARCH_PROVIDER=tavily``) plus a built Chroma
DB (``python -m agents.lit_search.rag.ingest``).
"""

from __future__ import annotations

import argparse
import asyncio
import textwrap

from dotenv import load_dotenv

from .pipeline import GatherResult, build_pipeline_from_env

# One per routing target, matching the Step 10 acceptance criteria.
DEFAULT_QUERIES = [
    # Clearly local: a training detail from DPR, one of the five papers.
    "what does DPR use for negative sampling?",
    # Clearly web: current, not in the five papers.
    "which models top the RAG leaderboards in 2025 and what retrievers do they use?",
    # Deliberately vague / broad — aims to trigger a reformulate-and-retry.
    "how do you make it better?",
]

_PREVIEW_WIDTH = 110


def _preview(text: str, width: int = _PREVIEW_WIDTH) -> str:
    flat = " ".join(str(text).split())
    return flat if len(flat) <= width else flat[: width - 1] + "…"


def _print_result(result: GatherResult) -> None:
    outcome = result.outcome
    print("=" * 100)
    print(f"SUB-QUERY: {result.sub_query}")
    print("=" * 100)

    d = result.initial_decision
    print(f"router  -> {d.route.upper()}")
    print(f"           reason: {d.reasoning}")

    print("\nself-correction loop:")
    for entry in outcome["history"]:
        if "reformulated_query" in entry:
            print(
                f"  ↻ retry {entry['attempt']}: reformulated -> {entry['reformulated_query']!r}"
                f"\n      route={entry['route'].upper()}  ({entry['route_reason']})"
                f"\n      fresh={entry['fresh_count']} source(s), merged={entry['merged_count']}"
            )
        else:
            verdict = "SUFFICIENT" if entry["sufficient"] else "INSUFFICIENT"
            print(
                f"  ⚖ judge (attempt {entry['attempt']}, {entry['evidence_count']} source(s)): {verdict}"
                f"\n      reason: {entry['reasoning']}"
            )

    print(
        f"\nSETTLED: sufficient={outcome['sufficient']}  "
        f"retries={outcome['retry_count']}  final_query={outcome['final_query']!r}"
    )

    counts: dict[str, int] = {}
    for s in result.sources:
        counts[s.get("origin", "?")] = counts.get(s.get("origin", "?"), 0) + 1
    tally = ", ".join(f"{k} x{v}" for k, v in counts.items()) or "(none)"
    print(f"evidence ({len(result.sources)}): {tally}")
    for i, s in enumerate(result.sources, start=1):
        print(f"\n  [{i}] ({s.get('origin', '?')}) {s.get('title', '(untitled)')}")
        print(f"      {s.get('url', '')}")
        print(textwrap.indent(_preview(s.get("snippet", "")), "      | "))
    print()


async def _run(queries: list[str], max_retries: int | None) -> None:
    pipeline = build_pipeline_from_env(max_retries=max_retries)
    for query in queries:
        result = await pipeline.gather(query)
        _print_result(result)


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--query",
        action="append",
        dest="queries",
        help="Sub-query (repeatable). Defaults to the built-in local / web / vague set.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=None,
        help="Override the judge's max reformulate-and-retry cycles (default 2).",
    )
    args = parser.parse_args(argv)
    asyncio.run(_run(args.queries or DEFAULT_QUERIES, args.max_retries))


if __name__ == "__main__":
    main()
