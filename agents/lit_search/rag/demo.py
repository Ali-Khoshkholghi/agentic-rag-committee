"""Standalone acceptance check for SPEC Step 9.

Runs ``ingest.py`` against ``agents/lit_search/data/corpus/``, then calls
the hybrid retriever with one or more sample queries and prints the
ranked, reranked chunks with their source metadata and the per-stage
scores (RRF, cross-encoder) and ranks (dense, sparse) that produced them.

    python -m agents.lit_search.rag.demo
    python -m agents.lit_search.rag.demo --query "your question here" --top-k 3
    python -m agents.lit_search.rag.demo --skip-ingest      # reuse existing DB

Deliberately not wired into lit_search's agent — this only exercises the
Step 9 modules directly.
"""

from __future__ import annotations

import argparse
import textwrap

from .ingest import ingest, load_embedder
from .retriever import DEFAULT_TOP_K, HybridRetriever

# Default set leans on near-duplicate papers on purpose: CRAG (2401.15884)
# and Self-RAG (2310.11511) both describe an evaluate-then-act RAG loop in
# very similar language, and REALM (2002.08909) and DPR (2004.04906) both
# describe a dense retriever feeding a reader. A working hybrid retriever
# should still pull the source that owns each mechanism to the top.
DEFAULT_QUERIES = [
    # CRAG-specific: the retrieval evaluator's confidence labels.
    "retrieval evaluator that outputs a Correct, Incorrect, or Ambiguous "
    "confidence label for the retrieved documents",
    # Self-RAG-specific: reflection / critique tokens.
    "special reflection tokens ISREL ISSUP ISUSE generated to critique "
    "whether a passage is relevant and supports the output",
    # CRAG-specific: decompose-then-recompose knowledge refinement + web search.
    "decompose-then-recompose algorithm to filter retrieved documents and "
    "large-scale web search as a knowledge source extension",
    # DPR-specific: dual-encoder trained with in-batch negatives.
    "dual-encoder dense passage retriever trained with in-batch negative "
    "examples and BM25 hard negatives",
    # REALM-specific: retriever pre-trained end-to-end with masked language modeling.
    "retriever pre-trained from scratch with a masked language modeling "
    "objective using a latent knowledge retriever",
    # Shared vocabulary, no single obvious owner — sanity check.
    "using retrieved passages to improve open-domain question answering",
]

_PREVIEW_WIDTH = 100


def _preview(text: str, width: int = _PREVIEW_WIDTH) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= width else flat[: width - 1] + "…"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--query",
        action="append",
        dest="queries",
        help="Sample query (repeatable). Defaults to a built-in set.",
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument(
        "--skip-ingest",
        action="store_true",
        help="Skip rebuilding the Chroma DB; query whatever is already there.",
    )
    args = parser.parse_args(argv)
    queries = args.queries or DEFAULT_QUERIES

    # One embedder instance, shared by ingest and the retriever.
    embedder = load_embedder()

    if args.skip_ingest:
        print("== Ingestion: skipped (--skip-ingest) ==\n")
    else:
        print("== Ingestion ==")
        summary = ingest(embedder=embedder)
        print(
            f"{summary.documents} document(s) -> {summary.chunks} chunk(s) "
            f"in collection {summary.collection!r}"
        )
        print(f"persisted at: {summary.chroma_path}")
        for source, count in sorted(summary.per_document.items()):
            print(f"  - {source}: {count} chunk(s)")
        print()

    retriever = HybridRetriever(embedder=embedder)

    for query in queries:
        print("=" * 96)
        print(f"QUERY: {query}")
        print("=" * 96)
        results = retriever.retrieve(query, top_k=args.top_k)
        if not results:
            print("  (no results)\n")
            continue

        counts: dict[str, int] = {}
        for chunk in results:
            counts[chunk.source] = counts.get(chunk.source, 0) + 1
        tally = ", ".join(f"{src} x{n}" for src, n in counts.items())
        print(f"top-{len(results)} by source: {tally}")
        print(f"rank 1: {results[0].source}")

        for rank, chunk in enumerate(results, start=1):
            print(
                f"\n[{rank}] {chunk.source_ref()}"
                f"\n    id={chunk.id}"
                f"\n    rerank={chunk.rerank_score:+.4f}  rrf={chunk.rrf_score:.5f}"
                f"  dense_rank={chunk.dense_rank}  sparse_rank={chunk.sparse_rank}"
            )
            body = textwrap.indent(_preview(chunk.text), "    | ")
            print(body)
        print()


if __name__ == "__main__":
    main()
