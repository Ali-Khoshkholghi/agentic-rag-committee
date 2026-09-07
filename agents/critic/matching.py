"""Naive claim-verification matching for the critic agent.

This is intentionally a crude keyword-overlap heuristic — not real
NLP or an LLM call — just enough to prove the DataPart-in/DataPart-out
shape works end-to-end (SPEC.md Step 3). Swap for a real fact-checking
approach later.
"""

from __future__ import annotations

from common.text import split_sentences, tokenize

SUPPORT_THRESHOLD = 0.4

# Crude content-hygiene filter, not real claim extraction: the naive
# ". "/"! "/"? " sentence splitter above treats bare numbered-list markers
# and stray fragments ("1.", "2.", "FromFlow") pulled in from scraped
# snippets as if they were claims, which is nonsense to run a keyword-
# overlap check against. Dropping anything under ~4 words before scoring
# clears that out. Messier drafts will still let some fragments through;
# that's an accepted v1 limitation, not something to chase further now.
MIN_CLAIM_WORDS = 4


def best_matching_source(claim: str, sources: list[dict]) -> tuple[dict | None, float]:
    """Picks the source with the highest keyword-overlap ratio against `claim`.

    Returns (None, -1.0) if there are no sources or the claim has no
    tokens to score against. Shared by the naive checker below and the
    LangGraph verifier's retrieve_candidate_source node (Step 7) — the
    ranking itself is cheap and doesn't need an LLM call.
    """
    claim_tokens = tokenize(claim)
    if not sources or not claim_tokens:
        return None, -1.0

    best_source = sources[0]
    best_ratio = -1.0
    for source in sources:
        source_text = f"{source.get('title', '')} {source.get('snippet', '')}"
        source_tokens = tokenize(source_text)
        ratio = len(claim_tokens & source_tokens) / len(claim_tokens)
        if ratio > best_ratio:
            best_ratio = ratio
            best_source = source

    return best_source, best_ratio


def verify_claim(claim: str, sources: list[dict]) -> dict:
    """Scores a single claim sentence against every source, keeping the best match."""
    best_source, best_ratio = best_matching_source(claim, sources)
    if best_source is None:
        return {
            "claim": claim,
            "supported": False,
            "note": "Unable to verify — insufficient data",
        }

    supported = best_ratio >= SUPPORT_THRESHOLD
    title = best_source.get("title", "?")
    if supported:
        note = f"Matches source: {title}"
    else:
        note = f"No source with sufficient overlap (best: {title}, ratio: {best_ratio:.2f})"
    return {"claim": claim, "supported": supported, "note": note}


def split_claims(draft: str) -> list[str]:
    """Splits a drafted answer into claim sentences, dropping junk fragments."""
    return [c for c in split_sentences(draft) if len(c.split()) >= MIN_CLAIM_WORDS]


def verify_claims(draft: str, sources: list[dict]) -> list[dict]:
    """Splits a drafted answer into claim sentences and verifies each one."""
    return [verify_claim(claim, sources) for claim in split_claims(draft)]
