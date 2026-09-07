"""Selects critic's claim-verification backend.

Step 7 of the SPEC: CRITIC_MODE=naive keeps the original keyword-overlap
checker (agents/critic/matching.py, Step 3); CRITIC_MODE=langgraph swaps
in the LangGraph + Gemini verifier (agents/critic/langgraph_verifier.py).
Same zero-key-required principle as SYNTHESIS_PROVIDER (Step 6) and
LIT_SEARCH_PROVIDER (Step 5): with no GEMINI_API_KEY, critic still runs
out of the box on the naive checker. With a key present and no explicit
CRITIC_MODE, critic defaults to the LangGraph-backed verifier.
"""

from __future__ import annotations

import os
from typing import Any, Protocol

from agents.critic.langgraph_verifier import CriticProviderConfigError, build_verifier_from_env
from agents.critic.matching import verify_claims as naive_verify_claims


class ClaimVerifier(Protocol):
    async def verify_claims(self, draft: str, sources: list[dict[str, Any]]) -> list[dict[str, Any]]: ...


class NaiveVerifier:
    """Wraps the original synchronous keyword-overlap checker."""

    async def verify_claims(self, draft: str, sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return naive_verify_claims(draft, sources)


def build_verifier() -> ClaimVerifier:
    """Selects the verifier from CRITIC_MODE (default: naive, or
    langgraph if GEMINI_API_KEY is set and CRITIC_MODE is unset).

    Fails loudly (CriticProviderConfigError) if 'langgraph' is selected
    (explicitly or by the key-present default) but GEMINI_API_KEY isn't
    set, so misconfiguration is caught at startup rather than on the
    first verify-claims request.
    """
    mode = os.environ.get("CRITIC_MODE", "").strip().lower()

    if not mode:
        mode = "langgraph" if os.environ.get("GEMINI_API_KEY") else "naive"

    if mode == "naive":
        return NaiveVerifier()

    if mode == "langgraph":
        return build_verifier_from_env()

    raise CriticProviderConfigError(f"Unknown CRITIC_MODE={mode!r} (expected 'naive' or 'langgraph')")
