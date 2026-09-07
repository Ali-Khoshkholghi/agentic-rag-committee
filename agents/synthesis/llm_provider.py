"""LLM provider abstraction for synthesis.

Step 6 of the SPEC: swap the naive templater for a real LLM call (Google
Gemini) without synthesis's executor (streaming/task logic) needing to
know which provider is behind it — it only ever calls draft() and gets
back an async iterator of text chunks, or a SynthesisProviderError it
turns into a failed task. Mirrors the SearchProvider adapter pattern
from agents/lit_search/search_provider.py (Step 5).
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

import httpx

from agents.synthesis.drafting import build_draft, parse_prioritized_index
from common.text import split_sentences

# Verified reachable and working against the live API as of this writing;
# override via GEMINI_MODEL if the free-tier lineup changes later (an
# earlier default, gemini-2.0-flash, was already retired mid-development —
# Google's own 404 response named this replacement).
DEFAULT_GEMINI_MODEL = "gemini-3.6-flash"


class SynthesisProviderConfigError(RuntimeError):
    """Raised at startup when the selected provider is misconfigured
    (e.g. SYNTHESIS_PROVIDER=gemini but no GEMINI_API_KEY) — synthesis
    should fail loudly here, the same pattern as chair's Agent Card
    validation and lit_search's search provider, rather than discovering
    it on the first draft request."""


class SynthesisProviderError(RuntimeError):
    """Raised when a live draft call fails: an API error, rate limit,
    timeout, network failure, or an empty/safety-blocked response.
    Caught by the executor and mapped to a failed task state with a
    clear message."""


class SynthesisProvider(ABC):
    """A source of streamed text chunks drafting an answer to a topic."""

    @abstractmethod
    def draft(
        self,
        topic: str,
        sources: list[dict[str, Any]],
        guidance: str | None = None,
    ) -> AsyncIterator[str]:
        """Yields text chunks that concatenate into a drafted answer.

        `guidance` is an optional free-text instruction from a human's
        input-required follow-up reply (e.g. "prioritize source 1",
        "present both perspectives").

        Raises SynthesisProviderError if drafting fails.
        """


class FakeSynthesisProvider(SynthesisProvider):
    """Deterministic templated draft — no API key required.

    The default provider, and useful for running debug_client.py
    without burning real LLM API quota. Reuses the Step 3/4 templater
    (agents/synthesis/drafting.py) and streams it out sentence by
    sentence, exactly as before Step 6.
    """

    async def draft(
        self,
        topic: str,
        sources: list[dict[str, Any]],
        guidance: str | None = None,
    ) -> AsyncIterator[str]:
        prioritized_index = None
        present_both = False
        if guidance:
            prioritized_index = parse_prioritized_index(guidance, sources)
            present_both = prioritized_index is None and "both" in guidance.lower()

        text = build_draft(
            topic,
            sources,
            prioritized_index=prioritized_index,
            present_both=present_both,
        )
        for index, sentence in enumerate(split_sentences(text)):
            yield sentence if index == 0 else f" {sentence}"


def _build_prompt(
    topic: str, sources: list[dict[str, Any]], guidance: str | None
) -> str:
    if sources:
        source_lines = [
            f"[{i}] title: {s.get('title', '')}\n"
            f"    url: {s.get('url', '')}\n"
            f"    snippet: {s.get('snippet', '')}"
            for i, s in enumerate(sources)
        ]
        sources_block = "\n".join(source_lines)
    else:
        sources_block = "(no sources provided)"

    guidance_block = f"\n\nAdditional guidance from the user: {guidance}" if guidance else ""

    return (
        "You are a research-synthesis assistant. Write a short, well-organized "
        f"answer to the following research topic:\n\n{topic}\n\n"
        "Base your answer ONLY on the sources listed below — do not use any "
        "outside knowledge, and do not add facts that are not supported by "
        "these sources. Write in your own words: paraphrase and synthesize "
        "the source material. Do not quote sources verbatim and do not copy "
        "sentences from the snippets directly.\n\n"
        f"Sources:\n{sources_block}"
        f"{guidance_block}\n\n"
        "Write the answer as plain prose — no headings, no bullet points, no "
        "markdown formatting."
    )


def _extract_chunk_text(chunk: dict[str, Any]) -> tuple[str, str | None]:
    """Pulls text (if any) and a block reason (if any) out of one Gemini
    streamGenerateContent SSE chunk."""
    block_reason = chunk.get("promptFeedback", {}).get("blockReason")
    if block_reason:
        return "", block_reason

    candidates = chunk.get("candidates") or []
    if not candidates:
        return "", None

    candidate = candidates[0]
    finish_reason = candidate.get("finishReason")
    parts = candidate.get("content", {}).get("parts", [])
    text = "".join(p.get("text", "") for p in parts)

    if not text and finish_reason and finish_reason not in ("STOP", "MAX_TOKENS"):
        return "", finish_reason

    return text, None


class GeminiSynthesisProvider(SynthesisProvider):
    """Real drafting via the Google Gemini API (generativelanguage.googleapis.com)."""

    DEFAULT_API_URL = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_GEMINI_MODEL,
        api_url: str = DEFAULT_API_URL,
        # Generous default: some Gemini models do internal "thinking" before
        # the first visible token, which can add real latency even for
        # short prompts (observed ~20s for a trivial one-line prompt).
        timeout_seconds: float = 60.0,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._api_url = api_url
        self._timeout_seconds = timeout_seconds

    async def draft(
        self,
        topic: str,
        sources: list[dict[str, Any]],
        guidance: str | None = None,
    ) -> AsyncIterator[str]:
        prompt = _build_prompt(topic, sources, guidance)
        url = f"{self._api_url}/{self._model}:streamGenerateContent"
        payload = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
        params = {"alt": "sse", "key": self._api_key}

        got_any_text = False
        try:
            async with httpx.AsyncClient(timeout=self._timeout_seconds) as client:
                async with client.stream(
                    "POST", url, json=payload, params=params
                ) as response:
                    if response.status_code in (401, 403):
                        raise SynthesisProviderError(
                            f"Gemini rejected the API key (HTTP {response.status_code}) "
                            "— check GEMINI_API_KEY"
                        )
                    if response.status_code == 429:
                        raise SynthesisProviderError(
                            "Gemini API rate limit exceeded (HTTP 429)"
                        )
                    if response.status_code >= 400:
                        body = await response.aread()
                        raise SynthesisProviderError(
                            f"Gemini API error (HTTP {response.status_code}): "
                            f"{body.decode(errors='replace')[:200]}"
                        )

                    async for line in response.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        data_str = line[len("data:") :].strip()
                        if not data_str:
                            continue
                        try:
                            chunk = json.loads(data_str)
                        except ValueError as exc:
                            raise SynthesisProviderError(
                                f"Gemini returned invalid JSON: {exc}"
                            ) from exc

                        text, block_reason = _extract_chunk_text(chunk)
                        if block_reason:
                            raise SynthesisProviderError(
                                f"Gemini blocked or stopped the response "
                                f"early ({block_reason})"
                            )
                        if text:
                            got_any_text = True
                            yield text
        except httpx.TimeoutException as exc:
            raise SynthesisProviderError(
                f"Gemini request timed out after {self._timeout_seconds}s"
            ) from exc
        except httpx.HTTPError as exc:
            raise SynthesisProviderError(f"network error calling Gemini: {exc}") from exc

        if not got_any_text:
            raise SynthesisProviderError("Gemini returned an empty response")


def build_provider() -> SynthesisProvider:
    """Selects the synthesis provider from SYNTHESIS_PROVIDER (default: fake).

    Fails loudly (SynthesisProviderConfigError) if 'gemini'/'adk' is
    selected but GEMINI_API_KEY isn't set, so misconfiguration is caught
    at startup rather than on the first draft request.
    """
    provider_name = os.environ.get("SYNTHESIS_PROVIDER", "fake").strip().lower()

    if provider_name == "fake":
        return FakeSynthesisProvider()

    if provider_name == "gemini":
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise SynthesisProviderConfigError(
                "SYNTHESIS_PROVIDER=gemini but GEMINI_API_KEY is not set. "
                "Set GEMINI_API_KEY, or set SYNTHESIS_PROVIDER=fake (or "
                "leave it unset) to use the built-in templated draft instead."
            )
        model = os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
        api_url = os.environ.get("GEMINI_API_URL", GeminiSynthesisProvider.DEFAULT_API_URL)
        return GeminiSynthesisProvider(api_key=api_key, model=model, api_url=api_url)

    if provider_name == "adk":
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise SynthesisProviderConfigError(
                "SYNTHESIS_PROVIDER=adk but GEMINI_API_KEY is not set. "
                "Set GEMINI_API_KEY, or set SYNTHESIS_PROVIDER=fake (or "
                "leave it unset) to use the built-in templated draft instead."
            )
        # Lazy import: keeps google-adk out of the import graph for
        # anyone running fake/gemini mode, and avoids any import-order
        # coupling with this module (adk_provider imports SynthesisProvider
        # and SynthesisProviderError from here).
        from agents.synthesis.adk_provider import ADKSynthesisProvider

        model = os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
        return ADKSynthesisProvider(model=model)

    raise SynthesisProviderConfigError(
        f"Unknown SYNTHESIS_PROVIDER={provider_name!r} (expected 'fake', 'gemini', or 'adk')"
    )
