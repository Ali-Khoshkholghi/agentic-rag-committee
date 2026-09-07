"""Google ADK-backed drafting for the synthesis agent.

Step 8 of the SPEC: rebuild synthesis's LLM-calling logic on the
**Google ADK** (`google-adk`), a genuine agentic framework, as a second
proof point alongside critic's LangGraph rebuild (Step 7) — both backed
by Gemini, both interoperating over their agent's unchanged A2A
contract. This module is a drop-in SynthesisProvider (same ABC as
GeminiSynthesisProvider in llm_provider.py): same draft() signature,
same streamed-text-chunk contract, same SynthesisProviderError on
failure.

Deliberately a single ADK `Agent` with no tools and no sub-agents — the
task doesn't call for a multi-step tool-use loop, just one
well-structured agentic call, using ADK's own streaming Runner to keep
emitting chunks incrementally rather than blocking for the whole draft.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid4

from google.adk.agents import Agent
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import errors as genai_errors
from google.genai import types

from agents.synthesis.llm_provider import DEFAULT_GEMINI_MODEL, SynthesisProvider, SynthesisProviderError

logger = logging.getLogger(__name__)

APP_NAME = "synthesis"
USER_ID = "synthesis"

# Same grounding rules as GeminiSynthesisProvider's prompt (Step 6),
# split out as a system instruction rather than folded into one user
# message — ADK agents take instruction/input separately.
SYSTEM_INSTRUCTION = (
    "You are a research-synthesis assistant. Base your answer ONLY on "
    "the sources given to you in the request — do not use any outside "
    "knowledge, and do not add facts that are not supported by these "
    "sources. Write in your own words: paraphrase and synthesize the "
    "source material. Do not quote sources verbatim and do not copy "
    "sentences from the snippets directly. Write the answer as plain "
    "prose — no headings, no bullet points, no markdown formatting."
)


def _build_user_message(
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
        "Write a short, well-organized answer to the following research "
        f"topic:\n\n{topic}\n\n"
        f"Sources:\n{sources_block}"
        f"{guidance_block}"
    )


def _extract_text(event: Any) -> str:
    if event.content is None or not event.content.parts:
        return ""
    return "".join(part.text or "" for part in event.content.parts)


class ADKSynthesisProvider(SynthesisProvider):
    """Real drafting via a single Google ADK Agent backed by Gemini."""

    def __init__(
        self,
        model: str = DEFAULT_GEMINI_MODEL,
        # Same generous default as GeminiSynthesisProvider: Gemini's
        # internal "thinking" can add real latency before the first token.
        timeout_seconds: float = 60.0,
    ) -> None:
        self._agent = Agent(
            name="synthesis_agent",
            model=model,
            instruction=SYSTEM_INSTRUCTION,
        )
        self._session_service = InMemorySessionService()
        self._runner = Runner(
            agent=self._agent, app_name=APP_NAME, session_service=self._session_service
        )
        self._timeout_seconds = timeout_seconds

    async def draft(
        self,
        topic: str,
        sources: list[dict[str, Any]],
        guidance: str | None = None,
    ) -> AsyncIterator[str]:
        session_id = str(uuid4())
        await self._session_service.create_session(
            app_name=APP_NAME, user_id=USER_ID, session_id=session_id
        )

        message = _build_user_message(topic, sources, guidance)
        content = types.Content(role="user", parts=[types.Part(text=message)])
        run_config = RunConfig(streaming_mode=StreamingMode.SSE)

        got_any_text = False
        try:
            async with asyncio.timeout(self._timeout_seconds):
                async for event in self._runner.run_async(
                    user_id=USER_ID,
                    session_id=session_id,
                    new_message=content,
                    run_config=run_config,
                ):
                    if event.error_message:
                        raise SynthesisProviderError(
                            f"ADK/Gemini error: {event.error_message}"
                        )
                    if not event.partial:
                        # The final event re-emits the full concatenated
                        # text non-partial — already streamed via the
                        # partial chunks above, so skip it here.
                        continue
                    text = _extract_text(event)
                    if text:
                        got_any_text = True
                        yield text
        except SynthesisProviderError:
            raise
        except TimeoutError as exc:
            raise SynthesisProviderError(
                f"ADK/Gemini request timed out after {self._timeout_seconds}s"
            ) from exc
        except genai_errors.APIError as exc:
            raise SynthesisProviderError(f"Gemini API error via ADK: {exc}") from exc
        except Exception as exc:  # noqa: BLE001 - collapse any other ADK/SDK failure
            raise SynthesisProviderError(f"ADK agent call failed: {exc}") from exc
        finally:
            await self._session_service.delete_session(
                app_name=APP_NAME, user_id=USER_ID, session_id=session_id
            )

        if not got_any_text:
            raise SynthesisProviderError("ADK/Gemini returned an empty response")
