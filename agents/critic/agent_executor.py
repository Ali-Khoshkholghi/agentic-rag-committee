"""AgentExecutor for the critic agent.

Step 3 of the SPEC, rebuilt in Step 7 on a pluggable ClaimVerifier
(agents/critic/verifier.py) — the naive keyword-overlap checker, or a
LangGraph + Gemini graph. Non-streaming internally: one `working`
status, one final artifact, then `completed`/`failed`. No
input-required branch — critic always has enough to attempt a (possibly
all-unsupported) verdict list, though a live LangGraph/Gemini call can
still fail outright (network error, rate limit, empty response), which
is mapped to a failed task the same way synthesis handles Gemini errors.
"""

from __future__ import annotations

import logging
from uuid import uuid4

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import DataPart, Part, Task, UnsupportedOperationError
from a2a.utils import new_agent_text_message, new_task
from a2a.utils.errors import ServerError

from agents.critic.langgraph_verifier import CriticProviderError
from agents.critic.verifier import ClaimVerifier
from common.parts import get_data_part

logger = logging.getLogger(__name__)

# TODO(v1): no auth/security hardening yet — see SPEC.md.


class CriticAgentExecutor(AgentExecutor):
    """Verifies each claim sentence in a draft answer against its sources."""

    def __init__(self, verifier: ClaimVerifier) -> None:
        self._verifier = verifier

    async def execute(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        task = context.current_task
        if task is None:
            task = new_task(context.message)
            await event_queue.enqueue_event(task)

        updater = TaskUpdater(event_queue, task.id, task.context_id)
        await updater.start_work()

        payload = get_data_part(context.message) or {}
        draft = payload.get("draft", "")
        sources = payload.get("sources", [])

        try:
            verdicts = await self._verifier.verify_claims(draft, sources)
        except CriticProviderError as exc:
            await self._fail(updater, task, exc)
            return
        logger.info("critic: verified %d claim(s)", len(verdicts))

        await updater.add_artifact(
            parts=[Part(root=DataPart(data={"verdicts": verdicts}))],
            artifact_id=str(uuid4()),
            name="verdicts",
            last_chunk=True,
        )
        await updater.complete()

    async def _fail(
        self, updater: TaskUpdater, task: Task, exc: CriticProviderError
    ) -> None:
        logger.error("critic: %s", exc)
        await updater.failed(
            message=new_agent_text_message(
                f"Claim verification failed: {exc}", task.context_id, task.id
            )
        )

    async def cancel(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        raise ServerError(error=UnsupportedOperationError())
