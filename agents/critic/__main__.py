"""Runs the critic A2A server on port 9003.

Selects the claim-verification backend (naive keyword-overlap, or
LangGraph + Gemini) at startup and fails loudly — same pattern as
chair's Agent Card validation and synthesis's LLM provider — if it's
misconfigured, rather than discovering it on the first verify-claims
request.
"""

import logging

import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from dotenv import load_dotenv

from agents.critic.agent_card import PORT, build_agent_card
from agents.critic.agent_executor import CriticAgentExecutor
from agents.critic.langgraph_verifier import CriticProviderConfigError
from agents.critic.verifier import build_verifier
from common.logging_config import configure_logging

load_dotenv()

HOST = "0.0.0.0"

logger = logging.getLogger(__name__)


def main() -> None:
    configure_logging()

    try:
        verifier = build_verifier()
    except CriticProviderConfigError as exc:
        logger.error("critic: startup configuration failed — %s", exc)
        raise SystemExit(1) from exc

    logger.info("critic: using verifier %s", type(verifier).__name__)

    request_handler = DefaultRequestHandler(
        agent_executor=CriticAgentExecutor(verifier),
        task_store=InMemoryTaskStore(),
    )
    app = A2AStarletteApplication(
        agent_card=build_agent_card(),
        http_handler=request_handler,
    )

    uvicorn.run(app.build(), host=HOST, port=PORT)


if __name__ == "__main__":
    main()
