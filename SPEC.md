# Research Committee — A2A Multi-Agent Project Spec

## Goal

Build a small standalone "committee of agents" research assistant using the
**A2A (Agent2Agent) protocol** (Linux Foundation, v1.0). A human asks a
research question; a chair agent delegates to three specialist agents over
A2A, streaming progress the whole way, and returns a synthesized, checked
answer.

This is a learning project — prioritize a correct, minimal implementation of
each A2A concept over feature completeness. Build and verify one agent at a
time, in the order below. Do not skip ahead to the chair until each
downstream agent works in isolation.

## Tech choices (already decided — do not deviate)

- Language: **Python**, using the official A2A Python SDK.
- Transport: **JSON-RPC 2.0** with **SSE streaming** (`message/stream`),
  not REST or gRPC.
- Local-only for now: all agents run on `localhost` on different ports.
- No auth/security hardening needed for v1 — flag it as a TODO, don't
  implement it yet.

## Agents

| Agent | Port | Skill (Agent Card) | Streams? | Output |
|---|---|---|---|---|
| `lit_search` | 9001 | `find-sources` | yes — one source at a time via `TaskArtifactUpdateEvent` | list of `{title, url, snippet}` as DataParts |
| `synthesis` | 9002 | `draft-answer` | yes — text chunks via `TaskArtifactUpdateEvent` with `append: true` | drafted paragraph(s), TextPart; internally a naive templater, a raw Gemini call, or a Google ADK agent (Step 8), same contract either way |
| `critic` | 9003 | `verify-claims` | no — single final DataPart | list of `{claim, supported: bool, note}`; internally naive keyword-overlap or LangGraph+Gemini (Step 7), same contract either way |
| `chair` | 9000 | `research-question` (the only skill a human calls) | yes — relays committee progress upward | final answer + critique summary |

Each agent (including `chair`) must publish a valid A2A **Agent Card** at
`/.well-known/agent-card.json` with accurate `name`, `description`, `url`,
`capabilities.streaming`, `defaultInputModes`, `defaultOutputModes`, and a
`skills` array describing exactly what it does. Treat agents as opaque to
each other — no agent should assume internal implementation details of
another; they only know each other via Agent Cards.

## Project structure

```
research-committee/
├── agents/
│   ├── lit_search/     # A2A server, port 9001
│   ├── synthesis/      # A2A server, port 9002
│   └── critic/          # A2A server, port 9003
├── chair/                # A2A server, port 9000 — also an A2A client to the above three
├── common/                # shared helpers (e.g. task/event plumbing, logging)
├── run_all.py             # launches all four agents locally
└── SPEC.md                # this file
```

## Build order (do not parallelize — each step must run before the next starts)

### Step 1 — `lit_search` agent (fake data first)
Minimal A2A server. Publishes its Agent Card. Implements `message/stream`
for skill `find-sources`. For now, fake 3–5 sources with a short `sleep()`
between each, streamed as separate `TaskArtifactUpdateEvent`s (DataPart per
source), ending with `lastChunk: true` and task state `completed`. No real
web search yet — the goal is to prove the A2A streaming/task/artifact
mechanics work end-to-end.

### Step 2 — raw JSON-RPC/SSE test client
A standalone script (not the chair) that calls `lit_search` directly via
`message/stream` and prints each `TaskStatusUpdateEvent` /
`TaskArtifactUpdateEvent` as it arrives. This is a debugging tool to
visually confirm the stream — keep it in the repo as
`common/debug_client.py`.

### Step 3 — `synthesis` and `critic` agents
Same A2A server pattern as `lit_search`, different skills:
- `synthesis` (`draft-answer`): takes a topic + list of sources (DataParts),
  streams back a drafted answer as TextPart chunks with `append: true`,
  final chunk has `lastChunk: true`.
- `critic` (`verify-claims`): takes the drafted answer + sources, returns
  ONE final DataPart with per-claim verdicts. No streaming needed here —
  use `message/send` semantics for this one internally, but still expose
  `message/stream` on the Agent Card for consistency (can just emit a
  single event).

Each should support the `input-required` task state for at least one
realistic case (e.g. `lit_search` asks for date-range/source-count if not
specified; `synthesis` asks which sources to prioritize if `lit_search`
returned conflicting or ambiguous results). Implement this properly — it's
the part of A2A most people skip, and it's a core learning goal here.

### Step 4 — `chair` agent
On startup, fetches all three Agent Cards (`lit_search`, `synthesis`,
`critic`) to confirm they're reachable and capable before accepting any
human request. Publishes its own Agent Card with skill `research-question`.
On a human request:
1. Calls `lit_search` via `message/stream`, relays/logs its
   `TaskStatusUpdateEvent`/`TaskArtifactUpdateEvent`s as they arrive.
2. Feeds the completed source list into `synthesis`, same streaming
   relay pattern.
3. Feeds the drafted answer + sources into `critic`.
4. Returns the final synthesized answer plus the critic's verdict list as
   the chair's own task artifact/result.

Correlate all sub-tasks under a shared `contextId` so the whole exchange is
traceable as one logical conversation, even though each is a separate A2A
Task.

### Step 5 — swap in real data
Only after steps 1–4 work end-to-end with fake data: replace `lit_search`'s
fake source generator with a real web search call. Everything else should
be unaffected, since the A2A contract (Agent Card + task/artifact shape)
doesn't change.

**Status:** Implemented via a swappable `SearchProvider` adapter
(`agents/lit_search/search_provider.py`), with `FakeSearchProvider` (default)
and `TavilySearchProvider`, selected by `LIT_SEARCH_PROVIDER=fake|tavily`.
Live Tavily API verification not completed in this session due to
sandboxed egress restrictions (api.tavily.com not reachable). Provider
adapter logic verified via stub success/error-mode tests; failure-handling
path additionally verified against a genuine live network failure in this
session. Outstanding: a real-key smoke test against the live Tavily API to
confirm response-shape parsing, to be run outside this sandbox.

### Step 6 — swap in a real LLM for `synthesis`
Only after step 5 works end-to-end: replace `synthesis`'s naive templater
with a real LLM call. Everything else should be unaffected, since the A2A
contract doesn't change.

**Status:** Implemented via a swappable `SynthesisProvider` adapter
(`agents/synthesis/llm_provider.py`), with `FakeSynthesisProvider` (default)
and `GeminiSynthesisProvider` (Google Gemini, `GEMINI_API_KEY`), selected by
`SYNTHESIS_PROVIDER=fake|gemini`. Verified live against the Gemini API in
this session.

### Step 7 — rebuild `critic`'s internals on a second agent framework
Only after step 6 works end-to-end: rebuild `critic`'s internal
claim-verification logic on **LangGraph**, backed by the Gemini API (the
same `GEMINI_API_KEY` already used by `synthesis`), while its A2A server
layer, Agent Card, skill (`verify-claims`), port (9003), and DataPart
shape (`{draft, sources}` in → `{verdicts: [{claim, supported, note}]}`
out) don't change at all — this is the interoperability proof: no other
agent (`chair`, `synthesis`, `lit_search`) needs to know or care that
`critic` is now internally built on a different agent framework than the
rest of the committee.

The graph (`agents/critic/langgraph_verifier.py`) walks, per claim:
1. `retrieve_candidate_source` — cheap keyword-overlap ranking (reused
   from the naive checker, no LLM call) to narrow to the one most
   plausible source for that claim.
2. `judge_claim` — a single Gemini call asking strictly whether that
   source genuinely supports the claim, with a one-sentence
   justification. A response that can't be parsed into a clear yes/no
   falls back to `supported: False` with an "inconclusive" note, not a
   crash.
3. `assemble` — produces the same `{claim, supported, note}` shape as
   the naive checker, `note` now being the LLM's real justification.

**Status:** Implemented via a swappable `ClaimVerifier`
(`agents/critic/verifier.py`), with `NaiveVerifier` (the original
keyword-overlap checker) and `LangGraphVerifier` (LangGraph + Gemini),
selected by `CRITIC_MODE=naive|langgraph` — defaulting to `langgraph` if
`GEMINI_API_KEY` is set and `CRITIC_MODE` is left unset, otherwise `naive`
(same zero-key-required principle as `LIT_SEARCH_PROVIDER` and
`SYNTHESIS_PROVIDER`). A live Gemini network/API failure inside the graph
is mapped to a `failed` task, the same pattern as `synthesis`'s Gemini
error handling. Verified live in this session: naive mode unchanged
byte-for-byte; `langgraph` mode standalone against `critic` (port 9003)
with a hand-planted false claim, correctly flagged `supported: false` with
a real justification alongside genuinely supported claims flagged `true`;
and the full `chair` pipeline end-to-end with `SYNTHESIS_PROVIDER=gemini`
+ `CRITIC_MODE=langgraph`, confirming `critic`'s verdicts carry real
per-claim reasoning rather than overlap ratios.

### Step 8 — rebuild `synthesis`'s internals on a second agentic framework
Only after step 7 works end-to-end: rebuild `synthesis`'s LLM-calling logic
on the **Google ADK** (`google-adk`), backed by the same `GEMINI_API_KEY`,
while its A2A server layer, Agent Card, skill (`draft-answer`), port
(9002), streamed TextPart chunk semantics (`append`/`last_chunk`), and its
existing `input-required` pause/resume for thin/contradictory sources
don't change at all. Together with Step 7 (`critic` on LangGraph), this
makes the committee's two genuinely agentic nodes — `synthesis` and
`critic` — each built on a *different* agent framework, both Gemini-backed,
interoperating over their respective unchanged A2A contracts; `lit_search`
and `chair` remain simpler non-agentic A2A services by design. This is the
project's actual framework-level interoperability proof-point.

`agents/synthesis/adk_provider.py` wraps a single ADK `Agent` — no tools,
no sub-agents, no multi-step tool-use loop, just one well-structured
agentic call — configured with the same grounding rules as Step 6's raw
Gemini prompt (answer only from the given sources, paraphrase in its own
words, no verbatim quoting, no outside knowledge, plain prose), split into
an ADK `instruction` (the fixed rules) and a per-request user message
(topic + sources + optional guidance). Uses ADK's own streaming `Runner`
(`StreamingMode.SSE`) so the executor keeps emitting incremental
`TaskArtifactUpdateEvent`s exactly as before, not one blocking call. A
live ADK/Gemini failure (API error, timeout, empty response) raises
`SynthesisProviderError`, the same exception the raw Gemini provider
raises, so the executor's existing failure handling (`TaskState.failed`)
needs no changes either.

**Status:** Implemented by extending the existing `SynthesisProvider`
adapter (`agents/synthesis/llm_provider.py`) with `ADKSynthesisProvider`,
selected by `SYNTHESIS_PROVIDER=fake|gemini|adk` (`gemini`, Step 6's raw
call, kept for comparison/fallback; `fake` remains the zero-key default).
Verified live in this session: `fake` and `gemini` modes unchanged;
`adk` mode standalone against `synthesis` (port 9002) produced a genuinely
paraphrased draft (not verbatim source text) with the correct streaming
sequence (`append=false` first chunk, `append=true`/`last_chunk=true`
final chunk); and the full `chair` pipeline end-to-end with
`SYNTHESIS_PROVIDER=adk` + `CRITIC_MODE=langgraph` completed successfully,
`critic` producing a sensible mix of `supported`/`unsupported` verdicts
with real reasoning against ADK's drafted output. The full-pipeline run
hit the Gemini free-tier's per-model daily request quota (20/day) on the
default `gemini-3.6-flash` partway through this session's testing (a
`RESOURCE_EXHAUSTED` 429, itself a genuine live exercise of the
failure-handling path — it correctly surfaced as `TaskState.failed`); the
final full-pipeline verification run was completed live by pointing
`GEMINI_MODEL` at a different model (`gemini-flash-lite-latest`, same API,
separate quota bucket) for that one run only — no code or default model
changed, this was purely a quota workaround for verification.

## Definition of done for each step

- Agent Card is fetchable and schema-valid.
- `message/stream` produces a correctly ordered sequence of
  `TaskStatusUpdateEvent`/`TaskArtifactUpdateEvent`s ending in a terminal
  task state (`completed`, `failed`, or `canceled`).
- At least one `input-required` round-trip is implemented somewhere in the
  system (per step 3).
- `common/debug_client.py` can hit any agent independently and print a
  readable stream.

## Explicitly out of scope for v1

- Auth / Agent Card signature verification.
- Multi-tenancy.
- gRPC/REST bindings.
- Deployment beyond localhost.
- Persistent storage of tasks (in-memory is fine).
