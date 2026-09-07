# research-committee

## What this is

A small multi-agent research assistant — a "committee of agents" — built to
learn the **A2A (Agent2Agent) protocol** properly, end-to-end, rather than
just reading about it. It's a portfolio/learning repo, not a production
service: the goal is a correct, minimal demonstration of Agent Cards, the
A2A task lifecycle, JSON-RPC+SSE streaming, and multi-hop `input-required`
pause/resume — not feature completeness or real answer quality.

A human asks a research question. Four independent agents collaborate over
A2A to answer it: `lit_search` finds sources, `synthesis` drafts an answer,
`critic` checks the draft's claims against those sources, and `chair`
orchestrates the other three and talks to the human. Each agent is its own
process, on its own port, speaking A2A's JSON-RPC/SSE protocol — nothing is
imported or called directly between them.

## Why A2A (briefly)

A2A and MCP solve different problems. MCP is agent-to-*tool*: a model calls
a well-known function with a defined schema. A2A is agent-to-*agent*: two
independent, possibly differently-implemented agents negotiate a
multi-turn task, and neither has to know anything about the other's
internals. Agents in this repo are deliberately treated as opaque peers —
`chair` never imports `synthesis`'s code, it only knows `synthesis` exists,
what it can do, and how to talk to it because `synthesis` publishes an
**Agent Card** describing exactly that.

For the protocol itself — Agent Cards, Tasks, `message/stream`, artifacts,
task states — see the spec: https://a2a-protocol.org. This README only
covers how this repo uses it.

## The agents

| Agent | Port | Skill (Agent Card) | Streams? | Role |
|---|---|---|---|---|
| `lit_search` | 9001 | `find-sources` | yes — one source at a time | Finds candidate sources for a topic |
| `synthesis` | 9002 | `draft-answer` | yes — text chunks | Drafts an answer from those sources |
| `critic` | 9003 | `verify-claims` | no — single final artifact | Checks the draft's claims against the sources |
| `chair` | 9000 | `research-question` | yes — relays committee progress | Orchestrates the other three; the only agent a human talks to |

### `lit_search` (9001)

Streams candidate sources one at a time — a separate
`TaskArtifactUpdateEvent` per source (`append`/`lastChunk` sequencing),
not one batch dump — so a caller sees results arrive incrementally
regardless of how the underlying search actually behaves.

Search itself sits behind a swappable `SearchProvider` adapter
(`agents/lit_search/search_provider.py`): `FakeSearchProvider` (default,
deterministic, no API key) or `TavilySearchProvider` (real web search).
The executor's streaming/task logic doesn't know or care which one is
behind it — the A2A contract never depends on the search API.

### `synthesis` (9002)

Streams a drafted answer as incremental `TextPart` chunks (`append=true`,
final chunk `lastChunk=true`) rather than returning the whole draft at
once.

Drafting sits behind a swappable `SynthesisProvider` adapter
(`agents/synthesis/llm_provider.py`), the same pattern as `lit_search`'s
search provider: `FakeSynthesisProvider` (default, deterministic
templater, no API key) or `GeminiSynthesisProvider` (real drafting via
the Google Gemini API). The executor just streams whatever chunks the
provider yields — it doesn't know or care which one is behind it.

The Gemini model currently defaulted to (`gemini-3.6-flash`) does
internal "thinking" before it emits any visible text, which adds real,
noticeable latency: expect roughly 20–35 seconds before the *first*
streamed chunk arrives, even for a short prompt. That's normal, not a
hang — `debug_client.py`'s elapsed-time prefix will show it plainly.
`FakeSynthesisProvider` has no such delay.

It also implements a real `input-required` pause: before drafting, it
checks whether the sources are too thin (`<2`) or contradictory (opposing
sentiment keywords like "increases" vs. "decreases"). The contradiction
check is gated behind a **topic-relatedness threshold**
(`TOPIC_RELATEDNESS_THRESHOLD`) — worth calling out specifically, because
it was a real bug caught during verification: without the gate, two
*unrelated* sources that each happened to contain one opposite-sentiment
keyword (e.g. "increases" describing solar panel efficiency, "decreases"
describing recycling costs) would falsely trigger a pause. The threshold
requires the two sources to also share enough vocabulary to plausibly be
about the same thing before treating opposite sentiment as a genuine
disagreement.

### `critic` (9003)

Single-shot, non-streaming: one `working` status, one final `DataPart`
artifact with a per-claim verdict list, then `completed`/`failed`. Its
A2A contract has never changed since Step 3 — same Agent Card, same
`verify-claims` skill, same port, same `{draft, sources}` in →
`{verdicts: [{claim, supported, note}]}` out shape — but its *internal*
verification logic is swappable via `CRITIC_MODE`
(`agents/critic/verifier.py`):

- `naive` (`agents/critic/matching.py`) — the original keyword-overlap
  heuristic against a named `SUPPORT_THRESHOLD` constant. No LLM call, no
  key required.
- `langgraph` (`agents/critic/langgraph_verifier.py`) — a small
  **LangGraph** graph, backed by the Gemini API (same `GEMINI_API_KEY` as
  `synthesis`). For each claim: a `retrieve_candidate_source` node picks
  the most likely source via the same cheap keyword-overlap ranking as
  the naive checker (no LLM call needed there), then a `judge_claim` node
  makes one Gemini call asking strictly whether that specific source
  supports the claim, with a one-sentence justification — a response
  that can't be parsed into a clear yes/no falls back to
  `supported: false` with an "inconclusive" note rather than crashing,
  and a network/API failure (timeout, rate limit, empty response) fails
  the whole task, the same pattern as `synthesis`'s Gemini error
  handling.

Defaults to `langgraph` if `GEMINI_API_KEY` is set and `CRITIC_MODE` is
left unset, otherwise `naive` — same zero-key-required principle as
`LIT_SEARCH_PROVIDER`/`SYNTHESIS_PROVIDER`. This is the repo's
interoperability proof: `critic` is built on a completely different
agent framework (LangGraph) than the rest of the committee (direct API
calls / naive logic), and no other agent needed to change at all for
that to work — they only ever talked to `critic` through its Agent Card
and DataPart contract.

Against the naive templater's near-verbatim output, the naive checker's
overlap heuristic is close to trivial (an honestly-echoed sentence will
essentially always clear `SUPPORT_THRESHOLD`). Against real Gemini
output it does meaningful work purely as a side effect of the LLM
paraphrasing in its own words — but that's `critic` catching *vocabulary
drift*, not verifying the claim is actually true. The `langgraph`
verifier is the real thing: it reads the actual source text and judges
whether it substantively supports the claim, so it can (and does) flag a
paraphrased-but-accurate claim as supported where the naive checker
would miss it on vocabulary alone, and flag a fabricated claim as
unsupported even when it shares no obviously-absent keywords with any
source.

### `chair` (9000)

The only agent a human talks to. On startup it fetches and validates all
three downstream Agent Cards (reachable + advertising the expected skill)
*before* accepting any request — if one is missing or misconfigured, it
fails loudly at startup with a clear log line, rather than discovering
the problem mid-conversation.

It also handles the hardest case in this repo: **two-hop `input-required`
propagation**. When `synthesis` pauses, `chair` doesn't guess or auto-fill
an answer — it pauses its own task too, forwards `synthesis`'s exact
question up to the human unmodified, and waits. When the human replies,
`chair` resumes and forwards that reply to the correct `synthesis`
sub-task (recovered from `chair`'s own `Task.metadata`, stashed there for
exactly this purpose) to continue where it left off.

## Architecture

```
                         human
                           │
                           │ message/stream ("research-question")
                           ▼
                   ┌───────────────┐
                   │     chair     │  :9000
                   │ (orchestrator)│
                   └───┬───┬───┬───┘
           ┌───────────┘   │   └───────────┐
           ▼                ▼               ▼
   ┌───────────────┐ ┌───────────────┐ ┌───────────────┐
   │   lit_search   │ │   synthesis   │ │    critic     │
   │     :9001      │ │     :9002     │ │     :9003     │
   │ find-sources   │ │ draft-answer  │ │ verify-claims │
   └────────────────┘ └───────────────┘ └───────────────┘
```

Request flow for one research question:

1. `chair` receives the question from the human and generates one
   `contextId` for the whole exchange.
2. `chair` calls `lit_search` (streamed), relaying each found source
   upward as its own status updates.
3. `chair` feeds the collected sources into `synthesis` (streamed),
   relaying each drafted chunk upward the same way.
4. `chair` feeds the finished draft + sources into `critic` (single call).
5. `chair` returns one final artifact: the answer plus `critic`'s verdicts.

Every one of those four calls is a **separate A2A Task** — a
`lit_search` task, a `synthesis` task, a `critic` task, and `chair`'s own
task to the human — but they all carry the same `contextId`, so the whole
exchange is traceable as one logical conversation even though there's no
single shared Task spanning it.

## Why `input-required` matters

Most tutorial-level A2A implementations skip `input-required` entirely —
it's easy to build a one-shot "send request, stream response" flow and
call it done. But a protocol that can't pause mid-task, ask a clarifying
question, and resume from a caller's reply is just a fancy RPC call, not
a collaboration protocol. `synthesis` and `chair` are the two places in
this repo where that round trip is implemented for real, including the
harder version — a pause that has to propagate up through an
intermediary (`chair`) rather than terminating at the immediate caller.

## Running it locally

```bash
git clone <this-repo>
cd research-committee-A2A
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Configuration

Copy `.env.example` to `.env` and fill it in:

```bash
cp .env.example .env
```

```
TAVILY_API_KEY=
LIT_SEARCH_PROVIDER=tavily

GEMINI_API_KEY=
SYNTHESIS_PROVIDER=gemini
# GEMINI_MODEL=gemini-3.6-flash

CRITIC_MODE=langgraph
```

- `LIT_SEARCH_PROVIDER` — `fake` (default if unset) or `tavily`. `fake`
  needs no key and is safe for repeated `debug_client.py` runs.
- `TAVILY_API_KEY` — only required when `LIT_SEARCH_PROVIDER=tavily`. If
  it's missing in that case, `lit_search` fails loudly at startup instead
  of on the first search.
- `SYNTHESIS_PROVIDER` — `fake` (default if unset) or `gemini`. `fake`
  needs no key and has no response latency.
- `GEMINI_API_KEY` — only required when `SYNTHESIS_PROVIDER=gemini`, same
  fail-loud-at-startup behavior if missing. `GEMINI_MODEL` is optional
  (defaults to `gemini-3.6-flash`) — override it if that model is ever
  retired; see the `synthesis` section above for the model's latency
  characteristics.
- `CRITIC_MODE` — `naive` or `langgraph`. If left unset, defaults to
  `langgraph` when `GEMINI_API_KEY` is set, otherwise `naive` — same
  zero-key-required principle as the other two toggles above.
  `langgraph` fails loudly at startup if `GEMINI_API_KEY` is missing,
  same as `SYNTHESIS_PROVIDER=gemini`.
- `.env` is loaded automatically (`python-dotenv`) by `lit_search`,
  `synthesis`, `critic`, and `chair`; you can also just export the vars
  in your shell instead of using a `.env` file.

### Start the agents

```bash
python -m agents.lit_search &
python -m agents.synthesis &
python -m agents.critic &
python -m chair &
```

`chair` validates the other three Agent Cards at startup, so they need to
be up first.

### Drive it with `debug_client.py`

Fetch an Agent Card directly:

```bash
python -m common.debug_client --url http://localhost:9001 --text "check the card"
```

Drive a full request through `chair`:

```bash
python -m common.debug_client --url http://localhost:9000 \
  --text "What does the evidence say about sleep and memory consolidation?"
```

By default, `[STATUS]`/`[ARTIFACT]` output is truncated for readability.
Add `--full` to print complete, untruncated field values instead —
useful for actually inspecting `chair`'s final answer text or `critic`'s
full verdict list rather than skimming a trace:

```bash
python -m common.debug_client --url http://localhost:9000 \
  --text "A2A protocol adversarial robustness" --full
```

`debug_client.py` also supports DataPart payloads (`--data`/`--data-file`,
for hitting `synthesis`/`critic` directly) and replying to a paused task
(`--task-id`/`--context-id`) — see its own `--help` and docstring.

## Current limitations / known TODOs

Stated plainly, since this is meant to be read, not buried:

- **`synthesis`'s default mode is a naive templater, not an LLM.** With no
  `GEMINI_API_KEY` set (or `SYNTHESIS_PROVIDER=fake` explicitly),
  `FakeSynthesisProvider` concatenates source snippets into a fixed
  sentence template — not real summarization or paraphrasing. This is
  deliberate (zero-setup, no key needed, no latency) rather than an
  oversight; see `agents/synthesis/drafting.py`. With
  `SYNTHESIS_PROVIDER=gemini` it's a real LLM call instead — see the
  `synthesis` section above.
- **`critic`'s default mode depends on whether a Gemini key is present.**
  With no `GEMINI_API_KEY` set (or `CRITIC_MODE=naive` explicitly), it
  falls back to the original keyword-overlap heuristic, which has no real
  grounding verification of its own — it trusts that whatever `synthesis`
  sends it is the actual draft, and scores each claim against the
  *best*-matching source by shared vocabulary, nothing more. Against the
  fake templater's near-verbatim output that overlap check is close to
  trivial; against real Gemini output it does meaningful work purely as a
  side effect of the LLM paraphrasing in its own words, but that's
  catching *vocabulary drift*, not verifying the claim is actually true.
  See `agents/critic/matching.py`. With `CRITIC_MODE=langgraph` (the
  default when a key is present), a Gemini call actually reads the
  candidate source and judges substantive support — see the `critic`
  section above.
- **`SUPPORT_THRESHOLD` and `TOPIC_RELATEDNESS_THRESHOLD` are empirically
  tuned on a small sample**, not rigorously validated — reasonable
  starting points, not numbers to trust blindly.
- **No auth or Agent Card signature verification** — flagged as a `v1`
  TODO in every agent's executor. Fine for localhost; not for anything
  beyond it. Explicitly out of scope for `v1` (see `SPEC.md`).
- **Localhost-only, no persistent storage, no multi-tenancy, no
  gRPC/REST bindings** — also explicitly out of scope for `v1`.

## Build history

`SPEC.md` is the original design doc this was built from, step by step
(fake `lit_search` → debug client → `synthesis`/`critic` → `chair` →
real search → real `synthesis` LLM → `critic` rebuilt on LangGraph).
Worth reading if you want the full build rationale, not just the result.
