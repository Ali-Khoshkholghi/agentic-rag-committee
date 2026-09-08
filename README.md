# agentic-rag-committee

## What this is

The **agentic-RAG extension** of
[`research-committee-A2A`](https://github.com/Ali-Khoshkholghi/research-committee-A2A) —
the same four-agent "committee of agents" over the **A2A (Agent2Agent)
protocol**, with `lit_search` extended to retrieve from a **local corpus**
via hybrid retrieval (dense + BM25 + cross-encoder rerank) alongside its
existing web search.

It is still a portfolio/learning repo, not a production service. The base
project's point was a correct, minimal demonstration of Agent Cards, the
A2A task lifecycle, JSON-RPC+SSE streaming, and multi-hop `input-required`
pause/resume. This fork's point is layering a real retrieval-augmented
pipeline into one agent **without changing its A2A contract or any other
agent**.

**The base project is unchanged here.** `chair`, `synthesis`, `critic`,
and `lit_search`'s original Tavily-only search behaviour are byte-for-byte
what they are upstream — same Agent Cards, same skills, same ports, same
streamed-chunk semantics, same `.env` toggles. Only `lit_search` gained
new *internals* (`agents/lit_search/rag/`), and as of Step 9 they are not
yet wired into the live request path — see **Current status** below.

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
| `lit_search` | 9001 | `find-sources` | yes — one source at a time | Finds candidate sources for a topic. Web search today; a local hybrid-retrieval corpus is being layered in (Steps 9–11) |
| `synthesis` | 9002 | `draft-answer` | yes — text chunks | Drafts an answer from those sources — agentic, Google ADK-backed by default |
| `critic` | 9003 | `verify-claims` | no — single final artifact | Checks the draft's claims against the sources — agentic, LangGraph-backed by default |
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

**New in this fork:** a standalone local-retrieval stack under
`agents/lit_search/rag/` (see the next section). It is fully independent
of the A2A executor for now — `agent_executor.py` still calls only the
`SearchProvider` — so `lit_search`'s external behaviour is exactly the
upstream project's until Step 10 wires the two together.

### Agentic RAG in `lit_search` (`agents/lit_search/rag/`)

A standalone local-retrieval stack: a corpus of documents in
`agents/lit_search/data/corpus/` turned into an index that hybrid-searches
better than either dense or keyword search alone. It reads no environment
variables and needs no API key — the embedding and rerank models run
locally on CPU (downloaded once from the Hugging Face hub on first use).
Three small modules:

- **`ingest.py`** — reads every `.pdf`, `.md` / `.markdown`, or `.txt`
  file in the corpus directory (PDF text extracted with **`pypdf`**), and
  splits each document with a structure-aware **recursive splitter** —
  heading → paragraph → line → sentence → word — into **~500-token
  windows with ~50-token verbatim overlap**. Each chunk is embedded with
  `sentence-transformers` **`all-MiniLM-L6-v2`** (384-dim; its native
  256-token input limit raised to 512 so a full chunk embeds without
  truncation), and chunks + embeddings + per-chunk source metadata
  (source filename, document title, chunk index) are persisted to a local
  **Chroma** DB at `agents/lit_search/data/chroma_db/` (generated,
  git-ignored). Each run rebuilds the collection from scratch.

- **`retriever.py`** — `HybridRetriever` runs two retrievers over the
  *same* stored chunks: **dense** cosine search against Chroma, and
  **sparse** BM25 (**`rank_bm25`**) built in memory from those exact chunk
  texts. The two rank lists are merged with **Reciprocal Rank Fusion**
  (`k = 60`), and the fused candidate pool (default 20) is **reranked with
  a cross-encoder** (`cross-encoder/ms-marco-MiniLM-L-6-v2`) down to
  top-k (default 5). Each result carries its text, source/title/chunk
  index, and every intermediate signal — dense rank, sparse rank, RRF
  score, rerank score.

- **`demo.py`** — a standalone acceptance script: ingest the corpus, then
  run a set of sample queries through `HybridRetriever` and print the
  ranked/reranked chunks with their source metadata and scores. Not
  imported anywhere; it only exercises the two modules above.

The corpus today is five arXiv papers on retrieval and
retrieval-augmented generation — REALM, DPR, RAG, Self-RAG, and CRAG. Drop
more files into the corpus directory and re-run `ingest.py` to refresh
(see [Build the vector store](#build-the-vector-store) below).

### `synthesis` (9002)

Streams a drafted answer as incremental `TextPart` chunks (`append=true`,
final chunk `lastChunk=true`) rather than returning the whole draft at
once. Its A2A contract has never changed since Step 3 — same Agent Card,
same `draft-answer` skill, same port, same streamed-chunk semantics — but
its *internal* drafting logic is swappable via `SYNTHESIS_PROVIDER`
(`agents/synthesis/llm_provider.py`):

- `fake` (`agents/synthesis/drafting.py`) — a deterministic templater, no
  LLM call, no key required.
- `gemini` (`agents/synthesis/llm_provider.py`'s `GeminiSynthesisProvider`,
  Step 6) — one raw call to the Gemini `streamGenerateContent` REST
  endpoint, hand-rolled SSE parsing.
- `adk` (`agents/synthesis/adk_provider.py`, Step 8) — a single **Google
  ADK** `Agent` (no tools, no sub-agents, one well-structured agentic
  call), also backed by Gemini, using ADK's own streaming `Runner` to
  keep emitting chunks incrementally rather than blocking for the whole
  draft. Same grounding rules as the `gemini` provider's prompt (answer
  only from the given sources, paraphrase in its own words, no verbatim
  quoting, no outside knowledge), just split into an ADK `instruction`
  (the fixed rules) and a per-request user message (topic + sources +
  guidance) instead of one folded-together prompt string.

The executor just streams whatever chunks the selected provider
yields — it doesn't know or care which one is behind it, and a live
failure from any of the two real providers (API error, timeout, empty
response) raises the same `SynthesisProviderError`, mapped to
`TaskState.failed` the same way regardless of which provider raised it.

Together with `critic`'s LangGraph rebuild (Step 7), this makes
`synthesis` and `critic` the committee's two genuinely agentic
nodes — each built on a *different* agent framework (Google ADK vs.
LangGraph), both Gemini-backed, each interoperating over its own
unchanged A2A contract. `lit_search` and `chair` remain simpler
non-agentic A2A services by design (a search-provider adapter and an
orchestrator, respectively) — this project's actual proof-point is that
the framework behind an A2A agent is fully opaque to every other agent
in the system, no matter which two frameworks happen to be in play.

The Gemini model currently defaulted to (`gemini-3.6-flash`) does
internal "thinking" before it emits any visible text, which adds real,
noticeable latency: expect roughly 20–35 seconds before the *first*
streamed chunk arrives, even for a short prompt (both the `gemini` and
`adk` providers see this). That's normal, not a hang —
`debug_client.py`'s elapsed-time prefix will show it plainly.
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
`LIT_SEARCH_PROVIDER`/`SYNTHESIS_PROVIDER`. Together with `synthesis`'s
Google ADK rebuild (Step 8, see above), this is the repo's
interoperability proof: `critic` and `synthesis` are each built on a
completely different agent framework from one another, and no other
agent needed to change at all for either rebuild — every agent only ever
talks to another through its Agent Card and DataPart/TextPart contract.

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
   │  lit_search   │ │   synthesis   │ │    critic     │
   │     :9001     │ │     :9002     │ │     :9003     │
   │ find-sources  │ │ draft-answer  │ │ verify-claims │
   └───────┬───────┘ └───────────────┘ └───────────────┘
           :  (Step 10+; not wired into the A2A path yet)
           v
   agents/lit_search/rag/
     corpus (PDF/MD/TXT)  --ingest.py-->  Chroma DB
     query  -->  dense search  \
                                 >-- RRF --> cross-encoder rerank --> top-k
                 BM25 search   /
     (retriever.py)
```

`chair` orchestrates `synthesis`, `critic`, and `lit_search` over A2A
exactly as in the base project. The `rag/` box hangs off `lit_search`
but is reachable only through `demo.py` today; Step 10 connects it to the
executor.

Request flow for one research question (unchanged from the base project):

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
cd agenticRAG-research-committee-A2A
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` carries the base project's dependencies plus, for the
`agents/lit_search/rag/` stack:

- `chromadb` — the local persistent vector store.
- `sentence-transformers` — the `all-MiniLM-L6-v2` embedding model and the
  `ms-marco-MiniLM-L-6-v2` cross-encoder reranker (pulls in `torch`).
- `rank-bm25` — the sparse BM25 retriever.
- `pypdf` — PDF text extraction for the corpus.

These are only imported by `agents/lit_search/rag/`; running the
committee itself does not load them.

### Build the vector store

```bash
python -m agents.lit_search.rag.ingest
```

Reads every document in `agents/lit_search/data/corpus/` and writes the
Chroma DB to `agents/lit_search/data/chroma_db/` (git-ignored — rebuild it
from the corpus rather than committing it). Add or remove documents in the
corpus directory and re-run to refresh; each run rebuilds the collection
from scratch. First run also downloads the two MiniLM models (~175 MB).

Then exercise retrieval on its own:

```bash
python -m agents.lit_search.rag.demo                 # ingest + sample queries
python -m agents.lit_search.rag.demo --skip-ingest --query "your question" --top-k 3
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
SYNTHESIS_PROVIDER=adk
# GEMINI_MODEL=gemini-3.6-flash

CRITIC_MODE=langgraph
```

- `LIT_SEARCH_PROVIDER` — `fake` (default if unset) or `tavily`. `fake`
  needs no key and is safe for repeated `debug_client.py` runs.
- `TAVILY_API_KEY` — only required when `LIT_SEARCH_PROVIDER=tavily`. If
  it's missing in that case, `lit_search` fails loudly at startup instead
  of on the first search.
- `SYNTHESIS_PROVIDER` — `fake` (default if unset), `gemini`, or `adk`.
  `fake` needs no key and has no response latency; `gemini` and `adk`
  both need `GEMINI_API_KEY` and fail loudly at startup if it's missing.
- `GEMINI_API_KEY` — required for `SYNTHESIS_PROVIDER=gemini`/`adk` and
  for `CRITIC_MODE=langgraph`, all three sharing the one key. `GEMINI_MODEL`
  is optional (defaults to `gemini-3.6-flash`, shared by all three) —
  override it if that model is ever retired, hits a rate limit, or you
  just want a cheaper/faster one for a given run; see the `synthesis`
  section above for the default model's latency characteristics.
- `CRITIC_MODE` — `naive` or `langgraph`. If left unset, defaults to
  `langgraph` when `GEMINI_API_KEY` is set, otherwise `naive` — same
  zero-key-required principle as `SYNTHESIS_PROVIDER`. `langgraph` fails
  loudly at startup if `GEMINI_API_KEY` is missing, same as
  `SYNTHESIS_PROVIDER=gemini`/`adk`.
- `.env` is loaded automatically (`python-dotenv`) by `lit_search`,
  `synthesis`, `critic`, and `chair`; you can also just export the vars
  in your shell instead of using a `.env` file. The `rag/` stack reads no
  environment variables at all.

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

## Current status

- **Step 9 — ingestion + hybrid retrieval: complete.** `ingest.py` and
  `retriever.py` exist and work standalone; `demo.py` is the acceptance
  check. `lit_search`'s A2A behaviour is untouched — the whole committee
  still runs end to end exactly as in the base project.
- **Step 10 — routing + self-correction: next.** A single-LLM `router`
  that decides per sub-query whether to hit the vector store, web search,
  or both; a two-node LangGraph `judge` / `reformulate_and_retrieve` loop
  that scores merged evidence for sufficiency and retries with a
  reformulated query when it's weak; and the wiring of both into
  `agents/lit_search/agent_executor.py` behind the same A2A contract.
- **Step 11 — RAGAS evaluation: after that.** Context precision/recall
  against the retriever alone, faithfulness against full `lit_search`
  output, results saved under `eval/`.

See [`SPEC.md`](SPEC.md) for the full Step 9–11 spec.

## Retrieval discrimination check

`demo.py`, run against the five-paper corpus above — chosen because they
overlap heavily in vocabulary: two describe an evaluate-then-act RAG loop
(Self-RAG, CRAG), two describe a dense retriever feeding a reader (REALM,
DPR). Ingest produces 198 chunks:

| File | Paper | Chunks |
|---|---|---|
| `2002.08909v1.pdf` | REALM: Retrieval-Augmented LM Pre-Training | 31 |
| `2004.04906v3.pdf` | DPR: Dense Passage Retrieval for Open-Domain QA | 32 |
| `2005.11401v4.pdf` | RAG for Knowledge-Intensive NLP Tasks | 42 |
| `2310.11511v1.pdf` | Self-RAG | 60 |
| `2401.15884v3.pdf` | CRAG: Corrective RAG | 33 |

Six test queries — five targeting a mechanism that belongs to exactly one
paper, one deliberately ambiguous:

| Query targets | Expected owner | Rank 1 | Top-5 by source |
|---|---|---|---|
| "Correct / Incorrect / Ambiguous" confidence labels | CRAG | CRAG | CRAG ×5 |
| reflection tokens `ISREL` `ISSUP` `ISUSE` | Self-RAG | Self-RAG | Self-RAG ×5 |
| decompose-then-recompose + web-search extension | CRAG | CRAG | CRAG ×5 |
| dual-encoder, in-batch negatives, BM25 hard negatives | DPR | DPR | DPR ×5 |
| masked-LM-pretrained latent knowledge retriever | REALM | REALM | REALM ×5 |
| "retrieved passages to improve open-domain QA" (ambiguous) | none in particular | DPR | DPR ×4, RAG ×1 |

Clean top-5 separation on all five targeted queries, and a sensible split
on the ambiguous one (DPR and RAG both legitimately own it; no false
confidence forcing a single winner).

The hybrid pipeline is doing real work, not just running — visible in the
per-result ranks. For the Self-RAG token query:

- a chunk at **dense rank 12**, BM25 rank 1 (BM25 caught the literal token
  `ISREL` that the embedding buried) is reranked to result #3;
- a chunk **absent from the dense top-20 entirely**, BM25 rank 4, is
  reranked to result #5.

For the CRAG decompose-then-recompose query, a chunk at **dense rank 18**,
BM25 rank 1 is fused and reranked all the way to result #1. Without the
sparse arm or the reranker those chunks don't surface.

This is spot-checking on six hand-written queries, not a measured
evaluation — quantifying it with RAGAS is Step 11.

## Current limitations / known TODOs

Stated plainly, since this is meant to be read, not buried:

### Agentic RAG layer (this fork)

- **Not wired into the live agent yet.** `agents/lit_search/rag/` is
  reachable only via `demo.py`. `lit_search`'s A2A responses still come
  purely from web search. Connecting the two is Step 10, and until then
  the retrieval quality above has no effect on what the committee
  actually answers.
- **Small, older local models.** `all-MiniLM-L6-v2` (embeddings) and
  `ms-marco-MiniLM-L-6-v2` (reranker) are chosen for "runs on a laptop
  CPU with no key", not for being state of the art. The embedding model
  is natively a 256-token model run at 512; chunks near the ~500-token
  target can still be truncated by the cross-encoder's 512-token limit at
  rerank time.
- **PDF extraction is `pypdf`-quality.** Two-column academic layouts,
  ligatures, and math come through imperfectly; the derived document
  `title` is a best-effort guess from the first page and often includes
  a few author-name tokens. The `source` filename is the reliable
  identifier. Swap in `pymupdf` in `ingest.py` if a given paper extracts
  badly.
- **The corpus is five papers.** Enough to demonstrate discrimination,
  not enough to say anything about retrieval quality at scale. BM25 is
  rebuilt in memory from the full collection on every retriever startup;
  there is no incremental ingest — adding a document means a full
  re-embed and rebuild.
- **RRF weights, `k = 60`, the candidate-pool size, and the rerank cutoff
  are defaults, not tuned** — Step 11's evaluation is what's supposed to
  tune them.

### Base project (unchanged, carried over from `research-committee-A2A`)

- **`synthesis`'s default mode (no key) is a naive templater, not an
  LLM.** With no `GEMINI_API_KEY` set (or `SYNTHESIS_PROVIDER=fake`
  explicitly), `FakeSynthesisProvider` concatenates source snippets into
  a fixed sentence template — not real summarization or paraphrasing.
  This is deliberate (zero-setup, no key needed, no latency) rather than
  an oversight; see `agents/synthesis/drafting.py`. With a key present,
  `SYNTHESIS_PROVIDER=gemini` (raw API call) or `adk` (Google ADK agent,
  the default once a key exists) both do real, genuinely paraphrased
  drafting — see the `synthesis` section above.
- **Gemini's free-tier daily request quota is per-model and easy to
  exhaust during active development.** It's tracked per API key per
  model (`generativelanguage.googleapis.com/generate_content_free_tier_requests`,
  20/day on the default `gemini-3.6-flash` as observed in this repo's
  sandbox), shared across every agent/provider using that model —
  `synthesis` (`gemini`/`adk`) and `critic` (`langgraph`) all draw from
  the same daily bucket. Hitting it surfaces as a `429 RESOURCE_EXHAUSTED`
  from Gemini, which both `synthesis` and `critic` correctly map to
  `TaskState.failed` rather than crashing — a real, useful exercise of
  the failure-handling path, not a bug. `GEMINI_MODEL` can point at a
  different model with its own separate quota bucket as a workaround.
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

This fork picks up from
[`research-committee-A2A`](https://github.com/Ali-Khoshkholghi/research-committee-A2A)
at the end of its Step 8. That project's `SPEC.md` is the original design
doc it was built from, step by step (fake `lit_search` → debug client →
`synthesis`/`critic` → `chair` → real search → real `synthesis` LLM →
`critic` rebuilt on LangGraph → `synthesis` rebuilt on Google ADK).

This repo's [`SPEC.md`](SPEC.md) carries Steps 9–11: the agentic-RAG
extension of `lit_search` — ingestion + hybrid retrieval (Step 9, done),
routing + a self-correction loop (Step 10), and RAGAS evaluation
(Step 11).
