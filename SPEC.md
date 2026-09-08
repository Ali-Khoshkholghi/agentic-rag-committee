# SPEC.md — Committee of Agents (A2A) with Agentic RAG

## Project note

This folder (`agenticRAG-research-committee-A2A`) is a working copy of
`research-committee-A2A` (Steps 1–8, complete, merged to main in the original
repo). It exists to build Steps 9–11 (agentic RAG) without touching the
original repo. Step 0 of this folder's work is bringing the existing
codebase in — see Step 9 setup below.

## Architecture recap (Steps 1–8, unchanged in this folder)

- **chair** — orchestrator. Talks to `lit_search`, `synthesis`, `critic` over
  A2A with SSE streaming; handles two-hop input-required task updates.
- **lit_search** — gathers evidence for the chair's research question. Currently:
  Tavily web search only (Step 5).
- **synthesis** — drafts an answer from lit_search's evidence. Google ADK +
  Gemini (Step 8; chosen over Anthropic Agent SDK — no Anthropic key).
- **critic** — reviews synthesis's draft for unsupported claims. LangGraph +
  Gemini (Step 7).
- **common/** — shared utilities, incl. `debug_client.py`.

Ports: chair + three agents, one port each, in the 9000s range (see
`common/` or each agent's `__main__` for exact assignment).

Steps 9–11 below extend **lit_search only**. Chair, synthesis, and critic's
code does not change — they keep calling lit_search over A2A and receiving
evidence, same as today.

---

## Step 9 — Agentic RAG foundation: ingestion + hybrid retrieval

**Goal:** lit_search can retrieve from a local corpus, standalone and
testable, without yet being wired into the main agent flow.

**Setup (do this first):**
- Copy the existing `research-committee-A2A` codebase into this folder
  (or clone `github.com/Ali-Khoshkholghi/research-committee-A2A` and check
  out `main`) so `chair/`, `lit_search/`, `synthesis/`, `critic/`, and
  `common/` all exist here before any new work starts.

**Tasks:**
- Add `lit_search/rag/` module:
  - `ingest.py` — chunking (recursive/structure-aware split, ~500 token
    chunks, small overlap) + embedding (`sentence-transformers`,
    `all-MiniLM-L6-v2` to start) → persist to a local Chroma DB at
    `lit_search/data/chroma_db/`.
  - `retriever.py` — hybrid retrieval: dense search against Chroma +
    sparse BM25 (`rank_bm25`) over the same corpus, merged via Reciprocal
    Rank Fusion, then reranked with a small cross-encoder
    (`sentence-transformers` cross-encoder) down to top-k.
- Add `lit_search/data/corpus/` with a handful of source documents (a few
  papers or docs relevant to the committee's typical research questions) as
  the initial local knowledge base.
- Update `requirements.txt` (or the lit_search agent's own deps) with
  `chromadb`, `sentence-transformers`, `rank_bm25`.
- Do **not** wire this into `lit_search/agent.py` yet — that's Step 10.
  Leave the existing Tavily-only flow untouched and working.

**Acceptance criteria:**
- A standalone script or test can run `ingest.py` against
  `lit_search/data/corpus/`, then call `retriever.py` with a sample query and
  get back a ranked, reranked list of chunks with source metadata.
- Existing lit_search A2A behavior (Tavily-only) is unaffected — the whole
  committee still runs end to end exactly as before.

---

## Step 10 — Routing + self-correction loop

**Goal:** lit_search decides per sub-query whether to use the vector store,
web search, or both, and self-corrects when retrieved evidence is weak.

**Tasks:**
- Add `lit_search/rag/router.py` — single LLM call (Gemini) given the
  sub-query and short descriptions of both sources, returns a structured
  choice: `vector_store`, `web`, or `both`.
- Add `lit_search/rag/judge.py` — a two-node LangGraph graph:
  - `judge` node: scores merged evidence against the sub-query for
    sufficiency; checks `retry_count` against a max.
  - `reformulate_and_retrieve` node: on insufficient evidence, builds a new
    query from the judge's reasoning, re-runs retrieval (router → sources →
    merge), increments `retry_count`, loops back to `judge`.
  - Conditional edge out of `judge`: sufficient → END; insufficient with
    retries left → `reformulate_and_retrieve`; insufficient with retries
    exhausted → END (return best-effort evidence).
- Wire this into `lit_search/agent.py`: incoming sub-query → router →
  retrieval (vector store / web / both, per Step 9's retriever and existing
  Tavily code) → merge → judge/reformulate graph → evidence returned to
  chair over A2A, same response shape as before.

**Acceptance criteria:**
- lit_search's A2A interface is unchanged from the chair's perspective —
  same request/response shape, just better-sourced evidence.
- A query that clearly needs the local corpus (e.g. about a paper you
  ingested) is answered from the vector store; a query needing current
  information routes to web search; a deliberately hard/ambiguous query
  triggers at least one reformulate-and-retry cycle, observable in logs or
  the debug client.

---

## Step 11 — Evaluation with RAGAS

**Goal:** quantify retrieval and generation quality instead of relying on
spot-checking.

**Tasks:**
- Build a small eval set: a handful of sub-queries with known-good source
  chunks/documents from the local corpus.
- Add an eval script using RAGAS to compute:
  - Context precision / recall against the retriever alone (Step 9's
    `retriever.py`), independent of any LLM step — tunes chunking, hybrid
    weights, and rerank cutoff.
  - Faithfulness against full lit_search output (post Step 10) — catches
    cases where the judge or synthesis drifts beyond what was retrieved.
- Save eval results (a short report or JSON) somewhere in the repo, e.g.
  `eval/results/`.

**Acceptance criteria:**
- Running the eval script produces numeric scores for the current
  retriever + judge configuration, re-runnable after any future tuning
  change to check for regressions.
