# `lit_search/rag` — agentic RAG foundation (SPEC Step 9)

Standalone retrieval stack for `lit_search`. **Not yet wired into the A2A
flow** — `agents/lit_search/agent_executor.py` is untouched and still
Tavily-only. Step 10 wires this in.

## Modules

| File | What it does |
|---|---|
| `ingest.py` | Reads `../data/corpus/`, does structure-aware recursive chunking (~500 tokens, ~60 overlap), embeds each chunk with `sentence-transformers` `all-MiniLM-L6-v2`, and persists chunks + embeddings + source metadata to a local Chroma DB at `../data/chroma_db/`. |
| `retriever.py` | `HybridRetriever`: dense cosine search (Chroma) + BM25 (`rank_bm25`) over the *same* chunks, merged with Reciprocal Rank Fusion, then reranked with the `cross-encoder/ms-marco-MiniLM-L-6-v2` cross-encoder down to top-k. |
| `demo.py` | Standalone acceptance check: ingest, then retrieve for sample queries, printing ranked/reranked chunks with source metadata and per-stage scores. |

## Usage

```bash
# (re)build the vector store
python -m agents.lit_search.rag.ingest

# ingest + sample retrievals
python -m agents.lit_search.rag.demo

# one query against the existing DB
python -m agents.lit_search.rag.demo --skip-ingest --query "what is RRF?" --top-k 3
```

First run downloads the two MiniLM models (~120 MB total) from the
Hugging Face hub.

## Corpus

`../data/corpus/` holds a few short placeholder reference docs relevant to
this committee's typical questions (the A2A protocol, agentic RAG, hybrid
retrieval / RRF, cross-encoder reranking, sleep & memory consolidation).
Swap in real papers or docs — any `.md` / `.markdown` / `.txt` file in
that directory is picked up on the next ingest.

`../data/chroma_db/` is generated and git-ignored.
