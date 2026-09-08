"""Agentic RAG foundation for lit_search (SPEC Step 9).

Standalone, self-contained retrieval stack — ingestion into a local Chroma
DB plus hybrid (dense + BM25, RRF-merged, cross-encoder reranked)
retrieval over the same corpus. Nothing here is imported by
``agents/lit_search/agent_executor.py`` yet; wiring it into the live A2A
flow is Step 10. The existing Tavily-only behaviour is untouched.
"""
