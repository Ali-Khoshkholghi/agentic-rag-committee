"""SPEC Step 11 — RAGAS evaluation of lit_search's retrieval and output.

- ``eval_set.py`` — the hand-built eval set: local / web / vague sub-queries
  with known-good corpus sources for the local ones.
- ``run_eval.py`` — scores HybridRetriever (Step 9) alone for context
  precision/recall, and the full agentic pipeline (Step 10) for
  faithfulness + context relevance, and writes a report to ``results/``.
"""
