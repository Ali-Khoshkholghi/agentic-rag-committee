# lit_search evaluation (SPEC Step 11)

- generated: 2026-09-10T10:27:28+00:00
- git: `c39d21d`
- gemini model: `gemini-flash-lite-latest`
- retriever top-k: 5 | pipeline max-retries: 2 | gemini rate cap: 0.0/min | merged


## 1. Retriever alone (Step 9) — non-LLM context metrics

Over 9 `local` queries, gold = actual chunk texts resolved from each query's key phrases.

| metric | mean |
|---|---|
| context precision (vs gold chunks) | 0.806 |
| context recall (vs gold chunks) | 0.428 |
| corpus precision (retrieved chunks from the right paper) | 0.956 |

<details><summary>per query</summary>

| id | precision | recall | corpus_prec | gold chunks | retrieved papers |
|---|---|---|---|---|---|
| L1-dpr-negatives | 1.000 | 0.429 | 1.000 | 7 | 2004.04906v3.pdf |
| L2-dpr-encoder | 0.333 | 0.200 | 1.000 | 5 | 2004.04906v3.pdf |
| L3-selfrag-reflection-tokens | 0.333 | 0.333 | 1.000 | 3 | 2310.11511v1.pdf |
| L4-selfrag-retrieve-on-demand | 1.000 | 0.500 | 0.600 | 4 | 2310.11511v1.pdf, 2401.15884v3.pdf |
| L5-crag-evaluator-labels | 0.887 | 0.286 | 1.000 | 14 | 2401.15884v3.pdf |
| L6-crag-decompose-recompose | 0.700 | 1.000 | 1.000 | 2 | 2401.15884v3.pdf |
| L7-realm-pretraining | 1.000 | 0.250 | 1.000 | 4 | 2002.08909v1.pdf |
| L8-rag-sequence-vs-token | 1.000 | 0.357 | 1.000 | 14 | 2005.11401v4.pdf |
| L9-rag-nonparametric-memory | 1.000 | 0.500 | 1.000 | 10 | 2005.11401v4.pdf |

</details>


## 2. Full pipeline (Step 10) — faithfulness + context relevance

`faithfulness`: is the generated answer grounded in the evidence lit_search returned. `context_relevance`: are those evidence items actually on point for the question (RAGAS LLM context precision, no reference).

| category | n | faithfulness | context_relevance | mean retries | judge-sufficient rate |
|---|---|---|---|---|---|
| local | 9 | 0.978 | 0.727 | 0.667 | 1.000 |
| web | 3 | 0.778 | 0.472 | 0.333 | 0.667 |
| vague | 3 | 1.000 | 0.667 | 1.333 | 0.000 |
| ALL | 15 | 0.942 | 0.664 | 0.733 | 0.733 |

<details><summary>per query</summary>

| id | cat | routes | retries | suff | faithfulness | context_relevance | degraded |
|---|---|---|---|---|---|---|---|
| L1-dpr-negatives | local | vector_store->vector_store | 1 | True | 1.000 | 0.833 |  |
| L2-dpr-encoder | local | vector_store->vector_store | 1 | True | 1.000 | 0.833 |  |
| L3-selfrag-reflection-tokens | local | vector_store->vector_store | 1 | True | 1.000 | 0.893 |  |
| L4-selfrag-retrieve-on-demand | local | vector_store | 0 | True | 1.000 | 1.000 |  |
| L5-crag-evaluator-labels | local | vector_store->vector_store | 1 | True | 0.800 | 1.000 |  |
| L6-crag-decompose-recompose | local | vector_store | 0 | True | 1.000 | 1.000 |  |
| L7-realm-pretraining | local | vector_store | 0 | True | 1.000 | 0.000 |  |
| L8-rag-sequence-vs-token | local | vector_store->vector_store | 1 | True | 1.000 | 0.417 |  |
| L9-rag-nonparametric-memory | local | vector_store->both | 1 | True | 1.000 | 0.566 |  |
| W1-post-crag-methods | web | web | 1 | False | 0.333 | 0.583 |  |
| W2-production-vector-dbs | web | web | 0 | True | 1.000 | 0.833 |  |
| W3-mteb-sota-embedding | web | web | 0 | True | 1.000 | 0.000 |  |
| V1-make-it-better | vague | web | 1 | False | 1.000 | 1.000 |  |
| V2-improve-the-retriever-part | vague | vector_store->vector_store | 2 | False | 1.000 | 0.417 |  |
| V3-main-problems | vague | vector_store | 1 | False | 1.000 | 0.583 |  |

</details>


### Reading these numbers

- **Faithfulness does not separate the query types.** The answer-generation step summarises whatever evidence came back, so it stays ~1.0 even when that evidence is off-topic; the only dip is a web query that over-reached on thin sources. It is a grounding check, not a relevance check.
- **`context_relevance` + retry behaviour is what flags a weak run.** A vague sub-query that the reformulate step can't ground in the committee's domain (no salient noun, a pronoun with no antecedent) now exits after a single retry as `unreformulable` instead of drifting to a plausible-sounding but unrelated query and burning both retries on it — lower `judge_sufficient` is expected there and is the honest outcome, not a regression; watch `context_relevance` and `retry_count_mean` together to catch a reformulation prompt that's drifting off-topic again.
- **Retriever recall/precision move with candidate-pool width and chunk overlap.** Widening the pre-rerank candidate pool lets the cross-encoder see chunks the fused dense/sparse pool would otherwise cut before reranking; widening chunk overlap helps when a short, dense technical span (a definition table, a stale-index explanation) gets split across a chunk boundary and neither half scores well alone. If `local` `context_recall` or `judge_sufficient` regresses after a retriever change, pull the gold vs. retrieved chunks for the worst queries before assuming it's a rerank-model problem.
- **A `context_relevance` of 0.000 is not automatically a scoring bug.** `LLMContextPrecisionWithoutReference` verdicts each retrieved context in isolation against the *synthesized* answer; a faithful answer stitched from complementary chunks or sources (high `faithfulness`) can still score 0.0 if no single context alone covers enough of it to pass that per-context bar. Before treating a 0.000 as a pipeline defect, call the RAGAS judge prompt directly per context and read the raw verdict + reason — an exception/parse failure shows up as `None` (`n/a`) via this script's own try/except, never a silent `0.0`, so a real `0.000` in this table is a genuine all-contexts-insufficient verdict.
- **Retriever recall is strict by construction.** Gold is the specific answer-bearing chunk(s); a run can route to the right paper (`corpus_precision` ~1.0) yet still miss the exact passage (`context_recall` low). Both columns matter — the first for routing, the second for chunking / rerank-cutoff tuning.

