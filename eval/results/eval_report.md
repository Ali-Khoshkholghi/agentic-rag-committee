# lit_search evaluation (SPEC Step 11)

- generated: 2026-09-09T13:21:34+00:00
- git: `0ce9fbe`
- gemini model: `gemini-flash-lite-latest`
- retriever top-k: 5 | pipeline max-retries: 2 | gemini rate cap: 0.0/min | merged


## 1. Retriever alone (Step 9) — non-LLM context metrics

Over 9 `local` queries, gold = actual chunk texts resolved from each query's key phrases.

| metric | mean |
|---|---|
| context precision (vs gold chunks) | 0.621 |
| context recall (vs gold chunks) | 0.363 |
| corpus precision (retrieved chunks from the right paper) | 0.956 |

<details><summary>per query</summary>

| id | precision | recall | corpus_prec | gold chunks | retrieved papers |
|---|---|---|---|---|---|
| L1-dpr-negatives | 1.000 | 0.500 | 1.000 | 6 | 2004.04906v3.pdf |
| L2-dpr-encoder | 0.583 | 0.400 | 1.000 | 5 | 2004.04906v3.pdf |
| L3-selfrag-reflection-tokens | 0.000 | 0.000 | 1.000 | 3 | 2310.11511v1.pdf |
| L4-selfrag-retrieve-on-demand | 0.000 | 0.000 | 0.600 | 3 | 2310.11511v1.pdf, 2401.15884v3.pdf |
| L5-crag-evaluator-labels | 0.804 | 0.308 | 1.000 | 13 | 2401.15884v3.pdf |
| L6-crag-decompose-recompose | 1.000 | 1.000 | 1.000 | 1 | 2401.15884v3.pdf |
| L7-realm-pretraining | 0.333 | 0.250 | 1.000 | 4 | 2002.08909v1.pdf |
| L8-rag-sequence-vs-token | 1.000 | 0.385 | 1.000 | 13 | 2005.11401v4.pdf |
| L9-rag-nonparametric-memory | 0.867 | 0.429 | 1.000 | 7 | 2005.11401v4.pdf |

</details>


## 2. Full pipeline (Step 10) — faithfulness + context relevance

`faithfulness`: is the generated answer grounded in the evidence lit_search returned. `context_relevance`: are those evidence items actually on point for the question (RAGAS LLM context precision, no reference).

| category | n | faithfulness | context_relevance | mean retries | judge-sufficient rate |
|---|---|---|---|---|---|
| local | 9 | 1.000 | 0.783 | 0.667 | 0.889 |
| web | 3 | 0.778 | 0.922 | 1.333 | 0.333 |
| vague | 3 | 1.000 | 0.464 | 2.000 | 0.667 |
| ALL | 15 | 0.956 | 0.747 | 1.067 | 0.733 |

<details><summary>per query</summary>

| id | cat | routes | retries | suff | faithfulness | context_relevance | degraded |
|---|---|---|---|---|---|---|---|
| L1-dpr-negatives | local | vector_store->vector_store | 1 | True | 1.000 | 0.700 |  |
| L2-dpr-encoder | local | vector_store->vector_store | 1 | True | 1.000 | 0.854 |  |
| L3-selfrag-reflection-tokens | local | vector_store->vector_store->vector_store | 2 | False | 1.000 | 0.810 |  |
| L4-selfrag-retrieve-on-demand | local | vector_store->vector_store | 1 | True | 1.000 | 0.700 |  |
| L5-crag-evaluator-labels | local | vector_store | 0 | True | 1.000 | 0.804 |  |
| L6-crag-decompose-recompose | local | vector_store | 0 | True | 1.000 | 0.806 |  |
| L7-realm-pretraining | local | vector_store | 0 | True | 1.000 | 0.750 |  |
| L8-rag-sequence-vs-token | local | vector_store | 0 | True | 1.000 | 1.000 |  |
| L9-rag-nonparametric-memory | local | vector_store->vector_store | 1 | True | 1.000 | 0.625 |  |
| W1-post-crag-methods | web | web->web->web | 2 | False | 0.333 | 0.911 |  |
| W2-production-vector-dbs | web | web | 0 | True | 1.000 | 1.000 |  |
| W3-mteb-sota-embedding | web | web->web->web | 2 | False | 1.000 | 0.854 |  |
| V1-make-it-better | vague | web->web->web | 2 | True | 1.000 | 0.692 |  |
| V2-improve-the-retriever-part | vague | vector_store->both->both | 2 | True | 1.000 | 0.200 |  |
| V3-main-problems | vague | vector_store->vector_store->vector_store | 2 | False | 1.000 | 0.500 |  |

</details>


### Reading these numbers

- **Faithfulness does not separate the query types.** The answer-generation step summarises whatever evidence came back, so it stays ~1.0 even when that evidence is off-topic; the only dip is a web query that over-reached on thin sources. It is a grounding check, not a relevance check.
- **`context_relevance` + retry behaviour is what flags a weak run.** A vague sub-query drives the judge to spend its whole retry budget, never reach `sufficient`, and still end on sources the LLM judge scores as mostly irrelevant. That is the signal to tighten the reformulation prompt, not faithfulness.
- **Retriever recall is strict by construction.** Gold is the specific answer-bearing chunk(s); a run can route to the right paper (`corpus_precision` ~1.0) yet still miss the exact passage (`context_recall` low). Both columns matter — the first for routing, the second for chunking / rerank-cutoff tuning.

