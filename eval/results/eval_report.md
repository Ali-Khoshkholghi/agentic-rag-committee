# lit_search evaluation (SPEC Step 11)

- generated: 2026-09-10T09:36:48+00:00
- git: `36b9c85`
- gemini model: `gemini-flash-lite-latest`
- retriever top-k: 5 | pipeline max-retries: 2 | gemini rate cap: 0.0/min


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

This run is against the retriever fix (`DEFAULT_CANDIDATE_POOL` 20→40 in `retriever.py`, `CHUNK_OVERLAP_TOKENS` 50→100 in `ingest.py`, 225 chunks) and the reformulate-domain-grounding fix (`judge.py`), both landed before this eval. Every number above is from the same single run against that corpus — no merged/stale rows.

- **Faithfulness does not separate the query types.** The answer-generation step summarises whatever evidence came back, so it stays ~1.0 even when that evidence is off-topic; the one dip this run is W1-post-crag-methods (0.333), a web query that over-reached on thin sources. It is a grounding check, not a relevance check.
- **The retriever fix moved both retriever-alone numbers and it shows up downstream.** Context precision rose 0.621→0.806 and context recall 0.363→0.428 (Section 1) after widening the pre-rerank candidate pool and the chunk overlap — the earlier gap traced to the recursive chunker splitting short technical spans (a reflection-token table, a stale-index explanation) across a chunk boundary, so neither half scored well against the cross-encoder. `local` pipeline `judge_sufficient` is now 1.000 (was 0.889) and local retries stayed low, consistent with the judge getting usable evidence on the first or second pass more often.
- **`context_relevance` + retry behaviour is what flags a weak reformulation run.** A vague sub-query that the reformulate step can't ground in the committee's domain (no salient noun, a pronoun with no antecedent) now exits after a single retry as `unreformulable` instead of drifting to a plausible-sounding but unrelated query and burning both retries on it — the resulting `vague` `judge_sufficient` of 0.000 is the honest outcome (the pipeline no longer claims success on an unresolved vague query), not a regression. Watch `context_relevance` and `retry_count_mean` together to catch a reformulation prompt drifting off-topic again.
- **A `context_relevance` of 0.000 (L7-realm-pretraining, W3-mteb-sota-embedding) is a real per-context verdict, not a scoring bug.** Verified directly by calling RAGAS's judge prompt per retrieved context and reading the raw verdict + reason (see `eval/results/`); for L7 (deterministic local retrieval, same route reproduced exactly) all 5 contexts were individually verdicted "not useful" with sound reasoning each time (one REALM chunk covers the MIPS-refresh mechanism but not the Inverse Cloze Task init, another covers the reverse). No exception, no parse failure, no silent coercion anywhere in the path — this script's own `except Exception` around each metric call would have recorded `None` (`n/a`), not `0.0`, and no such warning appears in the run log for either query. `LLMContextPrecisionWithoutReference` checks whether **each context alone** could produce the synthesized answer; when a faithful answer is stitched from complementary chunks or sources (`faithfulness=1.0` for both L7 and W3), no single context clearing that strict bar can still yield a real `0.0` average precision — confirmed as the mechanism on a re-probe of W3 too (fresh web results, non-deterministic, so not the identical original context set — but same per-context judging, including one context that legitimately scored 1 when it happened to cover most of a synthesized answer).
- **Retriever recall is strict by construction.** Gold is the specific answer-bearing chunk(s); a run can route to the right paper (`corpus_precision` ~1.0) yet still miss the exact passage (`context_recall` low). Both columns matter — the first for routing, the second for chunking / rerank-cutoff tuning.

