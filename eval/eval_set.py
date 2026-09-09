"""The Step 11 eval set — 15 sub-queries across three routing regimes.

Categories:

- ``local``  — answerable from one of the five ingested papers. Carries a
  ``gold_source`` (the paper that owns the answer) and ``gold_contains``
  (verbatim phrases that occur only in that paper's chunks). ``run_eval``
  resolves those phrases to the actual chunk texts and uses them as
  ``reference_contexts`` for non-LLM context precision / recall — so the
  retrieval metrics need no LLM and no hand-transcribed passages.
- ``web``    — needs current / outside information the corpus doesn't have.
- ``vague``  — deliberately under-specified, including the two the human
  tested by hand in Step 10 ("how do you make it better?" and "how do
  they improve the retriever part?").

``reference`` is a short human answer, for the report's readability; the
scored retrieval metrics use ``gold_contains`` -> chunk text, not this.

Papers: 2002.08909 = REALM, 2004.04906 = DPR, 2005.11401 = RAG (Lewis
et al.), 2310.11511 = Self-RAG, 2401.15884 = CRAG.
"""

from __future__ import annotations

REALM = "2002.08909v1.pdf"
DPR = "2004.04906v3.pdf"
RAG_LEWIS = "2005.11401v4.pdf"
SELF_RAG = "2310.11511v1.pdf"
CRAG = "2401.15884v3.pdf"


EVAL_SET: list[dict] = [
    # -- local: answerable from exactly one ingested paper ----------------
    {
        "id": "L1-dpr-negatives",
        "category": "local",
        "question": "What negative passages does DPR use to train its retriever?",
        "gold_source": DPR,
        "gold_contains": ["in-batch negative", "BM25 negative"],
        "reference": (
            "DPR trains with in-batch negatives (the gold passages of the "
            "other questions in a mini-batch) plus one hard negative per "
            "question retrieved by BM25 that has high lexical overlap but "
            "does not contain the answer."
        ),
    },
    {
        "id": "L2-dpr-encoder",
        "category": "local",
        "question": "What encoder architecture does DPR use to embed questions and passages?",
        "gold_source": DPR,
        "gold_contains": ["dual-encoder"],
        "reference": (
            "A dual-encoder: two independent BERT encoders, one for the "
            "question and one for the passage, with relevance scored as the "
            "dot product of their [CLS] embeddings."
        ),
    },
    {
        "id": "L3-selfrag-reflection-tokens",
        "category": "local",
        "question": "What reflection tokens does Self-RAG generate and what are they for?",
        "gold_source": SELF_RAG,
        "gold_contains": ["four types of reflection", "ISREL , ISSUP"],
        "reference": (
            "Self-RAG emits special reflection tokens: a Retrieve token "
            "deciding whether to retrieve, and critique tokens ISREL "
            "(passage relevance), ISSUP (whether the output is supported by "
            "the passage), and ISUSE (overall usefulness)."
        ),
    },
    {
        "id": "L4-selfrag-retrieve-on-demand",
        "category": "local",
        "question": "How does Self-RAG decide when to retrieve during generation?",
        "gold_source": SELF_RAG,
        "gold_contains": ["retrieve on demand", "adaptive retrieval"],
        "reference": (
            "The model itself predicts a Retrieve reflection token on "
            "demand as it generates, retrieving only when the next segment "
            "needs external knowledge instead of retrieving a fixed number "
            "of passages up front."
        ),
    },
    {
        "id": "L5-crag-evaluator-labels",
        "category": "local",
        "question": "What confidence labels does CRAG's retrieval evaluator assign to retrieved documents?",
        "gold_source": CRAG,
        "gold_contains": ["retrieval evaluator", "Correct, Incorrect"],
        "reference": (
            "A lightweight retrieval evaluator scores each retrieved "
            "document and, via an upper and lower threshold, labels the "
            "retrieval result Correct, Incorrect, or Ambiguous, which "
            "triggers different downstream actions."
        ),
    },
    {
        "id": "L6-crag-decompose-recompose",
        "category": "local",
        "question": "What is the decompose-then-recompose step in CRAG for?",
        "gold_source": CRAG,
        "gold_contains": ["decompose-then-recompose"],
        "reference": (
            "A knowledge-refinement algorithm that splits each retrieved "
            "document into fine-grained strips, filters out the irrelevant "
            "strips, and recomposes the rest, so only the key knowledge is "
            "passed to the generator."
        ),
    },
    {
        "id": "L7-realm-pretraining",
        "category": "local",
        "question": "How is REALM's retriever pre-trained and how is its index kept current?",
        "gold_source": REALM,
        "gold_contains": ["latent knowledge retriever", "stale"],
        "reference": (
            "REALM pre-trains a latent knowledge retriever end-to-end with "
            "a masked-language-modeling objective, backpropagating through "
            "retrieval; the document index would go stale as the encoder "
            "updates, so it is asynchronously re-embedded and rebuilt every "
            "few hundred steps."
        ),
    },
    {
        "id": "L8-rag-sequence-vs-token",
        "category": "local",
        "question": "What is the difference between RAG-Sequence and RAG-Token?",
        "gold_source": RAG_LEWIS,
        "gold_contains": ["RAG-Sequence", "RAG-Token"],
        "reference": (
            "RAG-Sequence conditions the whole output on a single retrieved "
            "document; RAG-Token can attend to a different retrieved "
            "document for each generated token, marginalising over documents "
            "per token."
        ),
    },
    {
        "id": "L9-rag-nonparametric-memory",
        "category": "local",
        "question": "What does RAG (Lewis et al.) use as its non-parametric memory?",
        "gold_source": RAG_LEWIS,
        "gold_contains": ["non-parametric memory"],
        "reference": (
            "A dense vector index of a Wikipedia dump, accessed with a "
            "pretrained DPR retriever — the non-parametric memory paired "
            "with BART as the parametric generator."
        ),
    },
    # -- web: needs current / out-of-corpus information ------------------
    {
        "id": "W1-post-crag-methods",
        "category": "web",
        "question": (
            "What retrieval-augmented generation methods introduced after 2024 "
            "improve on CRAG and Self-RAG, and how?"
        ),
        "gold_source": None,
        "gold_contains": [],
        "reference": None,
    },
    {
        "id": "W2-production-vector-dbs",
        "category": "web",
        "question": "Which vector databases are most widely used for production RAG systems in 2025?",
        "gold_source": None,
        "gold_contains": [],
        "reference": None,
    },
    {
        "id": "W3-mteb-sota-embedding",
        "category": "web",
        "question": "What is the current state-of-the-art embedding model on the MTEB retrieval leaderboard?",
        "gold_source": None,
        "gold_contains": [],
        "reference": None,
    },
    # -- vague: deliberately under-specified ---------------------------
    {
        "id": "V1-make-it-better",
        "category": "vague",
        "question": "how do you make it better?",
        "gold_source": None,
        "gold_contains": [],
        "reference": None,
    },
    {
        "id": "V2-improve-the-retriever-part",
        "category": "vague",
        "question": "how do they improve the retriever part?",
        "gold_source": None,
        "gold_contains": [],
        "reference": None,
    },
    {
        "id": "V3-main-problems",
        "category": "vague",
        "question": "what are the main problems and how are they handled?",
        "gold_source": None,
        "gold_contains": [],
        "reference": None,
    },
]


def load_eval_set() -> list[dict]:
    return [dict(row) for row in EVAL_SET]
