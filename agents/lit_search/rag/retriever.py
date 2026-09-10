"""Hybrid retrieval over lit_search's local corpus (SPEC Step 9).

Given a query, this runs two retrievers over the *same* set of chunks
(the ones ``ingest.py`` wrote to Chroma):

1. **Dense** — cosine search against the Chroma collection using the same
   ``all-MiniLM-L6-v2`` embeddings ingestion used.
2. **Sparse** — BM25 (``rank_bm25``) built in memory from exactly those
   chunk texts, so the two retrievers rank identical units.

Their rank lists are merged with **Reciprocal Rank Fusion**, and the
fused candidate pool is **reranked with a cross-encoder**
(``cross-encoder/ms-marco-MiniLM-L-6-v2``) down to top-k.

Standalone — not imported by lit_search's A2A executor. Step 10 wires it
into the agent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .ingest import CHROMA_DIR, COLLECTION_NAME, load_embedder

RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RRF_K = 60
DEFAULT_TOP_K = 5
DEFAULT_CANDIDATE_POOL = 40

_WORD_RE = re.compile(r"[a-z0-9]+")


class RetrieverError(RuntimeError):
    """Raised when the retriever can't be built (missing / empty Chroma DB)."""


@dataclass
class RetrievedChunk:
    """One reranked result, with the provenance of how it got here."""

    id: str
    text: str
    source: str
    title: str
    chunk_index: int
    rerank_score: float
    rrf_score: float
    dense_rank: int | None
    sparse_rank: int | None

    def source_ref(self) -> str:
        return f"{self.source} #{self.chunk_index} — {self.title}"


def _tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


class HybridRetriever:
    """Dense + BM25 retrieval, RRF-merged, cross-encoder reranked."""

    def __init__(
        self,
        chroma_dir: Path = CHROMA_DIR,
        collection_name: str = COLLECTION_NAME,
        embedder=None,
        cross_encoder=None,
        rerank_model_name: str = RERANK_MODEL_NAME,
    ) -> None:
        import chromadb
        from rank_bm25 import BM25Okapi

        if not Path(chroma_dir).exists():
            raise RetrieverError(
                f"no Chroma DB at {chroma_dir} — run "
                f"`python -m agents.lit_search.rag.ingest` first"
            )

        client = chromadb.PersistentClient(path=str(chroma_dir))
        try:
            self._collection = client.get_collection(collection_name)
        except Exception as exc:  # noqa: BLE001
            raise RetrieverError(
                f"collection {collection_name!r} not found in {chroma_dir} — "
                f"run the ingest step first"
            ) from exc

        stored = self._collection.get(include=["documents", "metadatas"])
        self._ids: list[str] = list(stored["ids"])
        if not self._ids:
            raise RetrieverError(
                f"collection {collection_name!r} is empty — re-run ingest"
            )
        self._docs: list[str] = list(stored["documents"])
        self._metas: list[dict] = list(stored["metadatas"])
        self._doc_by_id = dict(zip(self._ids, self._docs))
        self._meta_by_id = dict(zip(self._ids, self._metas))

        self._embedder = embedder or load_embedder()
        self._bm25 = BM25Okapi([_tokenize(doc) for doc in self._docs])

        if cross_encoder is not None:
            self._cross_encoder = cross_encoder
        else:
            from sentence_transformers import CrossEncoder

            # ms-marco-MiniLM-L-6-v2 caps at 512 tokens for the
            # (query, passage) pair; set it explicitly so a long chunk is
            # truncated deterministically rather than warned about.
            self._cross_encoder = CrossEncoder(rerank_model_name, max_length=512)

    # -- individual retrievers ------------------------------------------------

    def _dense_ranking(self, query: str, pool: int) -> list[str]:
        query_embedding = self._embedder.encode(
            [query], normalize_embeddings=True, show_progress_bar=False
        ).tolist()
        result = self._collection.query(
            query_embeddings=query_embedding,
            n_results=min(pool, len(self._ids)),
        )
        return list(result["ids"][0])

    def _sparse_ranking(self, query: str, pool: int) -> list[str]:
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self._ids[i] for i in ranked[:pool] if scores[i] > 0.0]

    # -- fusion + rerank ----------------------------------------------------

    @staticmethod
    def _rrf(
        dense_ids: list[str], sparse_ids: list[str]
    ) -> tuple[dict[str, float], dict[str, int], dict[str, int]]:
        dense_rank = {cid: i + 1 for i, cid in enumerate(dense_ids)}
        sparse_rank = {cid: i + 1 for i, cid in enumerate(sparse_ids)}
        fused: dict[str, float] = {}
        for cid in set(dense_rank) | set(sparse_rank):
            score = 0.0
            if cid in dense_rank:
                score += 1.0 / (RRF_K + dense_rank[cid])
            if cid in sparse_rank:
                score += 1.0 / (RRF_K + sparse_rank[cid])
            fused[cid] = score
        return fused, dense_rank, sparse_rank

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        candidate_pool: int = DEFAULT_CANDIDATE_POOL,
    ) -> list[RetrievedChunk]:
        """Return the top-k reranked chunks for ``query``, best first."""
        if not query.strip():
            raise RetrieverError("query is empty")

        dense_ids = self._dense_ranking(query, candidate_pool)
        sparse_ids = self._sparse_ranking(query, candidate_pool)
        fused, dense_rank, sparse_rank = self._rrf(dense_ids, sparse_ids)

        candidates = sorted(fused, key=lambda cid: fused[cid], reverse=True)
        candidates = candidates[:candidate_pool]
        if not candidates:
            return []

        pairs = [[query, self._doc_by_id[cid]] for cid in candidates]
        rerank_scores = self._cross_encoder.predict(pairs)

        ordered = sorted(
            zip(candidates, rerank_scores),
            key=lambda pair: float(pair[1]),
            reverse=True,
        )

        results: list[RetrievedChunk] = []
        for cid, rerank_score in ordered[:top_k]:
            meta = self._meta_by_id.get(cid, {})
            results.append(
                RetrievedChunk(
                    id=cid,
                    text=self._doc_by_id[cid],
                    source=str(meta.get("source", "")),
                    title=str(meta.get("title", "")),
                    chunk_index=int(meta.get("chunk_index", -1)),
                    rerank_score=float(rerank_score),
                    rrf_score=float(fused[cid]),
                    dense_rank=dense_rank.get(cid),
                    sparse_rank=sparse_rank.get(cid),
                )
            )
        return results


def load_retriever(**kwargs) -> HybridRetriever:
    """Convenience constructor mirroring ``load_embedder``."""
    return HybridRetriever(**kwargs)
