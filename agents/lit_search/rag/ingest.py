"""Corpus ingestion for lit_search's local knowledge base (SPEC Step 9).

Pipeline: read every document in ``agents/lit_search/data/corpus/``
(``.md`` / ``.markdown`` / ``.txt`` verbatim, ``.pdf`` via text
extraction) → structure-aware recursive chunking to ~500-token windows
with a small overlap → embed each chunk with ``sentence-transformers``
(``all-MiniLM-L6-v2``) → persist chunks, embeddings, and per-chunk source
metadata to a local Chroma DB at ``agents/lit_search/data/chroma_db/``.

Run directly to (re)build the DB::

    python -m agents.lit_search.rag.ingest

This module is standalone — it is not imported by lit_search's A2A
executor. Wiring retrieval into the agent is Step 10.
"""

from __future__ import annotations

import argparse
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

# --- Locations -------------------------------------------------------------

_RAG_DIR = Path(__file__).resolve().parent
DATA_DIR = _RAG_DIR.parent / "data"
CORPUS_DIR = DATA_DIR / "corpus"
CHROMA_DIR = DATA_DIR / "chroma_db"
COLLECTION_NAME = "lit_search_corpus"

CORPUS_SUFFIXES = {".md", ".markdown", ".txt", ".pdf"}

# --- Model / chunking parameters ----------------------------------------------

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
# all-MiniLM-L6-v2 ships with max_seq_length=256 but its positional table
# supports 512; bumping it lets a ~500-token chunk embed without silent
# truncation, so the dense vector actually reflects the whole chunk.
EMBED_MAX_SEQ_TOKENS = 512

CHUNK_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 50

# Tried in order. Heading separators first so a section stays with its
# heading; then paragraph, line, sentence, word, and finally character.
_SEPARATORS = ["\n## ", "\n### ", "\n#### ", "\n\n", "\n", ". ", " ", ""]


class IngestError(RuntimeError):
    """Raised when ingestion cannot proceed (empty corpus, unreadable file)."""


@dataclass
class Chunk:
    """One embeddable unit of a source document."""

    id: str
    text: str
    source: str
    title: str
    chunk_index: int
    chunk_count: int
    token_count: int

    def metadata(self) -> dict[str, str | int]:
        return {
            "source": self.source,
            "title": self.title,
            "chunk_index": self.chunk_index,
            "chunk_count": self.chunk_count,
            "token_count": self.token_count,
        }


@dataclass
class IngestSummary:
    chroma_path: str
    collection: str
    documents: int = 0
    chunks: int = 0
    per_document: dict[str, int] = field(default_factory=dict)


# --- Embedder ---------------------------------------------------------------


def load_embedder(model_name: str = EMBED_MODEL_NAME):
    """Loads the sentence-transformers bi-encoder used for dense vectors.

    Imported lazily so merely importing this module (e.g. for its path
    constants) doesn't drag in torch.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    model.max_seq_length = EMBED_MAX_SEQ_TOKENS
    return model


def _token_counter(embedder) -> Callable[[str], int]:
    tokenizer = embedder.tokenizer

    def count(text: str) -> int:
        # verbose=False: we only want a length, not a "sequence too long"
        # warning when measuring a whole document before it's split.
        return len(
            tokenizer.encode(text, add_special_tokens=False, verbose=False)
        )

    return count


# --- Chunking -------------------------------------------------------------


def _hard_split_by_tokens(
    text: str, embedder, max_tokens: int, overlap_tokens: int
) -> list[str]:
    """Last-resort split for a fragment with no usable separator left."""
    tokenizer = embedder.tokenizer
    ids = tokenizer.encode(text, add_special_tokens=False, verbose=False)
    if not ids:
        return []
    step = max(1, max_tokens - overlap_tokens)
    out: list[str] = []
    for start in range(0, len(ids), step):
        window = ids[start : start + max_tokens]
        if not window:
            break
        out.append(tokenizer.decode(window).strip())
        if start + max_tokens >= len(ids):
            break
    return [piece for piece in out if piece]


def _recursive_fragments(
    text: str,
    separators: list[str],
    count_tokens: Callable[[str], int],
    max_tokens: int,
    embedder,
) -> list[str]:
    """Split ``text`` into fragments, each <= ``max_tokens`` where possible.

    Walks ``separators`` from coarse (headings) to fine (characters),
    recursing into any piece that is still too large with the remaining,
    finer separators.
    """
    if count_tokens(text) <= max_tokens:
        stripped = text.strip()
        return [stripped] if stripped else []

    sep, rest = separators[0], separators[1:]
    if sep == "":
        return _hard_split_by_tokens(text, embedder, max_tokens, CHUNK_OVERLAP_TOKENS)
    if sep not in text:
        return _recursive_fragments(text, rest, count_tokens, max_tokens, embedder)

    fragments: list[str] = []
    is_heading_sep = sep.startswith("\n#")
    for i, raw in enumerate(text.split(sep)):
        if not raw.strip():
            continue
        # Re-attach heading markers so "## Foo" leads its section.
        piece = raw if i == 0 or not is_heading_sep else sep.lstrip("\n") + raw
        if count_tokens(piece) <= max_tokens:
            fragments.append(piece.strip())
        else:
            fragments.extend(
                _recursive_fragments(piece, rest, count_tokens, max_tokens, embedder)
            )
    return [frag for frag in fragments if frag]


def _verbatim_tail(
    text: str, count_tokens: Callable[[str], int], overlap_tokens: int
) -> str:
    """The trailing ~``overlap_tokens`` tokens of ``text``, as-is.

    Word-boundary based so casing and punctuation are preserved — unlike
    ``tokenizer.decode``, which lowercases (MiniLM is uncased) and respaces
    punctuation, corrupting text that then feeds BM25 and the reranker.
    """
    if overlap_tokens <= 0 or not text.strip():
        return ""
    words = text.split()
    tail: list[str] = []
    for word in reversed(words):
        tail.insert(0, word)
        if count_tokens(" ".join(tail)) >= overlap_tokens:
            break
    return " ".join(tail)


def _merge_with_overlap(
    fragments: list[str],
    count_tokens: Callable[[str], int],
    max_tokens: int,
    overlap_tokens: int,
) -> list[str]:
    """Greedily pack fragments into ~``max_tokens`` windows.

    Overlap is the verbatim trailing ``overlap_tokens`` tokens of the
    just-emitted chunk, prepended to the next window — so every chunk is
    bounded to ``max_tokens + overlap_tokens`` regardless of fragment sizes.
    """
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    def flush() -> str:
        return "\n\n".join(current).strip()

    for frag in fragments:
        frag_tokens = count_tokens(frag)
        if current and current_tokens + frag_tokens > max_tokens:
            emitted = flush()
            chunks.append(emitted)
            overlap_text = _verbatim_tail(emitted, count_tokens, overlap_tokens)
            current = [overlap_text] if overlap_text else []
            current_tokens = count_tokens(overlap_text) if overlap_text else 0
        current.append(frag)
        current_tokens += frag_tokens

    if current:
        chunks.append(flush())

    # Drop a trailing chunk that is nothing but the carried-over overlap.
    if len(chunks) >= 2 and chunks[-1] and chunks[-1] in chunks[-2]:
        chunks.pop()
    return [chunk for chunk in chunks if chunk]


def chunk_text(
    text: str,
    embedder,
    max_tokens: int = CHUNK_TOKENS,
    overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
) -> list[str]:
    """Structure-aware recursive split of one document into ~max_tokens chunks."""
    count_tokens = _token_counter(embedder)
    fragments = _recursive_fragments(
        text, _SEPARATORS, count_tokens, max_tokens, embedder
    )
    return _merge_with_overlap(fragments, count_tokens, max_tokens, overlap_tokens)


# --- Corpus reading -------------------------------------------------------


def iter_corpus_files(corpus_dir: Path = CORPUS_DIR) -> list[Path]:
    if not corpus_dir.is_dir():
        raise IngestError(f"corpus directory does not exist: {corpus_dir}")
    files = sorted(
        p
        for p in corpus_dir.iterdir()
        if p.is_file() and p.suffix.lower() in CORPUS_SUFFIXES
    )
    if not files:
        raise IngestError(
            f"no documents ({', '.join(sorted(CORPUS_SUFFIXES))}) found in {corpus_dir}"
        )
    return files


def _document_title(text: str, fallback: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
        if stripped:
            break
    return fallback


# arXiv slaps a stamp on page 1 ("arXiv:2401.15884v3 [cs.CL] 5 Oct 2024");
# strip it before guessing a title. Also drop a leading "Preprint." marker.
_ARXIV_STAMP_RE = re.compile(
    r"arXiv:\d{4}\.\d{4,5}v?\d*\s*\[[^\]]*\]\s*\d{1,2}\s+\w+\s+\d{4}", re.I
)


def _title_from_pdf_text(first_page: str, fallback: str) -> str:
    """Best-effort paper title from the first page's opening text.

    PDF extraction gives no reliable line breaks between the title and the
    author list, so this just takes the leading run of text (arXiv stamp
    removed), repairs the common small-caps artifact ("L EARNING" ->
    "LEARNING"), and truncates at a word boundary. Approximate by design —
    the authoritative identifier is the ``source`` filename.
    """
    txt = _ARXIV_STAMP_RE.sub(" ", first_page)
    txt = re.sub(r"^\s*preprint\.?\s*", "", txt, flags=re.I)
    txt = re.sub(r"\b([A-Z]) (?=[A-Z]{2,}\b)", r"\1", txt)  # "G ENERATE" -> "GENERATE"
    txt = " ".join(txt.split())
    if not txt:
        return fallback
    if len(txt) > 90:
        txt = txt[:90].rsplit(" ", 1)[0]
    return txt.strip(" ,:-") or fallback


def _extract_pdf(path: Path) -> tuple[str, str]:
    """Return (plain text, best-effort title) for a PDF.

    Uses ``pypdf`` (pure-Python, MIT). Pages are joined with blank lines so
    the same recursive splitter — paragraph → line → sentence → word — runs
    on the result; academic PDFs rarely carry markdown headings, so the
    heading separators simply don't fire and it falls through to those.
    Swap in ``pymupdf`` here if extraction quality on a given paper is poor.
    """
    try:
        from pypdf import PdfReader
    except ModuleNotFoundError as exc:  # pragma: no cover - dep not installed
        raise IngestError(
            "reading .pdf files needs `pypdf` — `pip install -r requirements.txt`"
        ) from exc

    fallback_title = path.stem.replace("_", " ").replace("-", " ").strip()

    # pypdf logs a WARNING per CFF font it can't fully decode ("fontTools is
    # required ..."); the text still extracts fine, so mute it for the call.
    pypdf_logger = logging.getLogger("pypdf")
    prior_level = pypdf_logger.level
    pypdf_logger.setLevel(logging.ERROR)
    try:
        reader = PdfReader(str(path))
        pages = [(page.extract_text() or "").strip() for page in reader.pages]
    except Exception as exc:  # noqa: BLE001 - pypdf raises assorted errors
        raise IngestError(f"could not parse PDF {path.name}: {exc}") from exc
    finally:
        pypdf_logger.setLevel(prior_level)

    text = "\n\n".join(page for page in pages if page)
    if not text.strip():
        raise IngestError(
            f"no extractable text in {path.name} — is it a scanned/image-only PDF?"
        )

    title = fallback_title
    try:
        meta = (reader.metadata.title or "").strip() if reader.metadata else ""
        if len(meta) >= 6:
            title = meta
        elif pages:
            title = _title_from_pdf_text(pages[0], fallback_title)
    except Exception:  # noqa: BLE001 - metadata is frequently malformed
        title = _title_from_pdf_text(pages[0], fallback_title) if pages else fallback_title
    return text, title


def _load_source(path: Path) -> tuple[str, str]:
    """Return (text, title) for one corpus file, dispatching on suffix."""
    fallback_title = path.stem.replace("_", " ").replace("-", " ").strip()
    if path.suffix.lower() == ".pdf":
        return _extract_pdf(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:  # pragma: no cover - unreadable file
        raise IngestError(f"could not read {path}: {exc}") from exc
    return text, _document_title(text, fallback_title)


def build_chunks(
    corpus_dir: Path = CORPUS_DIR, embedder=None
) -> tuple[list[Chunk], object]:
    """Reads and chunks the whole corpus. Returns (chunks, embedder)."""
    embedder = embedder or load_embedder()
    count_tokens = _token_counter(embedder)

    chunks: list[Chunk] = []
    for path in iter_corpus_files(corpus_dir):
        text, title = _load_source(path)
        pieces = chunk_text(text, embedder)
        for index, piece in enumerate(pieces):
            chunks.append(
                Chunk(
                    id=f"{path.name}::{index}",
                    text=piece,
                    source=path.name,
                    title=title,
                    chunk_index=index,
                    chunk_count=len(pieces),
                    token_count=count_tokens(piece),
                )
            )

    if not chunks:
        raise IngestError(f"corpus produced no chunks: {corpus_dir}")
    return chunks, embedder


# --- Persistence -------------------------------------------------------------


def ingest(
    corpus_dir: Path = CORPUS_DIR,
    chroma_dir: Path = CHROMA_DIR,
    collection_name: str = COLLECTION_NAME,
    embedder=None,
    reset: bool = True,
) -> IngestSummary:
    """Chunk + embed the corpus and (re)write the persistent Chroma collection."""
    import chromadb

    chunks, embedder = build_chunks(corpus_dir, embedder)

    embeddings = embedder.encode(
        [chunk.text for chunk in chunks],
        normalize_embeddings=True,
        show_progress_bar=False,
    ).tolist()

    chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_dir))

    if reset:
        try:
            client.delete_collection(collection_name)
        except Exception:  # noqa: BLE001 - "not found" is fine on a first run
            pass

    collection = client.get_or_create_collection(
        name=collection_name, metadata={"hnsw:space": "cosine"}
    )
    collection.add(
        ids=[chunk.id for chunk in chunks],
        documents=[chunk.text for chunk in chunks],
        embeddings=embeddings,
        metadatas=[chunk.metadata() for chunk in chunks],
    )

    summary = IngestSummary(chroma_path=str(chroma_dir), collection=collection_name)
    for chunk in chunks:
        summary.per_document[chunk.source] = chunk.chunk_count
    summary.documents = len(summary.per_document)
    summary.chunks = len(chunks)
    return summary


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=CORPUS_DIR)
    parser.add_argument("--chroma-dir", type=Path, default=CHROMA_DIR)
    parser.add_argument("--collection", default=COLLECTION_NAME)
    parser.add_argument(
        "--no-reset",
        action="store_true",
        help="Add to the existing collection instead of rebuilding it.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(argv)
    summary = ingest(
        corpus_dir=args.corpus,
        chroma_dir=args.chroma_dir,
        collection_name=args.collection,
        reset=not args.no_reset,
    )
    print(
        f"Ingested {summary.documents} document(s) → {summary.chunks} chunk(s) "
        f"into collection {summary.collection!r} at {summary.chroma_path}"
    )
    for source, count in sorted(summary.per_document.items()):
        print(f"  {source}: {count} chunk(s)")


if __name__ == "__main__":
    main()
