"""SPEC Step 11 — RAGAS evaluation of lit_search.

Two independent measurements, so a regression can be traced to the layer
that caused it:

1. **Retriever alone** (Step 9's ``HybridRetriever``, no router / judge /
   LLM) — non-LLM context **precision** and **recall** over the ``local``
   eval queries, against the actual gold chunk texts resolved from each
   query's ``gold_contains`` phrases. Tunes chunking / hybrid weights /
   rerank cutoff without any generation step in the loop.

2. **Full agentic pipeline** (Step 10: router -> sources -> merge ->
   judge/reformulate) — **faithfulness** of an answer generated from the
   pipeline's returned evidence, plus **context relevance** (LLM context
   precision without reference: are the returned sources actually on
   point for the question). Reported per category — ``local`` / ``web`` /
   ``vague`` — so the "vague query drifts to irrelevant sources" failure
   mode shows up as a number instead of a vibe.

Results are written to ``eval/results/`` as ``eval_report.json`` (full
detail, re-runnable diff target) and ``eval_report.md`` (readable
summary), plus a copy of the exact eval set used.

    python -m eval.run_eval
    python -m eval.run_eval --top-k 5 --max-retries 2
    python -m eval.run_eval --only vague           # re-score one category, merge into report
    python -m eval.run_eval --skip-pipeline        # retriever metrics only (fast, no LLM)
    python -m eval.run_eval --rpm 10               # client-side Gemini cap if 429s dominate

On the Gemini free tier (15 req/min) a full run backs off on 429s for
20-40 min. ``--gemini-timeout`` / ``--gemini-max-retries`` (defaults
120s / 8) stop a stalled call from hanging the whole run; a row whose
router/judge fell back on a rate limit is flagged ``degraded`` and left
out of the category means.
"""

from __future__ import annotations

# Must precede any ``ragas`` import — see the module docstring there.
from eval import _ragas_compat  # noqa: F401  (import for side effect)

import argparse
import asyncio
import json
import logging
import os
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from agents.lit_search.rag.gemini import ainvoke_text, build_chat_model, resolve_model
from agents.lit_search.rag.ingest import CHROMA_DIR, COLLECTION_NAME
from agents.lit_search.rag.pipeline import build_pipeline_from_env
from agents.lit_search.rag.retriever import HybridRetriever
from eval.eval_set import load_eval_set

logger = logging.getLogger("eval")

RESULTS_DIR = Path(__file__).resolve().parent / "results"
DEFAULT_TOP_K = 5


# --------------------------------------------------------------------------
# Gold context resolution
# --------------------------------------------------------------------------

def _load_corpus_chunks() -> list[tuple[str, str, dict]]:
    import chromadb

    coll = chromadb.PersistentClient(path=str(CHROMA_DIR)).get_collection(COLLECTION_NAME)
    got = coll.get(include=["documents", "metadatas"])
    return list(zip(got["ids"], got["documents"], got["metadatas"]))


def resolve_gold_contexts(
    eval_set: list[dict], chunks: list[tuple[str, str, dict]]
) -> dict[str, list[str]]:
    """For each local query, phrase -> the chunk texts that contain it.

    Fails loudly if a ``gold_contains`` phrase matches nothing, or matches
    a chunk outside the query's ``gold_source`` — either means the eval
    set has drifted from the corpus and the numbers can't be trusted.
    """
    gold: dict[str, list[str]] = {}
    for row in eval_set:
        if row["category"] != "local":
            continue
        texts: dict[str, None] = {}
        for phrase in row["gold_contains"]:
            matches = [
                (doc, meta)
                for _id, doc, meta in chunks
                if phrase.lower() in doc.lower()
            ]
            if not matches:
                raise SystemExit(
                    f"[eval] {row['id']}: gold phrase {phrase!r} matches no chunk "
                    f"— fix eval_set.py or re-run ingest."
                )
            off_source = sorted(
                {m["source"] for _d, m in matches if m["source"] != row["gold_source"]}
            )
            if off_source:
                raise SystemExit(
                    f"[eval] {row['id']}: gold phrase {phrase!r} also appears in "
                    f"{off_source} (expected only {row['gold_source']}) — pick a "
                    f"more specific phrase."
                )
            for doc, _m in matches:
                texts[doc] = None
        gold[row["id"]] = list(texts)
        logger.info("gold: %s -> %d chunk(s)", row["id"], len(texts))
    return gold


# --------------------------------------------------------------------------
# Phase 1 — retriever only
# --------------------------------------------------------------------------

async def score_retriever(
    eval_set: list[dict],
    gold: dict[str, list[str]],
    retriever: HybridRetriever,
    top_k: int,
) -> list[dict]:
    from ragas.dataset_schema import SingleTurnSample
    from ragas.metrics import NonLLMContextPrecisionWithReference, NonLLMContextRecall

    precision = NonLLMContextPrecisionWithReference()
    recall = NonLLMContextRecall()

    rows: list[dict] = []
    for row in eval_set:
        if row["category"] != "local":
            continue
        chunks = retriever.retrieve(row["question"], top_k=top_k)
        retrieved_texts = [c.text for c in chunks]
        retrieved_sources = [c.source for c in chunks]
        sample = SingleTurnSample(
            user_input=row["question"],
            retrieved_contexts=retrieved_texts,
            reference_contexts=gold[row["id"]],
        )
        rec = {
            "id": row["id"],
            "question": row["question"],
            "gold_source": row["gold_source"],
            "retrieved_sources": retrieved_sources,
            "n_gold_chunks": len(gold[row["id"]]),
            "corpus_precision": round(
                sum(s == row["gold_source"] for s in retrieved_sources) / len(retrieved_sources),
                4,
            )
            if retrieved_sources
            else 0.0,
        }
        for name, metric in (
            ("context_precision", precision),
            ("context_recall", recall),
        ):
            try:
                rec[name] = round(float(await metric.single_turn_ascore(sample)), 4)
            except Exception as exc:  # noqa: BLE001
                logger.warning("retriever %s %s failed: %s", row["id"], name, exc)
                rec[name] = None
        rows.append(rec)
        logger.info(
            "retriever %-28s precision=%.3f recall=%.3f corpus_precision=%.3f",
            row["id"], rec["context_precision"] or 0.0,
            rec["context_recall"] or 0.0, rec["corpus_precision"],
        )
    return rows


# --------------------------------------------------------------------------
# Phase 2 — full pipeline
# --------------------------------------------------------------------------

_ANSWER_PROMPT = (
    "Answer the question using ONLY the numbered sources below. Paraphrase; "
    "do not copy sentences verbatim. Keep it to 2-4 sentences. Do not add "
    "facts that are not in the sources. If the sources genuinely contain "
    "nothing relevant, still write the most on-topic 2-3 sentences you can "
    "from them.\n\nQuestion: {question}\n\nSources:\n{sources}\n\nAnswer:"
)


def _format_sources(sources: list[dict]) -> str:
    out = []
    for i, s in enumerate(sources):
        snippet = " ".join(str(s.get("snippet", "")).split())
        out.append(f"[{i}] ({s.get('origin', '?')}) {s.get('title', '')}: {snippet}")
    return "\n".join(out) if out else "(no sources)"


# Hard ceiling per step, so one stuck call can't hang the whole run even if
# the model-level timeout is somehow bypassed. Generous: rate-limited runs
# legitimately spend minutes per metric (one LLM call per retrieved context).
_GATHER_TIMEOUT_S = 600
_ANSWER_TIMEOUT_S = 240
_METRIC_TIMEOUT_S = 900


async def score_pipeline(
    eval_set: list[dict],
    pipeline,
    raw_llm,
    ragas_llm,
    only: str | None,
) -> list[dict]:
    from ragas.dataset_schema import SingleTurnSample
    from ragas.metrics import Faithfulness, LLMContextPrecisionWithoutReference

    faithfulness = Faithfulness(llm=ragas_llm)
    ctx_relevance = LLMContextPrecisionWithoutReference(llm=ragas_llm)

    rows: list[dict] = []
    for row in eval_set:
        if only and row["category"] != only:
            continue
        t0 = time.monotonic()
        try:
            result = await asyncio.wait_for(
                pipeline.gather(row["question"]), timeout=_GATHER_TIMEOUT_S
            )
        except Exception as exc:  # noqa: BLE001  (incl. asyncio.TimeoutError)
            logger.warning("pipeline %s: gather failed/timed out (%s) — skipping row", row["id"], exc)
            rows.append({
                "id": row["id"], "category": row["category"], "question": row["question"],
                "routes": [], "retry_count": 0, "judge_sufficient": False,
                "n_sources": 0, "source_origins": [], "final_query": row["question"],
                "answer": "", "faithfulness": None, "context_relevance": None,
                "degraded": True, "degraded_notes": [f"gather error: {exc}"],
                "elapsed_s": round(time.monotonic() - t0, 1),
            })
            continue
        sources = result.sources
        ctx_texts = [str(s.get("snippet", "")) for s in sources]

        # A router/judge Gemini call that gave up (e.g. a 429 that outlived
        # the SDK's backoff) is caught inside the pipeline and recorded in
        # the judge history as an "... error:" / "inconclusive" reasoning.
        # Flag the row so a rate-limited verdict isn't read as a real one.
        degraded_notes = [
            str(h.get("reasoning", ""))
            for h in result.outcome["history"]
            if any(
                marker in str(h.get("reasoning", "")).lower()
                for marker in ("judge error:", "router error:", "inconclusive")
            )
        ]

        try:
            answer = (
                await asyncio.wait_for(
                    ainvoke_text(
                        raw_llm,
                        _ANSWER_PROMPT.format(
                            question=row["question"], sources=_format_sources(sources)
                        ),
                    ),
                    timeout=_ANSWER_TIMEOUT_S,
                )
            ).strip()
        except Exception as exc:  # noqa: BLE001  (incl. asyncio.TimeoutError)
            logger.warning("pipeline %s answer-gen failed/timed out: %s", row["id"], exc)
            answer = ""

        rec = {
            "id": row["id"],
            "category": row["category"],
            "question": row["question"],
            "initial_route": result.initial_decision.route,
            "routes": [d.route for d in result.per_query_routes],
            "retry_count": result.outcome["retry_count"],
            "judge_sufficient": result.outcome["sufficient"],
            "n_sources": len(sources),
            "source_origins": [s.get("origin", "?") for s in sources],
            "final_query": result.outcome["final_query"],
            "answer": answer,
            "degraded": bool(degraded_notes) or not answer,
            "degraded_notes": degraded_notes,
            "elapsed_s": round(time.monotonic() - t0, 1),
        }
        if rec["degraded"]:
            logger.warning(
                "pipeline %s: DEGRADED row (rate-limit fallback or empty answer) "
                "— excluded from aggregates", row["id"],
            )

        sample = SingleTurnSample(
            user_input=row["question"],
            response=answer or "(no answer generated)",
            retrieved_contexts=ctx_texts or ["(no sources retrieved)"],
        )
        for name, metric in (
            ("faithfulness", faithfulness),
            ("context_relevance", ctx_relevance),
        ):
            try:
                val = float(
                    await asyncio.wait_for(
                        metric.single_turn_ascore(sample), timeout=_METRIC_TIMEOUT_S
                    )
                )
                rec[name] = round(val, 4) if val == val else None  # NaN -> None
            except Exception as exc:  # noqa: BLE001  (incl. asyncio.TimeoutError)
                logger.warning("pipeline %s %s failed/timed out: %s", row["id"], name, exc)
                rec[name] = None
        rows.append(rec)
        logger.info(
            "pipeline %-28s [%-5s] route=%-11s retries=%d suff=%-5s "
            "faithfulness=%s context_relevance=%s",
            row["id"], row["category"], "->".join(rec["routes"]),
            rec["retry_count"], rec["judge_sufficient"],
            rec["faithfulness"], rec["context_relevance"],
        )
    return rows


# --------------------------------------------------------------------------
# Aggregation + reporting
# --------------------------------------------------------------------------

def _mean(values: list[Any]) -> float | None:
    nums = [v for v in values if isinstance(v, (int, float))]
    return round(statistics.fmean(nums), 4) if nums else None


def aggregate(retriever_rows: list[dict], pipeline_rows: list[dict]) -> dict:
    agg: dict[str, Any] = {}

    if retriever_rows:
        agg["retriever"] = {
            "n_queries": len(retriever_rows),
            "context_precision_mean": _mean([r["context_precision"] for r in retriever_rows]),
            "context_recall_mean": _mean([r["context_recall"] for r in retriever_rows]),
            "corpus_precision_mean": _mean([r["corpus_precision"] for r in retriever_rows]),
        }

    if pipeline_rows:
        clean = [r for r in pipeline_rows if not r.get("degraded")]
        by_cat: dict[str, Any] = {}
        cats = ["local", "web", "vague"]
        for cat in [*cats, "ALL"]:
            rs = clean if cat == "ALL" else [r for r in clean if r["category"] == cat]
            if not rs:
                continue
            by_cat[cat] = {
                "n_queries": len(rs),
                "faithfulness_mean": _mean([r["faithfulness"] for r in rs]),
                "context_relevance_mean": _mean([r["context_relevance"] for r in rs]),
                "retry_count_mean": _mean([r["retry_count"] for r in rs]),
                "judge_sufficient_rate": _mean([1.0 if r["judge_sufficient"] else 0.0 for r in rs]),
            }
        agg["pipeline_by_category"] = by_cat
        agg["pipeline_degraded_ids"] = [
            r["id"] for r in pipeline_rows if r.get("degraded")
        ]

    return agg


def _fmt(v: Any) -> str:
    return "  n/a" if v is None else f"{v:.3f}" if isinstance(v, float) else str(v)


def write_reports(payload: dict, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "eval_report.json"
    md_path = out_dir / "eval_report.md"
    json_path.write_text(json.dumps(payload, indent=2))

    cfg = payload["config"]
    agg = payload["aggregates"]
    lines: list[str] = []
    lines.append("# lit_search evaluation (SPEC Step 11)\n")
    lines.append(
        f"- generated: {cfg['generated_at']}\n"
        f"- git: `{cfg['git_sha']}`\n"
        f"- gemini model: `{cfg['gemini_model']}`\n"
        f"- retriever top-k: {cfg['top_k']} | pipeline max-retries: {cfg['max_retries']} "
        f"| gemini rate cap: {cfg.get('rate_limit_rpm', 'n/a')}/min"
        f"{' | merged' if cfg.get('merged') else ''}\n"
    )
    degraded = agg.get("pipeline_degraded_ids") or []
    if degraded:
        lines.append(
            f"\n> ⚠ **{len(degraded)} pipeline row(s) degraded** (a router/judge "
            f"Gemini call hit the rate limit and fell back, or no answer was "
            f"generated): {', '.join(degraded)}. Excluded from the category means "
            f"below; re-run with `--only <category> --rpm <lower>` to refresh them.\n"
        )

    if "retriever" in agg:
        r = agg["retriever"]
        lines.append("\n## 1. Retriever alone (Step 9) — non-LLM context metrics\n")
        lines.append(
            f"Over {r['n_queries']} `local` queries, gold = actual chunk texts "
            f"resolved from each query's key phrases.\n"
        )
        lines.append("| metric | mean |")
        lines.append("|---|---|")
        lines.append(f"| context precision (vs gold chunks) | {_fmt(r['context_precision_mean'])} |")
        lines.append(f"| context recall (vs gold chunks) | {_fmt(r['context_recall_mean'])} |")
        lines.append(f"| corpus precision (retrieved chunks from the right paper) | {_fmt(r['corpus_precision_mean'])} |")
        lines.append("\n<details><summary>per query</summary>\n")
        lines.append("| id | precision | recall | corpus_prec | gold chunks | retrieved papers |")
        lines.append("|---|---|---|---|---|---|")
        for row in payload["retriever"]:
            papers = ", ".join(sorted(set(row["retrieved_sources"])))
            lines.append(
                f"| {row['id']} | {_fmt(row['context_precision'])} | {_fmt(row['context_recall'])} "
                f"| {_fmt(row['corpus_precision'])} | {row['n_gold_chunks']} | {papers} |"
            )
        lines.append("\n</details>\n")

    if "pipeline_by_category" in agg:
        lines.append("\n## 2. Full pipeline (Step 10) — faithfulness + context relevance\n")
        lines.append(
            "`faithfulness`: is the generated answer grounded in the evidence "
            "lit_search returned. `context_relevance`: are those evidence "
            "items actually on point for the question (RAGAS LLM context "
            "precision, no reference).\n"
        )
        lines.append("| category | n | faithfulness | context_relevance | mean retries | judge-sufficient rate |")
        lines.append("|---|---|---|---|---|---|")
        for cat in ["local", "web", "vague", "ALL"]:
            c = agg["pipeline_by_category"].get(cat)
            if not c:
                continue
            lines.append(
                f"| {cat} | {c['n_queries']} | {_fmt(c['faithfulness_mean'])} "
                f"| {_fmt(c['context_relevance_mean'])} | {_fmt(c['retry_count_mean'])} "
                f"| {_fmt(c['judge_sufficient_rate'])} |"
            )
        lines.append("\n<details><summary>per query</summary>\n")
        lines.append("| id | cat | routes | retries | suff | faithfulness | context_relevance | degraded |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for row in payload["pipeline"]:
            lines.append(
                f"| {row['id']} | {row['category']} | {'->'.join(row['routes'])} "
                f"| {row['retry_count']} | {row['judge_sufficient']} "
                f"| {_fmt(row['faithfulness'])} | {_fmt(row['context_relevance'])} "
                f"| {'⚠ yes' if row.get('degraded') else ''} |"
            )
        lines.append("\n</details>\n")

        lines.append("\n### Reading these numbers\n")
        lines.append(
            "- **Faithfulness does not separate the query types.** The "
            "answer-generation step summarises whatever evidence came back, "
            "so it stays ~1.0 even when that evidence is off-topic; the only "
            "dip is a web query that over-reached on thin sources. It is a "
            "grounding check, not a relevance check.\n"
            "- **`context_relevance` + retry behaviour is what flags a weak "
            "run.** A vague sub-query that the reformulate step can't "
            "ground in the committee's domain (no salient noun, a pronoun "
            "with no antecedent) now exits after a single retry as "
            "`unreformulable` instead of drifting to a plausible-sounding "
            "but unrelated query and burning both retries on it — lower "
            "`judge_sufficient` is expected there and is the honest "
            "outcome, not a regression; watch `context_relevance` and "
            "`retry_count_mean` together to catch a reformulation prompt "
            "that's drifting off-topic again.\n"
            "- **Retriever recall/precision move with candidate-pool width "
            "and chunk overlap.** Widening the pre-rerank candidate pool "
            "lets the cross-encoder see chunks the fused dense/sparse pool "
            "would otherwise cut before reranking; widening chunk overlap "
            "helps when a short, dense technical span (a definition table, "
            "a stale-index explanation) gets split across a chunk boundary "
            "and neither half scores well alone. If `local` "
            "`context_recall` or `judge_sufficient` regresses after a "
            "retriever change, pull the gold vs. retrieved chunks for the "
            "worst queries before assuming it's a rerank-model problem.\n"
            "- **A `context_relevance` of 0.000 is not automatically a "
            "scoring bug.** `LLMContextPrecisionWithoutReference` verdicts "
            "each retrieved context in isolation against the *synthesized* "
            "answer; a faithful answer stitched from complementary chunks "
            "or sources (high `faithfulness`) can still score 0.0 if no "
            "single context alone covers enough of it to pass that "
            "per-context bar. Before treating a 0.000 as a pipeline defect, "
            "call the RAGAS judge prompt directly per context and read the "
            "raw verdict + reason — an exception/parse failure shows up as "
            "`None` (`n/a`) via this script's own try/except, never a "
            "silent `0.0`, so a real `0.000` in this table is a genuine "
            "all-contexts-insufficient verdict.\n"
            "- **Retriever recall is strict by construction.** Gold is the "
            "specific answer-bearing chunk(s); a run can route to the right "
            "paper (`corpus_precision` ~1.0) yet still miss the exact "
            "passage (`context_recall` low). Both columns matter — the "
            "first for routing, the second for chunking / rerank-cutoff "
            "tuning.\n"
        )

    md_path.write_text("\n".join(lines) + "\n")
    return json_path, md_path


# --------------------------------------------------------------------------
# Entrypoint
# --------------------------------------------------------------------------

def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _prior_rows(out_dir: Path, section: str) -> list[dict]:
    path = out_dir / "eval_report.json"
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text()).get(section, []) or []
    except (OSError, ValueError):
        return []


def _merge_rows(prior: list[dict], fresh: list[dict]) -> list[dict]:
    """Overlay ``fresh`` rows onto ``prior`` by ``id``, preserving eval-set order."""
    if not fresh:
        return prior
    order = {row["id"]: i for i, row in enumerate(load_eval_set())}
    by_id = {row["id"]: row for row in prior}
    for row in fresh:
        by_id[row["id"]] = row
    return sorted(by_id.values(), key=lambda r: order.get(r["id"], 1_000))


async def _run(args: argparse.Namespace) -> None:
    eval_set = load_eval_set()
    chunks = _load_corpus_chunks()
    gold = resolve_gold_contexts(eval_set, chunks)

    raw_llm = build_chat_model()  # GEMINI_API_KEY / GEMINI_MODEL
    retriever = HybridRetriever()

    retriever_rows: list[dict] = []
    if not args.skip_retriever and (not args.only or args.only == "local"):
        logger.info("=== Phase 1: retriever alone ===")
        retriever_rows = await score_retriever(eval_set, gold, retriever, args.top_k)

    pipeline_rows: list[dict] = []
    if not args.skip_pipeline:
        logger.info("=== Phase 2: full agentic pipeline ===")
        from ragas.llms import LangchainLLMWrapper

        ragas_llm = LangchainLLMWrapper(raw_llm)
        pipeline = build_pipeline_from_env(max_retries=args.max_retries)
        # Reuse the already-loaded retriever (avoids a second model load).
        pipeline._retriever = retriever  # noqa: SLF001
        pipeline_rows = await score_pipeline(
            eval_set, pipeline, raw_llm, ragas_llm, args.only
        )

    out_dir = Path(args.out_dir)

    merge = args.merge or args.only is not None
    if merge:
        retriever_rows = _merge_rows(_prior_rows(out_dir, "retriever"), retriever_rows)
        pipeline_rows = _merge_rows(_prior_rows(out_dir, "pipeline"), pipeline_rows)
        logger.info(
            "merged into existing report: %d retriever row(s), %d pipeline row(s)",
            len(retriever_rows), len(pipeline_rows),
        )

    payload = {
        "config": {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_sha": _git_sha(),
            "gemini_model": resolve_model(),
            "top_k": args.top_k,
            "max_retries": args.max_retries,
            "rate_limit_rpm": args.rpm,
            "gemini_timeout_s": args.gemini_timeout,
            "gemini_max_retries": args.gemini_max_retries,
            "only": args.only,
            "merged": merge,
            "eval_set_size": len(eval_set),
        },
        "aggregates": aggregate(retriever_rows, pipeline_rows),
        "retriever": retriever_rows,
        "pipeline": pipeline_rows,
    }
    json_path, md_path = write_reports(payload, out_dir)
    (out_dir / "eval_set.json").write_text(json.dumps(eval_set, indent=2))

    print("\n" + "=" * 78)
    print(md_path.read_text())
    print("=" * 78)
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    for noisy in (
        "httpx", "httpcore", "urllib3", "sentence_transformers", "chromadb",
        "google_genai.models", "google_genai.types",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    # Keep the 429/backoff notices — they explain a slow run on the free tier.
    logging.getLogger("google_genai._api_client").setLevel(logging.INFO)
    # Models are already cached from Step 9/10 — don't re-check the hub.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--only", choices=["local", "web", "vague"], default=None)
    parser.add_argument(
        "--rpm",
        type=float,
        default=0.0,
        help="Client-side Gemini request/min cap shared across every call "
        "(pipeline + answer-gen + RAGAS). Free tier is 15/min. Default 0 "
        "(off) — the SDK's own 429 backoff plus --gemini-max-retries is "
        "usually enough and avoids the limiter interacting with RAGAS's "
        "executor. Set e.g. 10 if 429s dominate a run.",
    )
    parser.add_argument(
        "--gemini-timeout",
        type=float,
        default=120.0,
        help="Per-request timeout (s) so a stalled socket fails and retries "
        "instead of hanging the run. 0 = library default (no timeout).",
    )
    parser.add_argument(
        "--gemini-max-retries",
        type=int,
        default=8,
        help="Gemini client retry attempts — rides out free-tier 429 bursts.",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge this run's rows into the existing eval_report.json "
        "instead of replacing it (implied by --only).",
    )
    parser.add_argument("--skip-retriever", action="store_true")
    parser.add_argument("--skip-pipeline", action="store_true")
    parser.add_argument("--out-dir", default=str(RESULTS_DIR))
    args = parser.parse_args(argv)

    # Applied by build_chat_model() to every ChatGoogleGenerativeAI — the
    # pipeline's router/judge, our answer-gen, and the RAGAS metric LLM.
    # Set before anything imports/builds a model.
    if args.rpm and args.rpm > 0:
        os.environ["GEMINI_REQUESTS_PER_MINUTE"] = str(args.rpm)
    if args.gemini_timeout and args.gemini_timeout > 0:
        os.environ["GEMINI_TIMEOUT"] = str(args.gemini_timeout)
    if args.gemini_max_retries and args.gemini_max_retries > 0:
        os.environ["GEMINI_MAX_RETRIES"] = str(args.gemini_max_retries)

    if not os.environ.get("GEMINI_API_KEY"):
        raise SystemExit("[eval] GEMINI_API_KEY is not set (needed for faithfulness + answer gen).")

    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
