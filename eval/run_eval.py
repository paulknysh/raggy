"""Evaluation harness for the raggy RAG pipeline.

Score each QA pair in ``eval/qa.json`` on retrieval quality (hit rate, MRR,
recall@k) and generation quality (BLEU-1, ROUGE-L, embedding similarity, and an
LLM-as-judge verdict from the configured provider). Embeddings are local (Ollama).

Run with:  uv run eval/run_eval.py
"""

from __future__ import annotations

import json
import logging
import math
import re
import sys
from pathlib import Path
from statistics import mean
from typing import Any

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaEmbeddings

from raggy import load_config, run_pipeline
from raggy.llm_factory import get_llm

logger = logging.getLogger(__name__)

EVAL_DIR = Path(__file__).resolve().parent
QA_PATH = EVAL_DIR / "qa.json"
RESULTS_PATH = EVAL_DIR / "results.json"
CONFIG_PATH = EVAL_DIR.parent / "config.yaml"

_cfg = load_config(str(CONFIG_PATH))


# --------------------------------------------------------------------------- #
# Lexical metrics (implemented locally to avoid extra dependencies)
# --------------------------------------------------------------------------- #


def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def bleu1(reference: str, candidate: str) -> float:
    """BLEU-1 with brevity penalty using unigram overlap."""
    ref = _tokens(reference)
    cand = _tokens(candidate)
    if not cand:
        return 0.0
    clen, rlen = len(cand), len(ref)

    ref_counts: dict[str, int] = {}
    for tok in ref:
        ref_counts[tok] = ref_counts.get(tok, 0) + 1
    cand_counts: dict[str, int] = {}
    for tok in cand:
        cand_counts[tok] = cand_counts.get(tok, 0) + 1
    clipped = sum(min(c, ref_counts.get(t, 0)) for t, c in cand_counts.items())
    precision = clipped / clen
    brevity = math.exp(min(0.0, 1.0 - rlen / clen))
    return precision * brevity


def _lcs_length(a: list[str], b: list[str]) -> int:
    """Length of the longest common subsequence (worst/double precision)."""
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b, 1):
            cur[j] = prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1])
        prev = cur
    return prev[-1]


def rouge_l(reference: str, candidate: str) -> float:
    """ROUGE-L F-measure on token-level longest common subsequence."""
    ref = _tokens(reference)
    cand = _tokens(candidate)
    if not ref or not cand:
        return 0.0
    lcs = _lcs_length(ref, cand)
    recall = lcs / len(ref)
    precision = lcs / len(cand)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _abstains(answer: str) -> bool:
    """Heuristic: did the model refuse to answer rather than attempt one?"""
    s = answer.strip().lower()
    return (not s) or any(
        phrase in s for phrase in ("i don't know", "i do not know", "unknown")
    )


# --------------------------------------------------------------------------- #
# LLM-as-judge (groundedness + answer relevance), via the configured provider
# --------------------------------------------------------------------------- #

JUDGE_SYSTEM = (
    "You are a strict RAG evaluator. Given a QUESTION, a reference EXPECTED "
    "ANSWER, a GENERATED ANSWER, and the retrieved CONTEXT, score two "
    "dimensions from 0.0 to 1.0:\n"
    "  faithfulness: is the generated answer supported by (grounded in) the "
    "  provided context, without hallucination? An 'I don't know' answer is "
    "  highly faithful when the context does not contain the information.\n"
    "  relevance: is the generated answer correct and complete relative to the "
    "  expected answer? Refusing to answer a question that has a verifiable "
    "  answer must score low.\n"
    'Respond with ONLY a JSON object like {{"faithfulness": 0.8, '
    '"relevance": 0.9}}.'
)


def _parse_judge_scores(raw: Any) -> dict[str, float] | None:
    """Validate a judge response; return None if it is not usable JSON scores."""
    if not isinstance(raw, dict):
        return None
    scores: dict[str, float] = {}
    for dim in ("faithfulness", "relevance"):
        value = raw.get(dim)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        score = float(value)
        if not 0.0 <= score <= 1.0:
            return None
        scores[dim] = score
    return scores


def judge_answer(
    llm,
    question: str,
    reference: str,
    answer: str,
    context: str,
) -> dict[str, float | None]:
    prompt = (
        f"QUESTION:\n{question}\n\nEXPECTED ANSWER:\n{reference}\n\n"
        f"GENERATED ANSWER:\n{answer}\n\nCONTEXT:\n{context}\n"
    )
    chain = (
        ChatPromptTemplate.from_messages(
            [("system", JUDGE_SYSTEM), ("human", "{input}")]
        )
        | llm
        | JsonOutputParser()
    )

    for attempt in range(2):
        try:
            raw = chain.invoke({"input": prompt})
        except Exception as e:  # noqa: BLE001
            logger.warning("Judge call failed (attempt %d): %s", attempt + 1, e)
            raw = None
        scores = _parse_judge_scores(raw)
        if scores is not None:
            return scores
        logger.warning(
            "Judge returned malformed scores for question %r; retrying once.",
            question,
        )

    logger.warning(
        "Judge failed to return valid scores for question %r; "
        "scores excluded from aggregates.",
        question,
    )
    return {"faithfulness": None, "relevance": None}


# --------------------------------------------------------------------------- #
# Retrieval metrics
# --------------------------------------------------------------------------- #


def source_name(doc: Any) -> str:
    return Path(doc.metadata.get("source", "unknown")).name


def retrieval_metrics(docs: list[Any], ground_truth: list[str]) -> dict[str, float]:
    """Hit rate (any correct source in top-k), MRR, and recall@k."""
    truth = {Path(s).name.lower() for s in ground_truth}
    retrieved = [source_name(d).lower() for d in docs]

    first_hit_rank: int | None = None
    hits = 0
    for rank, name in enumerate(retrieved, 1):
        if name in truth:
            hits += 1
            if first_hit_rank is None:
                first_hit_rank = rank

    hit_rate = 1.0 if first_hit_rank is not None else 0.0
    mrr = 1.0 / first_hit_rank if first_hit_rank else 0.0
    recall = min(hits, len(truth)) / len(truth) if truth else 0.0
    return {"hit_rate": hit_rate, "mrr": mrr, "recall@k": recall}


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def fmt(v: float) -> str:
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def main() -> None:
    if not QA_PATH.exists():
        logger.error("QA file not found: %s", QA_PATH)
        sys.exit(1)

    with QA_PATH.open("r", encoding="utf-8") as f:
        pairs = json.load(f)
    if not isinstance(pairs, list) or not pairs:
        logger.error("qa.json must contain a non-empty list of QA entries.")
        sys.exit(1)

    logger.info("Evaluating QA pairs...")

    embeddings = OllamaEmbeddings(model=_cfg["embedding_model"])
    judge_llm = get_llm(
        _cfg["llm_provider"], _cfg["llm_model"], temperature=_cfg["temperature"]
    )

    rows: list[dict[str, Any]] = []
    for item in pairs:
        qid = item.get("id", "?")
        question = item["question"]
        reference = item.get("reference", "")
        ground_truth = item.get("ground_truth_sources", [])
        verifiable = bool(item.get("answer_can_be_verified", True))

        logger.info(">>> [%s] %s", qid, question)
        try:
            answer, docs = run_pipeline(question)
            logger.info("    ANSWER [%s]: %s", qid, answer)
        except Exception as e:  # noqa: BLE001
            logger.warning("FAILED [%s]: %s", qid, e)
            rows.append({"id": qid, "error": str(e)})
            continue

        retrieval = retrieval_metrics(docs, ground_truth)
        context = "\n\n".join(d.page_content for d in docs)

        gem_score = None
        if not _abstains(answer):
            gem_score = cosine(
                embeddings.embed_query(answer),
                embeddings.embed_query(reference),
            )

        if verifiable:
            judge = judge_answer(judge_llm, question, reference, answer, context)
            if _abstains(answer) and retrieval["hit_rate"] == 1.0:
                judge["relevance"] = 0.0
        else:
            abstained = _abstains(answer)
            judge = {
                "faithfulness": 1.0 if abstained else 0.0,
                "relevance": 1.0 if abstained else 0.0,
            }

        rows.append(
            {
                "id": qid,
                "question": question,
                "answer": answer,
                "retrieval": retrieval,
                "generation": {
                    "bleu1": bleu1(reference, answer),
                    "rouge_l": rouge_l(reference, answer),
                    "semantic": gem_score,
                    "faithfulness": judge["faithfulness"],
                    "relevance": judge["relevance"],
                },
            }
        )

    _render_summary(rows)
    output = {
        "overall": compute_overall(rows),
        "runtime": {"n_entries": len(rows)},
        "entries": rows,
    }
    RESULTS_PATH.write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Saved per-pair + aggregate results to %s", RESULTS_PATH)


RETRIEVAL_METRICS = ["hit_rate", "mrr", "recall@k"]
GENERATION_METRICS = ["bleu1", "rouge_l", "semantic", "faithfulness", "relevance"]


def _metric_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    """All per-pair values for a metric, skipping None and errored rows."""
    if key in GENERATION_METRICS:
        values = [r.get("generation", {}).get(key) for r in rows]
    else:
        values = [r.get("retrieval", {}).get(key) for r in rows]
    return [v for v in values if v is not None]


def compute_overall(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    """Mean of every metric across non-errored entries."""
    overall: dict[str, float | None] = {}
    for key in RETRIEVAL_METRICS + GENERATION_METRICS:
        check = _metric_values(rows, key)
        overall[key] = mean(check) if check else None
    return overall


def _render_summary(rows: list[dict[str, Any]]) -> None:
    overall = compute_overall(rows)
    headers = ["metric", "overall"] + [str(i + 1) for i in range(len(rows))]
    widths = [max(len(h), 14) for h in headers]
    print("\n" + "  ".join(h.ljust(widths[j]) for j, h in enumerate(headers)))
    print("  ".join("-" * w for w in widths))

    for key in RETRIEVAL_METRICS + GENERATION_METRICS:
        if key in GENERATION_METRICS:
            per_row = [r.get("generation", {}).get(key) for r in rows]
        else:
            per_row = [r.get("retrieval", {}).get(key) for r in rows]
        if all(v is None for v in per_row):
            continue
        agg = overall[key]
        line = [key, f"{agg:.3f}" if agg is not None else "-"] + [
            fmt(v) if v is not None else "-" for v in per_row
        ]
        print("  ".join(c.ljust(widths[j]) for j, c in enumerate(line)))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    main()
