# Databricks notebook source
# MAGIC %md
# MAGIC ## Retrieval evaluation (gold CSV)
# MAGIC - Runs **`rag_pipeline`** from **`ai/src/retrieval.py`** against each query in a gold CSV and reports **hit rate**, **MRR**, and **nDCG** at **`k`**.
# MAGIC - **CSV columns:** `query_id`, `query_text`, `relevant_chunk_id`, `relevance_label`, optional **`split`**.
# MAGIC - **Params:** env vars or widgets — **`EVAL_GOLD_CSV`** (required path on driver / DBFS), **`EVAL_K`**, **`EVAL_RELEVANT_THRESHOLD`**, optional **`EVAL_SPLIT`**, **`EVAL_RERANK_TOP_K`**, **`EVAL_OUTPUT_PREDICTIONS_CSV`**, **`EVAL_OUTPUT_JSON`**.
# MAGIC - **Path:** ensure the repo’s **`ai/src`** is importable (this notebook prepends it from `__file__` / cwd, or set **`REPO_ROOT`** to the repo root).

# COMMAND ----------

from __future__ import annotations

import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def _ensure_ai_src_on_path() -> None:
    """Prepend `ai/src` so `from retrieval import rag_pipeline` works on the driver."""
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        cand = os.path.abspath(os.path.join(here, "..", "..", "ai", "src"))
        if os.path.isfile(os.path.join(cand, "retrieval.py")) and cand not in sys.path:
            sys.path.insert(0, cand)
            return
    except NameError:
        pass
    root = os.environ.get("REPO_ROOT", "").strip()
    if root:
        p = os.path.abspath(os.path.join(root, "ai", "src"))
        if os.path.isfile(os.path.join(p, "retrieval.py")) and p not in sys.path:
            sys.path.insert(0, p)
            return
    cwd = os.getcwd()
    for rel in ("ai/src", os.path.join("..", "ai", "src"), os.path.join("..", "..", "ai", "src")):
        p = os.path.abspath(os.path.join(cwd, rel))
        if os.path.isfile(os.path.join(p, "retrieval.py")) and p not in sys.path:
            sys.path.insert(0, p)
            return


_ensure_ai_src_on_path()
from retrieval import rag_pipeline  # noqa: E402


DEFAULT_COLUMNS = [
    "chunk_id",
    "company_id",
    "chunk_index",
    "chunk_text",
    "company_name",
    "country",
    "theme",
    "deal_year",
    "investor_names",
]


def load_gold_csv(path: str | Path, split: str | None = None) -> tuple[dict[str, str], dict[str, dict[str, int]]]:
    query_text_by_id: dict[str, str] = {}
    judgments_by_query: dict[str, dict[str, int]] = defaultdict(dict)

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"query_id", "query_text", "relevant_chunk_id", "relevance_label"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Gold CSV missing required columns: {sorted(missing)}")

        for row in reader:
            row_split = (row.get("split") or "").strip()
            if split and row_split and row_split != split:
                continue

            query_id = (row.get("query_id") or "").strip()
            query_text = (row.get("query_text") or "").strip()
            chunk_id = (row.get("relevant_chunk_id") or "").strip()
            if not query_id or not query_text or not chunk_id:
                continue

            try:
                label = int(str(row.get("relevance_label") or "0").strip())
            except ValueError:
                label = 0

            query_text_by_id.setdefault(query_id, query_text)
            prev = judgments_by_query[query_id].get(chunk_id, 0)
            if label > prev:
                judgments_by_query[query_id][chunk_id] = label

    if not query_text_by_id:
        raise ValueError("No valid rows found in gold CSV after filtering.")
    return query_text_by_id, judgments_by_query


def dcg_at_k(labels: list[int], k: int) -> float:
    score = 0.0
    for rank, rel in enumerate(labels[:k], start=1):
        gain = (2**rel) - 1
        score += gain / math.log2(rank + 1)
    return score


def ndcg_at_k(retrieved_chunk_ids: list[str], judgments: dict[str, int], k: int) -> float:
    actual_labels = [judgments.get(cid, 0) for cid in retrieved_chunk_ids[:k]]
    ideal_labels = sorted(judgments.values(), reverse=True)[:k]
    ideal = dcg_at_k(ideal_labels, k)
    if ideal == 0.0:
        return 0.0
    return dcg_at_k(actual_labels, k) / ideal


def hit_rate_at_k(
    retrieved_chunk_ids: list[str], judgments: dict[str, int], k: int, relevant_threshold: int
) -> float:
    return 1.0 if any(judgments.get(cid, 0) >= relevant_threshold for cid in retrieved_chunk_ids[:k]) else 0.0


def reciprocal_rank_at_k(
    retrieved_chunk_ids: list[str], judgments: dict[str, int], k: int, relevant_threshold: int
) -> float:
    for rank, cid in enumerate(retrieved_chunk_ids[:k], start=1):
        if judgments.get(cid, 0) >= relevant_threshold:
            return 1.0 / rank
    return 0.0


def mean(xs: Iterable[float]) -> float:
    vals = list(xs)
    return sum(vals) / len(vals) if vals else 0.0


def evaluate(
    *,
    gold_csv: str | Path,
    k: int,
    relevant_threshold: int,
    split: str | None,
    rerank_top_k: int,
    output_predictions_csv: str | Path | None = None,
) -> dict[str, Any]:
    query_text_by_id, judgments_by_query = load_gold_csv(gold_csv, split=split)

    per_query_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []

    for query_id, query_text in query_text_by_id.items():
        out = rag_pipeline(
            query_text,
            columns=DEFAULT_COLUMNS,
            rerank_top_k=max(k, rerank_top_k),
        )

        retrieved = out.get("chunks") or []
        retrieved_chunk_ids = [str(ch.get("chunk_id")) for ch in retrieved if ch.get("chunk_id") is not None]
        judgments = judgments_by_query.get(query_id, {})

        hr = hit_rate_at_k(retrieved_chunk_ids, judgments, k, relevant_threshold)
        rr = reciprocal_rank_at_k(retrieved_chunk_ids, judgments, k, relevant_threshold)
        ndcg = ndcg_at_k(retrieved_chunk_ids, judgments, k)

        per_query_rows.append(
            {
                "query_id": query_id,
                "query_text": query_text,
                "num_gold_chunks": len(judgments),
                "retrieved_count": len(retrieved_chunk_ids),
                f"hit_rate@{k}": hr,
                f"mrr@{k}": rr,
                f"ndcg@{k}": ndcg,
                "rewritten_queries": json.dumps((out.get("query_info") or {}).get("rewritten_queries", []), ensure_ascii=False),
                "filters": json.dumps((out.get("query_info") or {}).get("filters", {}), ensure_ascii=False),
            }
        )

        for rank, ch in enumerate(retrieved[:k], start=1):
            cid = str(ch.get("chunk_id"))
            prediction_rows.append(
                {
                    "query_id": query_id,
                    "query_text": query_text,
                    "rank": rank,
                    "chunk_id": cid,
                    "company_id": ch.get("company_id"),
                    "company_name": ch.get("company_name"),
                    "gold_label": judgments.get(cid, 0),
                    "chunk_text_preview": (str(ch.get("chunk_text") or "")[:300]).replace("\n", " "),
                }
            )

    metrics = {
        "queries_evaluated": len(per_query_rows),
        f"mean_hit_rate@{k}": mean(row[f"hit_rate@{k}"] for row in per_query_rows),
        f"mean_mrr@{k}": mean(row[f"mrr@{k}"] for row in per_query_rows),
        f"mean_ndcg@{k}": mean(row[f"ndcg@{k}"] for row in per_query_rows),
        "relevant_threshold": relevant_threshold,
        "split": split or "",
    }

    if output_predictions_csv:
        with open(output_predictions_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "query_id",
                    "query_text",
                    "rank",
                    "chunk_id",
                    "company_id",
                    "company_name",
                    "gold_label",
                    "chunk_text_preview",
                ],
            )
            writer.writeheader()
            writer.writerows(prediction_rows)

    return {"metrics": metrics, "per_query": per_query_rows}

# COMMAND ----------


def _get_widget(name: str) -> str | None:
    try:
        return dbutils.widgets.get(name)  # type: ignore[name-defined]
    except Exception:
        return None


def _get_param(name: str, default: str) -> str:
    v = os.environ.get(name)
    if v is not None and str(v).strip() != "":
        return str(v).strip()
    w = _get_widget(name)
    if w is not None and str(w).strip() != "":
        return str(w).strip()
    return default


def _get_param_opt(name: str) -> str | None:
    v = _get_param(name, "")
    return v if v else None


def _get_param_int(name: str, default: int) -> int:
    return int(_get_param(name, str(default)))


GOLD_CSV = _get_param("EVAL_GOLD_CSV", "")
K = _get_param_int("EVAL_K", 10)
RELEVANT_THRESHOLD = _get_param_int("EVAL_RELEVANT_THRESHOLD", 2)
SPLIT = _get_param_opt("EVAL_SPLIT")
RERANK_TOP_K = _get_param_int("EVAL_RERANK_TOP_K", 10)
OUTPUT_PREDICTIONS_CSV = _get_param_opt("EVAL_OUTPUT_PREDICTIONS_CSV")
OUTPUT_JSON = _get_param_opt("EVAL_OUTPUT_JSON")

if not GOLD_CSV:
    raise RuntimeError(
        "Set EVAL_GOLD_CSV (env or widget) to the driver path of the gold CSV, e.g. /dbfs/FileStore/gold_set.csv"
    )

# COMMAND ----------


results = evaluate(
    gold_csv=GOLD_CSV,
    k=K,
    relevant_threshold=RELEVANT_THRESHOLD,
    split=SPLIT,
    rerank_top_k=RERANK_TOP_K,
    output_predictions_csv=OUTPUT_PREDICTIONS_CSV,
)

print(json.dumps(results["metrics"], indent=2, ensure_ascii=False))

if OUTPUT_JSON:
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
