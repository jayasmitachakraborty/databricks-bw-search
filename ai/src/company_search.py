"""
Map vector/RAG chunk payloads to the `CompanyResult` shape expected by the UI.

Requires requesting these columns in ``rag_pipeline(..., columns=RETRIEVAL_UI_COLUMNS)``.
"""

from __future__ import annotations

import math
from typing import Any

# Columns to request from Vector Search (gold_rag_company_chunks + search score from the index).
# Do not list synthetic fields like the relevance score; it is returned by similarity_search.
RETRIEVAL_UI_COLUMNS: list[str] = [
    "chunk_id",
    "company_id",
    "chunk_index",
    "chunk_text",
    "company_name",
    "website",
    "theme",
    "main_category",
    "subcategory",
    "year_founded",
    "country",
    "region",
    "city",
    "noa_funding_round",
    "deal_type",
    "deal_type_2",
    "raised_to_date",
    "invested_equity",
    "deal_size",
    "keywords_text",
    "investor_names",
    "best_investor_rank",
]


def _as_float(x: Any) -> float | None:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return None
        return float(x)
    try:
        return float(str(x).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def _as_int(x: Any) -> int | None:
    f = _as_float(x)
    if f is None:
        return None
    return int(f)


def _format_investors_for_ui(raw: Any) -> str:
    s = (raw if raw is not None else "") or ""
    s = str(s).strip()
    if not s:
        return ""
    # Table uses comma-separated list; source data often uses " | ".
    if " | " in s:
        s = ", ".join(p.strip() for p in s.split("|") if p.strip())
    return s


def _investor_ranking_ui(raw: Any) -> str:
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return "Undefined"
    if isinstance(raw, (int, float)):
        n = int(raw) if not isinstance(raw, float) or raw == int(raw) else float(raw)
        if isinstance(n, int):
            if n <= 2:
                return "High"
            if n <= 5:
                return "Medium"
            return "Low"
        if raw <= 2.0:
            return "High"
        if raw <= 5.0:
            return "Medium"
        return "Low"
    t = str(raw).strip()
    if not t:
        return "Undefined"
    low = t.lower()
    for label in ("High", "Medium", "Low", "Undefined"):
        if low == label.lower():
            return label
    return t[:64]


def _latest_deal_type(row: dict[str, Any]) -> str:
    a = row.get("noa_funding_round")
    b = row.get("deal_type")
    for v in (a, b):
        if v is not None and str(v).strip():
            return str(v).strip()
    return ""


def chunk_row_to_company_result(row: dict[str, Any], *, default_score: float = 0.0) -> dict[str, Any]:
    """
    One chunk row (Vector Search / ``rag_pipeline`` payload) -> CompanyResult dict for the UI.
    """
    score = _as_float(row.get("score"))
    if score is None:
        score = default_score

    total = _as_float(row.get("raised_to_date"))
    if total is None:
        total = _as_float(row.get("deal_size"))
    if total is None:
        total = _as_float(row.get("invested_equity"))

    return {
        "score": float(score),
        "company_id": (str(row.get("company_id")).strip() if row.get("company_id") is not None else None),
        "company_name": (str(row.get("company_name") or "").strip() or None),
        "website": (str(row.get("website") or "").strip() or None),
        "linkedin": None,
        "theme": (str(row.get("theme") or "").strip() or None),
        "main_category": (str(row.get("main_category") or "").strip() or None),
        "subcategory": (str(row.get("subcategory") or "").strip() or None),
        "country": (str(row.get("country") or "").strip() or None),
        "region": (str(row.get("region") or "").strip() or None),
        "year_founded": _as_int(row.get("year_founded")),
        "total_equity_raised": (int(total) if total is not None else None),
        "latest_deal_type": _latest_deal_type(row) or None,
        "keywords": (str(row.get("keywords_text") or "").strip() or None),
        "preferred_investors": _format_investors_for_ui(row.get("investor_names")) or None,
        "investor_ranking": _investor_ranking_ui(row.get("best_investor_rank")),
        "description": (str(row.get("chunk_text") or "").strip() or None),
    }


def dedupe_companies_by_score(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Keep the highest-scoring chunk per ``company_id`` (fallback key: ``chunk_id``).
    """
    best: dict[str, dict[str, Any]] = {}
    order: list[str] = []

    for r in rows:
        if not isinstance(r, dict):
            continue
        cid = r.get("company_id")
        key = str(cid).strip() if cid is not None and str(cid).strip() else str(r.get("chunk_id") or "")
        if not key:
            key = f"row:{id(r)}"
        if key not in order:
            order.append(key)
        s = _as_float(r.get("score")) or 0.0
        prev = best.get(key)
        if prev is None:
            best[key] = r
            continue
        ps = _as_float(prev.get("score")) or 0.0
        if s > ps:
            best[key] = r

    return [best[k] for k in order if k in best]


def run_company_table_search(
    query: str,
    *,
    config_path: str | None = None,
) -> dict[str, Any]:
    """
    Run RAG retrieval and return ``companies`` rows for the investment table (JSON-serializable).
    """
    from retrieval import rag_pipeline  # same package when ai/src is on sys.path

    out = rag_pipeline(
        query,
        columns=RETRIEVAL_UI_COLUMNS,
        config_path=config_path,
        skip_answer=True,
    )

    raw_chunks: list[dict[str, Any]] = list(out.get("chunks") or [])
    raw_chunks = dedupe_companies_by_score(raw_chunks)
    companies = [chunk_row_to_company_result(r) for r in raw_chunks]
    return {
        "query_info": out.get("query_info", {}),
        "companies": companies,
    }
