"""End-to-end RAG pipeline.

Architecture:
User query
  ↓
Query understanding:
 - normalize text
 - detect language
 - detect query type
 - rewrite/expansion (LLM) + extract metadata constraints
  ↓
Hybrid retrieval (Databricks Vector Search index)
  ↓
Top 50 results
  ↓
Reranker (LLM or cross-encoder)
  ↓
Top 5–10 chunks
  ↓
LLM answer (with citation)
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal


DEFAULT_RESULT_COLUMNS = ("chunk_id", "company_id", "chunk_index", "chunk_text")
DEFAULT_TEXT_COL = "chunk_text"
DEFAULT_ID_COL = "chunk_id"
DEFAULT_PARENT_COL = "company_id"

DEFAULT_REWRITE_MODEL = os.environ.get("DATABRICKS_QUERY_UNDERSTANDING_ENDPOINT") or os.environ.get(
    "DATABRICKS_LLM_ENDPOINT"
)
DEFAULT_RERANK_MODEL = os.environ.get( "DATABRICKS_RERANKER_ENDPOINT")
DEFAULT_ANSWER_MODEL = os.environ.get("DATABRICKS_ANSWER_ENDPOINT") or os.environ.get("DATABRICKS_LLM_ENDPOINT")

QUERY_UNDERSTANDING_PROMPT = """You are a query understanding system for a RAG search engine.

Your job is to:
1. Rewrite the query for better retrieval
2. Extract structured filters for metadata search

Schema:
- country
- city
- region
- theme
- main_category
- subcategory
- deal_type
- deal_year
- funding_type (e.g. debt, equity)
- investor_name

Rules:
- Only extract a field if explicitly or strongly implied
- Use normalized values (e.g. "UK" → "United Kingdom")
- Keep rewritten queries diverse (keyword + semantic)
- Do NOT hallucinate filters

Return JSON with:
- "queries": list of 3–5 rewritten queries
- "filters": object with extracted fields

User query:
"{query}"
"""


@dataclass(frozen=True)
class QueryInfo:
    raw: str
    normalized: str
    language: str  # best-effort: "en" | "unknown"
    query_type: str  # "fact" | "comparison" | "entity" | "broad"
    rewritten_queries: list[str]
    filters: dict[str, Any]


@dataclass(frozen=True)
class RetrievalResult:
    chunk_id: str
    payload: dict[str, Any]
    score: float
    source: str  # "hybrid" | "rerank"
    rank: int


def _config_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "config"


def _load_vector_index_yaml(path: Path | None = None) -> dict[str, Any]:
    p = path or (_config_dir() / "vector_index.yml")
    if not p.is_file():
        return {}
    try:
        import yaml
    except ImportError:
        return {}
    with p.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def _merge_vs_settings(
    *,
    endpoint_name: str | None,
    index_name: str | None,
    config_path: Path | str | None,
) -> tuple[str | None, str | None]:
    cfg = _load_vector_index_yaml(
        Path(config_path) if config_path else None
    )
    vi = cfg.get("vector_index") if isinstance(cfg.get("vector_index"), dict) else {}

    ep = (
        endpoint_name
        or os.environ.get("DATABRICKS_VECTOR_SEARCH_ENDPOINT_NAME")
        or os.environ.get("DATABRICKS_VS_ENDPOINT_NAME")
        or vi.get("endpoint_name")
    )
    idx = (
        index_name
        or os.environ.get("DATABRICKS_VECTOR_SEARCH_INDEX_NAME")
        or os.environ.get("DATABRICKS_VS_INDEX_NAME")
        or vi.get("index_name")
    )
    if isinstance(ep, str) and not ep.strip():
        ep = None
    if isinstance(idx, str) and not idx.strip():
        idx = None
    return (ep, idx)


_SPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    s = unicodedata.normalize("NFKC", text or "")
    s = s.replace("\u00a0", " ")
    s = s.strip()
    s = _SPACE_RE.sub(" ", s)
    return s


def detect_language(text: str) -> str:
    # Lightweight heuristic: ASCII-heavy → "en", else "unknown".
    # (Avoids adding a hard dependency like langdetect.)
    s = text or ""
    if not s:
        return "unknown"
    ascii_ratio = sum(1 for ch in s if ord(ch) < 128) / max(1, len(s))
    return "en" if ascii_ratio > 0.9 else "unknown"


def detect_query_type(text: str) -> str:
    t = (text or "").lower()
    if any(w in t for w in (" vs ", "versus", "compare", "difference between")):
        return "comparison"
    if t.startswith(("who ", "what ", "when ", "where ", "why ", "how ")):
        return "fact"
    if len(t.split()) <= 3:
        return "entity"
    return "broad"


def _json_extract_loose(text: str) -> dict[str, Any] | None:
    """
    Best-effort JSON extractor for model outputs.

    Accepts:
    - raw JSON object
    - fenced ```json ... ```
    - text containing a single top-level JSON object
    """
    if not text:
        return None
    s = text.strip()
    if s.startswith("```"):
        # Strip a single fenced block, keeping its content.
        s = re.sub(r"^\s*```[a-zA-Z0-9_-]*\s*", "", s)
        s = re.sub(r"\s*```\s*$", "", s)
        s = s.strip()
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    # Try to locate the first JSON object in the string.
    start = s.find("{")
    end = s.rfind("}")
    if start >= 0 and end > start:
        candidate = s[start : end + 1]
        try:
            obj = json.loads(candidate)
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None
    return None


def _normalize_rewrite_payload(raw: Any) -> tuple[list[str], dict[str, Any]]:
    """
    Normalize model output into (queries, filters).
    """
    if raw is None:
        return ([], {})
    if isinstance(raw, dict):
        queries = raw.get("queries")
        filters = raw.get("filters")
        q_out = [normalize_text(str(q)) for q in queries] if isinstance(queries, list) else []
        q_out = [q for q in q_out if q]
        f_out = filters if isinstance(filters, dict) else {}
        return (q_out, f_out)
    return ([], {})


def _call_databricks_llm(endpoint: str, *, messages: list[dict[str, str]], temperature: float = 0.0) -> str:
    """
    Call a Databricks model serving chat endpoint via MLflow deployments client.

    This intentionally supports multiple response shapes used by different serving backends.
    """
    from mlflow.deployments import get_deploy_client

    client = get_deploy_client("databricks")
    resp = client.predict(endpoint=endpoint, inputs={"messages": messages, "temperature": float(temperature)})

    # Common shapes:
    # - OpenAI-like: {"choices":[{"message":{"content":"..."}}], ...}
    # - Predictions: {"predictions":[{"content":"..."}]} or {"predictions":["..."]}
    # - Data: {"data":[...]}
    if isinstance(resp, dict):
        if isinstance(resp.get("choices"), list) and resp["choices"]:
            msg = resp["choices"][0].get("message") if isinstance(resp["choices"][0], dict) else None
            if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                return msg["content"]
        preds = resp.get("predictions")
        if isinstance(preds, list) and preds:
            p0 = preds[0]
            if isinstance(p0, dict) and isinstance(p0.get("content"), str):
                return p0["content"]
            if isinstance(p0, str):
                return p0
        data = resp.get("data")
        if isinstance(data, list) and data:
            d0 = data[0]
            if isinstance(d0, dict) and isinstance(d0.get("content"), str):
                return d0["content"]
            if isinstance(d0, str):
                return d0
    if isinstance(resp, str):
        return resp
    return json.dumps(resp, ensure_ascii=False)


def understand_query(query: str) -> QueryInfo:
    norm = normalize_text(query)
    lang = detect_language(norm)
    qtype = detect_query_type(norm)
    rewritten_queries: list[str] = []
    filters: dict[str, Any] = {}

    # LLM rewrite + metafilters (optional; falls back safely).
    if DEFAULT_REWRITE_MODEL and norm:
        prompt = QUERY_UNDERSTANDING_PROMPT.format(query=norm)
        content = _call_databricks_llm(
            DEFAULT_REWRITE_MODEL,
            messages=[
                {"role": "system", "content": "Return only valid JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        parsed = _json_extract_loose(content)
        q_out, f_out = _normalize_rewrite_payload(parsed)
        rewritten_queries = q_out
        filters = f_out

    # Always include the normalized query as a backstop.
    if norm:
        rewritten_queries = [*rewritten_queries, norm]
    # Dedupe while preserving order and bound to 5.
    seen: set[str] = set()
    deduped: list[str] = []
    for q in rewritten_queries:
        qq = normalize_text(q)
        if not qq or qq in seen:
            continue
        seen.add(qq)
        deduped.append(qq)
        if len(deduped) >= 5:
            break

    return QueryInfo(
        raw=query,
        normalized=norm,
        language=lang,
        query_type=qtype,
        rewritten_queries=deduped,
        filters=filters,
    )


def embed_query(
    query: str,
    *,
    endpoint: str | None = None,
    batch_size: int | None = None,
) -> list[float]:
    """Embed a single query string (same endpoint semantics as ``embedding.embed_texts``)."""
    try:
        from .embedding import embed_texts
    except ImportError:
        from embedding import embed_texts

    vecs = embed_texts([query], endpoint=endpoint, batch_size=batch_size)
    if not vecs:
        raise RuntimeError("Embedding endpoint returned no vector for the query.")
    return vecs[0]


def _normalize_similarity_hits(raw: Any) -> list[dict[str, Any]]:
    """Turn Vector Search ``similarity_search`` payloads into plain dict rows."""
    if raw is None:
        return []
    if isinstance(raw, list):
        out: list[dict[str, Any]] = []
        for row in raw:
            if isinstance(row, dict):
                out.append(dict(row))
        return out
    if not isinstance(raw, dict):
        return []

    # Common SDK shapes: { "result": { "data_array": [...] } } or flat list under "data"
    result = raw.get("result", raw)
    if isinstance(result, list):
        return _normalize_similarity_hits(result)

    data_array = result.get("data_array")
    columns = result.get("columns") or result.get("column_names")
    if isinstance(data_array, list) and isinstance(columns, list) and columns:
        rows: list[dict[str, Any]] = []
        for tup in data_array:
            if not isinstance(tup, (list, tuple)):
                continue
            rows.append({str(columns[i]): tup[i] for i in range(min(len(columns), len(tup)))})
        return rows

    # Single "rows" or "data" list of dicts
    for key in ("rows", "data", "matches", "documents"):
        chunk = result.get(key)
        if isinstance(chunk, list):
            return _normalize_similarity_hits(chunk)

    return []


def _safe_str(x: Any) -> str:
    if x is None:
        return ""
    return str(x)


def _get_first(row: dict[str, Any], keys: Iterable[str]) -> Any:
    for k in keys:
        if k in row and row[k] is not None:
            return row[k]
    return None


def _as_retrieval_results(
    rows: list[dict[str, Any]],
    *,
    source: str,
    id_col: str = DEFAULT_ID_COL,
    score_keys: tuple[str, ...] = ("score", "distance", "_distance", "similarity"),
) -> list[RetrievalResult]:
    out: list[RetrievalResult] = []
    for i, r in enumerate(rows):
        cid = _get_first(r, (id_col, "chunk_id", "id", "pk"))
        if cid is None:
            continue
        raw_score = _get_first(r, score_keys)
        score = float(raw_score) if raw_score is not None else 0.0
        out.append(
            RetrievalResult(
                chunk_id=_safe_str(cid),
                payload=dict(r),
                score=score,
                source=source,
                rank=i + 1,
            )
        )
    return out


def _dedupe_by_id_keep_best(
    results: list[RetrievalResult],
    *,
    prefer_source: str | None = None,
) -> list[RetrievalResult]:
    """
    Dedupe results by chunk_id, keeping the highest score (tie-breaker: prefer_source).
    """
    best: dict[str, RetrievalResult] = {}
    for r in results:
        prev = best.get(r.chunk_id)
        if prev is None:
            best[r.chunk_id] = r
            continue
        if r.score > prev.score:
            best[r.chunk_id] = r
        elif r.score == prev.score and prefer_source and r.source == prefer_source and prev.source != prefer_source:
            best[r.chunk_id] = r
    out = sorted(best.values(), key=lambda x: x.score, reverse=True)
    return [RetrievalResult(**{**r.__dict__, "rank": i + 1}) for i, r in enumerate(out)]


def _collapse_overlaps(
    results: list[RetrievalResult],
    *,
    parent_col: str = DEFAULT_PARENT_COL,
    index_col: str = "chunk_index",
    max_per_parent: int = 6,
) -> list[RetrievalResult]:
    """
    Best-effort overlap collapse: keep a limited number of chunks per parent (company/doc),
    preferring higher fused scores and spreading across chunk_index.
    """
    buckets: dict[str, list[RetrievalResult]] = {}
    for r in results:
        parent = _get_first(r.payload, (parent_col, "parent_id", "doc_id"))
        p = _safe_str(parent) or "__unknown__"
        buckets.setdefault(p, []).append(r)

    out: list[RetrievalResult] = []
    for _, lst in buckets.items():
        lst_sorted = sorted(lst, key=lambda x: x.score, reverse=True)
        picked: list[RetrievalResult] = []
        used_idx: set[int] = set()
        for r in lst_sorted:
            idx = _get_first(r.payload, (index_col,))
            try:
                ii = int(idx) if idx is not None else None
            except Exception:
                ii = None
            if ii is not None and ii in used_idx:
                continue
            picked.append(r)
            if ii is not None:
                used_idx.add(ii)
            if len(picked) >= max_per_parent:
                break
        out.extend(picked)

    # preserve global ordering by score
    out = sorted(out, key=lambda x: x.score, reverse=True)
    for i, r in enumerate(out):
        out[i] = RetrievalResult(**{**r.__dict__, "rank": i + 1})
    return out


def _diversify(
    results: list[RetrievalResult],
    *,
    parent_col: str = DEFAULT_PARENT_COL,
    max_per_parent: int = 3,
) -> list[RetrievalResult]:
    out: list[RetrievalResult] = []
    counts: dict[str, int] = {}
    for r in results:
        parent = _safe_str(_get_first(r.payload, (parent_col, "parent_id", "doc_id")) or "__unknown__")
        n = counts.get(parent, 0)
        if n >= max_per_parent:
            continue
        counts[parent] = n + 1
        out.append(r)
    for i, r in enumerate(out):
        out[i] = RetrievalResult(**{**r.__dict__, "rank": i + 1})
    return out


def rerank_candidates(
    query: str,
    candidates: list[RetrievalResult],
    *,
    top_n: int = 50,
    rerank_top_k: int = 10,
    rerank_model_endpoint: str | None = None,
    text_col: str = DEFAULT_TEXT_COL,
) -> list[RetrievalResult]:
    """
    Rerank the top-N candidates and return top rerank_top_k.

    Strategy:
    - If a rerank endpoint is provided, call it (assumed cross-encoder-like).
    - Otherwise, fall back to an LLM scoring prompt if an answer model is configured.
    - If neither is available, return the original ranking (truncated).
    """
    if not candidates:
        return []

    top_n = max(1, int(top_n))
    rerank_top_k = max(1, int(rerank_top_k))
    subset = candidates[:top_n]

    ep = rerank_model_endpoint or DEFAULT_RERANK_MODEL
    if ep:
        # Best-effort: assume the reranker endpoint accepts {"query":..., "documents":[...]} and returns scores.
        from mlflow.deployments import get_deploy_client

        client = get_deploy_client("databricks")
        docs = [normalize_text(_safe_str(r.payload.get(text_col))) for r in subset]
        resp = client.predict(endpoint=ep, inputs={"query": query, "documents": docs})
        scores: list[float] = []
        if isinstance(resp, dict) and isinstance(resp.get("predictions"), list):
            preds = resp["predictions"]
            if preds and isinstance(preds[0], dict) and "score" in preds[0]:
                scores = [float(p.get("score", 0.0)) for p in preds]
            elif preds and isinstance(preds[0], (int, float)):
                scores = [float(x) for x in preds]
        elif isinstance(resp, list) and resp and isinstance(resp[0], (int, float)):
            scores = [float(x) for x in resp]

        if len(scores) == len(subset):
            reranked = [
                RetrievalResult(
                    chunk_id=r.chunk_id,
                    payload=r.payload,
                    score=float(scores[i]),
                    source="rerank",
                    rank=i + 1,
                )
                for i, r in enumerate(subset)
            ]
            reranked = sorted(reranked, key=lambda x: x.score, reverse=True)
            reranked = [RetrievalResult(**{**r.__dict__, "rank": i + 1}) for i, r in enumerate(reranked)]
            return reranked[:rerank_top_k]

    # LLM fallback scoring
    if DEFAULT_ANSWER_MODEL:
        prompt_lines = [
            "Score each passage for relevance to the query on a 0-100 scale.",
            "Return ONLY JSON: {\"scores\": [..]} with one score per passage in order.",
            "",
            f"Query: {query}",
            "",
        ]
        for i, r in enumerate(subset, start=1):
            txt = normalize_text(_safe_str(r.payload.get(text_col)))
            txt = txt[:2500]  # safety bound
            prompt_lines.append(f"Passage {i}:\n{txt}\n")
        content = _call_databricks_llm(
            DEFAULT_ANSWER_MODEL,
            messages=[
                {"role": "system", "content": "You are a careful ranking system."},
                {"role": "user", "content": "\n".join(prompt_lines)},
            ],
            temperature=0.0,
        )
        parsed = _json_extract_loose(content) or {}
        raw_scores = parsed.get("scores")
        if isinstance(raw_scores, list) and len(raw_scores) == len(subset):
            try:
                scores = [float(x) for x in raw_scores]
                reranked = [
                    RetrievalResult(
                        chunk_id=r.chunk_id,
                        payload=r.payload,
                        score=float(scores[i]),
                        source="rerank",
                        rank=i + 1,
                    )
                    for i, r in enumerate(subset)
                ]
                reranked = sorted(reranked, key=lambda x: x.score, reverse=True)
                reranked = [RetrievalResult(**{**r.__dict__, "rank": i + 1}) for i, r in enumerate(reranked)]
                return reranked[:rerank_top_k]
            except Exception:
                pass

    return subset[:rerank_top_k]


def retrieve(
    query: str,
    top_k: int = 5,
    *,
    index_name: str | None = None,
    endpoint_name: str | None = None,
    columns: list[str] | None = None,
    query_type: str | None = None,
    filters: dict[str, Any] | None = None,
    embedding_endpoint: str | None = None,
    use_query_vector: bool = False,
    config_path: Path | str | None = None,
) -> list[dict[str, Any]]:
    """
    Return top-``top_k`` chunk rows for ``query`` via Databricks Vector Search.

    **Setup**

    - Install ``databricks-vectorsearch`` on the cluster / environment.
    - Set ``DATABRICKS_VECTOR_SEARCH_ENDPOINT_NAME`` and
      ``DATABRICKS_VECTOR_SEARCH_INDEX_NAME`` (Unity Catalog index FQN), *or*
      fill ``vector_index.endpoint_name`` / ``vector_index.index_name`` in
      ``ai/config/vector_index.yml``.

    **Query modes**

    - Default: ``query_text=query`` (Delta Sync index with model endpoint, or hybrid).
    - ``use_query_vector=True``: embed ``query`` with ``embedding.embed_texts`` and call
      ``similarity_search`` with ``query_vector`` (precomputed-embedding indexes).

    Environment (optional): ``DATABRICKS_EMBEDDING_ENDPOINT``, ``EMBED_BATCH_SIZE`` —
    same as ``embedding.embed_texts``.
    """
    ep, idx = _merge_vs_settings(
        endpoint_name=endpoint_name,
        index_name=index_name,
        config_path=config_path,
    )
    if not idx:
        raise ValueError(
            "Vector Search index is not configured. Set "
            "DATABRICKS_VECTOR_SEARCH_INDEX_NAME (and typically "
            "DATABRICKS_VECTOR_SEARCH_ENDPOINT_NAME), or add "
            "vector_index.index_name / vector_index.endpoint_name to ai/config/vector_index.yml."
        )

    try:
        from databricks.vector_search.client import VectorSearchClient
    except ImportError as e:
        raise ImportError(
            "Retrieval requires the databricks-vectorsearch package. "
            "On a cluster: %pip install databricks-vectorsearch"
        ) from e

    client = VectorSearchClient()
    if ep:
        index = client.get_index(endpoint_name=ep, index_name=idx)
    else:
        index = client.get_index(index_name=idx)

    cols = list(columns) if columns is not None else list(DEFAULT_RESULT_COLUMNS)
    kwargs: dict[str, Any] = {
        "columns": cols,
        "num_results": top_k,
    }
    if filters is not None:
        kwargs["filters"] = filters
    if query_type is not None:
        kwargs["query_type"] = query_type

    if use_query_vector:
        kwargs["query_vector"] = embed_query(query, endpoint=embedding_endpoint)
    else:
        kwargs["query_text"] = query

    raw = index.similarity_search(**kwargs)
    return _normalize_similarity_hits(raw)


def hybrid_retrieve_top50(
    qi: QueryInfo,
    *,
    index_name: str | None = None,
    endpoint_name: str | None = None,
    columns: list[str] | None = None,
    query_type: str | None = None,
    embedding_endpoint: str | None = None,
    use_query_vector: bool = False,
    config_path: Path | str | None = None,
    extra_filters: dict[str, Any] | None = None,
    per_query_top_k: int = 50,
) -> list[RetrievalResult]:
    """
    Hybrid retrieval against the Databricks Vector Search index for each rewritten query.

    Returns a single deduped ranked list of up to ~ (per_query_top_k * num_queries) candidates,
    sorted by the index-provided score.
    """
    merged_filters = dict(extra_filters or {})
    merged_filters.update(qi.filters or {})

    all_rows: list[dict[str, Any]] = []
    for q in qi.rewritten_queries[:5] if qi.rewritten_queries else [qi.normalized]:
        if not q:
            continue
        rows = retrieve(
            q,
            top_k=int(per_query_top_k),
            index_name=index_name,
            endpoint_name=endpoint_name,
            columns=columns,
            query_type=query_type,
            filters=merged_filters or None,
            embedding_endpoint=embedding_endpoint,
            use_query_vector=use_query_vector,
            config_path=config_path,
        )
        # Preserve which rewrite produced it (useful for debugging)
        for r in rows:
            if isinstance(r, dict):
                r.setdefault("_rewrite_query", q)
        all_rows.extend(rows)

    results = _as_retrieval_results(all_rows, source="hybrid")
    results = _dedupe_by_id_keep_best(results, prefer_source="hybrid")
    return results


def answer_with_citations(
    query: str,
    chunks: list[dict[str, Any]],
    *,
    answer_model_endpoint: str | None = None,
    text_col: str = DEFAULT_TEXT_COL,
    id_col: str = DEFAULT_ID_COL,
    max_context_chars: int = 14000,
) -> str:
    """
    Produce a final answer that includes citations of the form [chunk_id].
    """
    ep = answer_model_endpoint or DEFAULT_ANSWER_MODEL
    if not ep:
        return ""

    context_blocks: list[str] = []
    used = 0
    for ch in chunks:
        cid = normalize_text(_safe_str(ch.get(id_col) or ch.get("chunk_id") or ch.get("id"))) or "unknown"
        txt = normalize_text(_safe_str(ch.get(text_col)))
        if not txt:
            continue
        block = f"[{cid}]\n{txt}"
        if used + len(block) > max_context_chars:
            break
        context_blocks.append(block)
        used += len(block)

    prompt = "\n\n".join(
        [
            "Answer the question using ONLY the provided passages.",
            "When you use a passage, cite it using its bracketed id like [chunk_id].",
            "If the passages do not contain the answer, say you don't have enough information.",
            "",
            f"Question: {query}",
            "",
            "Passages:",
            "\n\n".join(context_blocks) if context_blocks else "(none)",
        ]
    )

    return _call_databricks_llm(
        ep,
        messages=[
            {"role": "system", "content": "You are a careful RAG assistant that cites sources."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    ).strip()


def rag_pipeline(
    query: str,
    *,
    # Retrieval config
    index_name: str | None = None,
    endpoint_name: str | None = None,
    columns: list[str] | None = None,
    query_type: str | None = None,
    filters: dict[str, Any] | None = None,
    embedding_endpoint: str | None = None,
    use_query_vector: bool = False,
    config_path: Path | str | None = None,
    # Rerank config
    rerank_top_n: int = 50,
    rerank_top_k: int = 10,
    rerank_model_endpoint: str | None = None,
    # Context assembly
    collapse_per_parent: int = 6,
    max_per_parent: int = 3,
    # Answer config
    answer_model_endpoint: str | None = None,
) -> dict[str, Any]:
    """
    End-to-end pipeline matching the requested architecture.
    """
    qi = understand_query(query)

    # 1) Hybrid retrieval from index (candidates)
    candidates = hybrid_retrieve_top50(
        qi,
        index_name=index_name,
        endpoint_name=endpoint_name,
        columns=columns,
        query_type=query_type,
        embedding_endpoint=embedding_endpoint,
        use_query_vector=use_query_vector,
        config_path=config_path,
        extra_filters=filters,
        per_query_top_k=50,
    )

    # 2) Rerank and select top 5–10 chunks
    reranked = rerank_candidates(
        qi.normalized,
        candidates,
        top_n=rerank_top_n,
        rerank_top_k=rerank_top_k,
        rerank_model_endpoint=rerank_model_endpoint,
    )

    assembled = _collapse_overlaps(reranked, max_per_parent=collapse_per_parent)
    assembled = _diversify(assembled, max_per_parent=max_per_parent)
    final_chunks = assembled[: max(5, min(10, int(rerank_top_k)))]

    # 3) Answer with citations
    answer = answer_with_citations(
        qi.normalized,
        [r.payload for r in final_chunks],
        answer_model_endpoint=answer_model_endpoint,
    )

    return {
        "query_info": {
            "raw": qi.raw,
            "normalized": qi.normalized,
            "language": qi.language,
            "query_type": qi.query_type,
            "rewritten_queries": qi.rewritten_queries,
            "filters": qi.filters,
        },
        "candidates_top50": [r.payload for r in candidates[:50]],
        "chunks": [r.payload for r in final_chunks],
        "answer": answer,
    }


# Backwards-compatible alias (old name returned only chunks + query_info).
def retrieve_pipeline(query: str, **kwargs: Any) -> dict[str, Any]:
    out = rag_pipeline(query, **kwargs)
    return {"query_info": out.get("query_info", {}), "chunks": out.get("chunks", [])}

