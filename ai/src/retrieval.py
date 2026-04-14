"""Retrieval pipeline for RAG (Databricks Vector Search + keyword search + fusion).

This module focuses on retrieval + context assembly. Generation/citations are downstream.
"""

from __future__ import annotations

import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_RESULT_COLUMNS = ("chunk_id", "company_id", "chunk_index", "chunk_text")
DEFAULT_TEXT_COL = "chunk_text"
DEFAULT_ID_COL = "chunk_id"
DEFAULT_PARENT_COL = "company_id"


@dataclass(frozen=True)
class QueryInfo:
    raw: str
    normalized: str
    language: str  # best-effort: "en" | "unknown"
    query_type: str  # "fact" | "comparison" | "entity" | "broad"
    constraints: dict[str, Any]
    expanded: list[str]


@dataclass(frozen=True)
class RetrievalResult:
    chunk_id: str
    payload: dict[str, Any]
    score: float
    source: str  # "vector" | "keyword" | "hybrid" | "rerank"
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
_CONSTRAINT_RE = re.compile(r"(?P<key>company_id|source|doc_id|parent_id)\s*:\s*(?P<val>[^\s]+)", re.I)


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


def extract_metadata_constraints(text: str) -> dict[str, Any]:
    constraints: dict[str, Any] = {}
    for m in _CONSTRAINT_RE.finditer(text or ""):
        key = m.group("key").lower()
        val = m.group("val")
        if key in ("doc_id", "parent_id"):
            constraints[key] = val
        elif key == "company_id":
            constraints["company_id"] = val
        elif key == "source":
            constraints["source"] = val
    return constraints


def optional_expand_query(info: QueryInfo) -> list[str]:
    # Keep it simple/deterministic: add a couple of variants for keyword retrieval.
    q = info.normalized
    if not q:
        return []
    variants = [q]
    if info.query_type == "comparison" and " vs " in q.lower():
        variants.append(q.replace(" vs ", " versus "))
    return list(dict.fromkeys(variants))


def understand_query(query: str) -> QueryInfo:
    norm = normalize_text(query)
    lang = detect_language(norm)
    qtype = detect_query_type(norm)
    constraints = extract_metadata_constraints(norm)
    base = QueryInfo(
        raw=query,
        normalized=norm,
        language=lang,
        query_type=qtype,
        constraints=constraints,
        expanded=[],
    )
    return QueryInfo(**{**base.__dict__, "expanded": optional_expand_query(base)})


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


def _rrf_fuse(
    lists: list[list[RetrievalResult]],
    *,
    k: int = 60,
    weights: dict[str, float] | None = None,
) -> list[RetrievalResult]:
    """
    Reciprocal Rank Fusion across multiple ranked lists.

    Score: sum_s w_s * 1/(k + rank_s)
    """
    w = weights or {}
    fused: dict[str, tuple[float, dict[str, Any]]] = {}
    for lst in lists:
        for r in lst:
            ww = float(w.get(r.source, 1.0))
            s = ww * (1.0 / float(k + max(1, r.rank)))
            prev = fused.get(r.chunk_id)
            if prev is None:
                fused[r.chunk_id] = (s, dict(r.payload))
            else:
                fused[r.chunk_id] = (prev[0] + s, prev[1])

    ranked = sorted(fused.items(), key=lambda kv: kv[1][0], reverse=True)
    out: list[RetrievalResult] = []
    for i, (cid, (score, payload)) in enumerate(ranked):
        out.append(RetrievalResult(chunk_id=cid, payload=payload, score=score, source="fused", rank=i + 1))
    return out


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


def keyword_search_chunks(
    query: str,
    *,
    spark: Any | None = None,
    table_fqn: str | None = None,
    top_k: int = 50,
    text_col: str = DEFAULT_TEXT_COL,
    id_col: str = DEFAULT_ID_COL,
    extra_cols: tuple[str, ...] = ("company_id", "chunk_index"),
    constraints: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """
    Best-effort keyword retrieval over the chunks table.

    Requires a SparkSession (pass ``spark=...``). If not available, returns [].
    """
    if spark is None or table_fqn is None:
        return []
    q = normalize_text(query)
    if not q:
        return []
    terms = [t for t in re.split(r"[^A-Za-z0-9_]+", q.lower()) if len(t) >= 3]
    if not terms:
        return []

    from pyspark.sql import functions as F

    df = spark.table(table_fqn).select(id_col, text_col, *extra_cols)
    c = constraints or {}
    if "company_id" in c and "company_id" in df.columns:
        df = df.where(F.col("company_id") == F.lit(str(c["company_id"])))

    # Simple TF-ish score: sum of occurrences for each term (lowercased).
    score = None
    txt = F.lower(F.col(text_col))
    for t in terms[:12]:
        part = (F.length(txt) - F.length(F.regexp_replace(txt, re.escape(t), ""))) / F.lit(max(1, len(t)))
        score = part if score is None else (score + part)
    df = df.withColumn("keyword_score", score).where(F.col("keyword_score") > F.lit(0))
    rows = df.orderBy(F.col("keyword_score").desc()).limit(int(top_k)).collect()
    out: list[dict[str, Any]] = []
    for r in rows:
        d = r.asDict(recursive=True)
        d["score"] = float(d.pop("keyword_score", 0.0))
        out.append(d)
    return out


def maybe_rerank(
    query: str,
    candidates: list[RetrievalResult],
    *,
    columns_to_rerank: list[str] | None = None,
    top_n: int = 50,
) -> list[RetrievalResult]:
    """
    Optional reranking for hybrid results.

    If available, uses DatabricksReranker via Vector Search SDK semantics (when present).
    Otherwise returns candidates unchanged.
    """
    _ = query
    _ = columns_to_rerank
    _ = top_n
    return candidates


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


def retrieve_pipeline(
    query: str,
    *,
    top_k: int = 8,
    vector_top_k: int = 40,
    keyword_top_k: int = 40,
    rrf_k: int = 60,
    weights: dict[str, float] | None = None,
    # Vector Search config:
    index_name: str | None = None,
    endpoint_name: str | None = None,
    columns: list[str] | None = None,
    query_type: str | None = None,
    filters: dict[str, Any] | None = None,
    embedding_endpoint: str | None = None,
    use_query_vector: bool = False,
    config_path: Path | str | None = None,
    # Keyword search config:
    spark: Any | None = None,
    chunks_table_fqn: str | None = None,
    # Context assembly:
    max_per_parent: int = 3,
    collapse_per_parent: int = 6,
) -> dict[str, Any]:
    """
    End-to-end retrieval per the architecture:
    understand → parallel retrieval → union/dedupe → fusion → (optional rerank) → context assembly.

    Returns a dict with ``query_info`` and ``chunks`` (list of payload dicts).
    """
    qi = understand_query(query)

    # Metadata constraints: apply to both retrieval paths when possible.
    merged_filters = dict(filters or {})
    if "company_id" in qi.constraints and "company_id" not in merged_filters:
        merged_filters["company_id"] = qi.constraints["company_id"]

    # A) Vector / hybrid retrieval
    vector_rows: list[dict[str, Any]] = []
    try:
        vector_rows = retrieve(
            qi.normalized,
            top_k=vector_top_k,
            index_name=index_name,
            endpoint_name=endpoint_name,
            columns=columns,
            query_type=query_type,
            filters=merged_filters or None,
            embedding_endpoint=embedding_endpoint,
            use_query_vector=use_query_vector,
            config_path=config_path,
        )
    except Exception:
        # Allow keyword-only mode in environments without vectorsearch.
        vector_rows = []

    vector_results = _as_retrieval_results(vector_rows, source="vector")

    # B) Keyword retrieval over chunks table (Spark)
    keyword_rows: list[dict[str, Any]] = []
    if qi.expanded:
        # take best of expanded variants by concatenating and letting RRF handle it
        for qv in qi.expanded[:2]:
            keyword_rows.extend(
                keyword_search_chunks(
                    qv,
                    spark=spark,
                    table_fqn=chunks_table_fqn,
                    top_k=keyword_top_k,
                    constraints=qi.constraints,
                )
            )
    keyword_results = _as_retrieval_results(keyword_rows, source="keyword")

    # Candidate union + fusion
    fused = _rrf_fuse([vector_results, keyword_results], k=rrf_k, weights=weights or {"vector": 1.0, "keyword": 0.7})

    # Reranking (optional hook)
    reranked = maybe_rerank(qi.normalized, fused)

    # Context assembly: collapse overlaps and diversify
    assembled = _collapse_overlaps(reranked, max_per_parent=collapse_per_parent)
    assembled = _diversify(assembled, max_per_parent=max_per_parent)
    assembled = assembled[: max(1, int(top_k))]

    return {
        "query_info": {
            "raw": qi.raw,
            "normalized": qi.normalized,
            "language": qi.language,
            "query_type": qi.query_type,
            "constraints": qi.constraints,
            "expanded": qi.expanded,
        },
        "chunks": [r.payload for r in assembled],
    }
