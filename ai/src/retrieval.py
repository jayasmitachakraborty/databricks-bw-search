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
Top-K results (hybrid retrieval per rewrite)
  ↓
Optional Vector Search reranker, then optional second-stage rerank (endpoint or LLM)
  ↓
Top 5–10 chunks
  ↓
LLM answer (with citation)
"""

from __future__ import annotations

import base64
import json
import os
import re
import unicodedata
import urllib.error
import urllib.request
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote


DEFAULT_RESULT_COLUMNS = ("chunk_id", "company_id", "chunk_index", "chunk_text")
DEFAULT_TEXT_COL = "chunk_text"
DEFAULT_ID_COL = "chunk_id"
DEFAULT_PARENT_COL = "company_id"

DEFAULT_REWRITE_MODEL = os.environ.get("DATABRICKS_QUERY_UNDERSTANDING_ENDPOINT") or os.environ.get(
    "DATABRICKS_LLM_ENDPOINT"
)
DEFAULT_RERANK_MODEL = os.environ.get("DATABRICKS_RERANK_ENDPOINT")
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


def _vector_index_section(config_path: Path | str | None) -> dict[str, Any]:
    cfg = _load_vector_index_yaml(Path(config_path) if config_path else None)
    vi = cfg.get("vector_index")
    return vi if isinstance(vi, dict) else {}


def _merge_vs_settings(
    *,
    endpoint_name: str | None,
    index_name: str | None,
    config_path: Path | str | None,
    vi: dict[str, Any] | None = None,
) -> tuple[str | None, str | None]:
    vi = vi if vi is not None else _vector_index_section(config_path)

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


def _resolve_vector_search_rerank_columns(
    rerank_columns: list[str] | None,
    *,
    config_path: Path | str | None,
    vi: dict[str, Any] | None = None,
) -> list[str] | None:
    """
    Columns for Mosaic AI Vector Search built-in reranker (`DatabricksReranker`).

    - ``rerank_columns`` not ``None``: use as-is (empty list = disable built-in reranker).
    - ``rerank_columns`` is ``None``: read ``DATABRICKS_VS_RERANK_COLUMNS`` (comma-separated)
      or ``vector_index.rerank_columns`` in ``vector_index.yml``.
    """
    if rerank_columns is not None:
        return list(rerank_columns) if rerank_columns else None

    raw = os.environ.get("DATABRICKS_VS_RERANK_COLUMNS", "").strip()
    if raw:
        return [c.strip() for c in raw.split(",") if c.strip()]

    vi = vi if vi is not None else _vector_index_section(config_path)
    rc = vi.get("rerank_columns")
    if isinstance(rc, list) and rc:
        return [str(x).strip() for x in rc if str(x).strip()]
    return None


def _env_truthy(name: str) -> bool | None:
    raw = os.environ.get(name, "").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    return None


def _resolve_use_query_vector(
    explicit: bool | None,
    *,
    config_path: Path | str | None,
    vi: dict[str, Any] | None = None,
) -> bool:
    """
    Whether to embed the query client-side and pass ``query_vector`` (vs ``query_text``).

    Resolution order: explicit arg → ``DATABRICKS_USE_QUERY_VECTOR`` →
    ``vector_index.use_query_vector`` in YAML → default ``False``.
    """
    if explicit is not None:
        return bool(explicit)
    ev = _env_truthy("DATABRICKS_USE_QUERY_VECTOR")
    if ev is not None:
        return ev
    vi = vi if vi is not None else _vector_index_section(config_path)
    v = vi.get("use_query_vector")
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("1", "true", "yes", "on"):
            return True
        if s in ("0", "false", "no", "off"):
            return False
    return False


def _embedding_endpoint_from_config(config_path: Path | str | None) -> str | None:
    cfg = _load_vector_index_yaml(Path(config_path) if config_path else None)
    if not isinstance(cfg, dict):
        return None
    emb = cfg.get("embedding")
    if isinstance(emb, dict):
        ep = emb.get("endpoint")
        if isinstance(ep, str):
            s = ep.strip()
            return s if s else None
    return None


def _resolve_embedding_endpoint(
    explicit: str | None,
    *,
    config_path: Path | str | None,
) -> str | None:
    """Serving endpoint name for ``embed_query`` / ``embedding.embed_texts``."""
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    env = os.environ.get("DATABRICKS_EMBEDDING_ENDPOINT", "").strip()
    if env:
        return env
    return _embedding_endpoint_from_config(config_path)


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


def _spark_databricks_workspace_host() -> str | None:
    """Resolve workspace URL on Databricks Runtime (driver has Spark)."""
    try:
        from pyspark.sql import SparkSession

        spark = SparkSession.getActiveSession()
        if spark is None:
            return None
        u = (spark.conf.get("spark.databricks.workspaceUrl") or "").strip()
        if not u:
            return None
        if u.startswith("http://") or u.startswith("https://"):
            return u.rstrip("/")
        return f"https://{u}".rstrip("/")
    except Exception:
        return None


def _databricks_workspace_host() -> str | None:
    for key in ("DATABRICKS_HOST", "DATABRICKS_WORKSPACE_URL"):
        v = os.environ.get(key, "").strip()
        if v:
            return v.rstrip("/") if v.startswith("http") else f"https://{v}".rstrip("/")
    return _spark_databricks_workspace_host()


def _databricks_rest_token() -> str | None:
    for key in ("DATABRICKS_TOKEN", "DATABRICKS_API_TOKEN", "TOKEN", "DATABRICKS_SERVICE_TOKEN"):
        t = os.environ.get(key, "").strip()
        if t:
            return t
    try:
        from IPython import get_ipython

        ip = get_ipython()
        if ip is not None:
            dbutils = ip.user_ns.get("dbutils")
            if dbutils is not None:
                ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
                api_tok = ctx.apiToken().get()
                if api_tok:
                    return str(api_tok)
    except Exception:
        pass
    # On Databricks Runtime with classic Spark, the API token may be available via Spark conf.
    # However, on Spark Connect this can trigger a gRPC config fetch that errors and is noisy.
    # So we only attempt Spark conf as a last resort.
    try:
        from pyspark.sql import SparkSession

        spark = SparkSession.getActiveSession()
        if spark:
            t = (spark.conf.get("spark.databricks.api.token", "") or "").strip()
            if t:
                return t
    except Exception:
        pass
    return None


def _coerce_llm_response_dict(resp: Any) -> dict[str, Any] | None:
    if resp is None:
        return None
    if isinstance(resp, dict):
        return resp
    try:
        if hasattr(resp, "model_dump"):  # pydantic v2
            out = resp.model_dump()
            return out if isinstance(out, dict) else None
        if hasattr(resp, "dict"):
            out = resp.dict()  # type: ignore[no-untyped-call]
            return out if isinstance(out, dict) else None
        if hasattr(resp, "as_dict"):
            out = resp.as_dict()
            return out if isinstance(out, dict) else None
    except Exception:
        pass
    return None


def _text_from_llm_payload(resp: Any) -> str:
    """
    Normalize serving responses to assistant text.

    Handles OpenAI-like chat completions, predictions arrays, MLflow wrappers, etc.
    """
    d = _coerce_llm_response_dict(resp)
    if isinstance(d, dict):
        if isinstance(d.get("choices"), list) and d["choices"]:
            ch0 = d["choices"][0]
            if isinstance(ch0, dict):
                msg = ch0.get("message")
                if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                    return msg["content"]
                if isinstance(ch0.get("content"), str):
                    return ch0["content"]
        preds = d.get("predictions")
        if isinstance(preds, list) and preds:
            p0 = preds[0]
            if isinstance(p0, dict) and isinstance(p0.get("content"), str):
                return p0["content"]
            if isinstance(p0, str):
                return p0
        data = d.get("data")
        if isinstance(data, list) and data:
            d0 = data[0]
            if isinstance(d0, dict) and isinstance(d0.get("content"), str):
                return d0["content"]
            if isinstance(d0, str):
                return d0
    if isinstance(resp, str):
        return resp
    return json.dumps(resp, ensure_ascii=False)


def _invoke_chat_via_openai_sdk(
    endpoint: str,
    *,
    messages: list[dict[str, str]],
    temperature: float,
) -> dict[str, Any] | None:
    """
    Chat completions via the OpenAI client against Databricks serving.

    ``base_url`` must be ``{workspace}/serving-endpoints`` and ``model`` must be the
    serving endpoint name — this matches the wire format external OpenAI models expect.
    See: https://learn.microsoft.com/azure/databricks/machine-learning/model-serving/query-chat-models
    """
    try:
        from openai import OpenAI
    except ImportError:
        return None

    host = _databricks_workspace_host()
    token = _databricks_rest_token()
    if not host or not token:
        return None

    try:
        client = OpenAI(api_key=token, base_url=f"{host}/serving-endpoints")
        msgs = [{"role": str(m.get("role", "user")), "content": str(m["content"])} for m in messages]
        comp = client.chat.completions.create(
            model=endpoint,
            messages=msgs,
            temperature=float(temperature),
        )
        if hasattr(comp, "model_dump"):
            d = comp.model_dump()
            return d if isinstance(d, dict) else None
        if hasattr(comp, "dict"):
            d = comp.dict()  # type: ignore[no-untyped-call]
            return d if isinstance(d, dict) else None
    except Exception:
        return None
    return None


def _invoke_chat_via_http(
    endpoint: str,
    *,
    messages: list[dict[str, str]],
    temperature: float,
) -> dict[str, Any]:
    """
    POST OpenAI-style chat JSON to ``/serving-endpoints/{name}/invocations``.

    External-model endpoints expect top-level ``messages``; some MLflow client versions
    send a shape the gateway does not forward correctly to OpenAI.
    """
    host = _databricks_workspace_host()
    token = _databricks_rest_token()
    if not host or not token:
        raise RuntimeError("missing host or token")

    url = f"{host}/serving-endpoints/{quote(endpoint, safe='')}/invocations"
    payload = {"messages": list(messages), "temperature": float(temperature)}
    body = json.dumps(payload).encode("utf-8")

    def _post(authz: str) -> str:
        req = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={"Authorization": authz, "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=180) as r:
            return r.read().decode("utf-8")

    try:
        raw = _post(f"Bearer {token}")
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        if e.code == 401:
            try:
                basic = base64.b64encode(f"token:{token}".encode()).decode()
                raw = _post(f"Basic {basic}")
            except urllib.error.HTTPError:
                raise RuntimeError(f"Databricks serving HTTP {e.code} for {endpoint!r}: {detail}") from e
        else:
            raise RuntimeError(f"Databricks serving HTTP {e.code} for {endpoint!r}: {detail}") from e
    return json.loads(raw) if raw else {}


def _invoke_chat_via_workspace_sdk(
    endpoint: str,
    *,
    messages: list[dict[str, str]],
    temperature: float,
) -> dict[str, Any] | None:
    """Use Databricks SDK data-plane or control-plane ``query`` (chat schema for external models)."""
    try:
        from databricks.sdk import WorkspaceClient
        from databricks.sdk.service.serving import ChatMessage, ChatMessageRole
    except ImportError:
        return None

    role_of = {
        "system": ChatMessageRole.SYSTEM,
        "user": ChatMessageRole.USER,
        "assistant": ChatMessageRole.ASSISTANT,
    }
    try:
        w = WorkspaceClient()
        msgs = [
            ChatMessage(
                role=role_of.get(str(m.get("role", "user")).lower(), ChatMessageRole.USER),
                content=m["content"],
            )
            for m in messages
        ]
        for api_name in ("serving_endpoints_data_plane", "serving_endpoints"):
            api = getattr(w, api_name, None)
            if api is None:
                continue
            try:
                q = api.query(name=endpoint, messages=msgs, temperature=float(temperature))
            except Exception:
                continue
            out = _coerce_llm_response_dict(q)
            if out:
                return out
    except Exception:
        return None
    return None


def _call_databricks_llm(endpoint: str, *, messages: list[dict[str, str]], temperature: float = 0.0) -> str:
    """
    Call a Databricks chat model serving endpoint.

    Order (unless ``DATABRICKS_LLM_FORCE_MLFLOW``): (1) ``openai`` package with
    ``base_url={workspace}/serving-endpoints`` and ``model=endpoint`` — correct for
    external OpenAI-backed endpoints; (2) ``databricks.sdk`` data-plane then control-plane
    ``query``; (3) raw REST ``/invocations`` with top-level ``messages``; (4) MLflow
    ``predict`` (legacy; may break external OpenAI routing).
    """
    force_mlflow = os.environ.get("DATABRICKS_LLM_FORCE_MLFLOW", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )

    if not force_mlflow:
        oa = _invoke_chat_via_openai_sdk(endpoint, messages=messages, temperature=temperature)
        if oa:
            return _text_from_llm_payload(oa)

        sdk_resp = _invoke_chat_via_workspace_sdk(endpoint, messages=messages, temperature=temperature)
        if sdk_resp:
            return _text_from_llm_payload(sdk_resp)

        if _databricks_workspace_host() and _databricks_rest_token():
            try:
                raw = _invoke_chat_via_http(endpoint, messages=messages, temperature=temperature)
                return _text_from_llm_payload(raw)
            except RuntimeError:
                raise
            except Exception:
                pass

    from mlflow.deployments import get_deploy_client

    client = get_deploy_client("databricks")
    resp = client.predict(endpoint=endpoint, inputs={"messages": messages, "temperature": float(temperature)})
    return _text_from_llm_payload(resp)


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


def _rerank_results_from_scores(
    subset: list[RetrievalResult],
    scores: list[float],
    *,
    rerank_top_k: int,
) -> list[RetrievalResult]:
    """Sort ``subset`` by parallel ``scores`` (same length) and return top-``rerank_top_k`` with ranks."""
    ranked = sorted(
        (replace(r, score=float(scores[i]), source="rerank") for i, r in enumerate(subset)),
        key=lambda x: x.score,
        reverse=True,
    )
    return [replace(r, rank=i + 1) for i, r in enumerate(ranked[:rerank_top_k])]


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
            return _rerank_results_from_scores(subset, scores, rerank_top_k=rerank_top_k)

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
                return _rerank_results_from_scores(subset, scores, rerank_top_k=rerank_top_k)
            except (TypeError, ValueError):
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
    use_query_vector: bool | None = None,
    config_path: Path | str | None = None,
    rerank_columns: list[str] | None = None,
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
      ``similarity_search`` with ``query_vector`` (precomputed-embedding / direct-access indexes).
      When ``use_query_vector`` is omitted, it is resolved from ``DATABRICKS_USE_QUERY_VECTOR``
      or ``vector_index.use_query_vector`` in ``ai/config/vector_index.yml`` (else ``False``).

    **Built-in reranker (Mosaic AI Vector Search)**

    Pass ``rerank_columns`` (or set env ``DATABRICKS_VS_RERANK_COLUMNS`` / YAML
    ``vector_index.rerank_columns``) to enable ``DatabricksReranker`` on the query.
    Each column name must appear in the index and is included in reranking context
    (see Databricks docs: order matters; large text is truncated per column).

    Environment (optional): ``DATABRICKS_EMBEDDING_ENDPOINT``, ``EMBED_BATCH_SIZE`` —
    same as ``embedding.embed_texts``. When ``use_query_vector`` resolves to true, the
    embedding endpoint is resolved from (in order): ``embedding_endpoint`` argument,
    ``DATABRICKS_EMBEDDING_ENDPOINT``, ``embedding.endpoint`` in ``ai/config/vector_index.yml``.
    """
    vi_section = _vector_index_section(config_path)
    use_qv = _resolve_use_query_vector(use_query_vector, config_path=config_path, vi=vi_section)
    ep, idx = _merge_vs_settings(
        endpoint_name=endpoint_name,
        index_name=index_name,
        config_path=config_path,
        vi=vi_section,
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

    # Suppress noisy "notebook authentication token" notices in notebooks.
    # (Auth behavior is unchanged; this only disables the warning spam.)
    client = VectorSearchClient(disable_notice=True)
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

    vs_rerank_cols = _resolve_vector_search_rerank_columns(
        rerank_columns, config_path=config_path, vi=vi_section
    )
    if vs_rerank_cols:
        try:
            from databricks.vector_search.reranker import DatabricksReranker
        except ImportError as e:
            raise ImportError(
                "Built-in Vector Search reranker requires databricks-vectorsearch with "
                "databricks.vector_search.reranker.DatabricksReranker. "
                "Upgrade: %pip install -U databricks-vectorsearch"
            ) from e
        kwargs["reranker"] = DatabricksReranker(columns_to_rerank=vs_rerank_cols)

    if use_qv:
        ep_embed = _resolve_embedding_endpoint(embedding_endpoint, config_path=config_path)
        if not ep_embed:
            raise ValueError(
                "Query-vector retrieval requires a Model Serving embedding endpoint. Set "
                "DATABRICKS_EMBEDDING_ENDPOINT, pass embedding_endpoint=..., or set "
                "embedding.endpoint in ai/config/vector_index.yml to an endpoint that exists "
                "in this workspace (Serving → Endpoints)."
            )
        kwargs["query_vector"] = embed_query(query, endpoint=ep_embed)
    else:
        kwargs["query_text"] = query

    raw = index.similarity_search(**kwargs)
    return _normalize_similarity_hits(raw)


DEFAULT_HYBRID_PER_QUERY_TOP_K = 50
# Clamp answer context to a small band regardless of rerank_top_k (generation cost vs coverage).
_MIN_CONTEXT_CHUNKS = 5
_MAX_CONTEXT_CHUNKS = 10


def hybrid_retrieve_top50(
    qi: QueryInfo,
    *,
    index_name: str | None = None,
    endpoint_name: str | None = None,
    columns: list[str] | None = None,
    query_type: str | None = None,
    embedding_endpoint: str | None = None,
    use_query_vector: bool | None = None,
    config_path: Path | str | None = None,
    extra_filters: dict[str, Any] | None = None,
    per_query_top_k: int = DEFAULT_HYBRID_PER_QUERY_TOP_K,
    rerank_columns: list[str] | None = None,
) -> list[RetrievalResult]:
    """
    Hybrid retrieval against the Databricks Vector Search index for each rewritten query.

    Returns a single deduped ranked list of up to ~ (per_query_top_k * num_queries) candidates,
    sorted by the index-provided score.

    When ``rerank_columns`` / env / YAML enable the Vector Search ``DatabricksReranker``,
    each ``retrieve`` call returns results already reranked by Mosaic AI Vector Search.
    """
    merged_filters = {**(extra_filters or {}), **(qi.filters or {})}

    subqueries = qi.rewritten_queries[:5] if qi.rewritten_queries else []
    if not subqueries and qi.normalized:
        subqueries = [qi.normalized]

    all_rows: list[dict[str, Any]] = []
    for q in subqueries:
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
            rerank_columns=rerank_columns,
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
    use_query_vector: bool | None = None,
    config_path: Path | str | None = None,
    hybrid_per_query_top_k: int = DEFAULT_HYBRID_PER_QUERY_TOP_K,
    vector_search_rerank_columns: list[str] | None = None,
    # Rerank config (second stage; optional if Vector Search reranker already applied)
    post_rerank: bool = True,
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
    End-to-end pipeline.

    Enable the Mosaic AI Vector Search built-in reranker by setting
    ``vector_search_rerank_columns``, or env ``DATABRICKS_VS_RERANK_COLUMNS`` / YAML
    ``vector_index.rerank_columns``. If you only want that reranker, set
    ``post_rerank=False`` so the second-stage ``rerank_candidates`` step is skipped.
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
        per_query_top_k=hybrid_per_query_top_k,
        rerank_columns=vector_search_rerank_columns,
    )

    # 2) Optional second-stage rerank (custom endpoint or LLM scoring)
    if post_rerank:
        reranked = rerank_candidates(
            qi.normalized,
            candidates,
            top_n=rerank_top_n,
            rerank_top_k=rerank_top_k,
            rerank_model_endpoint=rerank_model_endpoint,
        )
    else:
        n = max(1, int(rerank_top_n))
        k = max(1, int(rerank_top_k))
        reranked = candidates[:n][:k]

    assembled = _collapse_overlaps(reranked, max_per_parent=collapse_per_parent)
    assembled = _diversify(assembled, max_per_parent=max_per_parent)
    ctx_cap = max(_MIN_CONTEXT_CHUNKS, min(_MAX_CONTEXT_CHUNKS, int(rerank_top_k)))
    final_chunks = assembled[:ctx_cap]

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
        "candidates_top50": [r.payload for r in candidates[:hybrid_per_query_top_k]],
        "chunks": [r.payload for r in final_chunks],
        "answer": answer,
    }


# Backwards-compatible alias (old name returned only chunks + query_info).
def retrieve_pipeline(query: str, **kwargs: Any) -> dict[str, Any]:
    out = rag_pipeline(query, **kwargs)
    return {"query_info": out.get("query_info", {}), "chunks": out.get("chunks", [])}

