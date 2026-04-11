"""Vector search retrieval for RAG (Databricks Vector Search + embedding endpoint)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


DEFAULT_RESULT_COLUMNS = ("chunk_id", "company_id", "chunk_index", "chunk_text")


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
