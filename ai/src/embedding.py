"""Batch embedding via Databricks model serving (MLflow deployments client)."""

from __future__ import annotations

import os
from typing import Any


def _parse_embedding_response(resp: Any) -> list[list[float]]:
    """Normalize serving responses to a list of float vectors."""
    if resp is None:
        return []
    if isinstance(resp, dict):
        if "predictions" in resp:
            preds = resp["predictions"]
        elif "data" in resp:
            data = resp["data"]
            if data and isinstance(data[0], dict) and "embedding" in data[0]:
                return [list(map(float, d["embedding"])) for d in data]
            preds = data
        else:
            preds = resp.get("output", resp.get("outputs", []))
        if preds is None:
            return []
        if len(preds) > 0 and isinstance(preds[0], dict):
            key = "embedding" if "embedding" in preds[0] else next(
                (k for k in ("predictions", "values", "vector") if k in preds[0]), None
            )
            if key:
                return [list(map(float, p[key])) for p in preds]
        return [list(map(float, row)) for row in preds]
    if isinstance(resp, list):
        return [list(map(float, row)) for row in resp]
    raise TypeError(f"Unexpected embedding response type: {type(resp)}")


def embed_texts(
    texts: list[str],
    *,
    endpoint: str | None = None,
    batch_size: int | None = None,
) -> list[list[float]]:
    """
    Return one embedding vector per input string (same order).

    Uses ``mlflow.deployments.get_deploy_client("databricks").predict``.

    Environment:
        DATABRICKS_EMBEDDING_ENDPOINT — required if ``endpoint`` is omitted (your workspace
        Model Serving embedding endpoint name).

        EMBED_BATCH_SIZE — max texts per predict call (default 32).
    """
    if not texts:
        return []

    ep = (endpoint or os.environ.get("DATABRICKS_EMBEDDING_ENDPOINT") or "").strip()
    if not ep:
        raise ValueError(
            "Embedding endpoint is not configured. Set DATABRICKS_EMBEDDING_ENDPOINT to your "
            "workspace Model Serving endpoint name, or pass endpoint=... to embed_texts()."
        )
    bs = batch_size if batch_size is not None else int(os.environ.get("EMBED_BATCH_SIZE", "32"))

    from mlflow.deployments import get_deploy_client

    client = get_deploy_client("databricks")
    out: list[list[float]] = []
    for i in range(0, len(texts), bs):
        batch = texts[i : i + bs]
        resp = client.predict(endpoint=ep, inputs={"input": batch})
        vecs = _parse_embedding_response(resp)
        if len(vecs) != len(batch):
            raise RuntimeError(
                f"Embedding endpoint returned {len(vecs)} vectors for {len(batch)} inputs "
                f"(endpoint={ep!r}). Check endpoint schema and response shape."
            )
        out.extend(vecs)
    return out
