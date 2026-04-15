# Databricks notebook source
# MAGIC %md
# MAGIC - Ensures a **Mosaic AI Vector Search** serving endpoint exists in this workspace.
# MAGIC - If **`rag-hybrid-endpoint`** (or **`VECTOR_SEARCH_ENDPOINT_NAME`**) is missing, creates it with **`STANDARD`** capacity; otherwise reuses the existing endpoint.
# MAGIC - Prints **`STATUS=SUCCESS`** or **`STATUS=FAILURE`** for pipeline / job log filtering.
# MAGIC - Cluster libraries: install **`databricks-vectorsearch`** if it is not on the image (e.g. one-time **`%pip install databricks-vectorsearch`**).

# COMMAND ----------

from __future__ import annotations

import os
import sys

from databricks.vector_search.client import VectorSearchClient


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


ENDPOINT_NAME = _get_param("VECTOR_SEARCH_ENDPOINT_NAME", "rag-hybrid-endpoint")
ENDPOINT_TYPE = "STANDARD"


def _emit_failure(exc: BaseException) -> None:
    err = f"{type(exc).__name__}: {exc}"
    print("STATUS=FAILURE", flush=True)
    print(f"STEP=vector_search_endpoint name={ENDPOINT_NAME!r}", flush=True)
    print(f"ERROR={err}", flush=True)


def _run() -> str:
    client = VectorSearchClient()

    if client.endpoint_exists(ENDPOINT_NAME):
        detail = f"endpoint_exists name={ENDPOINT_NAME!r}"
        print(f"Vector Search endpoint already exists; using {ENDPOINT_NAME!r}.")
        return detail

    print(f"Creating Vector Search endpoint {ENDPOINT_NAME!r} (type={ENDPOINT_TYPE!r})...")
    client.create_endpoint(name=ENDPOINT_NAME, endpoint_type=ENDPOINT_TYPE)
    client.wait_for_endpoint(ENDPOINT_NAME, verbose=True)
    print(f"Endpoint {ENDPOINT_NAME!r} is online.")
    return f"created name={ENDPOINT_NAME!r} type={ENDPOINT_TYPE!r}"


try:
    outcome = _run()
    print("STATUS=SUCCESS", flush=True)
    print(f"STEP=vector_search_endpoint name={ENDPOINT_NAME!r}", flush=True)
    print(f"DETAIL={outcome}", flush=True)
except Exception as e:
    _emit_failure(e)
    sys.stderr.write(f"Vector Search endpoint task failed: {e}\n")
    raise
