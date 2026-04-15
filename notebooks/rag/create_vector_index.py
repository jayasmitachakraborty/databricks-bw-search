# Databricks notebook source
# MAGIC %md
# MAGIC - Creates a **Delta Sync** Vector Search index on **`gold_rag_company_chunks`** (hybrid-ready: dense vectors + keyword/BM25 on **`search_text`**).
# MAGIC - **Prereq:** Endpoint (e.g. **`rag-hybrid-endpoint`**) must exist — run **`create_vector_search_endpoint`** first.
# MAGIC - **`embedding_dimension`** must match the **`embedding`** column in the source table (default **1024** for `databricks-bge-large-en`; e.g. **1536** for some OpenAI models). Override with **`VECTOR_SEARCH_EMBEDDING_DIMENSION`**.
# MAGIC - The REST/UI field **`text_column`** corresponds to **`embedding_source_column`** in the Python SDK.
# MAGIC - **Serverless / many runtimes:** the Vector Search SDK is not pre-installed. Run the next cell first (or add **`databricks-vectorsearch`** to the job environment). If the import still fails, run **`dbutils.library.restartPython()`** once after install, then run the rest of the notebook.

# COMMAND ----------
# MAGIC %pip install databricks-vectorsearch

# COMMAND ----------

from __future__ import annotations

import os

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


def _get_param_int(name: str, default: int) -> int:
    return int(_get_param(name, str(default)))


ENDPOINT_NAME = _get_param("VECTOR_SEARCH_ENDPOINT_NAME", "rag-hybrid-endpoint")
INDEX_NAME = _get_param("VECTOR_SEARCH_INDEX_NAME", "bw_ai_search.gold_rag_company_chunks_index")
SOURCE_TABLE_NAME = _get_param(
    "VECTOR_SEARCH_SOURCE_TABLE",
    "bw_ai_search.03_gold.gold_rag_company_chunks",
)
PIPELINE_TYPE = _get_param("VECTOR_SEARCH_PIPELINE_TYPE", "TRIGGERED")
EMBEDDING_DIMENSION = _get_param_int("VECTOR_SEARCH_EMBEDDING_DIMENSION", 1024)

client = VectorSearchClient()

if client.index_exists(endpoint_name=ENDPOINT_NAME, index_name=INDEX_NAME):
    print(f"Vector Search index already exists; using {INDEX_NAME!r} on endpoint {ENDPOINT_NAME!r}.")
else:
    print(f"Creating Delta Sync index {INDEX_NAME!r} on {ENDPOINT_NAME!r}...")
    client.create_delta_sync_index_and_wait(
        endpoint_name=ENDPOINT_NAME,
        index_name=INDEX_NAME,
        source_table_name=SOURCE_TABLE_NAME,
        primary_key="chunk_id",
        pipeline_type=PIPELINE_TYPE,
        embedding_dimension=EMBEDDING_DIMENSION,
        embedding_vector_column="embedding",
        embedding_source_column="search_text",
        verbose=True,
    )
    print(f"Index {INDEX_NAME!r} is ready.")
