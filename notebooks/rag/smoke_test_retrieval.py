# Databricks notebook source
# MAGIC %md
# MAGIC ## Smoke test: `retrieval.py`
# MAGIC Quick end-to-end check of **query understanding**, **hybrid retrieval**, and **reranking**.
# MAGIC - Requires Databricks env / endpoints configured for **`ai/src/retrieval.py`** (same as production jobs).
# MAGIC - Prepends **`ai/src`** so `from retrieval import …` works (via `__file__`, **`REPO_ROOT`**, or cwd).
# MAGIC - **`pyyaml`** is required for `ai/config/vector_index.yml` to load; without it, set **`DATABRICKS_VECTOR_SEARCH_ENDPOINT_NAME`** and **`DATABRICKS_VECTOR_SEARCH_INDEX_NAME`**.

# COMMAND ----------

# MAGIC %pip install -q pyyaml

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from __future__ import annotations

import os
import sys


def _ensure_ai_src_on_path() -> None:
    """Prepend `ai/src` so `from retrieval import …` works on the driver."""
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


# COMMAND ----------

os.environ["DATABRICKS_QUERY_UNDERSTANDING_ENDPOINT"] = "bw-search-query-understanding"
os.environ["DATABRICKS_LLM_ENDPOINT"] = "bw-search-llm"
os.environ["DATABRICKS_ANSWER_ENDPOINT"] = "bw-search-llm"
os.environ["DATABRICKS_VECTOR_SEARCH_ENDPOINT_NAME"] = "rag-hybrid-endpoint"
os.environ["DATABRICKS_VECTOR_SEARCH_INDEX_NAME"] = "bw_ai_search.03_gold.gold_rag_company_chunks_index"

# COMMAND ----------

_ensure_ai_src_on_path()
from retrieval import hybrid_retrieve_top50, rerank_candidates, understand_query  # noqa: E402

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1. Query understanding

# COMMAND ----------

qi = understand_query("UK climate tech companies that raised in 2026")
print(qi)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2. Hybrid retrieval (top 50 pool)

# COMMAND ----------

cands = hybrid_retrieve_top50(qi, per_query_top_k=20)
print(len(cands))

for r in cands[:10]:
    print(
        r.rank,
        r.score,
        r.chunk_id,
        r.payload.get("company_id"),
        r.payload.get("chunk_text", "")[:200],
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3. Rerank

# COMMAND ----------

reranked = rerank_candidates(qi.normalized, cands, top_n=20, rerank_top_k=10)

for r in reranked:
    print(r.rank, r.score, r.chunk_id, r.payload.get("chunk_text", "")[:200])
