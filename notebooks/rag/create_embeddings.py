# Databricks notebook source
# MAGIC %md
# MAGIC - Embeddings from `bw_company_text_chunks`
# MAGIC - Reads **`bw_ai_search`.`02_silver`.bw_company_text_chunks**, calls a Databricks **embedding serving endpoint**
# MAGIC - (MLflow `get_deploy_client("databricks").predict`), writes **`bw_ai_search`.`02_silver`.bw_company_text_chunk_embeddings`**.
# MAGIC - **Setup:** Set **`DATABRICKS_EMBEDDING_ENDPOINT`** to your serving endpoint. Executors use **`mlflow`** only (no repo path required).

# COMMAND ----------

from __future__ import annotations

import os
from typing import Any

from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, DoubleType, IntegerType, StringType, StructField, StructType

CATALOG = "bw_ai_search"
SCHEMA = "02_silver"
CHUNKS_TABLE = "bw_company_text_chunks"
EMBEDDINGS_TABLE = "bw_company_text_chunk_embeddings"

EMBEDDING_ENDPOINT = os.environ.get("DATABRICKS_EMBEDDING_ENDPOINT", "databricks-bge-large-en")
EMBED_BATCH_SIZE = int(os.environ.get("EMBED_BATCH_SIZE", "32"))
EMBED_NUM_PARTITIONS = int(os.environ.get("EMBED_NUM_PARTITIONS", "8"))


def _require_table(catalog: str, schema: str, table: str) -> None:
    fq = f"`{catalog}`.`{schema}`.`{table}`"
    if not spark.catalog.tableExists(f"{catalog}.{schema}.{table}"):
        raise RuntimeError(
            f"Upstream table missing: {fq}. "
            "Run the upstream chunking task that creates it."
        )


def _parse_embedding_response(resp: Any) -> list[list[float]]:
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


def _embed_texts_batches(texts: list[str], endpoint: str, batch_size: int) -> list[list[float]]:
    if not texts:
        return []
    from mlflow.deployments import get_deploy_client

    client = get_deploy_client("databricks")
    out: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = client.predict(endpoint=endpoint, inputs={"input": batch})
        vecs = _parse_embedding_response(resp)
        if len(vecs) != len(batch):
            raise RuntimeError(
                f"Embedding endpoint returned {len(vecs)} vectors for {len(batch)} inputs (endpoint={endpoint!r})."
            )
        out.extend(vecs)
    return out


def main() -> None:
    spark.sql(f"USE CATALOG {CATALOG}")
    spark.sql(f"USE SCHEMA `{SCHEMA}`")

    _require_table(CATALOG, SCHEMA, CHUNKS_TABLE)

    df = (
        spark.table(CHUNKS_TABLE)
        .where(F.col("chunk_id").isNotNull() & F.col("chunk_text").isNotNull())
        .select("chunk_id", "company_id", "chunk_index", "chunk_text")
    )

    if df.limit(1).count() == 0:
        raise RuntimeError(
            f"Upstream table `{CATALOG}`.`{SCHEMA}`.`{CHUNKS_TABLE}` exists but has no rows to embed."
        )

    endpoint = EMBEDDING_ENDPOINT
    batch_size = EMBED_BATCH_SIZE
    n_parts = max(1, EMBED_NUM_PARTITIONS)

    schema_out = StructType(
        [
            StructField("chunk_id", StringType(), False),
            StructField("company_id", StringType(), True),
            StructField("chunk_index", IntegerType(), True),
            StructField("embedding", ArrayType(DoubleType(), containsNull=False), False),
            StructField("embedding_endpoint", StringType(), False),
        ]
    )

    def embed_partition(iterator):
        import pandas as pd

        for pdf in iterator:
            if pdf.empty:
                continue
            chunk_ids = pdf["chunk_id"].astype(str).tolist()
            company_ids = [None if pd.isna(x) else str(x) for x in pdf["company_id"].tolist()]
            chunk_indices = pdf["chunk_index"].fillna(0).astype(int).tolist()
            texts = pdf["chunk_text"].fillna("").astype(str).tolist()
            n = len(texts)
            all_vecs = _embed_texts_batches(texts, endpoint, batch_size)
            if len(all_vecs) != n:
                raise RuntimeError(f"Expected {n} embeddings, got {len(all_vecs)}")
            out = pd.DataFrame(
                {
                    "chunk_id": chunk_ids,
                    "company_id": company_ids,
                    "chunk_index": chunk_indices,
                    "embedding": all_vecs,
                    "embedding_endpoint": [endpoint] * n,
                }
            )
            yield out

    embedded = df.repartition(n_parts).mapInPandas(embed_partition, schema=schema_out)
    embedded = embedded.withColumn("embedded_at", F.current_timestamp())

    embedded.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(EMBEDDINGS_TABLE)

    preview = embedded.select(
        "chunk_id", "company_id", "chunk_index", "embedding_endpoint", "embedded_at"
    ).limit(20)
    try:
        display(preview)  # type: ignore[name-defined]
    except NameError:
        preview.show(truncate=80)

    fq = f"`{CATALOG}`.`{SCHEMA}`.`{EMBEDDINGS_TABLE}`"
    spark.sql(f"SELECT COUNT(*) AS n_rows, COUNT(DISTINCT chunk_id) AS n_chunks FROM {fq}").show()
    spark.sql(f"SELECT SIZE(embedding) AS embedding_dim FROM {fq} LIMIT 1").show()


'''
Databricks notebook / job runs this file with a name other than __main__; always execute.
'''
main()
