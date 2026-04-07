# Databricks notebook source
# MAGIC %md
# MAGIC # Chunk company semantic text (Unity Catalog)
# MAGIC Reads **`bw_ai_search`.`02_silver`.bw_company_semantic_text** (`semantic_text`), writes one row per chunk to **`bw_ai_search`.`02_silver`.bw_company_text_chunks`** with `chunk_id`, `company_id`, `chunk_index`, `chunk_text`.
# MAGIC Add **`ai/src`** to the cluster **PYTHONPATH** (or install this repo) so executors can import `chunking`.

import os
import sys

from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, StringType

# Allow `from chunking import chunk_text` when `ai/src` is on the path.
def _ensure_ai_src_on_path() -> None:
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        cand = os.path.abspath(os.path.join(here, "..", "..", "ai", "src"))
        if os.path.isfile(os.path.join(cand, "chunking.py")) and cand not in sys.path:
            sys.path.insert(0, cand)
            return
    except NameError:
        pass
    cwd = os.getcwd()
    for rel in ("ai/src", os.path.join("..", "ai", "src"), os.path.join("..", "..", "ai", "src")):
        p = os.path.abspath(os.path.join(cwd, rel))
        if os.path.isfile(os.path.join(p, "chunking.py")) and p not in sys.path:
            sys.path.insert(0, p)
            return


_ensure_ai_src_on_path()
from chunking import chunk_text  # noqa: E402

CATALOG = "bw_ai_search"
SCHEMA = "02_silver"
SOURCE_TABLE = "bw_company_semantic_text"
TEXT_COL = "semantic_text"
OUTPUT_TABLE = "bw_company_text_chunks"

# Match embedding context window / retrieval preference (see ai/src/chunking.py).
CHUNK_MAX_CHARS = 2500
CHUNK_OVERLAP = 300


def main() -> None:
    # Numeric schema name: set session catalog/schema so saveAsTable(table) lands in 02_silver.
    spark.sql(f"USE CATALOG {CATALOG}")
    spark.sql(f"USE SCHEMA `{SCHEMA}`")

    df = spark.table(SOURCE_TABLE).where(
        F.col("company_id").isNotNull()
        & F.col(TEXT_COL).isNotNull()
        & (F.length(F.trim(F.col(TEXT_COL))) > 0)
    )

    @F.udf(ArrayType(StringType()))
    def chunk_udf(t: str | None) -> list[str]:
        return chunk_text(t, max_chars=CHUNK_MAX_CHARS, overlap=CHUNK_OVERLAP)

    exploded = (
        df.withColumn("_chunks", chunk_udf(F.col(TEXT_COL)))
        .select("company_id", F.posexplode(F.col("_chunks")).alias("chunk_index", "chunk_text"))
    )

    # Stable id: same company_id + index + text => same chunk_id (useful for idempotent reloads).
    with_ids = exploded.withColumn(
        "chunk_id",
        F.sha2(
            F.concat_ws(
                "|",
                F.col("company_id").cast("string"),
                F.col("chunk_index").cast("string"),
                F.col("chunk_text"),
            ),
            256,
        ),
    )

    out_df = with_ids.select("chunk_id", "company_id", "chunk_index", "chunk_text")

    out_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(OUTPUT_TABLE)

    preview = out_df.limit(50)
    try:
        display(preview)  # type: ignore[name-defined]
    except NameError:
        preview.show(50, truncate=80)

    spark.sql(
        f"SELECT COUNT(*) AS n_chunks, COUNT(DISTINCT company_id) AS n_companies "
        f"FROM `{CATALOG}`.`{SCHEMA}`.`{OUTPUT_TABLE}`"
    ).show()


# Databricks notebook / job runs this file with a name other than __main__; always execute.
main()
