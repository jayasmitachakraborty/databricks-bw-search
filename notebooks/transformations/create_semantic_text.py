# Databricks notebook source
# MAGIC %md
# MAGIC - Build `bw_ai_search`.`02_silver`.bw_company_semantic_text`
# MAGIC - Runs `silver/sql/02_silver/company_semantic_text.sql`, then runs the data-quality script
# MAGIC validate_bw_company_semantic_text_company_count.sql` and **fails** if distinct `company_id`
# MAGIC - counts do not match market trends.
# MAGIC - Set **`REPO_ROOT`** if SQL files are not found from the driver cwd.

# COMMAND ----------

import os

SEMANTIC_SQL_REL = os.path.join("silver", "sql", "02_silver", "company_semantic_text.sql")
VALIDATE_SQL_REL = os.path.join("silver", "sql", "02_silver", "validate_bw_company_semantic_text_company_count.sql")


def _find_repo_root() -> str:
    """Resolve repo root for SQL files (Jobs often have wrong getcwd(); __file__ may be missing)."""
    env = os.environ.get("REPO_ROOT")
    if env:
        root = os.path.abspath(env)
        if os.path.isfile(os.path.join(root, SEMANTIC_SQL_REL)):
            return root
    try:
        start = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        start = os.getcwd()
    d = start
    for _ in range(12):
        root = os.path.abspath(d)
        if os.path.isfile(os.path.join(root, SEMANTIC_SQL_REL)):
            return root
        parent = os.path.dirname(root)
        if parent == root:
            break
        d = parent
    raise FileNotFoundError(
        f"Could not find {SEMANTIC_SQL_REL} starting from {start!r}. "
        "Set env REPO_ROOT to the clone root (folder that contains silver/sql/)."
    )

DQ_CHECK_NAME = "bw_company_semantic_text_company_id_count"
SEMANTIC_TABLE = "`bw_ai_search`.`02_silver`.bw_company_semantic_text"


def _read_sql(path: str) -> str:
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Missing {path}. Clone the repo on the driver or set env REPO_ROOT to the repo root."
        )
    with open(path, encoding="utf-8") as f:
        return f.read()


def _run_validate_sql_script(sql_text: str) -> None:
    """Run CREATE TABLE + INSERT as separate statements (Spark `sql()` is single-statement)."""
    upper = sql_text.upper()
    insert_at = upper.find("\nINSERT INTO")
    if insert_at == -1:
        insert_at = upper.find("INSERT INTO")
    if insert_at <= 0:
        raise ValueError("validate SQL must contain INSERT INTO after CREATE TABLE block")
    create_stmt = sql_text[:insert_at].strip()
    insert_stmt = sql_text[insert_at:].strip().rstrip(";")
    spark.sql(create_stmt)
    spark.sql(insert_stmt)


def _assert_company_id_counts_match() -> None:
    rows = spark.sql(
        f"""
        SELECT counts_match, market_trends_distinct, semantic_text_distinct, checked_at
        FROM `bw_ai_search`.`02_silver`.bw_data_quality_checks
        WHERE check_name = '{DQ_CHECK_NAME}'
        ORDER BY checked_at DESC
        LIMIT 1
        """
    ).collect()
    if not rows:
        raise RuntimeError(
            "Data quality check produced no row in bw_data_quality_checks "
            f"(check_name={DQ_CHECK_NAME!r})."
        )
    r = rows[0]
    if r.counts_match is True:
        return
    raise RuntimeError(
        "Data quality check failed: distinct company_id in bw_market_trends_anlzd does not match "
        f"bw_company_semantic_text (market_trends_distinct={r.market_trends_distinct}, "
        f"semantic_text_distinct={r.semantic_text_distinct}, checked_at={r.checked_at}). "
        "Fix company_semantic_text.sql grain or upstream data before continuing the pipeline."
    )


def main() -> None:
    root = _find_repo_root()
    print(f"create_semantic_text: repo root = {root}")
    semantic_sql_path = os.path.join(root, SEMANTIC_SQL_REL)
    validate_sql_path = os.path.join(root, VALIDATE_SQL_REL)

    semantic_ddl = _read_sql(semantic_sql_path)
    spark.sql(semantic_ddl)

    spark.sql(
        f"SELECT COUNT(*) AS n, SUM(LENGTH(semantic_text)) AS total_chars FROM {SEMANTIC_TABLE}"
    ).show()

    validate_sql = _read_sql(validate_sql_path)
    _run_validate_sql_script(validate_sql)
    _assert_company_id_counts_match()

    print("Data quality check passed: distinct company_id counts match.")


'''
Databricks notebook / job runs this file with a name other than __main__; always execute.
'''
main()
