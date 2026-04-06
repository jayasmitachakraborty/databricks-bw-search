-- Data quality: compare distinct company_id in market trends vs bw_company_semantic_text.
-- Appends one row per run to bw_data_quality_checks (silver). Query latest with:
--   SELECT * FROM `bw_ai_search`.`02_silver`.bw_data_quality_checks
--   WHERE check_name = 'bw_company_semantic_text_company_id_count'
--   ORDER BY checked_at DESC LIMIT 1;

CREATE TABLE IF NOT EXISTS `bw_ai_search`.`02_silver`.bw_data_quality_checks (
  check_name STRING NOT NULL,
  checked_at TIMESTAMP NOT NULL,
  counts_match BOOLEAN NOT NULL,
  market_trends_distinct BIGINT,
  semantic_text_distinct BIGINT
) USING DELTA;

INSERT INTO `bw_ai_search`.`02_silver`.bw_data_quality_checks
WITH market_trends AS (
  SELECT COUNT(DISTINCT company_id) AS n
  FROM `bw_ai_search`.`_01_bronze_bw_dashboard`.bw_market_trends_anlzd
  WHERE company_id IS NOT NULL
),
semantic_text AS (
  SELECT COUNT(DISTINCT company_id) AS n
  FROM `bw_ai_search`.`02_silver`.bw_company_semantic_text
  WHERE company_id IS NOT NULL
)
SELECT
  'bw_company_semantic_text_company_id_count' AS check_name,
  current_timestamp() AS checked_at,
  m.n = s.n AS counts_match,
  m.n AS market_trends_distinct,
  s.n AS semantic_text_distinct
FROM market_trends m
CROSS JOIN semantic_text s;
