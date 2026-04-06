CREATE OR REPLACE TABLE bw_ai_search.`02_silver`.bw_pitchbook_company_all_investors AS
SELECT
  mt.company_id,
  STRING_AGG(DISTINCT i.investor_name, ', ') AS all_investors
FROM bw_ai_search._01_bronze_bw_dashboard.bw_market_trends_anlzd mt
LEFT JOIN bw_ai_search._01_bronze_pitchbook.deals_investors di
  ON mt.deal_id = di.deal_id
LEFT JOIN bw_ai_search._01_bronze_pitchbook.investors i
  ON TRIM(SPLIT(di.investors, '\\(')[0]) = i.investor_name
WHERE mt.company_id IS NOT NULL
GROUP BY mt.company_id;