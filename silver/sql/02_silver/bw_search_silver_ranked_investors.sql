CREATE or REPLACE TABLE bw_ai_search.`02_silver`.bw_deals_investors_with_ranking AS
SELECT
  d.Deal_ID,
  i.Investor_Name,
  i.noa_ranking
FROM
  bw_ai_search._01_bronze_pitchbook.deals_investors d
JOIN
  bw_ai_search._01_bronze_pitchbook.investors i
ON
  TRIM(SPLIT(d.Investors, '\\(')[0]) = i.Investor_Name
WHERE
  d._fivetran_deleted = false
;