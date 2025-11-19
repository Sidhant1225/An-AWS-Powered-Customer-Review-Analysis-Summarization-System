CREATE EXTERNAL TABLE IF NOT EXISTS poc_processed_db.processed_fixed (
  reviewid        string,
  productid       string,
  reviewtext      string,
  reviewdate      string,
  positivescore   double,
  negativescore   double,
  neutralscore    double,
  mixedscore      double,
  keyphrases      array<string>,
  entities        array<string>,
  raw             map<string,string>,
  dynamoputstatus string
)
ROW FORMAT SERDE 'org.openx.data.jsonserde.JsonSerDe'
WITH SERDEPROPERTIES (
  'ignore.malformed.json'='true'
)
STORED AS TEXTFILE
LOCATION 's3://poc-sentiment-processed/processed/';




DROP TABLE IF EXISTS poc_processed_db.processed_clean;

CREATE TABLE poc_processed_db.processed_clean
WITH (
  format = 'PARQUET',
  external_location = 's3://poc-sentiment-processed/athena-tables/processed_clean/'
) AS
SELECT *
FROM poc_processed_db.processed_fixed;






CREATE EXTERNAL TABLE IF NOT EXISTS poc_processed_db.summaries_lines (
  line string
)
ROW FORMAT DELIMITED
  FIELDS TERMINATED BY '\n'
STORED AS TEXTFILE
LOCATION 's3://poc-sentiment-processed/summaries/';





CREATE TABLE poc_processed_db.summaries_clean
WITH (
  format = 'PARQUET',
  external_location = 's3://poc-sentiment-processed/athena-tables/summaries_clean/'
) AS
SELECT
  COALESCE(j_productid1, j_productid2, j_productid3, j_productid4)                            AS productid,
  COALESCE(j_summary1, j_summary2, j_summary3)                                                AS summary,
  COALESCE(CAST(j_aggs1 AS double), CAST(j_aggs2 AS double), CAST(j_aggs3 AS double))         AS aggregatesentiment,
  COALESCE(CAST(j_ts1 AS bigint), CAST(j_ts2 AS bigint), CAST(j_ts3 AS bigint), CAST(j_ts4 AS bigint)) AS timestamp,
  COALESCE(CAST(j_sa1 AS int), CAST(j_sa2 AS int), CAST(j_sa3 AS int))                         AS summaryattempts,
  COALESCE(j_m1, j_m2, j_m3)                                                                   AS method,
  COALESCE(j_u1, j_u2, j_u3)                                                                   AS uuid,
  source_path
FROM (
  SELECT
    "$path" AS source_path,
    trim(regexp_replace(line, '^[^\\{]*', '')) AS raw_guess,
    json_extract_scalar(trim(regexp_replace(line, '^[^\\{]*', '')), '$.ProductID')            AS j_productid1,
    json_extract_scalar(trim(regexp_replace(line, '^[^\\{]*', '')), '$.productId')            AS j_productid2,
    json_extract_scalar(trim(regexp_replace(line, '^[^\\{]*', '')), '$.product_id')           AS j_productid3,
    json_extract_scalar(trim(regexp_replace(line, '^[^\\{]*', '')), '$.Item.ProductID.S')     AS j_productid4,
    json_extract_scalar(trim(regexp_replace(line, '^[^\\{]*', '')), '$.Summary')              AS j_summary1,
    json_extract_scalar(trim(regexp_replace(line, '^[^\\{]*', '')), '$.summary')              AS j_summary2,
    json_extract_scalar(trim(regexp_replace(line, '^[^\\{]*', '')), '$.Item.Summary.S')       AS j_summary3,
    json_extract_scalar(trim(regexp_replace(line, '^[^\\{]*', '')), '$.AggregateSentiment')  AS j_aggs1,
    json_extract_scalar(trim(regexp_replace(line, '^[^\\{]*', '')), '$.aggregateSentiment')  AS j_aggs2,
    json_extract_scalar(trim(regexp_replace(line, '^[^\\{]*', '')), '$.Item.AggregateSentiment.N') AS j_aggs3,
    json_extract_scalar(trim(regexp_replace(line, '^[^\\{]*', '')), '$.Timestamp')            AS j_ts1,
    json_extract_scalar(trim(regexp_replace(line, '^[^\\{]*', '')), '$.timestamp')            AS j_ts2,
    json_extract_scalar(trim(regexp_replace(line, '^[^\\{]*', '')), '$.GeneratedAt')          AS j_ts3,
    json_extract_scalar(trim(regexp_replace(line, '^[^\\{]*', '')), '$.SummaryTimestamp')     AS j_ts4,
    json_extract_scalar(trim(regexp_replace(line, '^[^\\{]*', '')), '$.SummaryAttempts')      AS j_sa1,
    json_extract_scalar(trim(regexp_replace(line, '^[^\\{]*', '')), '$.summaryAttempts')      AS j_sa2,
    json_extract_scalar(trim(regexp_replace(line, '^[^\\{]*', '')), '$.Item.SummaryAttempts.N') AS j_sa3,
    json_extract_scalar(trim(regexp_replace(line, '^[^\\{]*', '')), '$.Method')               AS j_m1,
    json_extract_scalar(trim(regexp_replace(line, '^[^\\{]*', '')), '$.method')               AS j_m2,
    json_extract_scalar(trim(regexp_replace(line, '^[^\\{]*', '')), '$.Item.Method.S')        AS j_m3,
    json_extract_scalar(trim(regexp_replace(line, '^[^\\{]*', '')), '$.UUID')                 AS j_u1,
    json_extract_scalar(trim(regexp_replace(line, '^[^\\{]*', '')), '$.uuid')                 AS j_u2,
    json_extract_scalar(trim(regexp_replace(line, '^[^\\{]*', '')), '$.Item.UUID.S')          AS j_u3
  FROM poc_processed_db.summaries_lines
  WHERE "$path" LIKE '%/p%.json'
) t
WHERE raw_guess IS NOT NULL AND length(raw_guess) > 2;





CREATE OR REPLACE VIEW poc_processed_db.vw_combined_product_data AS
SELECT
  s.productid,
  s.summary,
  s.aggregatesentiment,
  from_unixtime(COALESCE(s.timestamp, 0))                                           AS summary_time,
  s.method,
  s.summaryattempts,
  regexp_extract(s.summary, 'Strengths:\\s*([^;]+)', 1)                             AS strengths,
  regexp_extract(s.summary, 'Weaknesses:\\s*([^;]+)', 1)                            AS weaknesses,
  CASE
    WHEN s.aggregatesentiment >= 0.5  THEN 'Very Positive'
    WHEN s.aggregatesentiment >= 0.2  THEN 'Positive'
    WHEN s.aggregatesentiment > -0.2 THEN 'Neutral'
    WHEN s.aggregatesentiment > -0.5 THEN 'Negative'
    ELSE 'Very Negative'
  END AS sentiment_bucket,

  -- Explicitly list review-side columns (exclude p.productid to avoid duplicate)
  p.reviewid,
  p.reviewtext,
  p.reviewdate,
  p.positivescore,
  p.negativescore,
  p.neutralscore,
  p.mixedscore,
  p.keyphrases,
  p.entities,
  p.raw AS raw_map,
  p.dynamoputstatus

FROM poc_processed_db.summaries_clean s
LEFT JOIN poc_processed_db.processed_clean p
  ON s.productid = p.productid;
