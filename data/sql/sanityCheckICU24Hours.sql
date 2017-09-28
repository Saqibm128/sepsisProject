-- Sanity check, see if chart events at least exist for most patients within window of interest??
WITH possibleRanges as (
  SELECT hadm_id, admittime + interval "24 hour" as endTime
  FROM admissions
), firstICU as (
  SELECT distinct hadm_id, min(charttime) as firstICU
  FROM chartevents
  GROUP BY hadm_id
),
SELECT possibleRanges.hadm_id, endtime, firstICU
FROM possibleRanges
LEFT JOIN firstICU on firstICU.hadm_id = possibleRanges.hadm_id
