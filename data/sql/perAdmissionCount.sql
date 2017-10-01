-- @author Mohammed Saqib
-- This sql query counts the occurrence of lab/chart event per admission, as well as average number of measurements per admission

WITH timeranges as (
SELECT hadm_id, admittime, admittime + interval '4 hour' as endtime
FROM admissions
),
labeventsInRange as (
  SELECT labevents.hadm_id, itemid, charttime, value, valuenum
  FROM labevents
  LEFT JOIN timeranges on timeranges.hadm_id = labevents.hadm_id
  WHERE (charttime, charttime) OVERLAPS (timeranges.admittime, timeranges.endtime) AND subject_id in <INSERT IDS HERE>
),
charteventsInRange as (
  SELECT chartevents.hadm_id, itemid, charttime, value, valuenum
  FROM chartevents
  LEFT JOIN timeranges on timeranges.hadm_id = chartevents.hadm_id
  WHERE (charttime, charttime) OVERLAPS (timeranges.admittime, timeranges.endtime) AND subject_id in <INSERT IDS HERE>
),
distinctChartAdmissions as (
  -- We take distinct combination of lab tests and each hospital admission
  SELECT DISTINCT hadm_id, label, charteventsInRange.itemid, count(charteventsInRange.itemid) as count
  FROM charteventsInRange
  LEFT JOIN d_items ON charteventsInRange.itemid = d_items.itemid
  GROUP BY hadm_id, charteventsInRange.itemid, label
),
chartEventCount as (
  -- If we take multiple tests each hospital admission, we only count it once
  SELECT itemid, label, COUNT(itemid) as countPerAdmission, AVG(count)
  FROM distinctChartAdmissions
  GROUP BY itemid, label
),
distinctLabAdmissions as (
  -- We take distinct combination of lab tests and each hospital admission
  SELECT DISTINCT hadm_id, label, labeventsInRange.itemid, count(labeventsInRange.itemid) as count
  FROM labeventsInRange
  LEFT JOIN d_labitems ON labeventsInRange.itemid = d_labitems.itemid
  GROUP BY hadm_id, labeventsInRange.itemid, label
),
labEventCount as (
  -- If we take multiple tests each hospital admission, we only count it once
  SELECT itemid, label, COUNT(itemid) as countPerAdmission, AVG(count) as avgPerAdmission
  FROM distinctLabAdmissions
  GROUP BY itemid, label
)
  SELECT * FROM labEventCount
  UNION
  SELECT * FROM chartEventCount
  ORDER BY countPerAdmission desc
