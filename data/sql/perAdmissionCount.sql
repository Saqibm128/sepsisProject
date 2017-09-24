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
  WHERE (charttime, charttime) OVERLAPS (timeranges.admittime, timeranges.endtime)
),
charteventsInRange as (
  SELECT chartevents.hadm_id, itemid, charttime, value, valuenum
  FROM chartevents
  LEFT JOIN timeranges on timeranges.hadm_id = chartevents.hadm_id
  WHERE (charttime, charttime) OVERLAPS (timeranges.admittime, timeranges.endtime)
),
distinctChartAdmissions as (
  -- We take distinct combination of lab tests and each hospital admission
  SELECT DISTINCT hadm_id, itemid, count(itemid) as count
  FROM charteventsInRange
  GROUP BY hadm_id, itemid
),
chartEventCount as (
  -- If we take multiple tests each hospital admission, we only count it once
  SELECT itemid, COUNT(itemid) as countPerAdmission, AVG(count)
  FROM distinctChartAdmissions
  GROUP BY itemid
),
distinctLabAdmissions as (
  -- We take distinct combination of lab tests and each hospital admission
  SELECT DISTINCT hadm_id, itemid, count(itemid) as count
  FROM labeventsInRange
  GROUP BY hadm_id, itemid
),
labEventCount as (
  -- If we take multiple tests each hospital admission, we only count it once
  SELECT itemid, COUNT(itemid) as countPerAdmission, AVG(count) as avgPerAdmission
  FROM distinctLabAdmissions
  GROUP BY itemid
),
allCounts as (
  SELECT * FROM labEventCount
  UNION
  SELECT * FROM chartEventCount
),
firstJoin as (SELECT allCounts.itemid, label, countPerAdmission, avgPerAdmission FROM allCounts
LEFT JOIN d_items on d_items.itemid = allCounts.itemid),
secondJoin as (SELECT allCounts.itemid, label, countPerAdmission, avgPerAdmission FROM allCounts
LEFT JOIN d_labitems on d_labitems.itemid = allCounts.itemid)
SELECT * FROM firstJoin UNION SELECT * FROM secondJoin
