-- @author Mohammed Saqib
-- This sql query counts the occurrence of lab event, as well as the occurrence of lab events for each unique patients

WITH noLabelNonDistinctCount as (
  -- This first query gets a "naive" count of the lab events
  SELECT chartevents.itemid, d_items.label, COUNT(chartevents.itemid) as nondistinctCountOfTests
  FROM chartevents
  RIGHT JOIN d_items on d_items.itemid = chartevents.itemid
  GROUP BY chartevents.itemid, d_items.label
),
distinctAdmissions as (
  -- We take distinct combination of lab tests and each hospital admission
  SELECT DISTINCT hadm_id, chartevents.itemid
  FROM chartevents
),
distinctAdmissionsCount as (
  -- If we take multiple tests each hospital admission, we only count it once
  SELECT itemid, COUNT(itemid) as charteventsCountsDistinctAdmissions
  FROM distinctAdmissions
  GROUP BY itemid
),
distinctSubjects as (
  -- same, except by subject
  SELECT DISTINCT subject_id, chartevents.itemid
    FROM chartevents
),
distinctSubjectsCounts as (
  -- if we take multiple tests for each patient, we only count it once
  SELECT itemid, COUNT(itemid) as charteventsCountsDistinctSubjects
  FROM distinctSubjects
  GROUP BY itemid
),
firstJoin as (
SELECT label, noLabelNonDistinctCount.itemid, nondistinctCountOfTests, charteventsCountsDistinctAdmissions
FROM noLabelNonDistinctCount
JOIN distinctAdmissionsCount on distinctAdmissionsCount.itemid = noLabelNonDistinctCount.itemid
)

SELECT label, firstJoin.itemid, nondistinctCountOfTests, charteventsCountsDistinctAdmissions, charteventsCountsDistinctSubjects
FROM firstJoin
JOIN distinctSubjectsCounts on distinctSubjectsCounts.itemid = firstJoin.itemid
ORDER BY nondistinctCountOfTests desc
