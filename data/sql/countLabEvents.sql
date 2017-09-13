-- @author Mohammed Saqib
-- This sql query counts the occurrence of lab event, as well as the occurrence of lab events for each unique patients

WITH noLabelNonDistinctCount as (
  -- This first query gets a "naive" count of the lab events
  SELECT labevents.itemid, d_labitems.label, COUNT(labevents.itemid) as nondistinctCountOfTests
  FROM labevents
  RIGHT JOIN d_labitems on d_labitems.itemid = labevents.itemid
  GROUP BY labevents.itemid, d_labitems.label
),
distinctAdmissions as (
  -- We take distinct combination of lab tests and each hospital admission
  SELECT DISTINCT hadm_id, labevents.itemid
  FROM labevents
),
distinctAdmissionsCount as (
  -- If we take multiple tests each hospital admission, we only count it once
  SELECT itemid, COUNT(itemid) as labEventsCountsDistinctAdmissions
  FROM distinctAdmissions
  GROUP BY itemid
),
distinctSubjects as (
  -- same, except by subject
  SELECT DISTINCT subject_id, labevents.itemid
    FROM labevents
),
distinctSubjectsCounts as (
  -- if we take multiple tests for each patient, we only count it once
  SELECT itemid, COUNT(itemid) as labEventsCountsDistinctSubjects
  FROM distinctSubjects
  GROUP BY itemid
),
firstJoin as (
SELECT label, noLabelNonDistinctCount.itemid, nondistinctCountOfTests, labEventsCountsDistinctAdmissions
FROM noLabelNonDistinctCount
JOIN distinctAdmissionsCount on distinctAdmissionsCount.itemid = noLabelNonDistinctCount.itemid
)

SELECT label, firstJoin.itemid, nondistinctCountOfTests, labEventsCountsDistinctAdmissions, labEventsCountsDistinctSubjects
FROM firstJoin
JOIN distinctSubjectsCounts on distinctSubjectsCounts.itemid = firstJoin.itemid
ORDER BY nondistinctCountOfTests desc
