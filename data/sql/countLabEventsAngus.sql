-- Literally copies Angus SQL code to narrow features down to sepsis suspect subpopulation

WITH infection_group AS (
	SELECT subject_id, hadm_id,
	CASE
		WHEN substring(icd9_code,1,3) IN ('001','002','003','004','005','008',
			   '009','010','011','012','013','014','015','016','017','018',
			   '020','021','022','023','024','025','026','027','030','031',
			   '032','033','034','035','036','037','038','039','040','041',
			   '090','091','092','093','094','095','096','097','098','100',
			   '101','102','103','104','110','111','112','114','115','116',
			   '117','118','320','322','324','325','420','421','451','461',
			   '462','463','464','465','481','482','485','486','494','510',
			   '513','540','541','542','566','567','590','597','601','614',
			   '615','616','681','682','683','686','730') THEN 1
		WHEN substring(icd9_code,1,4) IN ('5695','5720','5721','5750','5990','7110',
				'7907','9966','9985','9993') THEN 1
		WHEN substring(icd9_code,1,5) IN ('49121','56201','56203','56211','56213',
				'56983') THEN 1
		ELSE 0 END AS infection
	FROM diagnoses_icd),
-- Appendix 2: ICD9-codes (organ dysfunction)
	organ_diag_group as (
	SELECT subject_id, hadm_id,
		CASE
		-- Acute Organ Dysfunction Diagnosis Codes
		WHEN substring(icd9_code,1,3) IN ('458','293','570','584') THEN 1
		WHEN substring(icd9_code,1,4) IN ('7855','3483','3481',
				'2874','2875','2869','2866','5734')  THEN 1
		ELSE 0 END AS organ_dysfunction,
		-- Explicit diagnosis of severe sepsis or septic shock
		CASE
		WHEN substring(icd9_code,1,5) IN ('99592','78552')  THEN 1
		ELSE 0 END AS explicit_sepsis
	FROM diagnoses_icd),

-- Mechanical ventilation
	organ_proc_group as (
	SELECT subject_id, hadm_id,
		CASE
		WHEN substring(icd9_code,1,4) IN ('9670','9671','9672') THEN 1
		ELSE 0 END AS mech_vent
	FROM procedures_icd),

-- Aggregate
	aggregate as (
	SELECT subject_id, hadm_id,
		CASE
		WHEN hadm_id in (SELECT DISTINCT hadm_id
				FROM infection_group
				WHERE infection = 1) THEN 1
			ELSE 0 END AS infection,
		CASE
		WHEN hadm_id in (SELECT DISTINCT hadm_id
				FROM organ_diag_group
				WHERE explicit_sepsis = 1) THEN 1
			ELSE 0 END AS explicit_sepsis,
		CASE
		WHEN hadm_id in (SELECT DISTINCT hadm_id
				FROM organ_diag_group
				WHERE organ_dysfunction = 1) THEN 1
			ELSE 0 END AS organ_dysfunction,
		CASE
		WHEN hadm_id in (SELECT DISTINCT hadm_id
				FROM organ_proc_group
				WHERE mech_vent = 1) THEN 1
			ELSE 0 END AS mech_vent
	FROM admissions),
  angus_group as (
    -- List angus score for each admission
    SELECT subject_id, hadm_id, infection,
    	   explicit_sepsis, organ_dysfunction, mech_vent,
    	CASE
    	WHEN explicit_sepsis = 1 THEN 1
    	WHEN infection = 1 AND organ_dysfunction = 1 THEN 1
    	WHEN infection = 1 AND mech_vent = 1 THEN 1
    	ELSE 0 END AS Angus
    FROM aggregate;
  ),
  specific_lab_events as(
    SELECT * FROM labevents
    WHERE labevents.hadm_id in (SELECT hadm_id FROM angus_group)
  ),
  noLabelNonDistinctCount as (
    -- This first query gets a "naive" count of the lab events
    SELECT specific_lab_events.itemid, d_labitems.label, COUNT(specific_lab_events.itemid) as nondistinctCountOfTests
    FROM specific_lab_events
    RIGHT JOIN d_labitems on d_labitems.itemid = specific_lab_events.itemid
    GROUP BY specific_lab_events.itemid, d_labitems.label
  ),
  distinctAdmissions as (
    -- We take distinct combination of lab tests and each hospital admission
    SELECT DISTINCT hadm_id, specific_lab_events.itemid
    FROM specific_lab_events
  ),
  distinctAdmissionsCount as (
    -- If we take multiple tests each hospital admission, we only count it once
    SELECT itemid, COUNT(itemid) as labEventsCountsDistinctAdmissions
    FROM distinctAdmissions
    GROUP BY itemid
  ),
  distinctSubjects as (
    -- same, except by subject
    SELECT DISTINCT subject_id, specific_lab_events.itemid
      FROM specific_lab_events
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
