WITH timeranges as (
  SELECT hadm_id, admittime, admittime + interval '24 hour' as endtime
  FROM admissions
), topLabEvents as (
  SELECT * FROM labevents WHERE labevents.itemid in ('80', '211', '467', '51330', '51165', '51126', '31', '51477', '51349', '926')
), topChartEvents as (
  SELECT * FROM chartevents WHERE chartevents.itemid in ('80', '211', '467', '51330', '51165', '51126', '31', '51477', '51349', '926')
), labeventsInRange as (
  SELECT topLabEvents.hadm_id, itemid, charttime, value, valuenum
  FROM topLabEvents
  LEFT JOIN timeranges on timeranges.hadm_id = topLabEvents.hadm_id
  WHERE (charttime, charttime) OVERLAPS (timeranges.admittime, timeranges.endtime)
), charteventsInRange as (
  SELECT topChartEvents.hadm_id, itemid, charttime, value, valuenum
  FROM topChartEvents
  LEFT JOIN timeranges on timeranges.hadm_id = topChartEvents.hadm_id
  WHERE (charttime, charttime) OVERLAPS (timeranges.admittime, timeranges.endtime)
)
SELECT *
FROM labeventsInRange
UNION
SELECT *
FROM charteventsInRange
