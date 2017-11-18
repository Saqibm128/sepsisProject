import pickle
import commonDB
import pandas as pd
import numpy as np
import os.path
from preprocessing import preprocessing


# @author Mohammed Saqib
# This file is responsible for doing frequency counts of features
# and retrieving the attributes


def countFeatures(subject_ids=None, hadm_ids=None, path="data/sql/perAdmissionCount.sql", mapping=None):
    """
    This file goes and executes queries to count the most common features in 24 hour ranges
    as well as labels, itemids, and average occurrences of feature in each admission for the
    10,282 matched subset.
    :param path where sql query is stored
    :param subject_ids subjectIDS to restrict the count to; if None, then include all
    :param hadm_ids hadm_ids to restrict feature count to; if None, then include all
    :param mapping a DataFrame with columns itemid_x and itemid_y to deal with multiple mapping to same feature
    :return dataframe with the raw count data of features per admission
    """
    conn = commonDB.getConnection()

    with open(path, "r") as f:
        query = f.read()
    if subject_ids is None:
        query = query.replace("<INSERT IDS HERE>", "")
    else:
        query = query.replace("<INSERT IDS HERE>", "AND subject_id in " +
                              commonDB.convertListToSQL(subject_ids))
    if hadm_ids is None:
        query = query.replace("<INSERT labevents hadm_ids HERE>", "")
        query = query.replace("<INSERT chartevents hadm_ids HERE>", "")
    else:
        query = query.replace("<INSERT hadm_ids HERE>",
                              "AND timeranges.hadm_id in" + commonDB.convertListToSQL(hadm_ids))
        query = query.replace("<INSERT labevents hadm_ids HERE>",
                              "AND labevents.hadm_id in" + commonDB.convertListToSQL(hadm_ids))
        query = query.replace("<INSERT chartevents hadm_ids HERE>",
                              "AND chartevents.hadm_id in" + commonDB.convertListToSQL(hadm_ids))
    events = pd.read_sql(query, conn)
    # This was added in when we realized some different itemids map to same concept
    if (mapping is not None):
        toDrop = []
        for i in range(0, events.shape[1]):
            if events["itemid"][i] in set(mapping["itemid_x"]):
                ind = mapping[mapping["itemid_x"] == events["itemid"][i]].index
                itemid_y = mapping["itemid_y"][ind]
                countAdmissions = events["countAdmissionsPresent"][ind]
                perAdmissions = events["avgPerAdmission"][ind]
                countTotal = countAdmissions * perAdmissions
                # converts stats calculated during sql query to usable form here when using the mapping
                # (avg itemid_x * count itemid_x + avg itemid_y * count itemid_y) / total_counts
                events["avgPerAdmission"][events["itemid"] == itemid_y] = (events["avgPerAdmission"][events["itemid"] == itemid_y]
                                                                           * events["countAdmissionsPresent"][events["itemid"] == itemid_y]
                                                                           + countTotal) \
                    / (countAdmissions + events["countAdmissionsPresent"][events["itemid"] == itemid_y])
                events["countAdmissionsPresent"][events["itemid"] == itemid_y] += countTotal
                toDrop.append(ind)
        events.drop(toDrop, axis=1)
    return events


def getDataByHadmId(hadm_ids, itemids, mapping=None, mustinclude=None, ranges=None):
    '''
    This function turns the wide data into a narrower format and calls on other
    functions to clean up data for all hadm_ids chosen.
    :param hadm_ids a list of all hadm_ids to include and process
    :param itemids features to include in final data matrix
    :param mapping a DataFrame with columns itemid and variable to deal with items which are the same
                    if None, will use itemid only
    :param ranges a Dataframe that holds the possible and sane ranges for variables from the MIMIC dataset
    :param mustinclude a collection of itemids that must be included TODO
    :return dataframe containing all data, cleaned
    '''
    allPersons = pd.DataFrame()
    intermediateList = []
    for hadm_id in hadm_ids:
        dataEvents = getFirst24HrsDataValuesIndividually(hadm_id=hadm_id, itemids=itemids, mapping=mapping)
        dataEvents = preprocessing.clean_events(dataEvents, ranges=ranges)
        intermediateList.append(commonDB.consolidateEvents(dataEvents, hadm_id))
    allPersons = pd.concat(intermediateList)
    allPersons.set_index("hadm_id", inplace=True)
    # for column in allPersons.columns:
    #     allPersons[column].fillna(commonDB.cleanSeries(allPersons[column]), inplace=True)
    #     # remove outliers either above the 3rd std or below the 3rd std
    #     if allPersons[column].dtype == np.number:
    #         allPersons[column] = allPersons[column].apply(
    #             lambda ind: ((col.mean() + 3 * col.std()) if (ind > (col.mean() + 3 * col.std())) else ind
    #                          ))
    #         allPersons[column] = allPersons[column].apply(
    #             lambda ind: ((col.mean() - 3 * col.std()) if (ind < (col.mean() - 3 * col.std())) else ind
    #                          ))
    return allPersons


def getFirst24HrsDataValuesIndividually(hadm_id, itemids, mapping=None):
    """
    Runs an SQL Query to return featues that that returns features for top 100
    most frequent itemids of both chartevents and labevents (might overlap)
    HOWEVER, only for one hadm_id

    In addition uses the preprocessing csv stolen from mimic3 benchmark to deal with
    most common problems in clinical data
    :param itemids variable to use for counts of features
    :param hadm_id the admission id to run query and retrieve data for
    :param mapping a DataFrame with columns itemid and variable to translate from former to latter
            mapping is used to deal with multiple itemids that are essentially the same concept
    :return a Dataframe with the data
    """
    query = "WITH timeranges as (SELECT hadm_id, admittime, admittime + interval '24 hour' as endtime FROM admissions WHERE hadm_id = " + str(hadm_id) + "), \n"\
        + "topLabEvents as ( SELECT hadm_id, label, labevents.itemid, charttime, value, valuenum, valueuom FROM labevents  LEFT JOIN d_labitems on d_labitems.itemid = labevents.itemid WHERE labevents.itemid in  \n" \
        + commonDB.convertListToSQL(itemids) \
        + "\n AND hadm_id = " + str(hadm_id) + " AND charttime BETWEEN (SELECT admittime FROM timeranges) AND (SELECT endtime FROM timeranges)\n" \
        + "), topChartEvents as (SELECT hadm_id, label, chartevents.itemid, charttime, value, valuenum, valueuom FROM chartevents  LEFT JOIN d_items on d_items.itemid = chartevents.itemid WHERE chartevents.itemid in \n" \
        + commonDB.convertListToSQL(itemids) \
        + "\n AND hadm_id = " + str(hadm_id) + " AND charttime BETWEEN (SELECT admittime FROM timeranges) AND (SELECT endtime FROM timeranges) \n" \
        + " ) SELECT * FROM topLabEvents UNION SELECT * FROM topChartEvents ORDER BY charttime"
    conn = commonDB.getConnection()
    dataToReturn = pd.read_sql(query, conn)
    #default variable name is itemid if we cannot find the correct translation in the mapping dataframe we pass in
    if mapping is not None:
        mapping = mapping[["itemid", "variable"]] #when we merge, we want to discard miscellaneous columns for clarity TODO: do we need this line?
        dataToReturn = dataToReturn.merge(mapping, left_on=['itemid'], right_on=['itemid'], how='left')
        # dataToReturn.loc[:, "variable"].fillna(dataToReturn['itemid']) TODO: fix this?
    else:
        dataToReturn["variable"] = dataToReturn["itemid"]
    return dataToReturn
