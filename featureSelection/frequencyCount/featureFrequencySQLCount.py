import pickle
import commonDB
import pandas as pd
import numpy
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
    :param path where to write cached copy of counts of features
    :param subject_ids subjectIDS to restrict the count to; if None, then include all
    :param hadm_ids hadm_ids to restrict feature count to; if None, then include all
    :param mapping a DataFrame with columns itemid_x and itemid_y to deal with multiple mappings to same feature
    :return dataframe with the raw count data of features per admission
    """
    conn = commonDB.getConnection()

    with open(path, "r") as f:
        query = f.read()
    if subject_ids is None:
        query.replace("<INSERT IDS HERE>", "")
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
    if (mappings is not None):
        toDrop = []
        for i in range(0, events.shape[1]):
            if events["itemid"][i] in set(mappings["itemid_x"]):
                ind = mappings[mappings["itemid_x"] == events["itemid"][i]].index
                itemid_y = mappings["itemid_y"][ind]
                countAdmissions = events["countAdmissionsPresent"][ind]
                perAdmissions = events["avgPerAdmission"][ind]
                countTotal = countAdmissions * perAdmissions
                # (avg itemid_x * count itemid_x + avg itemid_y * count itemid_y) / total_counts
                events["avgPerAdmission"][events["itemid"] == itemid_y] = (events["avgPerAdmission"][events["itemid"] == itemid_y]
                                                                           * events["countAdmissionsPresent"][events["itemid"] == itemid_y]
                                                                           + countTotal) \
                    / (countAdmissions + events["countAdmissionsPresent"][events["itemid"] == itemid_y])
                events["countAdmissionsPresent"][events["itemid"] == itemid_y] += countTotal
                toDrop.append(ind)
        events.drop(toDrop, axis=1)
    return events


def getTopNItemIDs(itemids, numToFind=100):
    """
    :param numToFind top n features to return
    :param itemids a dataframe of itemids, ordered by count
    :return  a set of numbers representing itemids
    """
    featureItemCodes = set()  # using set because itemids may show up in both labevents AND chartevents
    i = 0
    while len(featureItemCodes) < numToFind or i < len(itemids["itemid"])
            featureItemCodes.add(itemids["itemid"][i])
            i = i + 1
    return featureItemCodes  # just return the set of itemids in int format


def getDataByHadmId(hadm_ids, nitems, itemids, mapping=None):
    '''
    This function turns the wide data into a narrower format and calls on other
    functions to clean up data for all hadm_ids chosen.
    :param hadm_ids a list of all hadm_ids to include and process
    :param nitems the number of the most common features to include in final data matrix
    :param itemids stores all the itemids, ordered by frequency of counts
    :param mapping a DataFrame with columns itemid_x and itemid_y to deal with items which are the same
    :return dataframe containing all data, cleaned
    '''
    allPersons = pd.DataFrame()
    intermediateList = []
    for hadm_id in hadm_ids:
        dataEvents = getFirst24HrsDataValuesIndividually(hadm_id=hadm_id, itemids=itemids, nitems=nitems)
        intermediateList.append(commonDB.cleanUpIndividual(dataEvents, hadm_id))
    allPersons = pd.concat(intermediateList)
    allPersons.set_index("hadm_id", inplace=True)
    for column in allPersons.columns:
        allPersons[column].fillna(commonDB.cleanSeries(allPersons[column]), inplace=True)
        # remove outliers either above the 3rd std or below the 3rd std
        if allPersons[column].dtype == numpy.number:
            allPersons[column] = allPersons[column].apply(
                lambda ind: ((col.mean() + 3 * col.std()) if (ind > (col.mean() + 3 * col.std())) else ind
                             ))
            allPersons[column] = allPersons[column].apply(
                lambda ind: ((col.mean() - 3 * col.std()) if (ind < (col.mean() - 3 * col.std())) else ind
                             ))
    return allPersons


def getFirst24HrsDataValuesIndividually(hadm_id, itemids, nitems=10):
    """
    Runs an SQL Query to return featues that that returns features for top 100
    most frequent itemids of both chartevents and labevents (might overlap)
    HOWEVER, only for one hadm_id
    
    In addition uses the preprocessing module stolen from mimic3 benchmark to deal with
    most common problems in clinical data
    :param itemids variable to use for counts of features
    :param hadm_id the admission id to run query and retrieve data for
    :param nitems number of most reported features to return
    :return a Dataframe with the data
    """
    itemIds = getTopNItemIDs(itemids, numToFind=nitems)
    query = "WITH timeranges as (SELECT hadm_id, admittime, admittime + interval '24 hour' as endtime FROM admissions WHERE hadm_id = " + str(hadm_id) + "), \n"\
        + "topLabEvents as ( SELECT hadm_id, label, labevents.itemid, charttime, value, valuenum FROM labevents  LEFT JOIN d_labitems on d_labitems.itemid = labevents.itemid WHERE labevents.itemid in  \n" \
        + itemIds \
        + "\n AND hadm_id = " + str(hadm_id) + " AND charttime BETWEEN (SELECT admittime FROM timeranges) AND (SELECT endtime FROM timeranges)\n" \
        + "), topChartEvents as (SELECT hadm_id, label, chartevents.itemid, charttime, value, valuenum FROM chartevents  LEFT JOIN d_items on d_items.itemid = chartevents.itemid WHERE chartevents.itemid in \n" \
        + itemIds \
        + "\n AND hadm_id = " + str(hadm_id) + " AND charttime BETWEEN (SELECT admittime FROM timeranges) AND (SELECT endtime FROM timeranges) \n" \
        + " ) SELECT * FROM topLabEvents UNION SELECT * FROM topChartEvents ORDER BY charttime"
    conn = commonDB.getConnection()
    dataToReturn = pd.read_sql(query, conn)
    # print(query) #debug method TODO: remove or comment this out
    return dataToReturn


def getFirst24HrsDataValues():
    """
    Runs the SQL Query to return features that match the itemids of the top
    100 most frequent itemids of both chartevents and labevents (might overlap)
    WARNING: Query will return a lot, uses tons of memory at once, don't use?
    :return a Dataframe with the data from the result of sql query GetFirst24Hours.sql
    """
    conn = commonDB.getConnection()
    with open("../../data/sql/GetFirst24HoursFull.sql") as f:
        query = f.read()
    first24HourData = pd.read_sql(query, conn)
    return first24HourData
