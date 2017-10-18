import pickle
import commonDB
import pandas as pd
import numpy
import os.path


## @author Mohammed Saqib
## This file is responsible for doing frequency counts of features
##      and retrieving the attributes



def countFeatures(subject_ids=None, hadm_ids=None, path="data/sql/perAdmissionCount.sql", write=True):
    """
    This file goes and executes queries to count the most common features in 24 hour ranges
    as well as labels, itemids, and average occurrences of feature in each admission for the
    10,282 matched subset.
    :param path where to write cached copy of counts of features
    :param write if this method should actually go ahead and write counts down
    :param subject_ids subjectIDS to restrict the count to; if None, then include all
    :param hadm_ids hadm_ids to restrict feature count to; if None, then include all
    :return dataframe with the raw count data of features per admission
    """
    conn = commonDB.getConnection()

    with open(path, "r") as f:
        query = f.read()
    if subject_ids is None:
        query.replace("<INSERT IDS HERE>", "")
    else:
        query = query.replace("<INSERT IDS HERE>", "AND subject_id in " + commonDB.convertListToSQL(subject_ids))
    if hadm_ids is None:
        query = query.replace("<INSERT lab_events hadm_ids HERE>", "")
        query = query.replace("<INSERT chart_events hadm_ids HERE>", "")
    else:
        query = query.replace("<INSERT hadm_ids HERE>", "AND timeranges.hadm_id in" + commonDB.convertListToSQL(hadm_ids))
        query = query.replace("<INSERT labevents hadm_ids HERE>", "AND labevents.hadm_id in" + commonDB.convertListToSQL(hadm_ids))
        query = query.replace("<INSERT chartevents hadm_ids HERE>", "AND chartevents.hadm_id in" + commonDB.convertListToSQL(hadm_ids))
    events = pd.read_sql(query, conn)
    return events

def getTopNItemIDs(numToFind = 100, sqlFormat = True, path="data/rawdatafiles/counts.csv", sqlPath="data/sql/perAdmissionCount.sql"):
    """
    :param numToFind top n features to return
    :param path where a cached copy of counts may be held
    :param sqlPath where a copy of the sql code to count frequencies is kept
    :param sqlFormat True to return sql format (string representation), else array of numbers (type int)
    :return string of itemid's if sqlFormat is true, properly formatted as per sqlFormat OR a set of numbers representing itemids
    """
    if not os.path.isfile(path):
        features = countFeatures(path=sqlPath)
    else:
        with open(path, "rb") as f:
            features =  pd.DataFrame.from_csv(path)
            features.to_csv("data/rawdatafiles/counts.csv")
    featureItemCodes = set() #using set because itemids may show up in both labevents AND chartevents
    for i in range(0, numToFind):
        if sqlFormat:
            featureItemCodes.add("\'" + str(features["itemid"][i]) + "\'")
        else:
            featureItemCodes.add(features["itemid"][i])
    if sqlFormat: #Go ahead and return a string
        return commonDB.convertListToSQL(featureItemCodes)
    return featureItemCodes #just return the set of itemids in int format
def getDataAllHadmId(hadm_ids, nitems, path="data/rawdatafiles/counts.csv"):
    '''
    :param hadm_ids a list of all hadm_ids to include and process
    :param nitems the number of the most common features to include in final data matrix
    :param path, where to read most common features from (Cached from earlier sql query)
    :return dataframe containing all data, cleaned
    '''
    allPersons = pd.DataFrame()
    intermediateList = []
    for hadm_id in hadm_ids:
        dataEvents = getFirst24HrsDataValuesIndividually(hadm_id=hadm_id, nitems = nitems, path=path)
        intermediateList.append(commonDB.cleanUpIndividual(dataEvents, hadm_id))
    allPersons = pd.concat(intermediateList)
    allPersons.set_index("hadm_id", inplace = True)
    for column in allPersons.columns:
        allPersons[column].fillna(commonDB.cleanSeries(allPersons[column]), inplace=True)
        # remove outliers either above the 3rd std or below the 3rd std
        if allPersons[column].dtype == numpy.number:
            allPersons[column] = allPersons[column].apply(
                lambda ind: ((col.mean() + 3 * col.std()) if  (ind > (col.mean() + 3 * col.std())) else ind \
                ))
            allPersons[column] = allPersons[column].apply(
                lambda ind: ((col.mean() - 3 * col.std()) if  (ind < (col.mean() - 3 * col.std())) else ind \
                ))
    return allPersons


def getFirst24HrsDataValuesIndividually(hadm_id, nitems = 10, path="../../data/rawdatafiles/counts.csv"):
    """
    Runs an SQL Query to return featues that that returns features for top 100
    most frequent itemids of both chartevents and labevents (might overlap)
    HOWEVER, only for one hadm_id
    :param path variable to use for cached counts of features
    :param hadm_id the admission id to run query and retrieve data for
    :param nitems number of most reported features to return
    :return a Dataframe with the data
    """
    itemIds = getTopNItemIDs(numToFind = nitems, path=path)
    query = "WITH timeranges as (SELECT hadm_id, admittime, admittime + interval '24 hour' as endtime FROM admissions WHERE hadm_id = " + str(hadm_id)+ "), \n"\
        + "topLabEvents as ( SELECT hadm_id, label, labevents.itemid, charttime, value, valuenum FROM labevents  LEFT JOIN d_labitems on d_labitems.itemid = labevents.itemid WHERE labevents.itemid in  \n" \
        + itemIds \
        + "\n AND hadm_id = " + str(hadm_id)  + " AND charttime BETWEEN (SELECT admittime FROM timeranges) AND (SELECT endtime FROM timeranges)\n" \
        + "), topChartEvents as (SELECT hadm_id, label, chartevents.itemid, charttime, value, valuenum FROM chartevents  LEFT JOIN d_items on d_items.itemid = chartevents.itemid WHERE chartevents.itemid in \n" \
        + itemIds \
        + "\n AND hadm_id = " + str(hadm_id)  + " AND charttime BETWEEN (SELECT admittime FROM timeranges) AND (SELECT endtime FROM timeranges) \n" \
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
