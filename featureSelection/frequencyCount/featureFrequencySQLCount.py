import pickle
import commonDB
import pandas as pd
import numpy
import os.path


## @author Mohammed Saqib
## This file is responsible for doing frequency counts of features
##      and retrieving the attributes



def countFeatures(path="../../data/sql/perAdmissionCount.sql"):
    """
    This file goes and executes queries to count the most common features in 24 hour ranges
    as well as labels, itemids, and average occurrences of feature in each admission.
    :param path where to write cached copy of counts of features
    :return dataframe with the raw count data of features per admission
    """
    conn = commonDB.getConnection()

    with open(path, "rb") as f:
        query = f.read()
    chartEvents = pd.read_sql(query, conn)
    # print(chartEvents)
    chartEvents.to_csv("data/rawdatafiles/counts.csv")
    return chartEvents

def getTopNItemIDs(numToFind = 100, sqlFormat = True, path="../../data/rawdatafiles/counts.csv", sqlPath="data/sql/perAdmissionCount.sql"):
    """
    :precondition labEventCountsAngus.p and chartEventCountsAngus.p were created, otherwise they will be recreated
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
            features =  pd.from_csv(path) #avoid weird conflict lines
    featureItemCodes = set() #using set because itemids may show up in both labevents AND chartevents
    for i in range(0, numToFind):
        if sqlFormat:
            featureItemCodes.add("\'" + str(features["itemid"][i]) + "\'")
        else:
            featureItemCodes.add(features["itemid"][i])
    if sqlFormat: #Go ahead and return a string
        toReturn = ""
        for itemIdString in featureItemCodes:
            toReturn = toReturn + itemIdString + ", "
        return toReturn[:-2] #remove last comma and space
    return featureItemCodes #just return the set of itemids in int format

def cleanUpIndividual(events, hadm_id):
    """
    Takes a dataframe of all events that occurred to an individual and goes through all of them. Then cleans the data up
    :param events dataframe of all events that occurred to one individual
    :param features a list of all features to look up in cleanUpIndividual
    :return a cleaned dataframe, with missing data interpolated, if possible from individual record
    """
    datamap = {}
    features = events["itemid"].unique() #Should return a series of unique itemids
    datamap["hadm_id"] = [hadm_id]
    for feature in features:
        featureVal = (events[events["itemid"] == feature])["value"]
        if featureVal.isnull().all(): #check and see if we had a null val
            featureVal = (events[events["itemid"] == feature])["valuenum"]
            featureVal = [featureVal.mean()]
        else:
            featureVal = [featureVal.mode()[0]] #get first mode of all nonnumeric data
        datamap[feature] = featureVal
    return pd.DataFrame(data=datamap, columns=features)



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
    query = "WITH timeranges as (SELECT hadm_id, admittime, admittime + interval '24 hour' as endtime FROM admissions WHERE hadm_id = " + str(hadm_id)+ "),"\
        + "topLabEvents as ( SELECT hadm_id, itemid, charttime, value, valuenum FROM labevents WHERE labevents.itemid in (" \
        + itemIds \
        + ") AND hadm_id = " + str(hadm_id)  + " AND charttime BETWEEN (SELECT admittime FROM timeranges) AND (SELECT endtime FROM timeranges)" \
        + "), topChartEvents as (SELECT hadm_id, itemid, charttime, value, valuenum FROM chartevents WHERE chartevents.itemid in (" \
        + itemIds \
        + ") AND hadm_id = " + str(hadm_id)  + " AND charttime BETWEEN (SELECT admittime FROM timeranges) AND (SELECT endtime FROM timeranges)" \
        + " ) SELECT * FROM topLabEvents UNION SELECT * FROM topChartEvents ORDER BY charttime"
    conn = commonDB.getConnection()
    dataToReturn = pd.read_sql(query, conn)
    # print(query) #debug method TODO: remove or comment this out
    return dataToReturn

def getFirst24HrsDataValues():
    """
    Runs the SQL Query to return features that match the itemids of the top
    100 most frequent itemids of both chartevents and labevents (might overlap)
    WARNING: Query will return a lot, uses tons of memory at once
    :return a Dataframe with the data from the result of sql query GetFirst24Hours.sql
    """
    conn = commonDB.getConnection()
    with open("../../data/sql/GetFirst24HoursFull.sql") as f:
        query = f.read()
    first24HourData = pd.read_sql(query, conn)
    return first24HourData
if __name__ == "__main__":
    counts = countFeatures()
    print(counts)
    counts.to_csv("../../data/rawdatafiles/counts.csv")
    # hadm_ids = commonDB.getAllHADMID()["hadm_id"]
    # allPersons = pd.DataFrame()
    # for hadm_id in hadm_ids:
    #     dataEvents = getFirst24HrsDataValuesIndividually(hadm_id=hadm_id, nitems = 10)
    #     allPersons = pd.concat([allPersons, cleanUpIndividual(dataEvents, hadm_id)])
    # allPersons.to_csv("data/rawdatafiles/testPersonsData.csv")
# # getCountOfFeaturesAngus()
# print(getTopNItemIDs(numToFind = 5))
#
# data = getFirst24HrsDataValues()
# data.to_csv("data/rawdatafiles/first24Hours.csv")
