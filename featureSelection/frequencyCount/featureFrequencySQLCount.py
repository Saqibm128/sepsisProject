import pickle
import commonDB
import pandas as pd
import numpy
import os.path


## @author Mohammed Saqib
## This file is responsible for doing frequency counts of features
##      and retrieving the attributes



def countFeatures(path="../../data/sql/perAdmissionCount.sql", write=True):
    """
    This file goes and executes queries to count the most common features in 24 hour ranges
    as well as labels, itemids, and average occurrences of feature in each admission.
    :param path where to write cached copy of counts of features
    :param write if this method should actually go ahead and write counts down
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
            features =  pd.DataFrame.from_csv(path)
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
        intermediateList.append(cleanUpIndividual(dataEvents, hadm_id))
    allPersons = pd.concat(intermediateList)
    allPersons.set_index("hadm_id", inplace = True)
    for column in allPersons.columns:
        allPersons[column].fillna(cleanSeries(allPersons[column]), inplace=True)
    return allPersons
def cleanUpIndividual(events, hadm_id):
    """
    Takes a dataframe of all events that occurred to an individual and goes through all of them. Then cleans the data up
    :param events dataframe of all events that occurred to one individual
    :param features a list of all features to look up in cleanUpIndividual
    :return a cleaned dataframe, with missing data interpolated, if possible from individual record
    """
    datamap = {}
    features = events["itemid"].unique() #Should return a series of unique itemids aka a list to iterate over for the features we decide to use
    datamap["hadm_id"] = [str(hadm_id)]
    for feature in features:
        featureVal = (events[events["itemid"] == feature])["value"]
        if featureVal.isnull().all(): #check and see if we had a null val
            featureVal = (events[events["itemid"] == feature])["valuenum"] #sets featureVal as temporary series that contains all values, then to take mean of
            featureVal = featureVal[~featureVal.duplicated(keep='first')]
            featureVal = [featureVal.mean()]
        else:
            featureVal = [featureVal.mode()[0]] #get first mode of all nonnumeric data
        datamap[str(feature)] = featureVal
    df = pd.DataFrame(datamap)
    df.set_index(["hadm_id"])
    return df

def cleanSeries(series):
    '''
    Used later on to help clean up data en masse
    :param series to clean, does a naive clean based on other stuff in series
    :return the data to fill Na data with
    '''
    if series.dtype == numpy.int_ or series.dtype == numpy.float_:
        return series.mean()
    else:
        return series.mode()[0]

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
        + "topLabEvents as ( SELECT hadm_id, label, labevents.itemid, charttime, value, valuenum FROM labevents  LEFT JOIN d_labitems on d_labitems.itemid = labevents.itemid WHERE labevents.itemid in ( \n" \
        + itemIds \
        + "\n) AND hadm_id = " + str(hadm_id)  + " AND charttime BETWEEN (SELECT admittime FROM timeranges) AND (SELECT endtime FROM timeranges)\n" \
        + "), topChartEvents as (SELECT hadm_id, label, chartevents.itemid, charttime, value, valuenum FROM chartevents  LEFT JOIN d_items on d_items.itemid = chartevents.itemid WHERE chartevents.itemid in (\n" \
        + itemIds \
        + ")\n AND hadm_id = " + str(hadm_id)  + " AND charttime BETWEEN (SELECT admittime FROM timeranges) AND (SELECT endtime FROM timeranges) \n" \
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
