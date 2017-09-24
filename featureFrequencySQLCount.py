import pickle
import commonDB
import pandas as pd
import numpy


## @author Mohammed Saqib
## This file is responsible for doing frequency counts of features
##      and retrieving the attributes



def countFeatures():
    """
    This file goes and executes the countLabEventsAngus and countChartEventsAngus sql
    query. In addition, writes the resulting data to file as pickle
    :return dataframe with the raw count data of features per admission
    """
    conn = commonDB.getConnection()
    with open("data/sql/countLabEventsAngus.sql") as f:
        query = f.read()
    labEvents = pd.read_sql(query, conn)

    pickle.dump(labEvents, open("data/rawdatafiles/labEventCountsAngus.p", "wb"))

    with open("data/sql/perAdmissionCount.sql") as f:
        query = f.read()
    chartEvents = pd.read_sql(query, conn)

    # print(chartEvents)

    return chartEvents

def getTopNItemIDs(numToFind = 100, sqlFormat = True):
    """
    :precondition labEventCountsAngus.p and chartEventCountsAngus.p were created, otherwise they will be recreated
    :param numToFind top n features to return
    :param sqlFormat True to return sql format (string representation), else array of numbers (type int)
    :return string of itemid's if sqlFormat is true, properly formatted as per sqlFormat OR a set of numbers representing itemids
    """
    if not os.path.isfile("data/rawdatafiles/labEventCountsAngus.p"):
        getCountOfFeaturesAngus()
    with open("data/rawdatafiles/labEventCountsAngus.p", "rb") as f:
        labEvents = pickle.load(f)
    with open("data/rawdatafiles/chartEventCountsAngus.p", "rb") as f:
        chartEvents = pickle.load(f)
    featureItemCodes = set() #using set because itemids may show up in both labevents AND chartevents
    for i in range(0, numToFind):
        if sqlFormat:
            featureItemCodes.add("\'" + str(labEvents.values[i, 0]) + "\'")
        else:
            featureItemCodes.add(labEvents.values[i, 0])
    for i in range(0, numToFind):
        if sqlFormat:
            featureItemCodes.add("\'" + str(chartEvents.values[i, 0]) + "\'")
        else:
            featureItemCodes.add(labEvents.values[i, 0])
    if sqlFormat: #Go ahead and return a string
        toReturn = ""
        for itemIdString in featureItemCodes:
            toReturn = toReturn + itemIdString + ", "
        return toReturn[:-2] #remove last comma and space
    return featureItemCodes #just return the set of python ints

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



def getFirst24HrsDataValuesIndividually(hadm_id, nitems = 10):
    """
    Runs an SQL Query to return featues that that returns features for top 100
    most frequent itemids of both chartevents and labevents (might overlap)
    HOWEVER, only for one hadm_id
    :param hadm_id the admission id to run query and retrieve data for
    :param nitems number of most reported features to return
    :return a Dataframe with the data
    """
    itemIds = getTopNItemIDs(numToFind = nitems)
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
    with open("data/sql/GetFirst24HoursFull.sql") as f:
        query = f.read()
    first24HourData = pd.read_sql(query, conn)
    return first24HourData
if __name__ == "__main__":
    counts = countFeatures()
    print(counts)
    counts.to_csv("data/rawdatafiles/counts.csv")
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
