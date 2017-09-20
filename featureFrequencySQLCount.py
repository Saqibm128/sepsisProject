import pickle
import commonDB
import pandas as pd
# from matplotlib import pyplot as plt


## @author Mohammed Saqib
## This file is responsible for getting out all results from sql queries.
##      It does not do much else, because getting the queries
##      takes a lot of time. Once done, it places the data as a raw pickle file


def getCountOfFeaturesAngus():
    """
    This file goes and executes the countLabEventsAngus and countChartEventsAngus sql
    query. In addition, writes the resulting data to file as pickle
    :postcondition writes to file a pickle holding the raw count data of features per admission
    """
    conn = commonDB.getConnection()
    with open("data/sql/countLabEventsAngus.sql") as f:
        query = f.read()
    labEvents = pd.read_sql(query, conn)

    pickle.dump(labEvents, open("data/rawdatafiles/labEventCountsAngus.p", "wb"))

    with open("data/sql/countChartEventsAngus.sql") as f:
        query = f.read()
    chartEvents = pd.read_sql(query, conn)

    # print(chartEvents)

    pickle.dump(chartEvents, open("data/rawdatafiles/chartEventCountsAngus.p", "wb"))

def getTopNItemIDs(numToFind = 100, sqlFormat = True):
    """
    :precondition labEventCountsAngus.p and chartEventCountsAngus.p were created, otherwise they will be recreated
    :param numToFind top n features to return
    :param sqlFormat true to return sql format (string representation), else array of numbers
    :return list of itemid's, properly formatted
    """

    with open("data/rawdatafiles/labEventCountsAngus.p", "rb") as f:
        labEvents = pickle.load(f)
    with open("data/rawdatafiles/chartEventCountsAngus.p", "rb") as f:
        chartEvents = pickle.load(f)
    featureItemCodes = set()
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
    return featureItemCodes
def getFirst24HrsDataValuesIndividually(hadm_id):
    """
    Runs an SQL Query to return featues that that returns features for top 100
    most frequent itemids of both chartevents and labevents (might overlap)
    HOWEVER, only for one hadm_id
    :param hadm_id the admission id to run query and retrieve data for
    :return a Dataframe with the data
    """
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
# getCountOfFeaturesAngus()
# for itemIdString in getTopNItemIDs():
#     print(itemIdString + ", ")

data = getFirst24HrsDataValues()
data.to_csv("data/rawdatafiles/first24Hours.csv")
