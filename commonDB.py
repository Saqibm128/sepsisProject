import psycopg2
import pandas as pd
import pickle
import numpy as np
import json

#represents common functions to use db

#Use the predefined MIT code to identify sepsis from the MIMIC3 database and create global var

def consolidateEvents(events, hadm_id):
    """
    Takes a dataframe of all events that occurred to an individual and goes through all of them. Then cleans the data up
    :param events dataframe of all events that occurred to one individual
    :param features a list of all features to look up in consolidateEvents
    :return a cleaned dataframe, with missing data interpolated, if possible from individual record
    """
    datamap = {} #dict to construct dataframe from
    features = events["variable"].unique() #Should return a series of unique itemids aka a list to iterate over for the features we decide to use
    datamap["hadm_id"] = [str(hadm_id)]
    for feature in features:
        featureVal = (events[events["variable"] == feature])["valuenum"]
        if featureVal.isnull().all():
            featureVal = np.nan #currently just ignore nonnumeric data
            # featureVal = [featureVal.mode()[0]] #TODO: get first mode of all nonnumeric data
        else: #valuenum field was populated, which means data was clean
            featureVal = (events[events["variable"] == feature])["valuenum"] #sets featureVal as temporary series that contains all values, then to take mean of
            featureVal = featureVal[~featureVal.duplicated(keep='first')] #if chartevents and labevents has the same exact item, ie times, dates, and values, then keep only one copy
            featureVal = [featureVal.mean()]
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
    if series.dtype == numpy.number:
        return series.mean()
    else:
        return series.mode()[0]

def read_sql(query, uppercase=True):
    """
    Wrapper around pd.read_sql
    we need to instantiate a connection every time and it gets old to
    deal with multiple new connections in code
    :param query the sql query (string)
    :param uppercase Due to attempting to work with mimic3 benchmark code
        some of the df may need to have columns that are uppercase
    :return the results of query (DataFrame)
    """
    conn = getConnection()
    df = pd.read_sql(query, conn)
    if uppercase:
        df.columns = df.columns.str.upper()
    return df

def getAllHADMID():
    """
    Quick utility method
    :return returns all the hospital admissions ids as series data
    """
    conn = getConnection()
    query = "SELECT hadm_id FROM admissions"
    return pd.read_sql(query, conn)["hadm_id"]

def specSubjectHadmId(subject_ids=None):
    '''
    Similar to getAllHADMID but only for hospital admissions from specific subjects
    :param subjectIDs to match to (list of integers)
    :return returns all the hospital admissions ids as series data
    '''
    conn = getConnection()
    if subject_ids is None:
        query = "SELECT hadm_id FROM admissions"
    else:
        query = "SELECT hadm_id FROM admissions WHERE subject_id in " + convertListToSQL(subject_ids)
    return pd.read_sql(query, conn)["hadm_id"]
def convertListToSQL(listItems):
    '''
    Transform a list of items, (usually ids)
    from type int to format "(itemId1, itemId2)"
    for sql
    :param listItems a python list of stuff
    :return string in sql format for "WHERE var IN" would work
    '''
    toRet = ""
    for item in listItems:
        toRet += str(item) + ", "
    toRet = "(" + toRet[0:-2] + ")"
    return toRet

def getConnection(port=5432):
    """
    :return: connection to database
    """
    try:
        with open("secrets.json") as file:
            dbCreds = json.load("secrets.json")
            conn = psycopg2.connect(dbname=dbCreds["db"], user=dbCreds["user"], host=dbCreds["host"], password=dbCreds["password"], port=port )
    except:
    	raise
    cur = conn.cursor()
    cur.execute("SET search_path TO mimiciii")
    return conn
