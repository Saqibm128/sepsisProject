import pickle
import commonDB
import pandas as pd
# import waveformUtil as wfutil TODO: remove these circular dependencies
# from matplotlib import pyplot as plt


## @author Mohammed Saqib
## This file is responsible for getting out all results from sql queries.
##      It does not do much else, because getting the queries
##      takes a lot of time. Once done, it places the data as a raw pickle file


def getCategorizations(hadm_ids = None):
    """
    This is a function to run the Angus.sql query, which is responsible for categorizing patients
    :postcondition stores results as data/rawdatafiles/classifiedAngusSepsis.p
    :param ids the hadm_ids to look at when doing the query; if None, run it on all
    :return the angusData
    """
    conn = commonDB.getConnection()
    with open("data/sql/angus.sql") as f:
        query = f.read()
    if hadm_ids is None:
        query = query.replace("<INSERT IDS HERE>", "")
    else:
        query = query.replace("<INSERT IDS HERE>", "WHERE hadm_id IN " + commonDB.convertListToSQL(hadm_ids))

    angusData = pd.read_sql(query, conn)
    angusData.set_index(["hadm_id"], inplace=True)
    return angusData
if __name__ == "__main__":
    getCategorizations(True)
