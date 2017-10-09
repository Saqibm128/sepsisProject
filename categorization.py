import pickle
import commonDB
import pandas as pd
# import waveformUtil as wfutil TODO: remove these circular dependencies
# from matplotlib import pyplot as plt


## @author Mohammed Saqib
## This file is responsible for getting out all results from sql queries.
##      It does not do much else, because getting the queries
##      takes a lot of time. Once done, it places the data as a raw pickle file


def getCategorizations(writeToCSV = True):
    """
    This is a function to run the Angus.sql query, which is responsible for categorizing patients
    :param writeToCSV If true, writes in csv format instead of pickle format
    :postcondition stores results as data/rawdatafiles/classifiedAngusSepsis.p
    :return the angusData
    """
    conn = commonDB.getConnection()
    with open("data/sql/angus.sql") as f:
        query = f.read()
    # query = query.replace("<INSERT IDS HERE>", commonDB.convertListToSQL(wfutil.listAllSubjects()))
    angusData = pd.read_sql(query, conn)
    angusData.set_index(["hadm_id"], inplace=True)
    if not writeToCSV:
        pickle.dump(angusData, open("data/rawdatafiles/classifiedAngusSepsis.p", "wb"))
    else:
        angusData.to_csv("data/rawdatafiles/classifiedAngusSepsis.csv")
    return angusData
if __name__ == "__main__":
    getCategorizations(True)
