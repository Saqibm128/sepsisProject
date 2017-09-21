import pickle
import commonDB
import pandas as pd
# from matplotlib import pyplot as plt


## @author Mohammed Saqib
## This file is responsible for getting out all results from sql queries.
##      It does not do much else, because getting the queries
##      takes a lot of time. Once done, it places the data as a raw pickle file


def getCategorizations(writeToCSV = False):
    """
    This is a function to run the Angus.sql query, which is responsible for categorizing patients
    :param writeToCSV If true, writes in csv format instead of pickle format
    :postcondition stores results as data/rawdatafiles/classifiedAngusSepsis.p
    """
    conn = commonDB.getConnection()
    with open("data/sql/angus.sql") as f:
        query = f.read()
    angusData = pd.read_sql(query, conn)
    if not writeToCSV:
        pickle.dump(angusData, open("data/rawdatafiles/classifiedAngusSepsis.p", "wb"))
    else:
        angusData.to_csv("data/rawdatafiles/classifiedAngusSepsis.csv")

getCategorizations(True)
