import pickle
import commonDB
import pandas as pd
# from matplotlib import pyplot as plt


## @author Mohammed Saqib
## This file is responsible for getting out all results from sql queries.
##      It does not do much else, because getting the queries
##      takes a lot of time. Once done, it places the data as a raw pickle file


conn = commonDB.getConnection()
with open("data/sql/angus.sql") as f:
    query = f.read()
angusData = pd.read_sql(query, conn)

## Probably shouldn't print but using the fact that python will only output a few lines
##      of a long output to get a preview of the result of the query
print(angusData)


pickle.dump(angusData, open("data/rawdatafiles/classifiedAngusSepsis.p", "wb"))
