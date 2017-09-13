import pickle
import commonDB
import pandas as pd
# from matplotlib import pyplot as plt


## @author Mohammed Saqib
## This file is responsible for getting out frequencies of specific
##      lab events.


conn = commonDB.getConnection()
with open("data/sql/countLabEvents.sql") as f:
    query = f.read()
sepsisClassified = pd.read_sql(query, conn)

print(sepsisClassified)


pickle.dump(sepsisClassified, open("data/rawdatafiles/labEventCounts.p", "wb"))
