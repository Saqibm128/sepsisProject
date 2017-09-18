import pickle
import commonDB
import pandas as pd
# from matplotlib import pyplot as plt


## @author Mohammed Saqib
## This file is responsible for getting out all results from sql queries.
##      It does not do much else, because getting the queries
##      takes a lot of time. Once done, it places the data as a raw pickle file


conn = commonDB.getConnection()
# with open("data/sql/countLabEventsAngus.sql") as f:
#     query = f.read()
# labEvents = pd.read_sql(query, conn)
#
# ## Probably shouldn't print but using the fact that python will only output a few lines
# ##      of a long output to get a preview of the result of the query
# print(labEvents)
# 
#
# pickle.dump(labEvents, open("data/rawdatafiles/labEventCountsAngus.p", "wb"))

with open("data/sql/countChartEventsAngus.sql") as f:
    query = f.read()
chartEvents = pd.read_sql(query, conn)

print(chartEvents)

pickle.dump(chartEvents, open("data/rawdatafiles/chartEventCountsAngus.p", "wb"))
