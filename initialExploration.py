import pickle
import commonDB
import pandas as pd


## Some other practice, especially to printout the labitems

# practiceSQL = "SELECT ITEMID, LABEL FROM D_LABITEMS"
# conn = commonDB.getConnection()
# allLabItems =  pd.read_sql(practiceSQL, conn)
# print(allLabItems)
# with open("data/rawdatafiles/labItems.txt", "wb") as f:
#     f.write(str.encode(allLabItems.to_csv()))
## Use predetermined criterion as explained by Angus paper and written by MIT

conn = commonDB.getConnection()
sepsisClassified = pd.read_sql(commonDB.sepsisSQLQuery, conn)

print(sepsisClassified)
pickle.dump(sepsisClassified, open("data/rawdatafiles/patientCodeClassified.p", "wb"))
