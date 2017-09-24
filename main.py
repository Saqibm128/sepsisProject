#This is the entry point for code
# @author Mohammed Saqib

import featureSelection.frequencyCount.featureFrequencySQLCount as freq
import pandas as pd
import commonDB

hadm_ids = commonDB.getAllHADMID()["hadm_id"]
allPersons = pd.DataFrame()
for hadm_id in hadm_ids:
    dataEvents = freq.getFirst24HrsDataValuesIndividually(hadm_id=hadm_id, nitems = 30, path="data/rawdatafiles/counts.csv")
    allPersons = pd.concat([allPersons, freq.cleanUpIndividual(dataEvents, hadm_id)])
allPersons.to_csv("data/rawdatafiles/testPersonsData.csv")
