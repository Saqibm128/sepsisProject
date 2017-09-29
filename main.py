#This is the entry point for code
# @author Mohammed Saqib

import featureSelection.frequencyCount.featureFrequencySQLCount as freq
import pandas as pd
import commonDB
import learning.logReg as logReg
import readWaveForm.waveformUtil as wfutil
import categorizationBySepsisStatus as catSepsis

catSepsis.getCategorizations(writeToCSV = True)
hadm_ids = commonDB.getAllHADMID()
allPersons = freq.getDataAllHadmId(hadm_ids, 40)
allPersons.to_csv("data/rawdatafiles/testPersonsData.csv")
print(allPersons)

allPersons = pd.DataFrame.from_csv("data/rawdatafiles/testPersonsData.csv")
classified = pd.DataFrame.from_csv("data/rawdatafiles/classifiedAngusSepsis.csv")
result = allPersons.join(classified["angus"], how="inner")
result.to_csv("data/rawdatafiles/all10features.csv")
#
# wfutil.generateAngusDF().to_csv("data/rawdatafiles/matchedWFDBAngus.csv")
scores = logReg.cross_val_score(result)
print(scores)
