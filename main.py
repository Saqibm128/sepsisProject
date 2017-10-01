#This is the entry point for code
# @author Mohammed Saqib

import featureSelection.frequencyCount.featureFrequencySQLCount as freq
import pandas as pd
import commonDB
import learning as logReg
import waveForm as wfutil
import categorization as catSepsis

catSepsis.getCategorizations(writeToCSV = True)
hadm_ids = commonDB.getAllHADMID()
allPersons = freq.getDataAllHadmId(hadm_ids, 40)
allPersons.to_csv("data/rawdatafiles/testPersonsData.csv")
print(allPersons)

allPersons = pd.DataFrame.from_csv("data/rawdatafiles/testPersonsData.csv")
classified = pd.DataFrame.from_csv("data/rawdatafiles/classifiedAngusSepsis.csv")
result = allPersons.join(classified["angus"], how="inner")
result.to_csv("data/rawdatafiles/all10features.csv")

#remove weird error about values being too large
normalized_df=(result-result.mean())/result.std() #https://stackoverflow.com/questions/26414913/normalize-columns-of-pandas-data-frame
#
# wfutil.generateAngusDF().to_csv("data/rawdatafiles/matchedWFDBAngus.csv")
scores = logReg.cross_val_score(result)
model = logReg.fully_train(result)
print(scores)
print(model.coef_)
print(model.intercept_)
