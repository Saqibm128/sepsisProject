#This is the entry point for code
# @author Mohammed Saqib

import featureSelection.frequencyCount.featureFrequencySQLCount as freq
import pandas as pd
import commonDB
import learning as logReg
import waveformUtil as wfutil
import categorization as catSepsis

#Get the data and write to disk (note we can comment out below lines if we have written already)
#If we have written to disk, then we should instantiate variables with pd.DataFrame.from_csv method
featureFrequencySQLCount.countFeatures()
catSepsis.getCategorizations(writeToCSV = True)
subject_ids = wfutil.listAllSubjects()
hadm_ids = commonDB.specSubjectHadmId(subject_ids=subject_ids)
allPersons = freq.getDataAllHadmId(hadm_ids, 40)
allPersons.to_csv("data/rawdatafiles/testPersonsData.csv")
print(allPersons) #debug print TODO: remove this

allPersons = pd.DataFrame.from_csv("data/rawdatafiles/testPersonsData.csv")
classified = pd.DataFrame.from_csv("data/rawdatafiles/classifiedAngusSepsis.csv")
result = allPersons.join(classified["angus"], how="inner")
result.to_csv("data/rawdatafiles/all10features.csv")
print(result) #debug print TODO: remove this

#remove weird error about values being too large
normalized_df=(result-result.mean())/result.std() #https://stackoverflow.com/questions/26414913/normalize-columns-of-pandas-data-frame
#
# wfutil.generateAngusDF().to_csv("data/rawdatafiles/matchedWFDBAngus.csv")
scores = logReg.cross_val_score(result)
model = logReg.fully_train(result)
print(scores)
print(model.coef_)
print(model.intercept_)
