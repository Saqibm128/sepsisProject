#This is the entry point for code
# @author Mohammed Saqib

import featureSelection.frequencyCount.featureFrequencySQLCount as freq
import pandas as pd
import commonDB
import learning.logReg as logReg
from readWaveForm import waveformUtil as wfutil
import categorization as catSepsis
import numpy
#Get the data and write to disk (note we can comment out below lines if we have written already)
#If we have written to disk, then we should instantiate variables with pd.DataFrame.from_csv method
subject_ids = wfutil.listAllSubjects()
#freq.countFeatures(ids=subject_ids)
#catSepsis.getCategorizations(writeToCSV = True)
#hadm_ids = commonDB.specSubjectHadmId(subject_ids=subject_ids)
#allPersons = freq.getDataAllHadmId(hadm_ids, 40)
#allPersons.to_csv("data/rawdatafiles/testPersonsData.csv")
#print(allPersons) #debug print TODO: remove this

allPersons = pd.DataFrame.from_csv("data/rawdatafiles/testPersonsData.csv")
classified = pd.DataFrame.from_csv("data/rawdatafiles/classifiedAngusSepsis.csv")
result = allPersons.join(classified["angus"], how="inner")
result.to_csv("data/rawdatafiles/all10features.csv")
print(result) #debug print TODO: remove this

#remove weird error about values being too large
result = result.select_dtypes([numpy.number])
angus = result["angus"]
normalized_df=(result-result.mean())/result.std() #https://stackoverflow.com/questions/26414913/normalize-columns-of-pandas-data-frame
normalized_df["angus"] = angus
#
# wfutil.generateAngusDF().to_csv("data/rawdatafiles/matchedWFDBAngus.csv")
scores = logReg.cross_val_score(normalized_df)
model = logReg.fully_train(normalized_df)
print(scores)
print(model.coef_)
print(model.intercept_)
