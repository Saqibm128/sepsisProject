#This is the entry point for code
# @author Mohammed Saqib

import featureSelection.frequencyCount.featureFrequencySQLCount as freq
import pandas as pd
import commonDB
import sklearn.linear_model as linMod
import sklearn.model_selection as modSel

def scorer(estimator, x, y):
    return estimator.score(x, y)

hadm_ids = commonDB.getAllHADMID()
allPersons = freq.getDataAllHadmId(hadm_ids, 10)
print(allPersons)
allPersons.to_csv("data/rawdatafiles/allPersonsData.csv")
# print(allPersons)
allPersons = pd.DataFrame.from_csv("data/rawdatafiles/allPersonsData.csv")
classified = pd.DataFrame.from_csv("data/rawdatafiles/classifiedAngusSepsis.csv")
classified.set_index(["hadm_id"], inplace = True)
print(classified["angus"])
result = allPersons.join(classified["angus"], how="inner")
print(result)
result.to_csv("data/rawdatafiles/all10features.csv")
print(result)
solver = linMod.LogisticRegression()
cvscore = modSel.cross_val_score(estimator=solver, X=result[result.columns[:-2]], y=result["angus"], scoring= scorer, cv = 10)
print(cvscore)
