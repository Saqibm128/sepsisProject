# This is the entry point for code
# @author Mohammed Saqib

import featureSelection.frequencyCount.featureFrequencySQLCount as freq
import pandas as pd
import commonDB
import learning.logReg as logReg
from readWaveform import waveformUtil as wfutil
import categorization as catSepsis
import numpy
# Get the data and write to disk (note we can comment out below lines if we have written already)
# If we have written to disk, tpythen we should instantiate variables with pd.DataFrame.from_csv method
# mappingItemids = pd.DataFrame.from_csv("data/rawdatafiles/selfCounts.csv") #stores which itemids are equivalent
# freqFeatOverall = freq.countFeatures(subject_ids=subject_ids, mapping=mappingItemids)
# freqFeatOverall.to_csv("data/rawdatafiles/freqOverallMatchedSubset.csv")
# 
# categorization = catSepsis.getCategorizations()
# sepsisCategorization = categorization[categorization["angus"] == 1]
# freqFeatSepsis = freq.countFeatures(subject_ids=subject_ids, hadm_ids=sepsisCategorization.index)
# freqFeatSepsis.to_csv("data/rawdatafiles/freqFeatSepsis.csv", \
#                         mapping=mappingItemids)
# 
# nonSepsisCategorization = categorization[categorization["angus"] == 0]
# freqFeatNonSepsis = freq.countFeatures(subject_ids=subject_ids, \
#                                         hadm_ids=nonSepsisCategorization.index, \
#                                         mappings=mappingItemids)
# freqFeatNonSepsis.to_csv("data/rawdatafiles/freqFeatNonSepsis.csv")
hadm_ids = commonDB.specSubjectHadmId()
#Read a cached copy of most common itemids for all subjects
itemids = pd.DataFrame.from_csv("preprocessing/resources/itemid.csv")["itemid"].as_matrix()
#Get a mapping of itemids to variables, since multiple itemids often map to same concept
itemidVariableMap = pd.DataFrame.from_csv("preprocessing/resources/itemid_to_variable_map.csv")
#Level2 is mapped to index, we want this column to be variable
itemidVariableMap["variable"] = itemidVariableMap.index
itemidVariableMap["itemid"] = itemidVariableMap["ITEMID"]
#Read a copy of mappings of itemids to variables and transform
#   into itemid to variable encoding (from mimiciii benchmark project)

allPersons = freq.getDataByHadmId(hadm_ids, itemids, mapping=itemidVariableMap)
allPersons.dropna(axis=0, how="any", subset=["Heart Rate", "Systolic blood pressure", "Diastolic blood pressure"])
allPersons.to_csv("data/rawdatafiles/allPersonsData2.csv")


# allPersons = pd.DataFrame.from_csv("data/rawdatafiles/allPersonsData.csv")
# classified = pd.DataFrame.from_csv("data/rawdatafiles/classifiedAngusSepsis.csv")
# result = allPersons.join(classified["angus"], how="inner")
# result.to_csv("data/rawdatafiles/testTrainSet.csv")


# remove weird error about values being too large
# result = result.select_dtypes([numpy.number])
# angus = result["angus"]
# normalized_df=(result-result.mean())/result.std() #https://stackoverflow.com/questions/26414913/normalize-columns-of-pandas-data-frame
# normalized_df["angus"] = angus
#
# wfutil.generateAngusDF().to_csv("data/rawdatafiles/matchedWFDBAngus.csv")

# testTrainSet = pd.DataFrame.from_csv("data/rawdatafiles/testTrainSet.csv")
# testTrainSet = testTrainSet.select_dtypes([numpy.number])
# cv_results, bestLogReg, score = logReg.test_train_validation(testTrainSet)
# print(score)
# cv_results = pd.DataFrame(cv_results)
# cv_results.to_csv("data/rawdatafiles/log_reg_cv_results.csv")


#
# fullScores = logReg.fully_test(testTrainSet, params)
# fullScores.to_csv("data/rawdatafiles/full_scores.csv")
# data = wfutil.compareAdmitToWF()
# data.to_csv("data/rawdatafiles/wfdetails.csv")
