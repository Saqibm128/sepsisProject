# This is the entry point for code
# @author Mohammed Saqib

import dataCollect.dataCollect as freq
import pandas as pd
import commonDB
import learning.logReg as logReg
import learning.svm
import learning.util
from readWaveform import waveformUtil as wfutil
import categorization as catSepsis
import numpy
from preprocessing import preprocessing
from pipeline.hadmid_reader import Hadm_Id_Reader

print("beginning to read all files in")
reader = Hadm_Id_Reader("./data/rawdatafiles/byHadmID0/")
testTrainSet = reader.getFullAvg()
#
# testTrainSet = pd.DataFrame.from_csv("./data/rawdatafiles/byHadmID0/full_data_matrix.csv") ## cached copy
classified = pd.DataFrame.from_csv("./data/rawdatafiles/classifiedAngusSepsis.csv")
# testTrainSet = reader.traditional_time_event_matrix()
testTrainSet.to_csv("./data/rawdatafiles/byHadmID0/full_avg_matrix.csv") ## cached copy
testTrainSet["angus"] = classified["angus"][testTrainSet.index]
testTrainSet.to_csv("./data/rawdatafiles/byHadmID0/full_data_matrix_with_angus.csv") ## caching some info
# testTrainSet = pd.DataFrame.from_csv("./data/rawdatafiles/byHadmID0/full_data_matrix_with_angus.csv")
print("beginning logReg grid search")
result = logReg.test_train_validation(testTrainSet)

print(result.best_score)
cv_results = pd.DataFrame(result.cv_results)
cv_results.to_csv("data/rawdatafiles/byHadmID/full_data_lr_cv_results.csv")

fullScores = learning.util.test(trainTuple=result.trainTuple, testTuple=result.testTuple, predictor=result.predictor)
fullScores.to_csv("data/rawdatafiles/byHadmID/full_data_lr_full_scores.csv")
# print("beginning svm gridsearch")
# result = learning.svm.test_train_validation(testTrainSet)
#
# print(result.best_score)
# cv_results = pd.DataFrame(result.cv_results)
# cv_results.to_csv("data/rawdatafiles/svm_cv_results.csv")
#
# fullScores = learning.util.test(trainTuple=result.trainTuple, testTuple=result.testTuple, predictor=result.predictor)
# fullScores.to_csv("data/rawdatafiles/svm_full_scores.csv")
# data = wfutil.compareAdmitToWF()
# data.to_csv("data/rawdatafiles/wfdetails.csv")

print("done!")
