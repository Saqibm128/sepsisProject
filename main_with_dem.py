# This is similar to main.py along with demographics, to double check some stuff
# @author Mohammed Saqib

import dataCollect.dataCollect as freq
import pandas as pd
import commonDB
import learning.logReg as logReg
import learning.svm
import learning.util
import learning.random_forest
import pipeline.feature_selection_wrapper as feat_sel
from readWaveform import waveformUtil as wfutil
import categorization as catSepsis
import numpy
from preprocessing import preprocessing
from pipeline.hadmid_reader import Hadm_Id_Reader
from pipeline.demographic_reader import DemReader
import sklearn.model_selection
from sklearn.preprocessing import MinMaxScaler
import time

def normalize(df):
    df = df.copy()
    for column in df.columns:
        values = df[column]
        if (values.min() - values.max() == 0):
            continue;
        else:
            df[column] = (df[column] - df[column].mean()) / df[column].std()
    return df

features_weights= pd.DataFrame()
runningTimes = pd.DataFrame(columns=["gridsearch", "run"])

print("beginning script")

reader = Hadm_Id_Reader("./data/rawdatafiles/byHadmID0/")
reader.use_multiprocessing(20)
un_normalized = reader.getFullAvg(endbound=24)
testTrainSet = normalize(un_normalized)

classified = pd.DataFrame.from_csv("./data/rawdatafiles/classifiedAngusSepsis.csv")
Y = classified["angus"][testTrainSet.index]
trainInd, testInd = sklearn.model_selection.train_test_split(pd.Series(testTrainSet.index), train_size=.9, stratify=Y, random_state = 0)

toKeep = feat_sel.chi2test(un_normalized.loc[trainInd], Y.loc[trainInd], pval_thresh=.05)
print("Finished feature selection on chartevents")
reader.set_features(toKeep.index)

toKeep.to_csv("./data/rawdatafiles/byHadmID0/pval5.csv")

testTrainSet = reader.getFullAvg(endbound=24) # regenerate testTrainSet with feature selection
testTrainSet = normalize(testTrainSet)
print("Finished generating data")
print(testTrainSet.shape)


model_name = "LogisticRegression average data feature selection 24 hours"
print("Beginning", model_name)
result = logReg.test_train_valid_explicit(Xtrain = testTrainSet.loc[trainInd], \
                                            Xtest = testTrainSet.loc[testInd], \
                                            Ytrain= Y.loc[trainInd], \
                                            Ytest = Y.loc[testInd], \
                                            n_jobs = -1, \
                                            validation_size=.1)
print("Model: ", model_name, "; Best Params: ", result.best_params)
fullScores = pd.concat([learning.util.test(trainTuple=result.trainTuple, \
                                testTuple=result.testTuple, \
                                predictor=result.predictor, \
                                name=model_name \
                                )])
features_weights_to_add = pd.DataFrame(result.predictor.coef_)
features_weights_to_add.index = [model_name]
features_weights_to_add.columns = testTrainSet.columns
features_weights= pd.concat([features_weights_to_add, features_weights])

model_name = "random_forest average 24 hours"
print("Beginning", model_name)
startTime = time.time()
result = learning.random_forest.test_train_valid_explicit(Xtrain = testTrainSet.loc[trainInd], \
                                            Xtest = testTrainSet.loc[testInd], \
                                            Ytrain= Y.loc[trainInd], \
                                            Ytest = Y.loc[testInd], \
                                            n_jobs = 8, \
                                            validation_size=.1)
endTime = time.time()

print("Model: ", model_name, result.best_score)
print("Model: ", model_name, "; Best Params: ", result.best_params)

fullScores = pd.concat([fullScores, learning.util.test(trainTuple=result.trainTuple, \
                                testTuple=result.testTuple, \
                                predictor=result.predictor, \
                                name=model_name \
                                )])
print("Getting demographics now")

dr = DemReader(hadms=testTrainSet.index)
testTrainSet = testTrainSet.join(dr.getHADMDems())
testTrainSet = normalize(testTrainSet) #Renormalize after adding back in demographics
print(testTrainSet.shape)


model_name = "LogisticRegression average data feature selection 24 hours with Demographics"
print("Beginning", model_name)
result = logReg.test_train_valid_explicit(Xtrain = testTrainSet.loc[trainInd], \
                                            Xtest = testTrainSet.loc[testInd], \
                                            Ytrain= Y.loc[trainInd], \
                                            Ytest = Y.loc[testInd], \
                                            n_jobs = -1, \
                                            validation_size=.1)
print("Model: ", model_name, "; Best Params: ", result.best_params)
fullScores = pd.concat([learning.util.test(trainTuple=result.trainTuple, \
                                testTuple=result.testTuple, \
                                predictor=result.predictor, \
                                name=model_name \
                                ), \
                                fullScores])
model_name = "random_forest average 24 hours with demographics"
print("Beginning", model_name)
result = learning.random_forest.test_train_valid_explicit(Xtrain = testTrainSet.loc[trainInd], \
                                            Xtest = testTrainSet.loc[testInd], \
                                            Ytrain= Y.loc[trainInd], \
                                            Ytest = Y.loc[testInd], \
                                            n_jobs = 8, \
                                            validation_size=.1)
endTime = time.time()

print("Model: ", model_name, result.best_score)
print("Model: ", model_name, "; Best Params: ", result.best_params)

fullScores = pd.concat([fullScores, learning.util.test(trainTuple=result.trainTuple, \
                                testTuple=result.testTuple, \
                                predictor=result.predictor, \
                                name=model_name \
                                )])

print(fullScores)
