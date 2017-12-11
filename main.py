# This is the entry point for code
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
import sklearn.model_selection
from sklearn.preprocessing import MinMaxScaler

print("beginning to read all files in")
reader = Hadm_Id_Reader("./data/rawdatafiles/byHadmID0/")
reader.use_multiprocessing(6)
# testTrainSet = reader.getFullAvg(endbound=24)
# testTrainSet.to_csv("./data/rawdatafiles/byHadmID0/avg_data.csv") #cached copy
classified = pd.DataFrame.from_csv("./data/rawdatafiles/classifiedAngusSepsis.csv")
testTrainSet = pd.DataFrame.from_csv("./data/rawdatafiles/byHadmID0/avg_data.csv")
Y = classified["angus"][testTrainSet.index]
trainInd, testInd = sklearn.model_selection.train_test_split(pd.Series(testTrainSet.index), train_size=.9, stratify=Y)
# print("beginning logReg grid search no feature selection")
# result = logReg.test_train_valid_explicit(Xtrain = testTrainSet.loc[trainInd], \
#                                             Xtest = testTrainSet.loc[testInd], \
#                                             Ytrain= Y.loc[trainInd], \
#                                             Ytest = Y.loc[testInd], \
#                                             n_jobs = 1, \
#                                             validation_size=.1)
#
# print(result.best_score)
# print(result.best_params)
# cv_results = pd.DataFrame(result.cv_results)
# cv_results.to_csv("data/rawdatafiles/byHadmID0/avg_feat_lr_valid.csv")
#
# #Note, artifact of old code...  tuples are same as train test split given
# fullScores = learning.util.test(trainTuple=result.trainTuple, \
#                                 testTuple=result.testTuple, \
#                                 predictor=result.predictor, \
#                                 name="LogisticRegression average data no feature selection")
#
#
# print("beginning logReg grid search with feature selection")
toKeep = feat_sel.chi2test(testTrainSet.loc[trainInd], Y.loc[trainInd], pval_thresh=.05)
reader.set_features(toKeep.index)
# toKeep.to_csv("./data/rawdatafiles/byHadmID0/pval5.csv")
#
# testTrainSet = testTrainSet[toKeep.index] # regenerate testTrainSet with feature selection
# result = logReg.test_train_valid_explicit(Xtrain = testTrainSet.loc[trainInd], \
#                                             Xtest = testTrainSet.loc[testInd], \
#                                             Ytrain= Y.loc[trainInd], \
#                                             Ytest = Y.loc[testInd], \
#                                             n_jobs = 1, \
#                                             validation_size=.1)
#
# pd.DataFrame(result.cv_results).to_csv("data/rawdatafiles/byHadmID0/avg_feat_lr_with_feat_sel_valid.csv")
#
# fullScores = pd.concat([learning.util.test(trainTuple=result.trainTuple, \
#                                 testTuple=result.testTuple, \
#                                 predictor=result.predictor, \
#                                 name="LogisticRegression average data feature selection" \
#                                 ), \
#                                 fullScores])
#
#
# print("beginning logReg grid search for traditional_time_event_matrix")
# testTrainSet = reader.traditional_time_event_matrix()
# result = logReg.test_train_valid_explicit(Xtrain = testTrainSet.loc[trainInd], \
#                                             Xtest = testTrainSet.loc[testInd], \
#                                             Ytrain= Y.loc[trainInd], \
#                                             Ytest = Y.loc[testInd], \
#                                             n_jobs = 1, \
#                                             validation_size=.1)
#
# pd.DataFrame(result.cv_results).to_csv("data/rawdatafiles/byHadmID0/full_feat_lr_with_feat_sel_valid.csv")
#
# fullScores = pd.concat([learning.util.test(trainTuple=result.trainTuple, \
#                                 testTuple=result.testTuple, \
#                                 predictor=result.predictor, \
#                                 name="LogisticRegression time data w feature selection no variance" \
#                                 ), \
#                                 fullScores])
# fullScores.to_csv("data/rawdatafiles/byHadmID0/final_scores.csv")

print("beginning rf")

# testTrainSet = pd.DataFrame.from_csv("./data/rawdatafiles/byHadmID0/avg_data.csv") ## cached copy

print("beginning rf grid search with feature selection")

testTrainSet = testTrainSet[toKeep.index] # regenerate testTrainSet with feature selection, (don't try to regen using Hadm_Id_Reader)
result = learning.random_forest.test_train_valid_explicit(Xtrain = testTrainSet.loc[trainInd], \
                                            Xtest = testTrainSet.loc[testInd], \
                                            Ytrain= Y.loc[trainInd], \
                                            Ytest = Y.loc[testInd], \
                                            n_jobs = 15, \
                                            validation_size=.1)

pd.DataFrame(result.cv_results).to_csv("data/rawdatafiles/byHadmID0/avg_feat_rf_with_feat_sel_valid.csv")

fullScores = learning.util.test(trainTuple=result.trainTuple, \
                                testTuple=result.testTuple, \
                                predictor=result.predictor, \
                                name="rf average data feature selection" \
                                )

print("beginning rf grid search for traditional_time_event_matrix")
testTrainSet = reader.traditional_time_event_matrix()
result = learning.random_forest.test_train_valid_explicit(Xtrain = testTrainSet.loc[trainInd], \
                                            Xtest = testTrainSet.loc[testInd], \
                                            Ytrain= Y.loc[trainInd], \
                                            Ytest = Y.loc[testInd], \
                                            n_jobs = 15, \
                                            validation_size=.1)

pd.DataFrame(result.cv_results).to_csv("data/rawdatafiles/byHadmID0/full_feat_rf_with_feat_sel_valid.csv")

fullScores = pd.concat([learning.util.test(trainTuple=result.trainTuple, \
                                testTuple=result.testTuple, \
                                predictor=result.predictor, \

                                name="rf time data w feature selection no variance" \
                                ), \
                                fullScores])


fullScores.to_csv("data/rawdatafiles/byHadmID0/final_scores2.csv")

print("done!")
