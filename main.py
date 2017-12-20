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
trainInd, testInd = sklearn.model_selection.train_test_split(pd.Series(testTrainSet.index), train_size=.9, stratify=Y)

model_name = "LogisticRegression average data no feature selection 24 hours"
print("beginning model:", model_name)

startTime = time.time()
result = logReg.test_train_valid_explicit(Xtrain = testTrainSet.loc[trainInd], \
                                            Xtest = testTrainSet.loc[testInd], \
                                            Ytrain= Y.loc[trainInd], \
                                            Ytest = Y.loc[testInd], \
                                            n_jobs = -1, \
                                            validation_size=.1)
endTime = time.time()
runningTimes = runningTimes.append(pd.DataFrame(columns=runningTimes.columns, index=[model_name]))
runningTimes.loc[model_name, "gridsearch"] = endTime - startTime
print("Model: ", model_name, result.best_score)
print("Model: ", model_name, "; Best Params: ", result.best_params)

features_weights_to_add = pd.DataFrame(result.predictor.coef_)
features_weights_to_add.index = [model_name]
features_weights_to_add.columns = testTrainSet.columns
features_weights= pd.concat([features_weights_to_add, features_weights])

cv_results = pd.DataFrame(result.cv_results)
cv_results.to_csv("data/rawdatafiles/byHadmID0/avg_feat_lr_valid.csv")

#Note, artifact of old code...  tuples are same as train test split given

startTime = time.time()
fullScores = learning.util.test(trainTuple=result.trainTuple, \
                                testTuple=result.testTuple, \
                                predictor=result.predictor, \
                                name=model_name)
endTime = time.time()
fullScores.to_csv("data/rawdatafiles/byHadmID0/final_scores.csv")
runningTimes.loc[model_name, "run"] = endTime - startTime

model_name = "LogisticRegression average data feature selection 24 hours"
# print("beginning logReg grid search with feature selection")
toKeep = feat_sel.chi2test(un_normalized.loc[trainInd], Y.loc[trainInd], pval_thresh=.05)
print(toKeep.index)
reader.set_features(toKeep.index)
toKeep.to_csv("./data/rawdatafiles/byHadmID0/pval5.csv")

testTrainSet = reader.getFullAvg(endbound=24) # regenerate testTrainSet with feature selection
testTrainSet = normalize(testTrainSet)

startTime = time.time()
result = logReg.test_train_valid_explicit(Xtrain = testTrainSet.loc[trainInd], \
                                            Xtest = testTrainSet.loc[testInd], \
                                            Ytrain= Y.loc[trainInd], \
                                            Ytest = Y.loc[testInd], \
                                            n_jobs = -1, \
                                            validation_size=.1)
endTime = time.time()
runningTimes = runningTimes.append(pd.DataFrame(columns=runningTimes.columns, index=[model_name]))
runningTimes.loc[model_name, "gridsearch"] = endTime - startTime

print("Model: ", model_name, result.best_score)
print("Model: ", model_name, "; Best Params: ", result.best_params)

features_weights_to_add = pd.DataFrame(result.predictor.coef_)
features_weights_to_add.index = [model_name]
features_weights_to_add.columns = testTrainSet.columns
features_weights= pd.concat([features_weights_to_add, features_weights])

pd.DataFrame(result.cv_results).to_csv("data/rawdatafiles/byHadmID0/avg_feat_lr_with_feat_sel_valid.csv")

startTime = time.time()
fullScores = pd.concat([learning.util.test(trainTuple=result.trainTuple, \
                                testTuple=result.testTuple, \
                                predictor=result.predictor, \
                                name=model_name \
                                ), \
                                fullScores])
endTime = time.time()
fullScores.to_csv("data/rawdatafiles/byHadmID0/final_scores.csv")
runningTimes.loc[model_name, "run"] = endTime - startTime

model_name = "LogisticRegression time data w feature selection  24 hours"
print("beginning ", model_name)
testTrainSet = reader.traditional_time_event_matrix()
testTrainSet = normalize(testTrainSet)

startTime = time.time()
result = logReg.test_train_valid_explicit(Xtrain = testTrainSet.loc[trainInd], \
                                            Xtest = testTrainSet.loc[testInd], \
                                            Ytrain= Y.loc[trainInd], \
                                            Ytest = Y.loc[testInd], \
                                            n_jobs = -1, \
                                            validation_size=.1)
endTime = time.time()
runningTimes = runningTimes.append(pd.DataFrame(columns=runningTimes.columns, index=[model_name]))
runningTimes.loc[model_name, "gridsearch"] = endTime - startTime

pd.DataFrame(result.cv_results).to_csv("data/rawdatafiles/byHadmID0/full_feat_lr_with_feat_sel_valid.csv")

print("Model: ", model_name, result.best_score)
print("Model: ", model_name, "; Best Params: ", result.best_params)

features_weights_to_add = pd.DataFrame(result.predictor.coef_)
features_weights_to_add.index = [model_name]
features_weights_to_add.columns = testTrainSet.columns
features_weights= pd.concat([features_weights_to_add, features_weights])

startTime = time.time()
fullScores = pd.concat([learning.util.test(trainTuple=result.trainTuple, \
                                testTuple=result.testTuple, \
                                predictor=result.predictor, \
                                name=model_name \
                                ), \
                                fullScores])
endTime = time.time()
fullScores.to_csv("data/rawdatafiles/byHadmID0/final_scores.csv")
runningTimes.loc[model_name, "run"] = endTime - startTime


print("beginning rf")

# testTrainSet = pd.DataFrame.from_csv("./data/rawdatafiles/byHadmID0/avg_data.csv") ## cached copy


model_name = "rf average data feature selection 24 hours"
print("beginning ", model_name)
testTrainSet = reader.getFullAvg(endbound=24) # regenerate testTrainSet with feature selection, (don't try to regen using Hadm_Id_Reader)
testTrainSet = normalize(testTrainSet)

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

runningTimes = runningTimes.append(pd.DataFrame(columns=runningTimes.columns, index=[model_name]))
runningTimes.loc[model_name, "gridsearch"] = endTime - startTime

features_weights_to_add = pd.DataFrame(result.predictor.feature_importances_).transpose()
features_weights_to_add.index = [model_name]
features_weights_to_add.columns = testTrainSet.columns
features_weights= pd.concat([features_weights_to_add, features_weights])

pd.DataFrame(result.cv_results).to_csv("data/rawdatafiles/byHadmID0/avg_feat_rf_with_feat_sel_valid.csv")

startTime = time.time()
fullScores = pd.concat([fullScores, learning.util.test(trainTuple=result.trainTuple, \
                                testTuple=result.testTuple, \
                                predictor=result.predictor, \
                                name=model_name \
                                )])
endTime = time.time()
fullScores.to_csv("data/rawdatafiles/byHadmID0/final_scores.csv")
runningTimes.loc[model_name, "run"] = endTime - startTime

model_name = "rf time data w feature selection  24 hours"
print("beginning ", model_name)
testTrainSet = reader.traditional_time_event_matrix()
testTrainSet = normalize(testTrainSet)

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

runningTimes = runningTimes.append(pd.DataFrame(columns=runningTimes.columns, index=[model_name]))
runningTimes.loc[model_name, "gridsearch"] = endTime - startTime

features_weights_to_add = pd.DataFrame(result.predictor.feature_importances_).transpose()
features_weights_to_add.index = [model_name]
features_weights_to_add.columns = testTrainSet.columns
features_weights= pd.concat([features_weights_to_add, features_weights])

pd.DataFrame(result.cv_results).to_csv("data/rawdatafiles/byHadmID0/full_feat_rf_with_feat_sel_valid.csv")

startTime = time.time()
fullScores = pd.concat([learning.util.test(trainTuple=result.trainTuple, \
                                testTuple=result.testTuple, \
                                predictor=result.predictor, \

                                name=model_name \
                                ), \
                                fullScores])
endTime = time.time()
fullScores.to_csv("data/rawdatafiles/byHadmID0/final_scores.csv")
runningTimes.loc[model_name, "run"] = endTime - startTime

model_name = "LogisticRegression average data feature selection 36 hours"
# print("beginning logReg grid search with feature selection")
testTrainSet = reader.getFullAvg(endbound=36) # regenerate testTrainSet with feature selection
testTrainSet = normalize(testTrainSet)

startTime = time.time()
result = logReg.test_train_valid_explicit(Xtrain = testTrainSet.loc[trainInd], \
                                            Xtest = testTrainSet.loc[testInd], \
                                            Ytrain= Y.loc[trainInd], \
                                            Ytest = Y.loc[testInd], \
                                            n_jobs = -1, \
                                            validation_size=.1)
endTime = time.time()
runningTimes = runningTimes.append(pd.DataFrame(columns=runningTimes.columns, index=[model_name]))
runningTimes.loc[model_name, "gridsearch"] = endTime - startTime

print("Model: ", model_name, result.best_score)
print("Model: ", model_name, "; Best Params: ", result.best_params)

features_weights_to_add = pd.DataFrame(result.predictor.coef_)
features_weights_to_add.index = [model_name]
features_weights_to_add.columns = testTrainSet.columns
features_weights= pd.concat([features_weights_to_add, features_weights])

pd.DataFrame(result.cv_results).to_csv("data/rawdatafiles/byHadmID0/avg_feat_lr_36_with_feat_sel_valid.csv")

startTime = time.time()
fullScores = pd.concat([learning.util.test(trainTuple=result.trainTuple, \
                                testTuple=result.testTuple, \
                                predictor=result.predictor, \
                                name=model_name \
                                ), \
                                fullScores])
endTime = time.time()
fullScores.to_csv("data/rawdatafiles/byHadmID0/final_scores.csv")
runningTimes.loc[model_name, "run"] = endTime - startTime


model_name = "LogisticRegression time data w feature selection  36 hours"
print("beginning ", model_name)
testTrainSet = reader.traditional_time_event_matrix(total_time=36)
testTrainSet = normalize(testTrainSet)

startTime = time.time()
result = logReg.test_train_valid_explicit(Xtrain = testTrainSet.loc[trainInd], \
                                            Xtest = testTrainSet.loc[testInd], \
                                            Ytrain= Y.loc[trainInd], \
                                            Ytest = Y.loc[testInd], \
                                            n_jobs = -1, \
                                            validation_size=.1)
endTime = time.time()
runningTimes = runningTimes.append(pd.DataFrame(columns=runningTimes.columns, index=[model_name]))
runningTimes.loc[model_name, "gridsearch"] = endTime - startTime

pd.DataFrame(result.cv_results).to_csv("data/rawdatafiles/byHadmID0/full_feat_lr_36_with_feat_sel_valid.csv")

print("Model: ", model_name, result.best_score)
print("Model: ", model_name, "; Best Params: ", result.best_params)

features_weights_to_add = pd.DataFrame(result.predictor.coef_)
features_weights_to_add.index = [model_name]
features_weights_to_add.columns = testTrainSet.columns
features_weights= pd.concat([features_weights_to_add, features_weights])

startTime = time.time()
fullScores = pd.concat([learning.util.test(trainTuple=result.trainTuple, \
                                testTuple=result.testTuple, \
                                predictor=result.predictor, \
                                name=model_name \
                                ), \
                                fullScores])
endTime = time.time()
fullScores.to_csv("data/rawdatafiles/byHadmID0/final_scores.csv")
runningTimes.loc[model_name, "run"] = endTime - startTime
# testTrainSet = pd.DataFrame.from_csv("./data/rawdatafiles/byHadmID0/avg_data.csv") ## cached copy


model_name = "rf average data feature selection 36 hours"
print("beginning ", model_name)
testTrainSet = reader.getFullAvg(endbound=36) # regenerate testTrainSet with feature selection, (don't try to regen using Hadm_Id_Reader)
testTrainSet = normalize(testTrainSet)


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

runningTimes = runningTimes.append(pd.DataFrame(columns=runningTimes.columns, index=[model_name]))
runningTimes.loc[model_name, "gridsearch"] = endTime - startTime

features_weights_to_add = pd.DataFrame(result.predictor.feature_importances_).transpose()
features_weights_to_add.index = [model_name]
features_weights_to_add.columns = testTrainSet.columns
features_weights= pd.concat([features_weights_to_add, features_weights])

pd.DataFrame(result.cv_results).to_csv("data/rawdatafiles/byHadmID0/avg_feat_rf_36_with_feat_sel_valid.csv")

startTime = time.time()
fullScores = pd.concat([fullScores, learning.util.test(trainTuple=result.trainTuple, \
                                testTuple=result.testTuple, \
                                predictor=result.predictor, \
                                name=model_name \
                                )])
endTime = time.time()
fullScores.to_csv("data/rawdatafiles/byHadmID0/final_scores.csv")
runningTimes.loc[model_name, "run"] = endTime - startTime


model_name = "rf time data w feature selection  36 hours"
print("beginning ", model_name)
testTrainSet = reader.traditional_time_event_matrix(total_time=36)
testTrainSet = normalize(testTrainSet)

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

runningTimes = runningTimes.append(pd.DataFrame(columns=runningTimes.columns, index=[model_name]))
runningTimes.loc[model_name, "gridsearch"] = endTime - startTime

features_weights_to_add = pd.DataFrame(result.predictor.feature_importances_).transpose()
features_weights_to_add.index = [model_name]
features_weights_to_add.columns = testTrainSet.columns
features_weights= pd.concat([features_weights_to_add, features_weights])

pd.DataFrame(result.cv_results).to_csv("data/rawdatafiles/byHadmID0/full_feat_rf_36_with_feat_sel_valid.csv")

startTime = time.time()
fullScores = pd.concat([learning.util.test(trainTuple=result.trainTuple, \
                                testTuple=result.testTuple, \
                                predictor=result.predictor, \
                                name=model_name \
                                ), \
                                fullScores])
endTime = time.time()
runningTimes.loc[model_name, "run"] = endTime - startTime

fullScores.to_csv("data/rawdatafiles/final_scores.csv")
runningTimes.to_csv("data/rawdatafiles/runningTimes.csv")
features_weights.to_csv("data/rawdatafiles/features_weights.csv")

print("done!")
