import pandas as pd
from readWaveform.segment_reader import SegmentReader
import learning.logReg as logReg
import learning.svm
import learning.util
import learning.random_forest
import pipeline.feature_selection_wrapper as feat_sel
import sklearn.model_selection
import time


if __name__=="__main__":
    segReader = SegmentReader();
    Y = segReader.labels()
    trainInd, testInd = sklearn.model_selection.train_test_split(pd.Series(Y.index), train_size=.8, stratify=Y, random_state=100)
    X = segReader.simpleStatsAll()

    Ytrain, Xtrain, Ytest, Xtest = Y.loc[trainInd], X.loc[trainInd], Y.loc[testInd], X.loc[testInd]
    model_name = "LogisticRegression"
    print("beginning model:", model_name)
    result = logReg.test_train_valid_explicit(Xtrain = Xtrain.values, \
                                                Xtest = Xtest.values, \
                                                Ytrain= Ytrain.values, \
                                                Ytest = Ytest.values, \
                                                n_jobs = -1, \
                                                validation_size=.1)
    print("Model: ", model_name, result.best_score)
    print("Model: ", model_name, "; Best Params: ", result.best_params)
    features_weights_to_add = pd.DataFrame(result.predictor.coef_)
    features_weights_to_add.index = [model_name]
    features_weights_to_add.columns = X.columns
    features_weights= pd.concat([features_weights_to_add])
    fullScores = learning.util.test(trainTuple=result.trainTuple, \
                                    testTuple=result.testTuple, \
                                    predictor=result.predictor, \
                                    name=model_name)
    fullScores.to_csv("data/rawdatafiles/byHadmIDNumRec/final_scores_lr.csv")
