import sklearn.model_selection as modSel
import sklearn.metrics
import sklearn.svm
from . import util
import numpy as np
import pandas as pd
from addict import Dict
# This script contains helper functions used to train and test on the data


def test_train_validation(joinedDataframe, train_size = .8, validation_size=.1):
    '''
    Does a test, train, and validation split, uses cross validation on the validation split
    and returns the results based on the test split
    :param joinedDataframe a dataframe with the binary classifier having column of "angus"
    :param train_validation_size how to split between train set and test set and validation set
    :return a dictionary of results for each param combination, the best LogisticRegression estimator, and the score on the test set
    '''
    X = joinedDataframe.drop(["angus"], axis = 1)
    Y = joinedDataframe["angus"]
    #https://stackoverflow.com/questions/34842405/parameter-stratify-from-method-train-test-split-scikit-learn
    Xtrain, Xtest, Ytrain, Ytest = modSel.train_test_split(X, Y, train_size = train_size + validation_size, stratify = Y)
    XvalidIndices, _ = modSel.train_test_split(list(range(0, len(Xtrain))), train_size = validation_size, stratify = Ytrain)
    Xvalid = np.full(len(Xtrain), -1)
    for index in XvalidIndices:
        Xvalid[index] = 0
    #we set up parameter search space here to look up
    params = Dict()
    params.C = [ .4, .8, 1.6, 3.2, 6.4]
    params.tol = [(10**-4), (10**-3), .01]
    polyParams = Dict(params) #copy to a new dict to avoid testing param sets that are the same
    params.kernel = ["linear", "rbf", "sigmoid"]
    params.gamma = ["auto", .001, .01,  .1]
    polyParams.kernel = ["poly"]
    polyParams.degree = [1, 2, 3, 4, 5]
    polyParams.coef0 = [0, 1, 5, -1, -5]
    svm = sklearn.svm.SVC()
    predef_split = modSel.PredefinedSplit(Xvalid)
    gridSearcher = modSel.GridSearchCV(svm, [params, polyParams], n_jobs=10, cv=predef_split)
    gridSearcher.fit(Xtrain, Ytrain)
    bestPredictor = gridSearcher.best_estimator_
    bestPredictor.fit(Xtrain, Ytrain)
    score = bestPredictor.score(Xtest, Ytest)
    return Dict({"cv_results": gridSearcher.cv_results_, \
            "predictor": bestPredictor, \
            "best_score":score, \
            "testTuple": (Xtest, Ytest), \
            "trainTuple": (Xtrain, Ytrain)})
