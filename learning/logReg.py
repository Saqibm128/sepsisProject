import sklearn.linear_model as linMod
import sklearn.model_selection as modSel
import sklearn.metrics
from . import util
import numpy as np
import pandas as pd
from addict import Dict #too lazy to deal with normal dict
# This script contains helper functions used to train and test on the data
# Specifically used for LogisticRegression tests as well as all the setup that goes along with it

def test(joinedDataframe, trainTuple, testTuple, params=None, logReg=None):
    '''
    removes nonnumeric data, then fully trains with set and provides key metrics back
    :param params a list of parameter dicts which will be assigned to LogisticRegression
        object before it fits and runs, if None will not be used
    :param testTuple a tuple of examples and the correct answer
    :param trainTuple a tuple of examples and the correct answer used for testing the final results
    :param logReg the fully trained model, if None will not be used, params will be used instead
    :return DataFrame of scores for train, test split pased on the params given
    '''
    if logReg is None:
        logReg = linMod.LogisticRegression()
    scores = pd.DataFrame(columns=["auc", "f1score", "matthews_corrcoef"])
    #https://stackoverflow.com/questions/34842405/parameter-stratify-from-method-train-test-split-scikit-learn
    Xtrain, Ytrain = trainTuple
    Xtest, Ytest = testTuple
    if params is not None:
        for i in range(0, len(params)):
            param = params[i]
            logReg.set_params(**param) #https://stackoverflow.com/questions/27122757/sklearn-set-params-takes-exactly-1-argument
            logReg.fit(Xtrain, Ytrain)
            Ypred = logReg.predict(Xtest)
            auc = sklearn.metrics.roc_auc_score(Ytest, Ypred)
            scores["auc"][i] = auc
            f1score = sklearn.metrics.f1_score(Ytest, Ypred)
            scores["f1score"][i] = f1score
            matC = sklearn.metrics.matthews_corrcoef(Ytest, Ypred)
            scores["matthews_corrcoef"][i]=matC
    else:
        Ypred = logReg.predict(Xtest)
        auc = sklearn.metrics.roc_auc_score(Ytest, Ypred)
        scores["auc"][0] = auc
        f1score = sklearn.metrics.f1_score(Ytest, Ypred)
        scores["f1score"][0] = f1score
        matC = sklearn.metrics.matthews_corrcoef(Ytest, Ypred)
        scores["matthews_corrcoef"][0]=matC
    return scores

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
    params.tol = np.arange(10 ** -4, 10 ** -3, 10** -4)
    params.solver = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
    params.penalty = ['l2']
    params.C = [.1, .2, .4, .8, 1.6, 3.2, 6.4]
    logReg = linMod.LogisticRegression();

    predef_split = modSel.PredefinedSplit(Xvalid)
    gridSearcher = modSel.GridSearchCV(logReg, params, n_jobs=1, cv=predef_split)
    gridSearcher.fit(Xtrain, Ytrain)
    bestLogReg = gridSearcher.best_estimator_
    bestLogReg.fit(Xtrain, Ytrain)
    score = bestLogReg.score(Xtest, Ytest)
    return Dict({"cv_results": gridSearcher.cv_results_, \
            "predictor":bestLogReg, \
            "best_score":score, \
            "testTuple": (Xtest, Ytest), \
            "trainTuple": (Xtrain, Ytrain)})
