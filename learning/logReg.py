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

    Wrapper around gridsearch

    :param joinedDataframe a dataframe with the binary classifier having column of "angus"
    :param train_validation_size how to split between train set and test set and validation set
    :return a dictionary of results for each param combination, the best LogisticRegression estimator, and the score on the test set
    '''
    X = joinedDataframe.drop(["angus"], axis = 1)
    Y = joinedDataframe["angus"]
    #https://stackoverflow.com/questions/34842405/parameter-stratify-from-method-train-test-split-scikit-learn
    Xtrain, Xtest, Ytrain, Ytest = modSel.train_test_split(X, Y, train_size = train_size + validation_size, stratify = Y)
    return test_train_valid_explicit(Xtrain, Xtest, Ytrain, Ytest)

def test_train_valid_explicit(Xtrain, Xtest, Ytrain, Ytest, validation_size=.1, n_jobs=1):
    '''
    Given a test, train split, does the validation split itself. Unlike test_train_validation,
    needs to be given the explicit train and test split (for featureSelection to avoid info leak)
    Does validation split itself based on Ytrain on the trainset

    Wrapper around util.gridsearchCV which itself wraps around sklearn.gridsearchCV

    Returns the best model based on gridseraching as well as score of the best model on the test value
    :param Xtrain the training feature vectors (instances)
    :param Ytrain the training answers/classlabels
    :param Xtest
    :param Ytest
    :param validation_size how to split
    :return a dictionary of results for each param combination, the best estimator
    '''
    #we set up parameter search space here to look up
    params = Dict()
    params.tol = [.001, .0001, .00001]
    params.solver = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
    params.penalty = ['l2']
    params.n_jobs = [1]
    params.C = [.1, .2, .4, .8, 1.6, 3.2]
    logReg = linMod.LogisticRegression();
    results = util.gridsearch_CV_wrapper(params, logReg, Xtrain, Ytrain, Xtest, Ytest, validation_size, n_jobs=n_jobs)
    results.weights = results.predictor.coef_
    return results
