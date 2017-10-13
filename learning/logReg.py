import sklearn.linear_model as linMod
import sklearn.model_selection as modSel
from . import util
import numpy as np
from addict import Dict #too lazy to deal with normal dict
# This script contains helper functions used to train and test on the data
# Specifically used for LogisticRegression tests as well as all the setup that goes along with it

def fully_train(joinedDataframe):
    '''
    removes nonnumeric data, then fully trains with set and provides key metrics back
    :param joinedDataframe a dataframe where all columns except the last one are feature data and the
        last column is the Angus sepsis specifier
    :return scores for 10-fold stratified cross validation
    '''
    regFunct = linMod.LogisticRegression()
    regFunct.fit(X=joinedDataframe.drop(["angus"], axis=1).select_dtypes([np.number]), y=joinedDataframe["angus"])
    return regFunct

def test_train_validation(joinedDataframe, train_size = .5, validation_size = .2):
    '''
    Does a test, train, and validation split, uses cross validation on the validation split
    and returns the results based on the test split
    :param joinedDataframe a dataframe with the binary classifier having column of "angus"
    :param train_size how to split between train set and test set and validation set
    :param validation_size how to split again to validation set TODO: did not implement Ytest
    :return a dictionary of results for each param combination, the best LogisticRegression estimator, and the score on the test set
    '''
    X = joinedDataframe.drop(["angus"], axis = 1)
    Y = joinedDataframe["angus"]
    #https://stackoverflow.com/questions/34842405/parameter-stratify-from-method-train-test-split-scikit-learn
    Xtrain, Xtest, Ytrain, Ytest = modSel.train_test_split(X, Y, train_size = train_size + validation_size, stratify = Y)
    # Xvalid, Xtest, Yvalid, Ytest = modSel.train_test_split(XtestValid, YtestValid, train_size = validation_size, stratify = YtestValid)

    #we set up parameter search space here to look up
    params = Dict()
    params.tol = np.arange(10 ** -4, 10 ** -3, 10** -4)
    params.solver = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
    params.penalty = ['l2']
    params.C = np.arange(.1, 10, .1)
    logReg = linMod.LogisticRegression();
    print(logReg.get_params().keys())
    gridSearcher = modSel.GridSearchCV(logReg, params)
    gridSearcher.fit(Xtrain, Ytrain)
    bestLogReg = gridSearcher.best_estimator_
    bestLogReg.fit(Xtrain, Ytrain)
    score = bestLogReg.score(Xtest, Ytest)
    return gridSearcher.cv_results_, bestLogReg, score

def cross_val_score(joinedDataframe, scorer = util.scorer):
    '''
    removes nonnumeric data, then tests with cross_val_score and provides key metrics back
    :param joinedDataframe a dataframe where all columns except the last one are feature data and the
        last column is the Angus sepsis specifier
    :return scores for 10-fold stratified cross validation
    '''
    regFunct = linMod.LogisticRegression()
    cvscore = modSel.cross_val_score(estimator=regFunct, \
        X=joinedDataframe.drop(["angus"], axis=1).select_dtypes([np.number]), \
        y=joinedDataframe["angus"], \
        scoring= scorer, \
        cv = 10)
    return cvscore
