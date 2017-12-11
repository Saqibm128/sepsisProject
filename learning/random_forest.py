import sklearn.model_selection as modSel
import sklearn.metrics
import sklearn.ensemble
from . import util
import numpy as np
import pandas as pd
from addict import Dict
# This script contains helper functions used to train and test on the data


def test_train_valid_explicit(Xtrain, Ytrain, Xtest, Ytest, validation_size=.1, n_jobs=4):
    '''
    Given a test, train split, does the validation split itself.
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
    params = Dict()
    params.criterion = ['gini', 'entropy']
    params.n_estimators = [20, 40, 60, 80, 160, 300]
    params.max_features = ['auto', 'log2', .1, .4, .8]
    params.max_depth = [None, 1, 4]
    params.min_samples_split = [2, 8]
    params.min_weight_fraction_leaf = [0, .2, .5]
    params.min_impurity_decrease = [0, .5, 1]
    params.n_jobs = [1]
    rf = sklearn.ensemble.RandomForestClassifier()
    return util.gridsearch_CV_wrapper(params=params, model=rf, Xtrain = Xtrain, Ytrain=Ytrain, Xtest=Xtest, Ytest=Ytest, validation_size=validation_size, n_jobs=n_jobs)
