import sklearn.linear_model as linMod
import sklearn.model_selection as modSel
from . import util
import numpy as np

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
def cross_val_score(joinedDataframe):
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
        scoring= util.scorer, \
        cv = 10)
    return cvscore
