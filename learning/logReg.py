import sklearn.linear_model as linMod
import sklearn.model_selection as modSel
from . import util
import numpy as np

# This function contains helper functions used to train and test on the data
# Specifically used for LogisticRegression tests as well as all the setup that goes along with it

def cross_val_score(joinedDataframe):
    '''
    removes nonnumeric data, then tests with cross_val_score and provides key metrics back
    :param joinedDataframe a dataframe where all columns except the last one are feature data and the
        last column is the Angus sepsis specifier
    :return TODO: figure it out
    '''
    regFunct = linMod.LogisticRegression()
    cvscore = modSel.cross_val_score(estimator=regFunct, X=joinedDataframe.drop(["angus"], axis=1).select_dtypes([np.number]), y=joinedDataframe["angus"], scoring= util.scorer, cv = 10)
