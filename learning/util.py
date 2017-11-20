import pandas as pd
import numpy as np
import sklearn.metrics
def test(trainTuple, testTuple, predictor=None):
    '''
    removes nonnumeric data, then fully trains with set and provides key metrics back
    :param params a list of parameter dicts which will be assigned to LogisticRegression
        object before it fits and runs, if None will not be used
    :param testTuple a tuple of examples and the correct answer
    :param trainTuple a tuple of examples and the correct answer used for testing the final results
    :param predictor the fully trained model, if None will not be used, params will be used instead
    :return DataFrame of scores for train, test split pased on the params given
    '''
    scores = pd.DataFrame(columns=["auc", "f1score", "matthews_corrcoef"])
    #https://stackoverflow.com/questions/34842405/parameter-stratify-from-method-train-test-split-scikit-learn
    Xtrain, Ytrain = trainTuple
    Xtest, Ytest = testTuple
    Ypred = predictor.predict(Xtest)
    auc = sklearn.metrics.roc_auc_score(Ytest, Ypred)
    scores.loc[0,"auc"] = auc
    f1score = sklearn.metrics.f1_score(Ytest, Ypred)
    scores.loc[0,"f1score"] = f1score
    matC = sklearn.metrics.matthews_corrcoef(Ytest, Ypred)
    scores.loc[0,"matthews_corrcoef"]=matC
    return scores
