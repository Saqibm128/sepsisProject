import pandas as pd
import numpy as np
import sklearn.metrics
import sklearn.model_selection as modSel
from addict import Dict
def test(trainTuple, testTuple, predictor=None, name=0):
    '''
    removes nonnumeric data, then fully trains with set and provides key metrics back
    :param params a list of parameter dicts which will be assigned to LogisticRegression
        object before it fits and runs, if None will not be used
    :param testTuple a tuple of examples and the correct answer
    :param trainTuple a tuple of examples and the correct answer used for testing the final results
    :param predictor the fully trained model, if None will not be used, params will be used instead
    :return DataFrame of scores for train, test split pased on the params given
    '''
    scores = pd.DataFrame(columns=["auc", "f1score", "matthews_corrcoef"], index=[name])
    #https://stackoverflow.com/questions/34842405/parameter-stratify-from-method-train-test-split-scikit-learn
    Xtrain, Ytrain = trainTuple
    Xtest, Ytest = testTuple
    Ypred = predictor.predict(Xtest)
    auc = sklearn.metrics.roc_auc_score(Ytest, Ypred)
    scores.loc[name,"auc"] = auc
    f1score = sklearn.metrics.f1_score(Ytest, Ypred)
    scores.loc[name,"f1score"] = f1score
    matC = sklearn.metrics.matthews_corrcoef(Ytest, Ypred)
    scores.loc[name,"matthews_corrcoef"]=matC
    scores.loc[name, "precision"] = sklearn.metrics.precision_score(Ytest, Ypred)
    scores.loc[name, "recall"] = sklearn.metrics.recall_score(Ytest, Ypred)
    return scores

def gridsearch_CV_wrapper(params, model, Xtrain, Ytrain, Xtest, Ytest, validation_size=.1, n_jobs=5):
    '''
    Wrapper around gridsearch
    '''
    XvalidIndices, _ = modSel.train_test_split(list(range(0, len(Xtrain))), train_size = validation_size, stratify = Ytrain)
    Xvalid = np.full(len(Xtrain), -1)
    for index in XvalidIndices:
        Xvalid[index] = 0
    predef_split = modSel.PredefinedSplit(Xvalid)
    gridSearcher = modSel.GridSearchCV(model, params, n_jobs=5, cv=predef_split)
    gridSearcher.fit(Xtrain, Ytrain)
    bestLogReg = gridSearcher.best_estimator_
    bestLogReg.fit(Xtrain, Ytrain)
    score = bestLogReg.score(Xtest, Ytest)
    return Dict({"cv_results": gridSearcher.cv_results_, \
            "predictor":bestLogReg, \
            "best_params": gridSearcher.best_params_, \
            "best_score":score, \
            "testTuple": (Xtest, Ytest), \
            "trainTuple": (Xtrain, Ytrain)})
