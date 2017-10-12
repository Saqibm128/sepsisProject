import sklearn.metrics.roc_auc_score

def scorer(estimator, x, y):
    '''
    Wrapper for score function becuase I'm bad at lambdas
    :param estimator any object that implements score
    :param x test set
    :param y outcome
    '''
    return estimator.score(x, y)
