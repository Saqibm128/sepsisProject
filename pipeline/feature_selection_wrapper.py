import sklearn.feature_selection
import pandas as pd

def chi2test(X, Y, pval_thresh=.01):
    '''
    '''
    (chi2, pval) = sklearn.feature_selection.chi2(X, Y)
    features = pd.DataFrame(index=X.columns)
    features['chi2'] = pd.Series(chi2, index=X.columns)
    features['pval'] = pd.Series(pval, index=X.columns)
    return features.loc[features['pval'] < pval_thresh];
