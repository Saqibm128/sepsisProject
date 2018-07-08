import sklearn.feature_selection
import pandas as pd

def chi2test(X, Y, pval_thresh=.05, num_features=47):
    '''
    With respect to Bonferroni correction
    '''
    # print(X.shape)
    # print(Y.shape)
    (chi2, pval) = sklearn.feature_selection.chi2(X, Y)
    features = pd.DataFrame(index=X.columns)
    features['chi2'] = pd.Series(chi2, index=X.columns)
    features['pval'] = pd.Series(pval, index=X.columns)
    return features.loc[features['pval'] < pval_thresh / num_features];
