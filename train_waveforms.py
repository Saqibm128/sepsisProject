import pandas as pd
from readWaveform.segment_reader import SegmentReader
import learning.logReg as logReg
import learning.svm
import learning.util
import learning.random_forest
import pipeline.feature_selection_wrapper as feat_sel
import sklearn.model_selection
from sklearn.preprocessing import MinMaxScaler
import time


if __name__=="__main__":
    mm = MinMaxScaler()
    segReader = SegmentReader();
    Y = segReader.labels()
    allFeatureWeights = None
    allFullScores = None
    for i in [1, 2, 3, 4]:
        trainInd, testInd = sklearn.model_selection.train_test_split(pd.Series(Y.index), train_size=.8, stratify=Y, random_state=i)
        X = segReader.simpleStatsAll()
        mm.fit(X.values)
        X = pd.DataFrame(mm.transform(X.values), index=X.index, columns=X.columns)

        Ytrain, Xtrain, Ytest, Xtest = Y.loc[trainInd], X.loc[trainInd], Y.loc[testInd], X.loc[testInd]
        print(Y.shape, Ytrain.shape)
        model_name = "LogisticRegression"
        print("beginning model:", model_name)
        result = logReg.test_train_valid_explicit(Xtrain = Xtrain.values, \
                                                    Xtest = Xtest.values, \
                                                    Ytrain= Ytrain.values, \
                                                    Ytest = Ytest.values, \
                                                    n_jobs = -1, \
                                                    validation_size=.2)
        print("Model: ", model_name, result.best_score)
        print("Model: ", model_name, "; Best Params: ", result.best_params)
        features_weights_to_add = pd.DataFrame(result.predictor.coef_)
        features_weights_to_add.index = [model_name]
        features_weights_to_add.columns = X.columns
        features_weights= pd.concat([features_weights_to_add])
        fullScores = learning.util.test(trainTuple=result.trainTuple, \
                                        testTuple=result.testTuple, \
                                        predictor=result.predictor, \
                                        name=model_name)
        # fullScores.to_csv("data/rawdatafiles/byHadmIDNumRec/final_scores.csv")

        model_name = "random_forest"
        print("beginning model:", model_name)
        result = learning.random_forest.test_train_valid_explicit(Xtrain = Xtrain.values, \
                                                    Xtest = Xtest.values, \
                                                    Ytrain= Ytrain.values, \
                                                    Ytest = Ytest.values, \
                                                    n_jobs = -1, \
                                                    validation_size=.2)
        print("Model: ", model_name, result.best_score)
        print("Model: ", model_name, "; Best Params: ", result.best_params)
        features_weights_to_add = pd.DataFrame(result.predictor.feature_importances_).transpose()
        features_weights_to_add.index = [model_name]
        features_weights_to_add.columns = X.columns
        features_weights= pd.concat([features_weights, features_weights_to_add])
        # features_weights.to_csv("data/rawdatafiles/byHadmIDNumRec/features_weights.csv")
        fullScores = pd.concat([fullScores, learning.util.test(trainTuple=result.trainTuple, \
                                        testTuple=result.testTuple, \
                                        predictor=result.predictor, \
                                        name=model_name \
                                        )])
        print(fullScores)
        # fullScores.to_csv("data/rawdatafiles/byHadmIDNumRec/final_scores.csv")


        X = segReader.simpleStatsAll(unittime = pd.Timedelta("30 minutes"))
        mm.fit(X.values)
        X = pd.DataFrame(mm.transform(X.values), index=X.index, columns=X.columns)

        Ytrain, Xtrain, Ytest, Xtest = Y.loc[trainInd], X.loc[trainInd], Y.loc[testInd], X.loc[testInd]
        print(Y.shape, Ytrain.shape)
        model_name = "LogisticRegression 30 minutes"
        print("beginning model:", model_name)
        result = logReg.test_train_valid_explicit(Xtrain = Xtrain.values, \
                                                    Xtest = Xtest.values, \
                                                    Ytrain= Ytrain.values, \
                                                    Ytest = Ytest.values, \
                                                    n_jobs = -1, \
                                                    validation_size=.2)
        print("Model: ", model_name, result.best_score)
        print("Model: ", model_name, "; Best Params: ", result.best_params)
        features_weights_to_add = pd.DataFrame(result.predictor.coef_)
        features_weights_to_add.index = [model_name]
        features_weights_to_add.columns = X.columns
        features_weights= pd.concat([ features_weights, features_weights_to_add])
        fullScores = pd.concat([fullScores, learning.util.test(trainTuple=result.trainTuple, \
                                        testTuple=result.testTuple, \
                                        predictor=result.predictor, \
                                        name=model_name)])
        # fullScores.to_csv("data/rawdatafiles/byHadmIDNumRec/final_scores.csv")

        model_name = "random_forest 30 minutes"
        print("beginning model:", model_name)
        result = learning.random_forest.test_train_valid_explicit(Xtrain = Xtrain.values, \
                                                    Xtest = Xtest.values, \
                                                    Ytrain= Ytrain.values, \
                                                    Ytest = Ytest.values, \
                                                    n_jobs = -1, \
                                                    validation_size=.2)
        print("Model: ", model_name, result.best_score)
        print("Model: ", model_name, "; Best Params: ", result.best_params)
        features_weights_to_add = pd.DataFrame(result.predictor.feature_importances_).transpose()
        features_weights_to_add.index = [model_name]
        features_weights_to_add.columns = X.columns
        features_weights= pd.concat([features_weights, features_weights_to_add])
        # features_weights.to_csv("data/rawdatafiles/byHadmIDNumRec/features_weights.csv")
        fullScores = pd.concat([fullScores, learning.util.test(trainTuple=result.trainTuple, \
                                        testTuple=result.testTuple, \
                                        predictor=result.predictor, \
                                        name=model_name \
                                        )])
        print(fullScores)
        # fullScores.to_csv("data/rawdatafiles/byHadmIDNumRec/final_scores.csv")

        X = segReader.simpleStatsAll(unittime = pd.Timedelta("20 minutes"))
        mm.fit(X.values)
        X = pd.DataFrame(mm.transform(X.values), index=X.index, columns=X.columns)

        Ytrain, Xtrain, Ytest, Xtest = Y.loc[trainInd], X.loc[trainInd], Y.loc[testInd], X.loc[testInd]
        print(Y.shape, Ytrain.shape)
        model_name = "LogisticRegression 20 minutes"
        print("beginning model:", model_name)
        result = logReg.test_train_valid_explicit(Xtrain = Xtrain.values, \
                                                    Xtest = Xtest.values, \
                                                    Ytrain= Ytrain.values, \
                                                    Ytest = Ytest.values, \
                                                    n_jobs = -1, \
                                                    validation_size=.2)
        print("Model: ", model_name, result.best_score)
        print("Model: ", model_name, "; Best Params: ", result.best_params)
        features_weights_to_add = pd.DataFrame(result.predictor.coef_)
        features_weights_to_add.index = [model_name]
        features_weights_to_add.columns = X.columns
        features_weights= pd.concat([ features_weights, features_weights_to_add])
        fullScores = pd.concat([fullScores, learning.util.test(trainTuple=result.trainTuple, \
                                        testTuple=result.testTuple, \
                                        predictor=result.predictor, \
                                        name=model_name)])
        # fullScores.to_csv("data/rawdatafiles/byHadmIDNumRec/final_scores.csv")

        model_name = "random_forest 20 minutes"
        print("beginning model:", model_name)
        result = learning.random_forest.test_train_valid_explicit(Xtrain = Xtrain.values, \
                                                    Xtest = Xtest.values, \
                                                    Ytrain= Ytrain.values, \
                                                    Ytest = Ytest.values, \
                                                    n_jobs = -1, \
                                                    validation_size=.2)
        print("Model: ", model_name, result.best_score)
        print("Model: ", model_name, "; Best Params: ", result.best_params)
        features_weights_to_add = pd.DataFrame(result.predictor.feature_importances_).transpose()
        features_weights_to_add.index = [model_name]
        features_weights_to_add.columns = X.columns
        features_weights= pd.concat([features_weights, features_weights_to_add])
        # features_weights.to_csv("data/rawdatafiles/byHadmIDNumRec/features_weights.csv")
        fullScores = pd.concat([fullScores, learning.util.test(trainTuple=result.trainTuple, \
                                        testTuple=result.testTuple, \
                                        predictor=result.predictor, \
                                        name=model_name \
                                        )])
        print(fullScores)
        # fullScores.to_csv("data/rawdatafiles/byHadmIDNumRec/final_scores.csv")

        X = segReader.simpleStatsAll(unittime = pd.Timedelta("10 minutes"))
        mm.fit(X.values)
        X = pd.DataFrame(mm.transform(X.values), index=X.index, columns=X.columns)

        Ytrain, Xtrain, Ytest, Xtest = Y.loc[trainInd], X.loc[trainInd], Y.loc[testInd], X.loc[testInd]
        print(Y.shape, Ytrain.shape)
        model_name = "LogisticRegression 10 minutes"
        print("beginning model:", model_name)
        result = logReg.test_train_valid_explicit(Xtrain = Xtrain.values, \
                                                    Xtest = Xtest.values, \
                                                    Ytrain= Ytrain.values, \
                                                    Ytest = Ytest.values, \
                                                    n_jobs = -1, \
                                                    validation_size=.2)
        print("Model: ", model_name, result.best_score)
        print("Model: ", model_name, "; Best Params: ", result.best_params)
        features_weights_to_add = pd.DataFrame(result.predictor.coef_)
        features_weights_to_add.index = [model_name]
        features_weights_to_add.columns = X.columns
        features_weights= pd.concat([ features_weights, features_weights_to_add])
        fullScores = pd.concat([fullScores, learning.util.test(trainTuple=result.trainTuple, \
                                        testTuple=result.testTuple, \
                                        predictor=result.predictor, \
                                        name=model_name)])
        # fullScores.to_csv("data/rawdatafiles/byHadmIDNumRec/final_scores.csv")

        model_name = "random_forest 10 minutes"
        print("beginning model:", model_name)
        result = learning.random_forest.test_train_valid_explicit(Xtrain = Xtrain.values, \
                                                    Xtest = Xtest.values, \
                                                    Ytrain= Ytrain.values, \
                                                    Ytest = Ytest.values, \
                                                    n_jobs = -1, \
                                                    validation_size=.2)
        print("Model: ", model_name, result.best_score)
        print("Model: ", model_name, "; Best Params: ", result.best_params)
        features_weights_to_add = pd.DataFrame(result.predictor.feature_importances_).transpose()
        features_weights_to_add.index = [model_name]
        features_weights_to_add.columns = X.columns
        features_weights= pd.concat([features_weights, features_weights_to_add])
        # features_weights.to_csv("data/rawdatafiles/byHadmIDNumRec/features_weights.csv")
        fullScores = pd.concat([fullScores, learning.util.test(trainTuple=result.trainTuple, \
                                        testTuple=result.testTuple, \
                                        predictor=result.predictor, \
                                        name=model_name \
                                        )])
        print(fullScores)
        # fullScores.to_csv("data/rawdatafiles/byHadmIDNumRec/final_scores.csv")

        X = segReader.simpleStatsAll(unittime = pd.Timedelta("5 minutes"))
        mm.fit(X.values)
        X = pd.DataFrame(mm.transform(X.values), index=X.index, columns=X.columns)

        Ytrain, Xtrain, Ytest, Xtest = Y.loc[trainInd], X.loc[trainInd], Y.loc[testInd], X.loc[testInd]
        print(Y.shape, Ytrain.shape)
        model_name = "LogisticRegression 5 minutes"
        print("beginning model:", model_name)
        result = logReg.test_train_valid_explicit(Xtrain = Xtrain.values, \
                                                    Xtest = Xtest.values, \
                                                    Ytrain= Ytrain.values, \
                                                    Ytest = Ytest.values, \
                                                    n_jobs = -1, \
                                                    validation_size=.2)
        print("Model: ", model_name, result.best_score)
        print("Model: ", model_name, "; Best Params: ", result.best_params)
        features_weights_to_add = pd.DataFrame(result.predictor.coef_)
        features_weights_to_add.index = [model_name]
        features_weights_to_add.columns = X.columns
        features_weights= pd.concat([ features_weights, features_weights_to_add])
        fullScores = pd.concat([fullScores, learning.util.test(trainTuple=result.trainTuple, \
                                        testTuple=result.testTuple, \
                                        predictor=result.predictor, \
                                        name=model_name)])
        # fullScores.to_csv("data/rawdatafiles/byHadmIDNumRec/final_scores.csv")

        model_name = "random_forest 5 minutes"
        print("beginning model:", model_name)
        result = learning.random_forest.test_train_valid_explicit(Xtrain = Xtrain.values, \
                                                    Xtest = Xtest.values, \
                                                    Ytrain= Ytrain.values, \
                                                    Ytest = Ytest.values, \
                                                    n_jobs = -1, \
                                                    validation_size=.2)
        print("Model: ", model_name, result.best_score)
        print("Model: ", model_name, "; Best Params: ", result.best_params)
        features_weights_to_add = pd.DataFrame(result.predictor.feature_importances_).transpose()
        features_weights_to_add.index = [model_name]
        features_weights_to_add.columns = X.columns
        features_weights= pd.concat([features_weights, features_weights_to_add])
        # features_weights.to_csv("data/rawdatafiles/byHadmIDNumRec/features_weights.csv")
        fullScores = pd.concat([fullScores, learning.util.test(trainTuple=result.trainTuple, \
                                        testTuple=result.testTuple, \
                                        predictor=result.predictor, \
                                        name=model_name \
                                        )])
        print(fullScores)
        # fullScores.to_csv("data/rawdatafiles/byHadmIDNumRec/final_scores.csv")
        if allFeatureWeights is None:
            allFeatureWeights = features_weights
            allFullScores = fullScores
        allFeatureWeights += features_weights
        allFullScores += fullScores
    allFeatureWeights /= 4
    allFullScores /= 4
    allFullScores.to_csv("data/rawdatafiles/byHadmIDNumRec/final_scores.csv")
    allFeatureWeights.to_csv("data/rawdatafiles/byHadmIDNumRec/features_weights.csv")
