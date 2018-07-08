import pandas as pd
from readWaveform.segment_reader import SegmentReader
import learning.logReg as lr
import learning.svm
import learning.util
import learning.random_forest as rf
import pipeline.feature_selection_wrapper as feat_sel
import sklearn.model_selection
from sklearn.preprocessing import MinMaxScaler
import time

num_splits = 3
def appendAll(fullScores, featureWeights, times, scoresToAppend, weightsToAppend, timesToAppend):
    return pd.concat([fullScores, scoresToAppend]), pd.concat([featureWeights, weightsToAppend]), pd.concat([times, timesToAppend])
def testTrain(X, Y, trainInd, testInd, model_name, test_train_valid_explicit):
    '''
    @param X data instances
    @param Y labels
    @param trainInd training split indices
    @param testInd testing split indices
    @param model_name
    @param test_train_valid_explicit the function we use to generate results (see learning.logReg for example)
    '''
    Ytrain, Xtrain, Ytest, Xtest = Y.loc[trainInd], X.loc[trainInd], Y.loc[testInd], X.loc[testInd]
    print("beginning model:", model_name)
    startTime = time.time()
    result = test_train_valid_explicit(Xtrain = Xtrain.values, \
                                                Xtest = Xtest.values, \
                                                Ytrain= Ytrain.values, \
                                                Ytest = Ytest.values, \
                                                n_jobs = -1, \
                                                validation_size=.2)
    print("Model: ", model_name, result.best_score)
    print("Model: ", model_name, "; Best Params: ", result.best_params)
    features_weights_to_add = pd.DataFrame(result.weights)
    features_weights_to_add.index = [model_name]
    features_weights_to_add.columns = X.columns
    features_weights= features_weights_to_add
    fullScores = learning.util.test(trainTuple=result.trainTuple, \
                                    testTuple=result.testTuple, \
                                    predictor=result.predictor, \
                                    name=model_name)
    # fullScores.to_csv("data/rawdatafiles/byHadmIDNumRec/final_scores.csv")
    endTime = time.time()
    times[model_name] = pd.Timedelta(seconds = endTime - startTime)
    return fullScores, features_weights, times

if __name__=="__main__":
    mm = MinMaxScaler()
    segReader = SegmentReader();
    Y = segReader.labels()
    allFeatureWeights = None
    allFullScores = None
    totalTimes = None
    for i in range(num_splits):
        times = pd.Series()

        trainInd, testInd = sklearn.model_selection.train_test_split(pd.Series(Y.index), train_size=.8, stratify=Y, random_state=i)
        X = segReader.simpleStatsAll()
        mm.fit(X.values)
        X = pd.DataFrame(mm.transform(X.values), index=X.index, columns=X.columns)

        fullScores, featureWeights, times = testTrain(X, Y, trainInd, testInd, "LR 1 hour", lr.test_train_valid_explicit)
        fullScoresToAppend, featureWeightsToAppend, timesToAppend = testTrain(X, Y, trainInd, testInd, "RF 1 hour", rf.test_train_valid_explicit)
        fullScores, featureWeights, times = appendAll(fullScores, featureWeights, times, fullScoresToAppend, featureWeightsToAppend, timesToAppend)

        X = segReader.simpleStatsAll(unittime = pd.Timedelta("40 minutes"))
        mm.fit(X.values)
        X = pd.DataFrame(mm.transform(X.values), index=X.index, columns=X.columns)
        fullScoresToAppend, featureWeightsToAppend, timesToAppend = testTrain(X, Y, trainInd, testInd, "LR 40 minutes", lr.test_train_valid_explicit)
        fullScores, featureWeights, times = appendAll(fullScores, featureWeights, times, fullScoresToAppend, featureWeightsToAppend, timesToAppend)
        fullScoresToAppend, featureWeightsToAppend, timesToAppend = testTrain(X, Y, trainInd, testInd, "RF 40 minutes", rf.test_train_valid_explicit)
        fullScores, featureWeights, times = appendAll(fullScores, featureWeights, times, fullScoresToAppend, featureWeightsToAppend, timesToAppend)

        X = segReader.simpleStatsAll(unittime = pd.Timedelta("20 minutes"))
        mm.fit(X.values)
        X = pd.DataFrame(mm.transform(X.values), index=X.index, columns=X.columns)
        fullScoresToAppend, featureWeightsToAppend, timesToAppend = testTrain(X, Y, trainInd, testInd, "LR 20 minutes", lr.test_train_valid_explicit)
        fullScores, featureWeights, times = appendAll(fullScores, featureWeights, times, fullScoresToAppend, featureWeightsToAppend, timesToAppend)
        fullScoresToAppend, featureWeightsToAppend, timesToAppend = testTrain(X, Y, trainInd, testInd, "RF 20 minutes", rf.test_train_valid_explicit)
        fullScores, featureWeights, times = appendAll(fullScores, featureWeights, times, fullScoresToAppend, featureWeightsToAppend, timesToAppend)

        X = segReader.simpleStatsAll(unittime = pd.Timedelta("10 minutes"))
        mm.fit(X.values)
        X = pd.DataFrame(mm.transform(X.values), index=X.index, columns=X.columns)
        fullScoresToAppend, featureWeightsToAppend, timesToAppend = testTrain(X, Y, trainInd, testInd, "LR 10 minutes", lr.test_train_valid_explicit)
        fullScores, featureWeights, times = appendAll(fullScores, featureWeights, times, fullScoresToAppend, featureWeightsToAppend, timesToAppend)
        fullScoresToAppend, featureWeightsToAppend, timesToAppend = testTrain(X, Y, trainInd, testInd, "RF 10 minutes", rf.test_train_valid_explicit)
        fullScores, featureWeights, times = appendAll(fullScores, featureWeights, times, fullScoresToAppend, featureWeightsToAppend, timesToAppend)

        # X = segReader.simpleStatsAll(unittime = pd.Timedelta("5 minutes"))
        # mm.fit(X.values)
        # X = pd.DataFrame(mm.transform(X.values), index=X.index, columns=X.columns)
        # fullScoresToAppend, featureWeightsToAppend, timesToAppend = testTrain(X, Y, trainInd, testInd, "LR 5 minutes", lr.test_train_valid_explicit)
        # fullScores, featureWeights, times = appendAll(fullScores, featureWeights, times, fullScoresToAppend, featureWeightsToAppend, timesToAppend)
        # fullScoresToAppend, featureWeightsToAppend, timesToAppend = testTrain(X, Y, trainInd, testInd, "RF 5 minutes", rf.test_train_valid_explicit)
        # fullScores, featureWeights, times = appendAll(fullScores, featureWeights, times, fullScoresToAppend, featureWeightsToAppend, timesToAppend)

        print(fullScores)
        if allFeatureWeights is None:
            allFeatureWeights = featureWeights
            allFullScores = fullScores
            totalTimes = times
        else:
            allFeatureWeights += featureWeights
            allFullScores += fullScores
            totalTimes += times
    allFeatureWeights /= num_splits
    allFullScores /= num_splits
    totalTimes /= num_splits
    allFullScores.to_csv("data/rawdatafiles/numRecResults/final_scores.csv")
    allFeatureWeights.to_csv("data/rawdatafiles/numRecResults/features_weights.csv")
    totalTimes.to_csv("data/rawdatafiles/numRecResults/total_times.csv")

    print(allFullScores)
