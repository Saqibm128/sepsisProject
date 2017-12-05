'''
This script provides key preliminary analysis on data to ensure some of the assumptions we make actually works
It provides analysis on the two or more itemids for 1 variable problem, as well as
providing verification of the location of waveform data's start time and if it falls within the first 24 hours of the hospital admission
'''
import pandas as pd
import numpy as np
import commonDB
from readWaveform import waveformUtil as wfutil
import sklearn.feature_selection

# wfutil.preliminaryCompareTimes().to_csv("data/rawdatafiles/comparedTimes.csv")

def countCheck2():
    '''
    This function provides analysis on counts between itemids map and the counts that were cached already
    '''
    labelDF = pd.read_csv("preprocessing/resources/itemid_to_variable_map_only_labeled.csv")
    countsDF = pd.read_csv("data/rawdatafiles/counts.csv")
    countsDF.columns = countsDF.columns.str.upper()
    mergedDF = labelDF.merge(countsDF, left_on="ITEMID", right_on="ITEMID")
    for variable in mergedDF["LEVEL2"]:
        idx = (mergedDF["LEVEL2"] == variable)
        totalCount = mergedDF["COUNTPERADMISSION"][idx].sum()
        firstRow = mergedDF[idx]
        firstRow["COUNTPERADMISSION"] = totalCount
        mergedDF = mergedDF[mergedDF["LEVEL2"] != variable]
        mergedDF.append(firstRow)
    mergedDF.to_csv("data/rawdatafiles/mergedCountsLabel.csv")

def selfJoinFix(data):
    '''
    This method deals with the fact that chartevents includes items from multiple different technologies
    To deal with this, the selfJoinFix runs through and does a join on label to provide a preliminary results
    unfortunately, may need to work on other things over time
    :param data path to the csv file (EX: data/rawdatafiles/counts.csv format)
    :return selfjoin where only if the dbsources are different but labels are some will the result come out
    :warning This is a bandaid fix, a sort of hack. Better solution currently implemented is stolen from MIMIC3 benchmark repo
    '''
    counts = pd.DataFrame.from_csv(data)
    countsCopy = counts.copy()
    selfJoin = pd.merge(counts, countsCopy, left_on=["label"], right_on="label")
    selfJoin = selfJoin.query("itemid_x > itemid_y")  # Keep nonunique, only one copy
    # remove LOINC code self links (apparently mixed in as well)
    # "redundant" LOINC codes are not redundant, but the qualifying details (i.e. bp on leg vs bp on arms) is missing in label
    selfJoin = selfJoin.query(
        "(not (itemid_x > 50000 and itemid_x < 60000)) or (not(itemid_y > 50000 and itemid_y < 60000))")
    return selfJoin



# selfJoin = selfJoinFix("data/rawdatafiles/counts.csv")
# selfJoin.to_csv("data/rawdatafiles/selfCounts.csv")
X = pd.DataFrame.from_csv("./data/rawdatafiles/byHadmID0/full_avg_matrix.csv")
Y = pd.DataFrame.from_csv("./data/rawdatafiles/classifiedAngusSepsis.csv")["angus"][X.index]

(chi2, pval) = sklearn.feature_selection.chi2(X, Y)
features = pd.DataFrame(index=X.columns)
features['chi2'] = pd.Series(chi2, index=X.columns)
features['pval'] = pd.Series(pval, index=X.columns)
features.to_csv("data/rawdatafiles/features.csv")
