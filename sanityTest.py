'''
alters through a self
Provides verification of the location of waveform data's start time and if it falls within the first 24 hours of the hospital admission
'''
import pandas as pd
import commonDB
from readWaveform import waveformUtil as wfutil

wfutil.compareTimes().to_csv("data/rawdatafiles/comparedTimes.csv")

def selfJoinFix(data):
    '''
    This method deals with the fact that chartevents includes items from multiple different technologies
    To deal with this, the selfJoinFix runs through and does a join on label to provide a preliminary results
    unfortunately, may need to work on other things over time
    :param data path to the csv file (EX: data/rawdatafiles/counts.csv format)
    :return selfjoin where only if the dbsources are different but labels are some will the result come out
    :warning This is a bandaid fix, a sort of hack. Any real solution to deal with chartevents will need to scale better than this
    '''
    counts = pd.DataFrame.from_csv(data)
    countsCopy = counts.copy()
    selfJoin = pd.merge(counts, countsCopy, left_on=["label"], right_on="label")
    selfJoin = selfJoin.query("itemid_x > itemid_y") #Keep nonunique, only one copy
    #remove LOINC code self links (apparently mixed in as well)
    # "redundant" LOINC codes are not redundant, but the qualifying details (i.e. bp on leg vs bp on arms) is missing in label
    selfJoin = selfJoin.query("(not (itemid_x > 50000 and itemid_x < 60000)) or (not(itemid_y > 50000 and itemid_y < 60000))")
    return selfJoin

# selfJoin = selfJoinFix("data/rawdatafiles/counts.csv")
# selfJoin.to_csv("data/rawdatafiles/selfCounts.csv")
