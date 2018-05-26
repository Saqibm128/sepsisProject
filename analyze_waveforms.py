from readWaveform.waveform_reader import WaveformReader
from readWaveform import waveformUtil
from multiprocessing import Process
from multiprocessing import Queue
from multiprocessing import Manager
from addict import Dict
from commonDB import read_sql
import pandas as pd
import numpy as np
import re
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, show, savefig, xlim, figure, \
                hold, ylim, legend, boxplot, setp, axes

__n_workers = 24
threshold = 1 # threshold is which section of the subjects to consider for a single variable
numericMapping = pd.read_csv("preprocessing/resources/numeric_waveform_to_variable_map.csv")
numericMapping["numeric"] = numericMapping["numeric"].str.upper()
numericMapping["high_level_var"] = numericMapping["high_level_var"].str.upper()

def processSubjectID(subject_id):
    reader = WaveformReader(numericMapping=numericMapping)
    reader.traverser.numeric = True
    toReturn = [pd.DataFrame()]
    fileToHADMID = reader.traverser.matchWithHADMID(subject_id)
    multiRecords = reader.traverser.getMultiRecordFiles(subject_id)
    if len(multiRecords) == 0:
        return pd.DataFrame() #if there are not numeric record files, we just skip
    for multiRecord in multiRecords:
        singleRecord = pd.DataFrame(index=[multiRecord])
        singleRecord["HADM_MAPPING"] = fileToHADMID[multiRecord] # Maps file to hadmid
        singleRecord["SUBJECT_ID"] = subject_id
        if fileToHADMID[multiRecord] != "NOT FOUND": #could not be matched for some reason
            admittime = read_sql("SELECT ADMITTIME from ADMISSIONS WHERE HADM_ID = " + str(fileToHADMID[multiRecord])).iloc[0, 0]
        try:
            data, fields = reader.getRecord(subject_id, multiRecord)
            for sig_name in data.columns:
                if (~(data[sig_name].apply(np.isnan))).all():
                    singleRecord[sig_name + "_PERCENT_MISSING"] = 0
                else:
                    singleRecord[sig_name + "_PERCENT_MISSING"] = pd.isnull(data[sig_name]).value_counts()[True] / len(data)
                #For each signal, find the percentage that is filled in the first 24 hours after admission
                if fileToHADMID[multiRecord] != "NOT FOUND":
                    admittime = pd.Timestamp(admittime)
                    firstDay = data.iloc[data.index < admittime + pd.Timedelta("1 days")]
                    if (pd.isnull(firstDay[sig_name])).all():
                        singleRecord[sig_name + "_PERCENT_MISSING_FIRST_24_HOURS"] = 1
                    else:
                        singleRecord[sig_name + "_PERCENT_MISSING_FIRST_24_HOURS"] =  1 - pd.isnull(firstDay[sig_name]).value_counts()[False] / (60*60*24*60)
            singleRecord["length"] = len(data)
        except:
            singleRecord["comment"] = "Could not getRecord"
        toReturn.append(singleRecord)
    return pd.concat(toReturn)


def helperWaveformRunner(toRunQueue, toReturnQueue):
    '''
    Uses queues to analyze waveforms for key prelim stats
    '''
    for subject_id in iter(toRunQueue.get, None):
        print(toRunQueue.qsize())
        toReturn = processSubjectID(subject_id)
        toReturnQueue.put(toReturn)




if __name__ == "__main__":

    # icuComp = waveformUtil.preliminaryCompareTimes()
    # icuComp.to_csv("data/rawdatafiles/prelimTimeCompare.csv")
    # careunits = icuComp["first_careunit"].unique()
    # careunitFreq = pd.DataFrame()
    # for unit in careunits:
    #     print((icuComp["first_careunit"]==unit))
    #     careunitFreq[unit] = (icuComp["first_careunit"]==unit).value_counts()
    # careunitFreq.to_csv("data/rawdatafiles/icustay_waveform_freq.csv")

    #
    reader = WaveformReader(numericMapping=numericMapping)
    reader.traverser.numeric = True

    manager = Manager()
    inQueue = manager.Queue()
    outQueue = manager.Queue()
    subjects = reader.traverser.getSubjects()
    [inQueue.put(subject) for subject in subjects]
    [inQueue.put(None) for i in range(__n_workers)]
    processes = [Process(target=helperWaveformRunner, args=(inQueue, outQueue)) for i in range(__n_workers)]
    [process.start() for process in processes]
    [process.join() for process in processes]
    allResults = []
    while not outQueue.empty():
        allResults.append(outQueue.get())
    allResults = pd.concat(allResults)
    allResults = allResults.fillna(1) #missing 100 % if we didnt find the waveform
    allResults.to_csv("data/rawdatafiles/numeric_prelim_analysis.csv")
    # allResults = pd.read_csv("data/rawdatafiles/numeric_prelim_analysis.csv")

    numericStats = pd.DataFrame(index=allResults.columns, columns=["Average Missing", "Num Missing"])
    twentyFourHourCoverage = []
    for col in allResults.columns:
        if re.search(r"_PERCENT_MISSING", col) is not None:
            numericStats["Average Missing"][col] = allResults[allResults[col] < threshold][col].mean() #Mean of all columns that measure amount of info missing
            numericStats["Num Missing"][col] =  allResults[~(allResults[col] < threshold)].shape[0] #Counts number of waveforms that don't have the signal
        if re.search(r"_PERCENT_MISSING_FIRST_24_HOURS$", col) is not None:
            twentyFourHourCoverage.append(pd.DataFrame({"Average Missing": [allResults[col].mean()]}, index=[col]))
    numericStats.to_csv("data/rawdatafiles/summarized_numeric_prelim_analysis.csv")
    (pd.concat(twentyFourHourCoverage)).to_csv("data/rawdatafiles/numeric_24_hour_analysis.csv")
