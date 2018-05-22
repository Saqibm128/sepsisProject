from readWaveform.waveform_reader import WaveformReader
from readWaveform import waveformUtil
from multiprocessing import Process
from multiprocessing import Queue
from multiprocessing import Manager
from addict import Dict
import pandas as pd
import numpy as np

__n_workers = 24

def processSubjectID(subject_id):
    reader = WaveformReader()
    reader.traverser.numeric = True
    toReturn = pd.DataFrame()
    fileToHADMID = reader.traverser.matchWithHADMID(subject_id)
    multiRecords = reader.traverser.getMultiRecordFiles(subject_id)
    for multiRecord in multiRecords:
        singleRecord = pd.DataFrame(index=[multiRecord])
        singleRecord["HADM_MAPPING"] = fileToHADMID[multiRecord] # Maps file to hadmid
        singleRecord["SUBJECT_ID"] = subject_id
        try:
            data, fields = reader.getRecord(subject_id, multiRecord)
            for sig_name in data.columns:
                if (~(data[sig_name].apply(np.isnan))).all():
                    singleRecord[sig_name + "_PERCENT_MISSING"] = 0
                else:
                    singleRecord[sig_name + "_PERCENT_MISSING"] = data[sig_name].apply(np.isnan).value_counts()[True] / len(data)
            singleRecord["length"] = len(data) / 125
        except: #hack
            singleRecord["Error"] = "Length was set to end" #Note: apparently some numeric records were incorrectly saved (i.e. p052848-2172-03-03-09-04n)
        toReturn = pd.concat([toReturn,singleRecord])
    return toReturn


def helperWaveformRunner(toRunQueue, toReturnQueue):
    '''
    Uses queues to analyze waveforms for key prelim stats
    '''
    for subject_id in iter(toRunQueue.get, None):
        toReturn = processSubjectID(subject_id)
        print(toRunQueue.qsize())
        toReturnQueue.put(toReturn)




if __name__ == "__main__":

    # icuComp = waveformUtil.preliminaryCompareTimesICU()
    # careunits = icuComp["first_careunit"].unique()
    # careunitFreq = pd.DataFrame()
    # for unit in careunits:
    #     print((icuComp["first_careunit"]==unit))
    #     careunitFreq[unit] = (icuComp["first_careunit"]==unit).value_counts()
    # careunitFreq.to_csv("data/rawdatafiles/icustay_waveform_freq.csv")
    reader = WaveformReader()
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
    allResults = pd.DataFrame()
    while not outQueue.empty():
        allResults = pd.concat([outQueue.get(), allResults])
    allResults = allResults.fillna(100) #missing 100 % if we didnt find the waveform
    print(allResults)
    allResults.to_csv("data/rawdatafiles/numeric_prelim_analysis.csv")


    # for subject in subjects:
    #     q.put(subject)
    # print(reader.access_subject("p010013"))
    # sig, fields = reader.get_record("p010013", reader.access_subject("p010013")[0])
    # print(fields)
    # print(len(sig))
