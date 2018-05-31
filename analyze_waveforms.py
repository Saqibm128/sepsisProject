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
# mpl.use('Agg')
import matplotlib.pyplot as plt
plt.switch_backend('agg')

from matplotlib.pyplot import plot, show, savefig, xlim, figure, \
                hold, ylim, legend, boxplot, setp, axes

__n_workers = 24
threshold = 1 # threshold is which section of the subjects to consider for a single variable
numericMapping = pd.read_csv("preprocessing/resources/numeric_waveform_to_variable_map.csv")
numericMapping["numeric"] = numericMapping["numeric"].str.upper()
numericMapping["high_level_var"] = numericMapping["high_level_var"].str.upper()
numericMapping = None

def processSubjectID(subject_id, numHours = 24):
    reader = WaveformReader(numericMapping=numericMapping)
    reader.traverser.numeric = True
    toReturn = [pd.DataFrame()]
    fileToHADMID = reader.traverser.matchWithHADMID(subject_id)
    multiRecords = reader.traverser.getMultiRecordFiles(subject_id)
    if len(multiRecords) == 0:
        return pd.DataFrame() #if there are not numeric record files, we just skip
    for multiRecord in multiRecords:
        singleRecord = pd.DataFrame(index=[multiRecord])
        singleRecord["HADM_MAPPING"] = fileToHADMID[multiRecord].hadmid # Maps file to hadmid
        singleRecord["SUBJECT_ID"] = subject_id
        if fileToHADMID[multiRecord].hadmid != "NOT FOUND": #could not be matched for some reason
            admittime = fileToHADMID[multiRecord].admittime
        try:
            data, fields = reader.getRecord(multiRecord, subject_id=subject_id)
            for sig_name in data.columns:
                if (~(data[sig_name].apply(np.isnan))).all():
                    singleRecord[sig_name + "_PERCENT_MISSING"] = 0
                else:
                    singleRecord[sig_name + "_PERCENT_MISSING"] = pd.isnull(data[sig_name]).value_counts()[True] / len(data[sig_name])
                #For each signal, find the percentage that is filled in the first 24 hours after admission
                if fileToHADMID[multiRecord].hadmid != "NOT FOUND":
                    admittime = pd.Timestamp(admittime)
                    firstDay = data.iloc[(data.index < admittime + pd.Timedelta(str(numHours) + " hours")) & (data.index > admittime)]
                    if (pd.isnull(firstDay[sig_name])).all():
                        singleRecord[sig_name + "_PERCENT_MISSING_FIRST_24_HOURS"] = 1
                    else:
                        singleRecord[sig_name + "_PERCENT_MISSING_FIRST_24_HOURS"] =  1 - pd.isnull(firstDay[sig_name]).value_counts()[False] / (60*numHours) # 60 minutes in an hour, 24 hours in a day
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

def plotRecord(allResults, ind, name, filename):
    plt.hist(allResults[ind][name.upper() + "_PERCENT_MISSING_FIRST_24_HOURS"], bins=20, rwidth=.5)
    plt.xlabel("Percent Missing in First 24 Hours")
    plt.title(name)
    plt.ylabel("Number of Numeric Records")
    plt.savefig("data/rawdatafiles/" + filename + ".png", dpi=300, bottom=-.1)
    plt.gcf().clear()



if __name__ == "__main__":

    ## preliminary analysis based solely on file names
    ##  Using this because files were not downloaded yet onto server, so depending on internet slow speed
    # icuComp = waveformUtil.preliminaryCompareTimes()
    # icuComp.to_csv("data/rawdatafiles/prelimTimeCompare.csv")
    # careunits = icuComp["first_careunit"].unique()
    # careunitFreq = pd.DataFrame()
    # for unit in careunits:
    #     print((icuComp["first_careunit"]==unit))
    #     careunitFreq[unit] = (icuComp["first_careunit"]==unit).value_counts()
    # careunitFreq.to_csv("data/rawdatafiles/icustay_waveform_freq.csv")

    # Specific file reader for waveform files on /data/mimic3wdb
    #   Generates a set of statistics for each waveform
    reader = WaveformReader(numericMapping=numericMapping)
    reader.traverser.numeric = True

    manager = Manager()
    inQueue = manager.Queue()
    outQueue = manager.Queue()
    subjects = reader.traverser.getSubjects()[0:500]
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
    allResults = pd.read_csv("data/rawdatafiles/numeric_prelim_analysis.csv")

    # Further analysis of statistics to provide average data
    # 22461 by 600 matrix is hard to parse, so we just use mean
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

    # Let's go through and find out how many samples remain depending on how we filter
    for threshold in [.2, .4, .6, .8, 1]:
        ind = ((allResults["HEART RATE_PERCENT_MISSING_FIRST_24_HOURS"] < threshold) & \
               (allResults["SYSTOLIC BLOOD PRESSURE_PERCENT_MISSING_FIRST_24_HOURS"] < threshold) & \
               (allResults["DIASTOLIC BLOOD PRESSURE_PERCENT_MISSING_FIRST_24_HOURS"] < threshold) &\
               (allResults["OXYGEN SATURATION_PERCENT_MISSING_FIRST_24_HOURS"] < threshold) &\
               (allResults["RESPIRATION RATE_PERCENT_MISSING_FIRST_24_HOURS"] < threshold)
               )
        print("Threshold:", threshold)
        print("Number of sample records:", ind.value_counts()[True])

    # Continuing off of last section, now we get
    #   representative stats for a well represented cohort of waveforms

    threshold = .8
    ind = ((allResults["HEART RATE_PERCENT_MISSING_FIRST_24_HOURS"] < threshold) & \
           (allResults["SYSTOLIC BLOOD PRESSURE_PERCENT_MISSING_FIRST_24_HOURS"] < threshold) & \
           (allResults["DIASTOLIC BLOOD PRESSURE_PERCENT_MISSING_FIRST_24_HOURS"] < threshold) &\
           (allResults["OXYGEN SATURATION_PERCENT_MISSING_FIRST_24_HOURS"] < threshold) &\
           (allResults["RESPIRATION RATE_PERCENT_MISSING_FIRST_24_HOURS"] < threshold)
           )


    plotRecord(allResults, ind, "Heart Rate", "heartrate")
    plotRecord(allResults, ind, "Respiration Rate", "respirationrate")
    plotRecord(allResults, ind, "Oxygen Saturation", "oxygen")
    plotRecord(allResults, ind, "Diastolic Blood Pressure", "diastolic")
    plotRecord(allResults, ind, "Systolic Blood Pressure", "systolic")


    print(allResults[ind][["HEART RATE_PERCENT_MISSING_FIRST_24_HOURS", \
                           "OXYGEN SATURATION_PERCENT_MISSING_FIRST_24_HOURS", \
                           "DIASTOLIC BLOOD PRESSURE_PERCENT_MISSING_FIRST_24_HOURS", \
                           "SYSTOLIC BLOOD PRESSURE_PERCENT_MISSING_FIRST_24_HOURS", \
                           "RESPIRATION RATE_PERCENT_MISSING_FIRST_24_HOURS"]].mean())

    # for non filtered
    threshold = 10000 # too lazy to fix this, so just did copy pasta and huge threshold
    ind = ((allResults["HEART RATE_PERCENT_MISSING_FIRST_24_HOURS"] < threshold) & \
           (allResults["SYSTOLIC BLOOD PRESSURE_PERCENT_MISSING_FIRST_24_HOURS"] < threshold) & \
           (allResults["DIASTOLIC BLOOD PRESSURE_PERCENT_MISSING_FIRST_24_HOURS"] < threshold) &\
           (allResults["OXYGEN SATURATION_PERCENT_MISSING_FIRST_24_HOURS"] < threshold)
           )
    plotRecord(allResults, ind, "Heart Rate", "filtered_heartrate")
    plotRecord(allResults, ind, "Respiration Rate", "filtered_respirationrate")
    plotRecord(allResults, ind, "Oxygen Saturation", "filtered_oxygen")
    plotRecord(allResults, ind, "Diastolic Blood Pressure", "filtered_diastolic")
    plotRecord(allResults, ind, "Systolic Blood Pressure", "filtered_systolic")
