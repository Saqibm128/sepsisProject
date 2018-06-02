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
hoursAfterAdmit = [12, 24, 36, 48] #Hours after admission
# numericMapping = None

def processSubjectID(subject_id, numHours = hoursAfterAdmit):
    reader = WaveformReader(numericMapping=numericMapping)
    reader.traverser.numeric = True
    toReturn = [pd.DataFrame()]
    fileToHADMID = reader.traverser.matchWithHADMID(subject_id)
    multiRecords = reader.traverser.getMultiRecordFiles(subject_id)
    if len(multiRecords) == 0:
        return pd.DataFrame() #if there are not numeric record files, we just skip
    for multiRecord in multiRecords:
        singRecStats = pd.DataFrame(index=[multiRecord])
        singRecStats["HADM_MAPPING"] = fileToHADMID[multiRecord].hadmid # Maps file to hadmid
        singRecStats["SUBJECT_ID"] = subject_id
        if fileToHADMID[multiRecord].hadmid != "NOT FOUND": #could not be matched for some reason
            admittime = fileToHADMID[multiRecord].admittime
        try:
            data, fields = reader.getRecord(multiRecord, subject_id=subject_id)
            for sig_name in data.columns:
                if (~(data[sig_name].apply(np.isnan))).all():
                    singRecStats[sig_name + "_PERCENT_MISSING"] = 0
                else:
                    singRecStats[sig_name + "_PERCENT_MISSING"] = pd.isnull(data[sig_name]).value_counts()[True] / len(data[sig_name])
                #For each signal, find the percentage that is filled in the first 24 hours after admission
                if fileToHADMID[multiRecord].hadmid != "NOT FOUND":
                    singRecStats = percentMissing(data, admittime, numHours, sig_name, singRecStats)
            singRecStats["length"] = len(data)
        except:
            singRecStats["comment"] = "Could not getRecord"
        toReturn.append(singRecStats)
    return pd.concat(toReturn)

def percentMissing(data, admittime, numHours, sig_name, singRecStats):
    '''
    Represent the amount of data missing for the number of hours noted, then writes to singRecStats (single record stats)
    :param data to analyzie
    :param admittime the admission time of a specific data record
    :param sig_name specific signal to consider
    :param numHours the exact number of hours after the admission to consider. if it is an array of number of hours, will iterate through that
    :param singRecStats the dataframe to write to
    '''
    if not isinstance(numHours, list):
        numHours = [numHours]
    for numHour in numHours:
        admittime = pd.Timestamp(admittime)
        firstDay = data.iloc[(data.index < admittime + pd.Timedelta(str(numHour) + " hours")) & (data.index > admittime)]
        # print(pd.isnull(firstDay[sig_name]).value_counts()[False])
        if (pd.isnull(firstDay[sig_name])).all(): #edge case where False is not a key
            singRecStats[sig_name + "_PERCENT_MISSING_FIRST_" + str(numHour) + "_HOURS"] = 1
        else:
            singRecStats[sig_name + "_PERCENT_MISSING_FIRST_" + str(numHour) + "_HOURS"] =  1 - pd.isnull(firstDay[sig_name]).value_counts()[False] / (60*numHour) # 60 minutes in an hour, 24 hours in a day
    return singRecStats



def helperWaveformRunner(toRunQueue, toReturnQueue):
    '''
    Uses queues to analyze waveforms for key prelim stats
    '''
    for subject_id in iter(toRunQueue.get, None):
        print(toRunQueue.qsize())
        toReturn = processSubjectID(subject_id)
        toReturnQueue.put(toReturn)

def plotRecord(allResults, ind, name, filename, numHour = 24):
    plt.hist(allResults[ind][name.upper() + "_PERCENT_MISSING_FIRST_" + str(numHour) + "_HOURS"], bins=20, rwidth=.5)
    plt.xlabel("Percent Missing in First " + str(numHour) + " Hours")
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
    allResults = pd.read_csv("data/rawdatafiles/numeric_prelim_analysis.csv", index_col=0)

    # Further analysis of statistics to provide average data
    # 22461 by 600 matrix is hard to parse, so we just use mean
    numericStats = pd.DataFrame(index=allResults.columns, columns=["Average Missing", "Num Missing"])
    twentyFourHourCoverage = []
    for col in allResults.columns:
        if re.search(r"_PERCENT_MISSING", col) is not None:
            numericStats["Average Missing"][col] = allResults[allResults[col] < threshold][col].mean() #Mean of all columns that measure amount of info missing
            numericStats["Num Missing"][col] =  allResults[~(allResults[col] < threshold)].shape[0] #Counts number of waveforms that don't have the signal
        if re.search(r"_PERCENT_MISSING_FIRST_\d+_HOURS$", col) is not None:
            twentyFourHourCoverage.append(pd.DataFrame({"Average Missing": [allResults[col].mean()]}, index=[col]))
    numericStats.to_csv("data/rawdatafiles/summarized_numeric_prelim_analysis.csv")
    (pd.concat(twentyFourHourCoverage)).to_csv("data/rawdatafiles/numeric_24_hour_analysis.csv")

    # Let's go through and find out how many samples remain depending on how we filter
    for hour in hoursAfterAdmit:
        for threshold in [.2, .4, .6, .8, 1]:
            ind = ((allResults["HEART RATE_PERCENT_MISSING_FIRST_{}_HOURS".format(hour)] < threshold) & \
                   (allResults["SYSTOLIC BLOOD PRESSURE_PERCENT_MISSING_FIRST_{}_HOURS".format(hour)] < threshold) & \
                   (allResults["DIASTOLIC BLOOD PRESSURE_PERCENT_MISSING_FIRST_{}_HOURS".format(hour)] < threshold) &\
                   (allResults["OXYGEN SATURATION_PERCENT_MISSING_FIRST_{}_HOURS".format(hour)] < threshold) &\
                   (allResults["RESPIRATORY RATE_PERCENT_MISSING_FIRST_{}_HOURS".format(hour)] < threshold)
                   )
            print("Threshold:", threshold, "Hours after Admit:", hour)
            print("Number of sample records:", len(ind) - ind.value_counts()[False]) #If all are false, value counts fails
        print()

    # Continuing off of last section, now we get
    #   representative stats for a well represented cohort of waveforms

    hour = 24

    threshold = 1000
    ind = ((allResults["HEART RATE_PERCENT_MISSING_FIRST_{}_HOURS".format(hour)] < threshold) & \
           (allResults["SYSTOLIC BLOOD PRESSURE_PERCENT_MISSING_FIRST_{}_HOURS".format(hour)] < threshold) & \
           (allResults["DIASTOLIC BLOOD PRESSURE_PERCENT_MISSING_FIRST_{}_HOURS".format(hour)] < threshold) &\
           (allResults["OXYGEN SATURATION_PERCENT_MISSING_FIRST_{}_HOURS".format(hour)] < threshold) &\
           (allResults["RESPIRATORY RATE_PERCENT_MISSING_FIRST_{}_HOURS".format(hour)] < threshold)
           )


    plotRecord(allResults, ind, "Heart Rate", "heartrate", numHour = hour)
    plotRecord(allResults, ind, "Respiratory Rate", "respirationrate", numHour = hour)
    plotRecord(allResults, ind, "Oxygen Saturation", "oxygen", numHour = hour)
    plotRecord(allResults, ind, "Diastolic Blood Pressure", "diastolic", numHour = hour)
    plotRecord(allResults, ind, "Systolic Blood Pressure", "systolic", numHour = hour)


    print(allResults[ind][["HEART RATE_PERCENT_MISSING_FIRST_{}_HOURS".format(hour), \
                           "OXYGEN SATURATION_PERCENT_MISSING_FIRST_{}_HOURS".format(hour), \
                           "DIASTOLIC BLOOD PRESSURE_PERCENT_MISSING_FIRST_{}_HOURS".format(hour), \
                           "SYSTOLIC BLOOD PRESSURE_PERCENT_MISSING_FIRST_{}_HOURS".format(hour), \
                           "RESPIRATORY RATE_PERCENT_MISSING_FIRST_{}_HOURS".format(hour)]].mean())

    # for non filtered
    threshold = .8
    ind = ((allResults["HEART RATE_PERCENT_MISSING_FIRST_{}_HOURS".format(hour)] < threshold) & \
           (allResults["SYSTOLIC BLOOD PRESSURE_PERCENT_MISSING_FIRST_{}_HOURS".format(hour)] < threshold) & \
           (allResults["DIASTOLIC BLOOD PRESSURE_PERCENT_MISSING_FIRST_{}_HOURS".format(hour)] < threshold) &\
           (allResults["OXYGEN SATURATION_PERCENT_MISSING_FIRST_{}_HOURS".format(hour)] < threshold) &\
           (allResults["RESPIRATORY RATE_PERCENT_MISSING_FIRST_{}_HOURS".format(hour)] < threshold)
           )
    plotRecord(allResults, ind, "Heart Rate", "filtered_heartrate", numHour = hour)
    plotRecord(allResults, ind, "Respiratory Rate", "filtered_respirationrate", numHour = hour)
    plotRecord(allResults, ind, "Oxygen Saturation", "filtered_oxygen", numHour = hour)
    plotRecord(allResults, ind, "Diastolic Blood Pressure", "filtered_diastolic", numHour = hour)
    plotRecord(allResults, ind, "Systolic Blood Pressure", "filtered_systolic", numHour = hour)
