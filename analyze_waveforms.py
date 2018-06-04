from readWaveform.waveform_reader import WaveformReader
from readWaveform import waveformUtil
from readWaveform.waveformUtil import percentMissing, processSubjectID, plotRecord
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
reader = WaveformReader(numericMapping=numericMapping)
reader.traverser.numeric = True


def helperWaveformRunner(toRunQueue, toReturnQueue, numericMapping):
    '''
    Uses queues to analyze waveforms for key prelim stats
    '''
    for subject_id in iter(toRunQueue.get, None):
        print(toRunQueue.qsize())
        toReturn = processSubjectID(subject_id, numHours = hoursAfterAdmit, numericMapping=numericMapping)
        toReturnQueue.put(toReturn)


def waveformStats(toRunQueue, toReturnQueue, numericMapping, columns):
    for recordName in iter(toRunQueue.get, None):
        print(toRunQueue.qsize())
        toReturn = pd.DataFrame(column=columns, index=[recordName])
        numericRec = reader.getRecord(recordName)
        for column in columns:
            if column in toReturn.columns:
                toReturn[column] = numericRec[column].mean()
        toReturnQueue.put(toReturn)

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


    # manager = Manager()
    # inQueue = manager.Queue()
    # outQueue = manager.Queue()
    # subjects = reader.traverser.getSubjects()
    # [inQueue.put(subject) for subject in subjects]
    # [inQueue.put(None) for i in range(__n_workers)]
    # processes = [Process(target=helperWaveformRunner, args=(inQueue, outQueue, numericMapping)) for i in range(__n_workers)]
    # [process.start() for process in processes]
    # [process.join() for process in processes]
    # allResults = []
    # while not outQueue.empty():
    #     allResults.append(outQueue.get())
    # allResults = pd.concat(allResults)
    # allResults = allResults.fillna(1) #missing 100 % if we didnt find the waveform
    # allResults.to_csv("data/rawdatafiles/numeric_prelim_analysis.csv")
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
                   (allResults["OXYGEN SATURATION_PERCENT_MISSING_FIRST_{}_HOURS".format(hour)] < threshold)
                   # &\
                   # (allResults["RESPIRATORY RATE_PERCENT_MISSING_FIRST_{}_HOURS".format(hour)] < threshold)
                   )
            print("Threshold:", threshold, "Hours after Admit:", hour)
            print("Number of sample records:", len(ind) - ind.value_counts()[False]) #If all are false, value counts fails
        print()

    # Continuing off of last section, now we get
    #   representative stats for a well represented cohort of waveforms

    hour = 24

    threshold = .8
    ind = ((allResults["HEART RATE_PERCENT_MISSING_FIRST_{}_HOURS".format(hour)] < threshold) & \
           (allResults["SYSTOLIC BLOOD PRESSURE_PERCENT_MISSING_FIRST_{}_HOURS".format(hour)] < threshold) & \
           (allResults["DIASTOLIC BLOOD PRESSURE_PERCENT_MISSING_FIRST_{}_HOURS".format(hour)] < threshold) &\
           (allResults["OXYGEN SATURATION_PERCENT_MISSING_FIRST_{}_HOURS".format(hour)] < threshold)
           # &\
           # (allResults["RESPIRATORY RATE_PERCENT_MISSING_FIRST_{}_HOURS".format(hour)] < threshold)
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
           (allResults["OXYGEN SATURATION_PERCENT_MISSING_FIRST_{}_HOURS".format(hour)] < threshold)
           )
    #  &\
    # (allResults["RESPIRATORY RATE_PERCENT_MISSING_FIRST_{}_HOURS".format(hour)] < threshold)


    plotRecord(allResults, ind, "Heart Rate", "filtered_heartrate", numHour = hour)
    plotRecord(allResults, ind, "Respiratory Rate", "filtered_respirationrate", numHour = hour)
    plotRecord(allResults, ind, "Oxygen Saturation", "filtered_oxygen", numHour = hour)
    plotRecord(allResults, ind, "Diastolic Blood Pressure", "filtered_diastolic", numHour = hour)
    plotRecord(allResults, ind, "Systolic Blood Pressure", "filtered_systolic", numHour = hour)

    classified = pd.DataFrame.from_csv("./data/rawdatafiles/classifiedAngusSepsis.csv")
    Y = classified["angus"].loc[allResults[ind]["HADM_MAPPING"].apply(int)]
    sepsis = allResults[ind]["HADM_MAPPING"].apply(lambda x: int(x) in Y[Y == 1].index)
    print("Number of Sepsis Cases:", len(sepsis))
    print("Number of NonSepsis Cases:", len(Y) - len(sepsis))
