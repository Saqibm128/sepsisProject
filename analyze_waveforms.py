from readWaveform.waveform_reader import WaveformReader
from readWaveform import waveformUtil
from readWaveform.waveformUtil import percentMissingFirstNHours, processSubjectID, plotRecord
from readWaveform.record_cleaner import Record_Cleaner
from multiprocessing import Process, Queue, Manager
from addict import Dict
from commonDB import read_sql
import pandas as pd
import numpy as np
import re
import pickle
import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from functools import reduce

from matplotlib.pyplot import plot, show, savefig, xlim, figure, \
                hold, ylim, legend, boxplot, setp, axes

__n_workers = 24
threshold = 1 # threshold is which section of the subjects to consider for a single variable
numericMapping = pd.read_csv("preprocessing/resources/numeric_waveform_to_variable_map.csv")
numericMapping["numeric"] = numericMapping["numeric"].str.upper()
numericMapping["high_level_var"] = numericMapping["high_level_var"].str.upper()
hoursAfterAdmit = [12, 24, 36, 48] #Hours after admission
# numericMapping = None
columnsToAnalyze = ["RESPIRATORY RATE", "HEART RATE", "DIASTOLIC BLOOD PRESSURE", "SYSTOLIC BLOOD PRESSURE", "OXYGEN SATURATION"]
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

def waveformStats(toRunQueue, toReturnQueue, numericMapping, columns=columnsToAnalyze):
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
    #
    #
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
    # classified = pd.DataFrame.from_csv("./data/rawdatafiles/classifiedAngusSepsis.csv")
    #
    #
    # # Further analysis of statistics to provide average data
    # # 22461 by 600 matrix is hard to parse, so we just use mean
    # numericStats = pd.DataFrame(index=allResults.columns, columns=["Average Missing", "Num Missing"])
    # twentyFourHourCoverage = []
    # for col in allResults.columns:
    #     if re.search(r"_PERCENT_MISSING", col) is not None:
    #         numericStats["Average Missing"][col] = allResults[allResults[col] < threshold][col].mean() #Mean of all columns that measure amount of info missing
    #         numericStats["Num Missing"][col] =  allResults[~(allResults[col] < threshold)].shape[0] #Counts number of waveforms that don't have the signal
    #     if re.search(r"_PERCENT_MISSING_FIRST_\d+_HOURS$", col) is not None:
    #         twentyFourHourCoverage.append(pd.DataFrame({"Average Missing": [allResults[col].mean()]}, index=[col]))
    # numericStats.to_csv("data/rawdatafiles/summarized_numeric_prelim_analysis.csv")
    # (pd.concat(twentyFourHourCoverage)).to_csv("data/rawdatafiles/numeric_24_hour_analysis.csv")
    #
    # toConcat = []
    # # Let's go through and find out how many samples remain depending on how we filter
    # for hour in hoursAfterAdmit:
    #     for threshold in [.2, .4, .6, .8, 1]:
    #         toAppend = pd.DataFrame(index=["HOUR: " + str(hour) + "THRESHOLD: " + str(threshold)])
    #         ind = ((allResults["HEART RATE_PERCENT_MISSING_FIRST_{}_HOURS".format(hour)] < threshold) & \
    #                (allResults["SYSTOLIC BLOOD PRESSURE_PERCENT_MISSING_FIRST_{}_HOURS".format(hour)] < threshold) & \
    #                (allResults["DIASTOLIC BLOOD PRESSURE_PERCENT_MISSING_FIRST_{}_HOURS".format(hour)] < threshold) &\
    #                (allResults["OXYGEN SATURATION_PERCENT_MISSING_FIRST_{}_HOURS".format(hour)] < threshold) &\
    #                (allResults["RESPIRATORY RATE_PERCENT_MISSING_FIRST_{}_HOURS".format(hour)] < threshold)
    #                )
    #         print("Hours after Admit:", hour, "Threshold:", threshold)
    #         print("Number of sample records:", ind.value_counts()[True]) #If all are false, value counts fails
    #         toAppend["NUM RECORDS"] = ind.value_counts()[True]
    #         Y = classified["angus"].loc[allResults[ind]["HADM_MAPPING"].apply(int)]
    #         sepsis = allResults[ind]["HADM_MAPPING"].apply(lambda x: int(x) in Y[Y == 1].index)
    #         print("Number of Sepsis Cases:", sepsis.value_counts()[True])
    #         toAppend["ANGUS"] = sepsis.value_counts()[True]
    #         nonSepsis = allResults[ind][~sepsis]
    #         sepsis = allResults[ind][sepsis]
    #         for col in columnsToAnalyze:
    #             colStr = col + " MEAN"
    #             toAppend["SEPSIS:" + col] = sepsis[colStr].mean()
    #             print("sepsis: ", col, " -> ", sepsis[colStr].mean())
    #             toAppend["NONSEPSIS:" + col] = nonSepsis[colStr].mean()
    #             print("nonsepsis: ", col, " -> ", nonSepsis[colStr].mean())
    #     print()
    #
    # # Continuing off of last section, now we get
    # #   representative stats for a well represented cohort of waveforms
    #
    hour = 24

    threshold = .8
    ind = ((allResults["HEART RATE_PERCENT_MISSING_FIRST_{}_HOURS".format(hour)] < threshold) & \
           (allResults["SYSTOLIC BLOOD PRESSURE_PERCENT_MISSING_FIRST_{}_HOURS".format(hour)] < threshold) & \
           (allResults["DIASTOLIC BLOOD PRESSURE_PERCENT_MISSING_FIRST_{}_HOURS".format(hour)] < threshold) &\
           (allResults["OXYGEN SATURATION_PERCENT_MISSING_FIRST_{}_HOURS".format(hour)] < threshold) &\
           (allResults["RESPIRATORY RATE_PERCENT_MISSING_FIRST_{}_HOURS".format(hour)] < threshold)
           )
    #
    #
    # plotRecord(allResults, ind, "Heart Rate", "heartrate", numHour = hour)
    # plotRecord(allResults, ind, "Respiratory Rate", "respirationrate", numHour = hour)
    # plotRecord(allResults, ind, "Oxygen Saturation", "oxygen", numHour = hour)
    # plotRecord(allResults, ind, "Diastolic Blood Pressure", "diastolic", numHour = hour)
    # plotRecord(allResults, ind, "Systolic Blood Pressure", "systolic", numHour = hour)
    #
    #
    # print(allResults[ind][["HEART RATE_PERCENT_MISSING_FIRST_{}_HOURS".format(hour), \
    #                        "OXYGEN SATURATION_PERCENT_MISSING_FIRST_{}_HOURS".format(hour), \
    #                        "DIASTOLIC BLOOD PRESSURE_PERCENT_MISSING_FIRST_{}_HOURS".format(hour), \
    #                        "SYSTOLIC BLOOD PRESSURE_PERCENT_MISSING_FIRST_{}_HOURS".format(hour), \
    #                        "RESPIRATORY RATE_PERCENT_MISSING_FIRST_{}_HOURS".format(hour)]].mean())
    #
    # # for non filtered
    # threshold = 1.1
    ind = ((allResults["HEART RATE_PERCENT_MISSING_FIRST_{}_HOURS".format(hour)] < threshold) & \
           (allResults["SYSTOLIC BLOOD PRESSURE_PERCENT_MISSING_FIRST_{}_HOURS".format(hour)] < threshold) & \
           (allResults["DIASTOLIC BLOOD PRESSURE_PERCENT_MISSING_FIRST_{}_HOURS".format(hour)] < threshold) &\
           (allResults["OXYGEN SATURATION_PERCENT_MISSING_FIRST_{}_HOURS".format(hour)] < threshold)   &\
           (allResults["RESPIRATORY RATE_PERCENT_MISSING_FIRST_{}_HOURS".format(hour)] < threshold)
           )
    #
    #
    #
    # plotRecord(allResults, ind, "Heart Rate", "filtered_heartrate", numHour = hour)
    # plotRecord(allResults, ind, "Respiratory Rate", "filtered_respirationrate", numHour = hour)
    # plotRecord(allResults, ind, "Oxygen Saturation", "filtered_oxygen", numHour = hour)
    # plotRecord(allResults, ind, "Diastolic Blood Pressure", "filtered_diastolic", numHour = hour)
    # plotRecord(allResults, ind, "Systolic Blood Pressure", "filtered_systolic", numHour = hour)
    #
    # Y = classified["angus"].loc[allResults[ind]["HADM_MAPPING"].apply(int)]
    # sepsis = allResults[ind]["HADM_MAPPING"].apply(lambda x: int(x) in Y[Y == 1].index)
    # print("Number of Sepsis Cases:", sepsis.value_counts()[True])
    # print("Number of NonSepsis Cases:", len(Y) - sepsis.value_counts()[True])

    # reader = WaveformReader(numericMapping=numericMapping)
    # reader.traverser.numeric = True
    #
    # toConcat = []
    # #deal with the duplicated hospital admission id issue and get some stats on how many are actually duplicated
    # duplicated = allResults[ind].loc[allResults[ind]["HADM_MAPPING"].duplicated()]["HADM_MAPPING"].unique()
    # for hadm in duplicated:
    #     duplicatedStats = pd.DataFrame(index=[hadm])
    #     duplicatedRecords = allResults[ind][allResults[ind]["HADM_MAPPING"] == hadm]
    #     duplicatedDataColumns = Dict()
    #     for column in columnsToAnalyze:
    #         duplicatedDataColumns[column] = []
    #     for duplicatedRecordID in duplicatedRecords.index:
    #         data, fields = reader.getRecord(duplicatedRecordID)
    #         firstHours = data[(data.index < pd.Timestamp(duplicatedRecords["ADMITTIME"].iloc[0]) + pd.Timedelta("{} hours".format(hour)))] #since these share hadmids, they share the same admittime
    #         for column in columnsToAnalyze:
    #             if column in firstHours.columns:
    #                 duplicatedDataColumns[column].append(firstHours[column])
    #     # go through each column and count amount of overlap
    #     for column in columnsToAnalyze:
    #         if len(duplicatedDataColumns[column]) != 0:
    #             toAnd = [~pd.isnull(data) for data in duplicatedDataColumns[column]]
    #             overlapping = reduce(lambda x, y: x & y, toAnd)
    #             duplicatedStats[column + "Number of Overlaps"] = overlapping.sum()
    #     toConcat.append(duplicatedStats)
    # fullDuplicatedStats = pd.concat(toConcat)
    # print(fullDuplicatedStats.mean())
    # rc = Record_Cleaner(columns=columnsToAnalyze, records = list(allResults[ind].index[0:10]), reader=reader)
    # dataDict, totalNumImputed = rc.cleanAll()
    # print(totalNumImputed.mean())
    # totalNumImputed.to_csv("data/rawdatafiles/recordsNumImputed.csv")
    # anyIndex = True #check to see all indices are same
    # anyColumn = True  #check to see all columns are same
    # noNulls = True #check to see nothing is null
    # expectedIndex = dataDict[149518]['data'].index
    # expectedColumn = dataDict[149518]['data'].columns
    #
    # for key in dataDict.keys():
    #     anyIndex = anyIndex & (expectedIndex.sort_values() == dataDict[key]['data'].index.sort_values()).all()
    #     anyColumn = anyColumn & (expectedColumn.sort_values() == dataDict[key]['data'].columns.sort_values()).all()
    #     noNulls = noNulls &  pd.isnull(dataDict[key]['data']).any().any()
    #     if (not anyColumn):
    #         print(key);
    #         break;
    # print("indices are same? :", anyIndex, "columns are same: ", anyColumn, "any nulls detected? :", noNulls)
    # with open('data/rawdatafiles/recordData.pickle', 'wb') as file:
    #     pickle.dump(dataDict, file) #store data to make things faster in future

    rc = Record_Cleaner(columns=columnsToAnalyze, records = list(allResults[ind].index[0:300]), reader=reader)
    dataDict, totalNumImputed = rc.cleanAll(shouldImpute=False)
    for variable in columnsToAnalyze:
        heartRate = pd.DataFrame()
        for key in dataDict.keys():
            heartRate[key] = dataDict[key]['data'][variable]
        plt.pcolormesh(pd.isnull(heartRate))
        plt.ylabel("Number of Seconds Since Admission")
        plt.title(variable)
        plt.savefig("data/rawdatafiles/" + variable + "Missing.png", dpi=300)
