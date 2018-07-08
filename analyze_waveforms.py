from readWaveform.waveform_reader import WaveformReader
from readWaveform import waveformUtil
from readWaveform.waveformUtil import percentMissingFirstNHours, processSubjectID
from readWaveform.record_cleaner import RecordCleaner
from readWaveform.record_segments import RecordSegmentsAnalyzer
from preprocessing.preprocessing import read_variable_ranges
from multiprocessing import Process, Queue, Manager
from addict import Dict
from commonDB import read_sql
import pandas as pd
import numpy as np
import re
import pickle
import os
import matplotlib as mpl
mpl.use('Agg')
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
columnsToAnalyze = ["RESPIRATORY RATE", "HEART RATE", "DIASTOLIC BLOOD PRESSURE", "SYSTOLIC BLOOD PRESSURE", "OXYGEN SATURATION"]
variable_ranges=read_variable_ranges("preprocessing/resources/variable_ranges.csv")
reader = WaveformReader(numericMapping=numericMapping, columnsToUse=columnsToAnalyze, variable_ranges=variable_ranges)
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
    #
    #
    manager = Manager()
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
    classified = pd.DataFrame.from_csv("./data/rawdatafiles/classifiedAngusSepsis.csv")
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
    # hour = 24
    #
    # threshold = .8
    # ind = ((allResults["HEART RATE_PERCENT_MISSING_FIRST_{}_HOURS".format(hour)] < threshold) & \
    #        (allResults["SYSTOLIC BLOOD PRESSURE_PERCENT_MISSING_FIRST_{}_HOURS".format(hour)] < threshold) & \
    #        (allResults["DIASTOLIC BLOOD PRESSURE_PERCENT_MISSING_FIRST_{}_HOURS".format(hour)] < threshold) &\
    #        (allResults["OXYGEN SATURATION_PERCENT_MISSING_FIRST_{}_HOURS".format(hour)] < threshold) &\
    #        (allResults["RESPIRATORY RATE_PERCENT_MISSING_FIRST_{}_HOURS".format(hour)] < threshold)
    #        )
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
    # ind = ((allResults["HEART RATE_PERCENT_MISSING_FIRST_{}_HOURS".format(hour)] < threshold) & \
    #        (allResults["SYSTOLIC BLOOD PRESSURE_PERCENT_MISSING_FIRST_{}_HOURS".format(hour)] < threshold) & \
    #        (allResults["DIASTOLIC BLOOD PRESSURE_PERCENT_MISSING_FIRST_{}_HOURS".format(hour)] < threshold) &\
    #        (allResults["OXYGEN SATURATION_PERCENT_MISSING_FIRST_{}_HOURS".format(hour)] < threshold)   &\
    #        (allResults["RESPIRATORY RATE_PERCENT_MISSING_FIRST_{}_HOURS".format(hour)] < threshold)
    #        )
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
    #     # go through each column and count amount of overlap between repeated hadmids
    #     for column in columnsToAnalyze:
    #         if len(duplicatedDataColumns[column]) != 0:
    #             toAnd = [~pd.isnull(data) for data in duplicatedDataColumns[column]]
    #             overlapping = reduce(lambda x, y: x & y, toAnd)
    #             duplicatedStats[column + "Number of Overlaps"] = overlapping.sum()
    #     toConcat.append(duplicatedStats)
    # fullDuplicatedStats = pd.concat(toConcat)
    # print(fullDuplicatedStats.mean())
    # rc = RecordCleaner(columns=columnsToAnalyze, records = list(allResults[ind].index[0:10]), reader=reader)
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
    #
    # rc = RecordCleaner(columns=columnsToAnalyze, records = list(allResults[ind].index[0:300]), reader=reader)
    # dataDict, totalNumImputed = rc.cleanAll(shouldImpute=False)
    # for variable in columnsToAnalyze:
    #     heartRate = pd.DataFrame()
    #     for key in dataDict.keys():
    #         heartRate[key] = dataDict[key]['data'][variable]
    #     plt.pcolormesh(pd.isnull(heartRate))
    #     plt.ylabel("Number of Seconds Since Admission")
    #     plt.title(variable)
    #     plt.savefig("data/rawdatafiles/" + variable + "Missing.png", dpi=300)
    rsa = RecordSegmentsAnalyzer(reader=reader, variable_ranges=variable_ranges)
    firstSeg = rsa.analyzeAll(hadmids=allResults[allResults['HADM_MAPPING'] != 'NOT FOUND']['HADM_MAPPING'].unique())
    with open('data/rawdatafiles/recordSegments.pickle', 'wb') as file:
        pickle.dump(firstSeg, file) #store data to make things faster in future

    with open('data/rawdatafiles/recordSegments.pickle', 'rb') as file:
        firstSeg = pickle.load(file)
    #
    # print("Number of HADM_IDS:", len(firstSeg))
    #
    # firstSegLengths = []
    # for hadmid in firstSeg.keys():
    #     if True in firstSeg[hadmid].isSeg:
    #         firstSegLengths.append(firstSeg[hadmid].isSeg[True].iloc[0])
    #     else:
    #         firstSegLengths.append(0) #length of nothing is false
    # plt.title("First Filtered Segment of Data in Waveform (Most Missing = 15)")
    # plt.hist(firstSegLengths, range=(50,4000))
    # plt.xlabel("Size of First Segment")
    # plt.ylabel("Number of First Segments")
    # plt.savefig("data/rawdatafiles/firstFilteredSegmentLengths.png", dpi=300)
    # plt.hist(firstSegLengths,)
    # plt.title("First Filtered Segment of Data in Record (Most Missing = 15)")
    # plt.xlabel("Size of First Segment")
    # plt.ylabel("Number of First Segments")
    # plt.savefig("data/rawdatafiles/firstSegmentLengths.png", dpi=300)
    # plt.gcf().clear()
    #
    # print("Median length of values for first segment of data for HADMID: ", np.median(np.array(firstSegLengths)))
    #
    # segmentLengths = []
    # for hadmid in firstSeg:
    #     if True in firstSeg[hadmid].isSeg:
    #         for i in range(len(firstSeg[hadmid].isSeg.loc[True])):
    #             segmentLengths.append(firstSeg[hadmid].isSeg.loc[True].iloc[i])
    #
    # plt.hist(segmentLengths)
    # plt.title("Lengths of All Segments of Data in Record (Most Missing = 15)")
    # plt.xlabel("Size of Segment")
    # plt.ylabel("Number of Segments")
    # plt.savefig("data/rawdatafiles/allSegmentLengths.png", dpi=300)
    # plt.gcf().clear()
    # print("Median length of segments for record for HADMID: ", np.median(np.array(segmentLengths)))
    #
    # afterAdmittime = []
    # for hadmid in firstSeg:
    #     if True in firstSeg[hadmid].isSeg:
    #         if 1 in firstSeg[hadmid].isSeg[True]:  #is the first block a segment?
    #             afterAdmittime.append(firstSeg[hadmid].admittimeDiff) #storing the admittimeDiff here
    #         else:
    #             afterAdmittime.append(firstSeg[hadmid].admittimeDiff + pd.Timedelta("{} seconds".format(firstSeg[hadmid].isSeg[False][1]))) # add the time from interceding first block of False block
    #
    # plt.hist(pd.Series(afterAdmittime) / pd.Timedelta("1 hours"))
    # plt.title("Gap Size Before First Segments of Data in Record (Most Missing = 15)")
    # plt.xlabel("Size of Gap before All Data (1 hour)")
    # plt.ylabel("Number of Segments")
    # plt.savefig("data/rawdatafiles/admittimeDiffWaveform.png", dpi=300)
    # plt.gcf().clear()
    # print("Median gap for record for HADMID before data starts (measured from admittime): ", np.median(np.array(afterAdmittime)))
    # missingSegments = []
    # for hadmid in firstSeg:
    #     if False in firstSeg[hadmid].isSeg:
    #         for i in range(len(firstSeg[hadmid].isSeg.loc[False])):
    #             missingSegments.append(firstSeg[hadmid].isSeg.loc[False].iloc[i])
    #
    # plt.hist(missingSegments)
    # plt.title("Missing Gaps of Data in Record (Most Missing = 15)")
    # plt.xlabel("Size of Missing Gap")
    # plt.ylabel("Number of Missing Gaps")
    # plt.savefig("data/rawdatafiles/missingGapLengths.png", dpi=300)
    # plt.gcf().clear()
    # print("Median length of missing Gaps for record for HADMID: ", np.median(np.array(missingSegments)))
    #
    # numberOfSegments = []
    # for hadmid in firstSeg:
    #     numberOfSegments.append(firstSeg[hadmid].isSeg.shape[0])
    # print("Median number of segments and gaps: ", np.median(np.array(numberOfSegments)))
    # plt.hist(numberOfSegments)
    # plt.title("Number Of Segments and Gaps for each HADM (Most Missing = 15)")
    # plt.xlabel("Number of Segments")
    # plt.ylabel("Number of HADMs")
    # plt.savefig("data/rawdatafiles/totalNumSegmentLengths.png", dpi=300)
    # plt.gcf().clear()
    #
    #
    # numberOfDataSegments = []
    # for hadmid in firstSeg:
    #     if True in firstSeg[hadmid].isSeg:
    #         numberOfDataSegments.append(firstSeg[hadmid].isSeg[True].shape[0])
    # print("Median number of data-filled segments: ", np.median(np.array(numberOfDataSegments)))
    # plt.hist(numberOfDataSegments)
    # plt.title("Number Of Segments for each HADM (Most Missing = 15)")
    # plt.xlabel("Number of Segments")
    # plt.ylabel("Number of HADMs")
    # plt.savefig("data/rawdatafiles/totalFilledSegmentLengths.png", dpi=300)
    # plt.gcf().clear()
    #
    #get data into usable form for first seg analysis

    for hadmid in firstSeg.keys():
        if True in firstSeg[hadmid].isSeg:
            if 1 in firstSeg[hadmid].isSeg[True]:  #is the first block a segment?
                firstSeg[hadmid].firstSegInd = 0 #storing the admittimeDiff here
            else:
                firstSeg[hadmid].firstSegInd = firstSeg[hadmid].isSeg[False][1] # add the time from interceding first block of False block
            firstSeg[hadmid].firstSegLength = firstSeg[hadmid].isSeg[True].iloc[0]

    def extractFirstSegment(hadmid):
        data, matching = reader.getRecordByHADMID(hadmid)
        if columnsToAnalyze is not None:
            for col in columnsToAnalyze:
                data.loc[data[col] < variable_ranges["OUTLIER_LOW"][col], col] = np.nan
                data.loc[data[col] > variable_ranges["OUTLIER_HIGH"][col], col] = np.nan
        data = data.fillna(method="ffill")
        data = data.fillna(method="bfill")
        return data.iloc[firstSeg[hadmid].firstSegInd:firstSeg[hadmid].firstSegInd + firstSeg[hadmid].firstSegLength]

    def helperExtract(toRunQ, toReturnQ):
        for hadmid in iter(toRunQ.get, None):
            print(toRunQ.qsize())
            data = extractFirstSegment(hadmid)
            toReturnQ.put((hadmid, data))
    firstSegFiltered = Dict() #only records meeting 6 hour criterion
    toRunQ = manager.Queue()
    toReturnQ = manager.Queue()
    [toRunQ.put(hadmid) for hadmid in firstSeg.keys() \
        if True in firstSeg[hadmid].isSeg and (firstSeg[hadmid].isSeg[True].iloc[0] > 6 * 60) \
        and (firstSeg[hadmid].admittimeDiff < pd.Timedelta("72 hours"))\
        and ((firstSeg[hadmid].admittime - firstSeg[hadmid].dob) > pd.Timedelta("{} days".format(365.25 * 18)))]
    [toRunQ.put(None) for i in range(24)]
    processes = [Process(target=helperExtract, args=[toRunQ, toReturnQ]) for i in range(24)]
    [process.start() for process in processes]
    [process.join() for process in processes]
    while not toReturnQ.empty():
        hadmid, recordData = toReturnQ.get()
        firstSegFiltered[hadmid].firstSegRec = recordData
        firstSegFiltered[hadmid].admittimeDiff = firstSeg[hadmid].admittimeDiff

    with open('data/rawdatafiles/recordFirstSegments.pickle', 'wb') as file:
        pickle.dump(firstSegFiltered, file) #store data to make things faster in future

    with open('data/rawdatafiles/recordFirstSegments.pickle', 'rb') as file:
        firstSegFiltered = pickle.load(file)
    Y = classified['angus'].loc[[int(hadmid) for hadmid in list(firstSegFiltered.keys())]]

    allRecHADMIDS = pd.DataFrame([list(firstSegFiltered.keys()), \
                                        [firstSegFiltered[hadmid].admittimeDiff for hadmid in list(firstSegFiltered.keys())], Y.tolist()]).T
    allRecHADMIDS.columns = ["hadmid", "admittimeDiff", "Y"]
    allRecHADMIDS = allRecHADMIDS.set_index(["hadmid"])
    # print(allRecHADMIDS.shape)
    allRecHADMIDS.to_csv("data/rawdatafiles/byHadmIDNumRec/allRecHADMIDS.csv")
    for hadmid in firstSegFiltered.keys():
        firstSixHourRec = firstSegFiltered[hadmid].firstSegRec
        if not os.path.isdir("data/rawdatafiles/byHadmIDNumRec/{}".format(hadmid)):
            os.mkdir("data/rawdatafiles/byHadmIDNumRec/{}".format(hadmid))
        firstSixHourRec.to_csv("data/rawdatafiles/byHadmIDNumRec/{}/sixHourSegment.csv".format(hadmid))


    def extractSignal(sigName, recordsDict, sigKey='firstSegRec', max_allowed=None): #cringy code...
        '''
        @param sigName the signal (i.e. "HEART RATE") to extract
        @param recordsDict holding all record data with hadmid as key
        @param sigKey which key in dict has record
        @param max_allowed how much time to cut off at (i.e. pd.Timedelta("6 hours"))
        @return a new dataframe with hadmids as columns, of only on signal name
        '''
        signals = pd.DataFrame()
        for hadmid in recordsDict.keys():
            signal = recordsDict[hadmid][sigKey].loc[:,sigName]
            signal.index = signal.index - signal.index[0]
            if max_allowed is not None:
                signal = signal[signal.index < max_allowed]
            signals[int(hadmid)] =  signal
        return signals

    for sigName in columnsToAnalyze:
        signals = extractSignal(sigName, firstSegFiltered, max_allowed = pd.Timedelta("6 hours"))
        sepsisSignals = signals[signals.columns[Y==1]].mean(axis=1)
        sepsisSignalsStd = signals[signals.columns[Y==1]].std(axis=1)
        nonSepsisSignals = signals[signals.columns[Y==0]].mean(axis=1)
        nonSepsisSignalsStd = signals[signals.columns[Y==0]].std(axis=1)
        x = sepsisSignals.index
        plt.plot(sepsisSignals.index/pd.Timedelta("1 hours"), sepsisSignals.values, color="Blue")
        plt.plot(nonSepsisSignals.index/pd.Timedelta("1 hours"), nonSepsisSignals.values, color="Orange")
        plt.title("Sepsis vs Nonsepsis")
        plt.ylabel(sigName)
        plt.xlabel("Time Since Start of First Segment (Hour)")
        plt.legend(["Sepsis Cohort", "Non-Sepsis Cohort"])
        plt.fill_between(sepsisSignals.index/pd.Timedelta("1 hours"), (sepsisSignals + sepsisSignalsStd).values, (sepsisSignals - sepsisSignalsStd).values, color='Blue', alpha=.5)
        plt.fill_between(nonSepsisSignals.index/pd.Timedelta("1 hours"), (nonSepsisSignals + nonSepsisSignalsStd).values, (nonSepsisSignals - nonSepsisSignalsStd).values, color='Orange', alpha=.5)
        plt.savefig("data/rawdatafiles/sepsisVSnonsepsis{}.png".format(sigName), dpi=300)
        plt.gcf().clear()
        print("First 6 hours segment, average mean of sepsis " + sigName, sepsisSignals.mean())
        print("First 6 hours segment, average mean of nonsepsis " + sigName, nonSepsisSignals.mean())
        print("First 6 hours segment, average std of sepsis " + sigName, sepsisSignalsStd.mean())
        print("First 6 hours segment, average std of nonsepsis " + sigName, nonSepsisSignalsStd.mean())
    #
    #
    # timesAfterAdmit = pd.Series()
    # for hadm in firstSegFiltered.keys():
    #     timesAfterAdmit[hadm] = firstSeg[hadm].admittimeDiff
    # print("Average time between admission and first segment", timesAfterAdmit.mean())
    # print("Median time between admission and first segment", timesAfterAdmit.median())
    #
    # plt.hist(timesAfterAdmit.apply(lambda time: time / pd.Timedelta("1 hours"))) #unit of 1 hour
    # plt.title("Time Between Admission and First Segment")
    # plt.xlabel("Time (Hours)")
    # plt.ylabel("Number of Hospital Admissions")
    # plt.savefig("data/rawdatafiles/admitvsfirstSeg.png", dpi=300)
    # plt.gcf().clear()
    #
    # plt.hist(timesAfterAdmit.apply(lambda time: time / pd.Timedelta("1 hours")), range=(-20, 72)) #unit of 1 hour
    # plt.title("Filtered Time Between Admission and First Segment")
    # plt.xlabel("Time (Hours)")
    # plt.ylabel("Number of Hospital Admissions")
    # plt.savefig("data/rawdatafiles/filteredAdmitvsfirstSeg.png", dpi=300)
