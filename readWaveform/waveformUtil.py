#This file is responsible for taking in waveform data from the MIMIC3 wfdb
#   and providing key utilities.
# @author Mohammed Saqib
import wfdb
import urllib.request as request
import pandas as pd
import numpy as np
import time
import datetime
import commonDB
import math
from readWaveform.waveform_reader import WaveformReader
from multiprocessing import Process
from multiprocessing import Queue
from multiprocessing import Manager
from addict import Dict
from commonDB import read_sql
import re
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.switch_backend('agg')

from matplotlib.pyplot import plot, show, savefig, xlim, figure, \
                hold, ylim
def preliminaryCompareTimesICU(num_days=1):
    '''
    This method is another sanity check method similar to preliminaryCompareTimes but uses ICUSTAYS instead of ADMISSIONS
    '''
    icustays = commonDB.read_sql("SELECT * FROM ICUSTAYS", uppercase=False)
    (subjects, times) = listAllMatchedWFSubjects()
    matchedWF = pd.DataFrame({"subject_id":subjects, "wfStartTime": times}) #times is when the start of recording for each wf
    matchedWF["subject_id"] = matchedWF["subject_id"].astype(np.number)
    matchedWF["wfStartTime"] = matchedWF["wfStartTime"].apply(preliminaryCompareTimesHelper) #convert weird time format into useful data
    admWfMerge = pd.merge(matchedWF, icustays, left_on="subject_id", right_on="subject_id")
    admWfMerge["timeDiff"] = admWfMerge["wfStartTime"].subtract(admWfMerge["intime"])
    admWfMerge = admWfMerge[(admWfMerge["timeDiff"] > pd.Timedelta(0))]
    admWfMerge = admWfMerge[(admWfMerge["timeDiff"] < pd.Timedelta(str(num_days) + " days"))] #don't consider waveform older than 15 days
    admWfMerge["rawTimeDiff"] = admWfMerge["timeDiff"].astype(np.int64)
    print(pd.Timedelta(admWfMerge["timeDiff"].astype(np.int64).mean()))
    return admWfMerge

def preliminaryCompareTimes(num_days=1):
    '''
    This method is a sanity check method to compare the admittime of patients to the waveform start time.
    It calculates the difference and returns the hospital admissions id that correspond to the waveform start time
    unlike compareAdmitToWF, this DOES NOT PULL WAVEFORM data
    as a result, it should be much faster, but means we miss a lot of possible stats

    TLDR: This function is simpler, faster, less comprehensive version of compareAdmitToWf
    :param num_days the max number of days the differnece between the admittime and
            the wfStartTime has to be to be considered allowed, default= 15
    :return a dataframe to characterize results
    '''
    admissions = commonDB.read_sql("SELECT subject_id, admittime FROM admissions", uppercase=False)
    (subjects, times) = listAllMatchedWFSubjects()
    matchedWF = pd.DataFrame({"subject_id":subjects, "wfStartTime": times}) #times is when the start of recording for each wf
    matchedWF["subject_id"] = matchedWF["subject_id"].astype(np.number)
    matchedWF["wfStartTime"] = matchedWF["wfStartTime"].apply(preliminaryCompareTimesHelper) #convert weird time format into useful data
    print(admissions.columns, matchedWF.columns)
    admWfMerge = pd.merge(matchedWF, admissions, left_on="subject_id", right_on="subject_id")
    admWfMerge["timeDiff"] = admWfMerge["wfStartTime"].subtract(admWfMerge["admittime"])
    admWfMerge = admWfMerge[(admWfMerge["timeDiff"] > pd.Timedelta(0))]
    admWfMerge = admWfMerge[(admWfMerge["timeDiff"] < pd.Timedelta(str(num_days) + " days"))] #don't consider waveform older than 15 days
    admWfMerge["rawTimeDiff"] = admWfMerge["timeDiff"].astype(np.int64)
    print(admWfMerge.shape)
    print(pd.Timedelta(admWfMerge["timeDiff"].astype(np.int64).mean()))
    return admWfMerge


def preliminaryCompareTimesHelper(time):
    '''
    A helper function that turns strings of format yyyy-mm-dd-hh-mm
    into a pandas Timestamp
    :param time a string of format yyyy-mm-dd-hh-mm
    :return pd.Timestamp equivalent
    '''
    year = int(time[0:4])
    month = int(time[5:7])
    day = int(time[8:10])
    hour = int(time[11:13])
    minute = int(time[14:16])
    return pd.Timestamp(year=year, month=month, day=day, hour=hour, minute=minute)
def listAllMatchedRecSubjects(url = "https://physionet.org/physiobank/database/mimic3wdb/matched/RECORDS-numerics"):
    '''
    Numeric records in mimic3wdb are averages per second (1hz) of waveform data
    method here returns the subject_id of the matched records for numeric records
        uses specified format given in documentions
    :param url (already set) which sets where to read the subject Ids
    :return a list of lists (subject_id, start time in string format) for matched numeric records in the waveform data
    '''
    resp = request.urlopen(url)
    data = resp.read()
    recordSubjects = [];
    recordTimes = [];
    for line in data.splitlines():
        line = str(line)
        sublines = line.split('/')
        subjectID = sublines[1][1:]
        time = sublines[2].replace("p" + subjectID, "")
        recordSubjects.append(subjectID)
        recordTimes.append(time[1:-1])
    return [recordSubjects, recordTimes]
def compareAdmitToWF():
    '''
    This method runs through and pulls waveforms out of the mimic3wdb and checks for
    1. number of waveforms found
    2. the proportion of all waveforms that was missing
    3. the total length of the waveforms
    4. if the first 24 hours after an admission are covered by the waveforms
    :return a dataframe summarizing these results
    '''
    admissions = pd.read_sql("SELECT subject_id, hadm_id, admittime FROM admissions limit 5")
    wfSub = listAllMatchedWFSubjects()
    wfSub = pd.DataFrame({"subject_id": wfSub[0], "startWFTime": wfSub[1]})[0:1] #TODO: Just use this subset until physionet api works or data downloads
    wfSub["endWFTime"] = pd.Series(np.full([wfSub.shape[0]], np.nan), index=wfSub.index)
    wfSub["percentMissing"] = pd.Series(np.full([wfSub.shape[0]], np.nan), index=wfSub.index)
    wfSub["numberOfWaveforms"] = pd.Series(np.full([wfSub.shape[0]], np.nan), index=wfSub.index)
    wfSub = wfSub.apply(helperFillOutWF, axis=1)
    # wfSub.set_index("subject_id", inplace=True)
    # admissions.set_index("subject_id", inplace=True)
    wfSub = admissions.join(wfSub, how="right", on="subject_id", lsuffix="_left", rsuffix="_right")
    #following column is if we capture any waveform in the first 24 hours of admittime, 1 if true
    wfSub["containsBegin"] = pd.Series(np.full([wfSub.shape[0]], np.nan), index=wfSub.index)
    #following column is if we capture entire 24 hours after admittime, 1 if true
    wfSub["coversAll"] =pd.Series(np.full([wfSub.shape[0]], np.nan), index=wfSub.index)
    #startWfTime - admittime (in hours)
    wfSub["differenceInTime"] = pd.Series(np.full([wfSub.shape[0]], np.nan), index=wfSub.index)
    # wfSub = wfSub.apply(helperCompareTimes, axis=1)
    return wfSub

count = 1; #TODO: remove this
count1 = 1;
def helperCompareTimes(row):
    '''
    Helper function to run through entire row in joined wfSub and admissions
    :param row the specific row to process
    :return the row containing if the admittime plus 24 hours past it contains the start of missing data, etc.
    :precondition containsBegin and coversAll columns are already in dataframe
    '''
    global count1
    print("cleaning data for :" + str(count1))
    count1 = count1 + 1
    daySecs = 24 * 60 * 60
    admittime = row["admittime"].value
    afterAdmittime = admittime + daySecs
    startWF = time.strptime(str(row["startWFTime"]), "%Y-%m-%d-%H-%M")
    endWFTime = row["endWFTime"]
    row["containsBegin"] = 1 if afterAdmittime >= time.mktime(startWF) else 0
    row["coversAll"] = 1 if(admittime >= time.mktime(startWF) \
                            and time.mktime(endWFTime) > afterAdmittime) else 0
    row["differenceInTime"] = float(time.mktime(startWF) - admittime) / (60**2)
    return row
def helperFillOutWF(row):
    '''
    Helper function to run through entire row of wfSub in compareAdmitToWF
    :param row the specific row to process
    :return the row with endWFTime, percentMissing, and numberOfWaveforms placed in
    :precondition subjectID and startWFTime are present in the row
    '''
    global count
    print("calling data for :" + str(count))
    count = count + 1
    data, fields = sampleWFSubject(row["subject_id"], row["startWFTime"])
    secLength = len(data) / 125.0 #sampling frequency for mimic3 is known to be 125 hz
    startWF = time.strptime(row["startWFTime"], "%Y-%m-%d-%H-%M") #turn into a struct_time object to do math with it
    endWF = datetime.date.fromtimestamp(time.mktime(startWF)) +\
                                        datetime.timedelta(0, secLength) #add the time in seconds
    row["endWFTime"] = endWF.strftime("%Y-%m-%d-%H-%M") #return to end time
    numMissing = 0
    for col in data.columns:
        numMissing += data[col].isnull().sum()
    row["percentMissing"]= numMissing / (data.shape[0] * data.shape[1])
    row["numberOfWaveforms"] = data.shape[1]
    return row

def listAllMatchedWFSubjects(url="https://physionet.org/physiobank/database/mimic3wdb/matched/RECORDS-waveforms"):
    '''
    Waveforms in mimic3wdb have a frequency of 125 hz and can contain multiple waveform data
    method here returns the subject_id of the matched records for waveforms
        uses specified format given in documentions
    :param url (already set) which sets where to read the subject ids
    :return a list n * 2(subject_id, start time in string format) for matched numeric records in the waveform data
    '''
    resp = request.urlopen(url)
    data = resp.read()
    recordSubjects = [];
    recordTime = [];
    for line in data.splitlines():
        line = str(line)
        sublines = line.split('/')
        subjectID = sublines[1][1:]
        time = sublines[2].replace("p" + subjectID, "")
        recordTime.append(time[1:-1])
        recordSubjects.append(subjectID)
    return [recordSubjects, recordTime]

def listAllSubjects(url="https://physionet.org/physiobank/database/mimic3wdb/matched/RECORDS-waveforms"):
    '''
    all subjects recorded in mimic3wdb, either numeric record or waveform record
    :param url (already set) which sets where to read the subject ids
    :return an ndarray of all subject ids
    '''
    subjects = np.append(listAllMatchedWFSubjects()[0], listAllMatchedRecSubjects()[0])
    subjects = pd.Series(subjects).unique()
    return subjects

def sampleWFSubject(subject_id, time):
    '''
    Wrapper function for wfdb.srdsamp3
    Returns the waveform data for a specific patient or NaN if not in matched set or some other error occurs
    :param subject_id the unique mimic identifier, used in the filename of the waveform
    :param time the starting time of the waveform of interest
    :return tuple of signal and fields, or NaN if nothing
    '''
    first2 = str(subject_id)[:2]
    record_name = 'p' + str(subject_id) + \
                '-' + str(time)
    pbdir = 'mimic3wdb/matched/' + 'p' + first2 + \
            '/p' + str(subject_id) + '/'
    try:
        data, fields = wfdb.srdsamp(recordname=record_name, pbdir=pbdir)
        return (pd.DataFrame(data), fields)
    except:
        print("Could not get data from: " + pbdir)
        raise
def applyInIntervals(applier, waveform, freq = 125, time=6):
    '''
    Applies a function to waveform data at certain hour intervals over 24 hours
    :param applier function to apply to waveform data
    :param waveform 1d numpy array to process
    :param freq the sampling frequency of the waveform, default 125 Hz
    :param time how long each subsection of waveform to process should be, in hours
    :return array of results of function applier
    '''
    toRet = []
    hour = 60 * 60 * freq #datapoints per hour
    segment = time * hour
    #iterate for each time segment of the waveform and use applier
    for i in range(0, len(waveform)/(hour * time) - 1):
        toRet.append(applier(waveform[segment*i:segment*(i+1)]))
    return toRet
def nameToTimestamp(recordName):
    time = pd.Timestamp(year=int(recordName[8:12]), month=int(recordName[13:15]), day=int(recordName[16:18]), \
                     hour=int(recordName[19:21]), minute=int(recordName[22:24]))
    return time

def matchRecordNameWithHADMID(recordName, time_error = '6 hours'):
    '''
    Unlike waveform_traverser, which maps all records under a subject id to a hospital admission at one time,
    this only maps one record to a hadm_id.
    :param recordName string containing subject id and time of record, as per mimic3wdb convention
    :param time_error the amount of leeway we give to matching to a hospital admission
    '''
    subjectid = recordName[1:7]
    admissions = read_sql("SELECT HADM_ID, ADMITTIME, DISCHTIME from ADMISSIONS where subject_id = {} and DISCHTIME > ADMITTIME".format(subjectid))
    print(recordName)
    time = nameToTimestamp(recordName)
    matching = admissions[(admissions["ADMITTIME"] - pd.Timedelta(time_error) < time) & (admissions["DISCHTIME"] + pd.Timedelta(time_error) > time)]
    if matching["ADMITTIME"].iloc[0] > time:
        admittime = time
    else:
        admittime = matching["ADMITTIME"].iloc[0]
    return matching["HADM_ID"].iloc[0], admittime

def processSubjectID(subject_id, numericMapping=None, numHours=24):
    '''
    Provides representative statistics for all records under a subjectid
    '''
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
        try: #some records are illformed
            data, fields = reader.getRecord(multiRecord, subject_id=subject_id)
            for sig_name in data.columns:
                singRecStats[sig_name + " MEAN"] = data[sig_name].mean()
                if (~(data[sig_name].apply(np.isnan))).all():
                    singRecStats[sig_name + "_PERCENT_MISSING"] = 0
                else:
                    singRecStats[sig_name + "_PERCENT_MISSING"] = pd.isnull(data[sig_name]).value_counts()[True] / len(data[sig_name])
                #For each signal, find the percentage that is filled in the first 24 hours after admission
                if fileToHADMID[multiRecord].hadmid != "NOT FOUND":
                    singRecStats = percentMissingFirstNHours(data, admittime, numHours, sig_name, singRecStats)
                    singRecStats["ADMITTIME"] = admittime
            singRecStats["LENGTH"] = len(data)
        except:
            singRecStats["comment"] = "Could not getRecord"
        toReturn.append(singRecStats)
    return pd.concat(toReturn)

def percentMissingFirstNHours(data, admittime, numHours, sig_name, singRecStats):
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




if __name__ == "__main__":
    # print(ListAllMatchedSubjectsWaveforms()[1:10])
    print(wfdb.srdsamp(recordname='3141595', pbdir='mimic3wdb/31/3141595/', sampfrom=0, sampto=100))
