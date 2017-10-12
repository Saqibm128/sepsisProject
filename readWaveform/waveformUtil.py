#This file is responsible for taking in waveform data from the MIMIC3 wfdb
#   and providing key utilities.
# @author Mohammed Saqib
import wfdb
import urllib.request as request
import pandas as pd
import numpy as np
import categorization as angus
import time
import datetime
import commonDB

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
    '''
    conn = commonDB.getConnection()
    admissions = pd.read_sql("SELECT subject_id, hadm_id, admittime FROM admissions", conn)
    wfSub = listAllMatchedWFSubjects()
    wfSub = pd.DataFrame({"subject_id": wfSub[0], "startWFTime": wfSub[1]})
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
    wfSub = wfSub.apply(helperCompareTimes, axis=1)
    return wfSub
def helperCompareTimes(row):
    '''
    Helper function to run through entire row in joined wfSub and admissions
    :param row the specific row to process
    :return the row containing if the admittime plus 24 hours past it contains the start of missing data, etc.
    :precondition containsBegin and coversAll columns are already in dataframe
    '''
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
def generateAngusDF(cachedAngus="data/rawdatafiles/classifiedAngusSepsis.csv", sqlFile = None):
    '''
    Run through and get the subject_id of all patients in the matched subset.
    Use that to generate an Angus dataframe which matches the subject_ids to sepsis status for preliminary analysis
    :param cachedAngus file path to read cached csv file of Angus sql result from.
    :param sqlFile None (default) if not to run, if path is given, run the query and use that instead of cached
    :return a dataframe with hadm_ids as the index joined with the angus data
    '''
    if sqlFile != None:
        categorization = angus.getCategorizations()
    else:
        categorization = pd.DataFrame.from_csv(cachedAngus)
    recordSubIds = listAllMatchedRecSubjects()
    wfSubIds = listAllMatchedWFSubjects()
    #ensure we input unique names but we allow for all subject_ids to come in for now,
    # even if missing any record or waveform
    np.append(recordSubIds, wfSubIds)
    subjectSeries = pd.Series(recordSubIds[:, 0]).astype(int).unique() # weird edge case with dtypes, make sure we don't do weird stuff with repeats

    df = pd.DataFrame({"hadm_id": subjectSeries})
    df.set_index(["hadm_id"], inplace=True)
    categorization.subject_id = categorization.subject_id.astype(int)
    categorization.set_index("subject_id", inplace = True)
    toReturn = df.join(categorization, how="inner")
    return toReturn
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
if __name__ == "__main__":
    # print(ListAllMatchedSubjectsWaveforms()[1:10])
    print(wfdb.srdsamp(recordname='3141595', pbdir='mimic3wdb/31/3141595/', sampfrom=0, sampto=100))
