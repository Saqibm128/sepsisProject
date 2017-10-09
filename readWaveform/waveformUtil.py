#This file is responsible for taking in waveform data from the MIMIC3 wfdb
#   and providing key utilities.
# @author Mohammed Saqib
import wfdb
import urllib.request as request
import pandas as pd
import numpy as np
import categorization as angus

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
        recordTimes.append(time[1:-2])
    return [recordSubjects, recordTimes]


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
        recordTime.append(time[1:-2])
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
def sampleWFSubject(subject_id, debug=False):
    '''
    Wrapper function for wfdb.srdsamp3
    Returns the waveform data for a specific patient or NaN if not in matched set or some other error occurs
    :param subject_id the unique mimic identifier, used in the filename of the waveform
    :param debug True if output should be printed, error raised, when data is not gathered
    :return tuple of signal and fields, or NaN if nothing
    '''
    first2 = str(subject_id)[:2]
    pbdir = 'mimic3wdb/' + first2 + '/' + str(subject_id + '/')
    try:
        data = wfdb.srdsamp(recordname=str(subject_id), pbdir=pbdir)
        return data
    except:
        if debug:
            print("Could not get data")
            raise
        return None

def applyInIntervals(applier, waveform, startIndex = 0, freq = 125, time=6):
    '''
    Applies a function to waveform data at certain hour intervals over 24 hours
    :param applier function to apply to waveform data
    :param waveform numpy array to process
    :param startIndex index to start applying from, default 0
    :param freq the sampling frequency of the waveform, default 125 Hz
    :param time how long each subsection of waveform to process should be, in hours
    :return array of results of function applier
    '''
    #TODO: do this function? or remove it!
    return None

if __name__ == "__main__":
    # print(ListAllMatchedSubjectsWaveforms()[1:10])
    print(wfdb.srdsamp(recordname='3141595', pbdir='mimic3wdb/31/3141595/', sampfrom=0, sampto=100))
