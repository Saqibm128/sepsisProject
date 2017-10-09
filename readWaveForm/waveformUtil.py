#This file is responsible for taking in waveform data from the MIMIC3 wfdb
#   and providing key utilities.
# @author Mohammed Saqib
import wfdb
import urllib.request as request
import pandas as pd
import numpy as np
from .. import categorization as angus

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
    return pd.DataFrame({"recordSubjects": recordSubjects, "recordTimes": recordTimes})
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
    return pd.DataFrame({recordSubjects: recordSubjects, "recordTimes": recordTime})

def listAllSubjects(url="https://physionet.org/physiobank/database/mimic3wdb/matched/RECORDS-waveforms"):
    '''
    all UNIQUE subjects recorded in mimic3wdb, either numeric record or waveform record
    :param url (already set) which sets where to read the subject ids
    :return an ndarray of all subject ids
    '''
    subjects = np.append(listAllMatchedWFSubjects()["recordSubjects"].as_matrix(), listAllMatchedRecSubjects()["recordSubjects"].as_matix()) #TODO: more elegant way
    subjects = pd.Series(subjects).unique() #TODO: more elegant way to do this?
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
        return pd.DataFrame(data)
    except:
        print("Could not get data from: " + pbdir)
        raise

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
    data = listAllMatchedWFSubjects()
    print(data[0][0])
    print(data[1][0])
    sampleWFSubject(data[0][0], data[1][0])[0]
