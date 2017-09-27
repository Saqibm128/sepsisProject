#This file is responsible for taking in waveform data from the MIMIC3 wfdb
#   and providing key utilities.
# @author Mohammed Saqib
import wfdb
import urllib.request as request

def listAllMatchedRecSubjects():
    # Numeric records in mimic3wdb are averages per second (1hz) of waveform data
    # :return a list of tuple (subject_id, start time in string format) for matched numeric records in the waveform data
    resp = request.urlopen("https://physionet.org/physiobank/database/mimic3wdb/matched/RECORDS-numerics")
    data = resp.read()
    recordSubjects = [];
    for line in data.splitlines():
        line = str(line)
        sublines = line.split('/')
        subjectID = sublines[1][1:]
        time = sublines[2].replace("p" + subjectID, "")
        recordSubjects.append((subjectID, time[1:-2]))
    return recordSubjects
def listAllMatchedWFSubjects():
    # Waveforms in mimic3wdb have a frequency of 125 hz and can contain multiple waveform data
    # :return a list of tuple (subject_id, start time in string format) for matched numeric records in the waveform data
    resp = request.urlopen("https://physionet.org/physiobank/database/mimic3wdb/matched/RECORDS-waveforms")
    data = resp.read()
    recordSubjects = [];
    for line in data.splitlines():
        line = str(line)
        sublines = line.split('/')
        subjectID = sublines[1][1:]
        time = sublines[2].replace("p" + subjectID, "")
        recordSubjects.append((subjectID, time[1:-2]))
    return recordSubjects

def sampleWFSubject(subject_id, debug=False):
    # Wrapper function for wfdb.srdsamp
    # Returns the waveform data for a specific patient or NaN if not in matched set or some other error occurs
    # :param subject_id the unique mimic identifier, used in the filename of the waveform
    # :param debug True if output should be printed, error raised, when data is not gathered
    # :return tuple of signal and fields, or NaN if nothing
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
    #Applies a function to waveform data at certain hour intervals over 24 hours
    # :param applier function to apply to waveform data
    # :param waveform numpy array to process
    # :param startIndex index to start applying from, default 0
    # :param freq the sampling frequency of the waveform, default 125 Hz
    # :param time how long each subsection of waveform to process should be
    # :return array of results of function applier
    #TODO
    return None

if __name__ == "__main__":
    # print(ListAllMatchedSubjectsWaveforms()[1:10])
    print(wfdb.srdsamp(recordname='3141595', pbdir='mimic3wdb/31/3141595/', sampfrom=0, sampto=100))
