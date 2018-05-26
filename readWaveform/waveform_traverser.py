### there is a specific manner in which files are retrieved from local storage for MIMIC3 matched subset

import os
import os.path as path
import re
from addict import Dict
from pandas import Timestamp, Timedelta
from commonDB import read_sql

class WaveformFileTraverser():
    def __init__(self, root="/data/mimic3wdb"):
        self.root = root
        self.numeric = False
    def getParentFolders(self):
        '''
        ex: for /data/mimic3wdb/matched/p00/p000020 returns p00 alongside others
        @return a list of parent folders.
        '''
        parentFolders = os.listdir(self.root + "/matched/")
        #only p01, p02, etc.
        parentFolders = [file for file in parentFolders if  re.match(r'p\d+', file) != None]
        return parentFolders
    def getSubjects(self):
        '''
        ex: for /data/mimic3wdb/matched/p00/p000020 returns p000020 alongside others
        @return a list of subject ids with p appended
        '''
        parentFolders = self.getParentFolders()
        subjects = []
        for parentFolder in parentFolders:
            subjects = subjects + os.listdir(self.root + "/matched/" + parentFolder)
        return [subject[1:] for subject in subjects]
    def getSubjectPath(self, subjectid, p_appended=False):
        '''
        @return the generated path for the waveform data for a specific subjectid
        '''
        if not p_appended:
            subjectid = "p" + subjectid
        return self.root + "/matched/" + subjectid[0:3] + "/" + subjectid

    def getMultiRecordFiles(self, subjectid, p_appended=False):
        '''
        p_appended: is the subject_id including a 'p' at beginning
        numeric: use minute by minute files instead of 125 hz files
        '''
        if self.numeric:
            fileRegex = r"p\d+-\d{4}-\d{2}-\d{2}-\d{2}-\d{2}n.hea"
        else:
            fileRegex = r"p\d+-\d{4}-\d{2}-\d{2}-\d{2}-\d{2}.hea"
        subjectPath = self.getSubjectPath(subjectid, p_appended)
        return [file[:-4] for file in os.listdir(subjectPath) if re.match(fileRegex, file) is not None]

    def matchWithHADMID(self, subjectid, p_appended=False, time_error="6 hours"):
        '''
        Goes through each fileDateMap entry to match each key (waveform file) to hospital admission
        time_error is the amount of time before admittime and after DISCHTIME to consider for admissions
        @return a dictionary which matches hadm_id with the waveform file
        '''
        subjectFiles = self.getMultiRecordFiles(subjectid, p_appended)
        fileDateMap = Dict()
        for subjectFile in subjectFiles:
            time = Timestamp(year=int(subjectFile[8:12]), month=int(subjectFile[13:15]), day=int(subjectFile[16:18]), \
                             hour=int(subjectFile[19:21]), minute=int(subjectFile[22:24]))
            fileDateMap[subjectFile] = time
        if p_appended:
            subjectid = subjectid[1:] #drop the p
        admissions = read_sql("SELECT HADM_ID, ADMITTIME, DISCHTIME from ADMISSIONS where subject_id = " + subjectid)
        fileAdmissionMap = Dict()

        for waveform in fileDateMap.keys():
            time = fileDateMap[waveform]
            matching = admissions["HADM_ID"][(admissions["ADMITTIME"] - Timedelta(time_error) < time) & (admissions["DISCHTIME"] + Timedelta(time_error) > time)]
            if (len(matching.values) != 0):
                fileAdmissionMap[waveform] = matching.iloc[0] #assume that admissions don't overlap for a single subject id
            else:
                fileAdmissionMap[waveform] = "NOT FOUND"
        return fileAdmissionMap
