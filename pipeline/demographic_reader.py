import pandas as pd
import numpy as np
import os
from preprocessing.preprocessing import read_variable_ranges
from addict import Dict
from multiprocessing import Process
from multiprocessing import Queue
from multiprocessing import Manager
from commonDB import read_sql

class DemReader():

    def __init__(self, hadms=[], num_workers=24):
        self.hadms = hadms
        self.__n_workers = num_workers
        self.manager = Manager()

    def getHADMDem(self, hadmid):
        df = pd.DataFrame(columns=[hadmid])
        dem = read_sql("WITH ADMITDATA as (SELECT GENDER, DOB, PATIENTS.SUBJECT_ID FROM PATIENTS WHERE SUBJECT_ID in (SELECT SUBJECT_ID FROM ADMISSIONS WHERE HADM_ID = {})),  SUBJECTDATA as (SELECT * FROM ADMISSIONS WHERE HADM_ID = {}) SELECT * FROM ADMITDATA JOIN SUBJECTDATA on ADMITDATA.SUBJECT_ID=SUBJECTDATA.SUBJECT_ID".format(hadmid, hadmid))
        temp = df[hadmid].append(dem['ETHNICITY'].value_counts().rename(lambda x: "ETHNICITY:" + x))
        temp = temp.append(dem['ADMISSION_TYPE'].value_counts().rename(lambda x: "ADMISSION_TYPE:" + x))
        temp = temp.append(dem['GENDER'].value_counts().rename(lambda x: "GENDER:" + x))
        df[hadmid] = temp
        df = df.T
        df.loc[hadmid, "AGE"] = dem["ADMITTIME"].iloc[0] - dem["DOB"].iloc[0]
        return df

    def getHADMDemsHelper(self, toRun, toReturn):
        for hadm in iter(toRun.get, None):
            toReturn.put(self.getHADMDem(hadm))
        return

    def getHADMDems(self):
        toReturn = self.manager.Queue()
        toRun = self.manager.Queue()
        [toRun.put(hadm) for hadm in self.hadms]
        [toRun.put(None) for i in range(self.__n_workers)]
        running = [Process(target=self.getHADMDemsHelper, args=(toRun, toReturn)) for i in range(self.__n_workers)]
        [runner.start() for runner in running]
        [runner.join() for runner in running]
        toConcat = []
        while not toReturn.empty():
            toConcat.append(toReturn.get())
        toRet = pd.concat(toConcat)
        toRet["AGE"] = toRet["AGE"].fillna(pd.Timedelta(toRet["AGE"].astype(np.int).mean())).astype(np.int)
        toRet = toRet.fillna(0)
        return toRet
