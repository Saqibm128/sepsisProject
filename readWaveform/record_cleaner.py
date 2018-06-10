### This cleans waveform data from a provided waveform reader using variable ranges from the MIMIC3benchmark project
###
### @author Mohammed Saqib

import pandas as pd
import numpy as np
import time
import datetime
import commonDB
import math
import os

from multiprocessing import Process, Queue, Manager
import wfdb
from preprocessing.preprocessing import read_variable_ranges
from readWaveform.waveform_traverser import WaveformFileTraverser
from readWaveform.waveform_reader import WaveformReader
from readWaveform import waveformUtil as wfutil
class Record_Cleaner:
    def __init__(self, variable_ranges=read_variable_ranges("preprocessing/resources/variable_ranges.csv"), reader = WaveformReader(), columns=[], records=None, num_workers=24, num_hours=24):
        '''
        Reader is an object capable of providing a record for cleaning
        variable_ranges is the DataFrame for variables, with OUTLIER_HIGH and OUTLIER_LOW values
        reader is an object which provides the record info
        numericMapping is a dataframe which maps variables from numeric mapping to
        columns is the variables to keep from the records
        records is a list of record names to use
        num_hours is the number of hours after admission to keep (helps with efficiency to specify here), if none keep all
        '''
        self.reader = reader
        self.variable_ranges = variable_ranges
        self.columns = columns
        self.manager = Manager()
        self.records = records
        self.num_workers = num_workers
        self.num_hours=24

    def clean(self, recordName):
        data, fields = self.reader.getRecord(recordName)

        hadmid, admittime = wfutil.matchRecordNameWithHADMID(recordName)

        # drop off seconds and milliseconds
        admittime = pd.Timestamp(year=admittime.year, month=admittime.month, day=admittime.day, hour=admittime.hour, minute=admittime.minute)
        data.index = data.index.map(lambda date: pd.Timestamp(year=date.year, month=date.month, day=date.day, hour=date.hour, minute=date.minute))
        data = data.join(pd.DataFrame(index=pd.date_range(admittime, admittime + pd.Timedelta('24 hours'), freq='1min')), how="outer") #force include the first 24 hours, so we correctly fill in data
        admittime = pd.Timestamp(year=admittime.year, month=admittime.month, day=admittime.day, hour=admittime.hour, minute=admittime.minute)


        cleaned = []
        for col in self.columns:
            if not col in data.columns:
                data[col] = (pd.Series(index=data.index).apply(lambda a: self.variable_ranges["IMPUTE"][col])) #Just use imputed data for completely missing variable
                continue
            data.loc[data[col] < self.variable_ranges["OUTLIER_LOW"][col], col] = np.nan
            data.loc[data[col] < self.variable_ranges["OUTLIER_HIGH"][col], col] = np.nan
            data[col] = data[col].fillna(method="ffill")
            data[col] = data[col].fillna(method="bfill")
            data[col] = data[col].fillna(self.variable_ranges["IMPUTE"][col])
        if self.num_hours is not None:
            data = data[data.index < admittime + pd.Timedelta("{} hours".format(self.num_hours))]

        data = data[self.columns]

        return (hadmid, data, recordName)

    def helperClean(self, toClean, cleaned):
        '''
        Uses queues to clean waveform
        '''
        for recordName in iter(toClean.get, None):
            print(toClean.qsize())
            toReturn = self.clean(recordName)
            cleaned.put(toReturn)

    def cleanAll(self, records = None):
        '''
        :param records is list of records to clean and return
        if not set, just use the set of records passed to object
        '''
        toClean = self.manager.Queue()
        cleaned = self.manager.Queue()
        if records is not None:
            [toClean.put(recordName) for recordName in records]
        else:
            [toClean.put(recordName) for recordName in self.records]
        [toClean.put(None) for i in range(self.num_workers)]
        processes = [Process(target=self.helperClean, args=(toClean, cleaned)) for i in range(self.num_workers)]
        [process.start() for process in processes]
        [process.join() for process in processes]
        allResults = {}
        while not cleaned.empty():
            hadmID, data, recordName = cleaned.get()
            if hadmID in allResults.keys():
                #combine records, choosing older record as base df
                #1)figure out earlier/older record
                if wfutil.nameToTimestamp(allResults[hadmID]["recordName"]) > wfutil.nameToTimestamp(recordName):
                    temp =  allResults[hadmID]["data"]
                    allResults[hadmID]["data"] = data
                    allResults[hadmID]["recordName"] = recordName
                    data = temp
                allResults[hadmID]["data"] = allResults[hadmID]["data"].fillna(data)
                #fill in inconsistencies from combining both together
                allResults[hadmID]["data"].fillna(method="ffill")
                allResults[hadmID]["data"].fillna(method="bfill")
            else:
                allResults[hadmID] = {}
                allResults[hadmID]["data"] = data
                allResults[hadmID]["recordName"] = recordName
        for hadmid in allResults.keys():
            allResults[hadmid]['data'].index = allResults[hadmid]['data'].index - allResults[hadmid]['data'].index[0]
        return allResults
