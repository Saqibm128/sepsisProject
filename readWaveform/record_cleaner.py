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

class Record_Cleaner:
    def __init__(self, variable_ranges=read_variable_ranges("preprocessing/resources/variable_ranges.csv"), reader = WaveformReader(), columns=[], records=None, num_workers=24):
        # Reader is an object capable of providing a record for cleaning
        # variable_ranges is the DataFrame for variables, with OUTLIER_HIGH and OUTLIER_LOW values
        # reader is an object which provides the record info
        # columns is the variables to keep from the records
        # records is a list of record names to use
        self.traverser = traverser
        self.reader = reader
        self.variable_ranges = variable_ranges
        self.columns = columns
        self.manager = Manager()
        self.records = records
        self.num_workers = num_workers

    def clean(self, data, subjectID=None):
        cleaned = []
        for col in self.columns:
            if not col in data.columns:
                cleaned.append(pd.Series(index=data.index).apply(lambda a: self.variable_ranges["IMPUTE"][col])) #Just use imputed data for completely missing variable
                continue
            dataColumn = data[col]
            dataColumn.loc[dataColumn < self.variable_ranges["OUTLIER_LOW"][variable]] = np.nan
            dataColumn.loc[dataColumn < self.variable_ranges["OUTLIER_HIGH"][variable]] = np.nan
            dataColumn = dataColumn.fillna(method="ffill")
            dataColumn = dataColumn.fillna(method="bfill")
            dataColumn = dataColumn.fillna(self.variable_ranges["IMPUTE"][col])
            cleaned.append(dataColumn)
        toReturn = pd.concat(cleaned)
        if subjectID is not None:
            toReturn.index = [[subjectID for i in range(len(self.columns))], toReturn.index] #Multiindex
        return toReturn

    def helperClean(self, toClean, cleaned):
        '''
        Uses queues to clean waveform
        '''
        for recordName in iter(toRunQueue.get, None):
            print(toRunQueue.qsize())
            data = self.reader.getRecord(recordName)
            toReturn = self.clean(data)
            toReturnQueue.put(toReturn)

    def cleanAll(self):
        toClean = self.manager.Queue()
        cleaned = self.manager.Queue()
        [toClean.put(recordName) for recordName in self.records]
        processes = [Process(target=helperWaveformRunner, args=(toClean, cleaned)) for i in range(self.num_workers)]
        [process.start() for process in processes]
        [process.join() for process in processes]
        allResults = []
        while not cleaned.empty():
            allResults.append(cleaned.get())
