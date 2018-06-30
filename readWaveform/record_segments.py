### After we realized that waveforms have giant gaps that we need to address, we go ahead and just filter waveforms for contiguous segments to analyze
### Very similar to record_cleaner.py
### @author Mohammed Saqib

import pandas as pd
import numpy as np
import time
import datetime
import commonDB
import math
import os
from addict import Dict

from multiprocessing import Process, Queue, Manager
import wfdb
from preprocessing.preprocessing import read_variable_ranges
from readWaveform.waveform_traverser import WaveformFileTraverser
from readWaveform.waveform_reader import WaveformReader
from readWaveform import waveformUtil as wfutil
class RecordSegmentsAnalyzer:
    def __init__(self, variable_ranges=read_variable_ranges("preprocessing/resources/variable_ranges.csv"), columns=None, reader = WaveformReader(), num_workers=24, num_missing_most=15):
        '''
        Reader is an object capable of providing a record for cleaning
        variable_ranges is the DataFrame for variables, with OUTLIER_HIGH and OUTLIER_LOW values
        reader is an object which provides the record info
        numericMapping is a dataframe which maps variables from numeric mapping to
        num_missing_most is the most number of CONSECUTIVE data points that can be missing in a segment
        '''
        self.reader = reader
        self.variable_ranges = variable_ranges
        self.manager = Manager()
        self.num_workers = num_workers
        self.num_missing_most = num_missing_most
        if columns is not None:
            reader.columnsToUse = columns


    def analyze(self, hadmID):
        data, matching = self.reader.getRecordByHADMID(hadmID)

        admittimeDiff = data.index[0] - matching.iloc[0]['admittime']
        data.index = data.index.map(lambda date: pd.Timestamp(year=date.year, month=date.month, day=date.day, hour=date.hour, minute=date.minute))


        if self.reader.columnsToUse is not None:
            for col in self.reader.columnsToUse:
                data.loc[data[col] < self.variable_ranges["OUTLIER_LOW"][col], col] = np.nan
                data.loc[data[col] > self.variable_ranges["OUTLIER_HIGH"][col], col] = np.nan

        partOfSegment = pd.DataFrame(index=pd.RangeIndex(data.shape[0]), columns=["isSeg", "block"]) #is truee or false, depending on window of data

        #figure out which parts of the data are part of a segment or part of a missing segment
        tempData = data.fillna(-1).reset_index(drop=True) #rolling window has incorrect behavior when analyzing nulls, use -1 as placeholder instead

        partOfSegment.loc[:,'isSeg'] = ~tempData.rolling(self.num_missing_most).apply(lambda win: (win == -1).all()).any(axis=1)
        #edge cases
        if data.shape[0] <= self.num_missing_most:
            partOfSegment.loc[:, 'isSeg'] = False
        else:
            partOfSegment.loc[0:self.num_missing_most, 'isSeg'] = partOfSegment.loc[self.num_missing_most, 'isSeg'] #we can always carry over the result for the first n
        partOfSegment['block'] = (partOfSegment['isSeg'].shift(1) != partOfSegment['isSeg']).astype(int).cumsum()
        segments = partOfSegment.groupby(['isSeg','block']).apply(len)

        return (hadmID, segments, admittimeDiff)

    def helperAnalyze(self, toAnalyze, analyzed):
        '''
        Uses queues to clean waveform
        '''
        for hadmid in iter(toAnalyze.get, None):
            print(toAnalyze.qsize())
            toReturn = self.analyze(hadmid)
            analyzed.put(toReturn)

    def analyzeAll(self, hadmids = None):
        '''
        :param hamdIDs is list of hadmIDs to clean and return
        '''
        toAnalyze = self.manager.Queue()
        analyzed = self.manager.Queue()
        [toAnalyze.put(hadmID) for hadmID in hadmids]
        [toAnalyze.put(None) for i in range(self.num_workers)]
        processes = [Process(target=self.helperAnalyze, args=(toAnalyze, analyzed)) for i in range(self.num_workers)]
        [process.start() for process in processes]
        [process.join() for process in processes]
        allResults = Dict()
        while not analyzed.empty():
            hadmid, isSeg, admittimeDiff = analyzed.get()
            allResults[hadmid].isSeg = isSeg
            allResults[hadmid].admittimeDiff = admittimeDiff
        return allResults
