import pandas as pd
import numpy as np
import time
import datetime
import commonDB
import math
import os
from addict import Dict
from multiprocessing import Process, Queue, Manager

class SegmentReader():
    """docstring for SegmentReader."""
    def __init__(self, root_folder="data/rawdatafiles/byHadmIDNumRec", info_df=None, \
                 df_name="allRecHADMIDS.csv", rec_df_name="sixHourSegment.csv", \
                 num_workers=24, max_allowed_admittime_diff = pd.Timedelta("72 hours")):
        """
        @param root_folder is the root of all hadmIDs, all data is underneath it
        @param info_df, holds the df which contains info on key info about other records for each hadmid, if None assume it is in root_folder
        @param df_name, used if info_df is not set
        @param rec_df_name, name of hadmid df held in subfolder underneath root_folder
        @param max_allowed_admittime_diff the most a segment can be allowed to occur after admission, if None, don't filter segments on this
        """
        self.root_folder = root_folder
        if info_df is None:
            self.info_df = pd.read_csv(root_folder + "/" + df_name, index_col=0)
        else:
            self.info_df = info_df
        self.info_df["admittimeDiff"] = self.info_df["admittimeDiff"].apply(pd.Timedelta)
        if max_allowed_admittime_diff is not None:
            self.info_df = self.info_df[self.info_df["admittimeDiff"] < max_allowed_admittime_diff]
        self.records = self.info_df.index
        self.rec_df_name = rec_df_name
        self.num_workers = num_workers

    def labels(self):
        return self.info_df['Y']

    def read(self, hadmid, max=pd.Timedelta('6 hours')):
        data = pd.read_csv(self.root_folder + "/{}/".format(hadmid) + self.rec_df_name, index_col=0)
        #fix index
        data.index = data.index.map(pd.Timestamp)
        data.index = data.index - data.index[0]
        return data[data.index < max]

    def simpleStats(self, data, unittime=pd.Timedelta('1 hours')):
        '''
        Given cleaned up data segment, generate some stats by the hour
        @return data with std, mean, min, and max for each column of data, per unit time
        '''
        resampler = data.resample(unittime)
        mean = resampler.mean()
        mean.columns = mean.columns + " MEAN"
        std = resampler.std()
        std.columns = std.columns + " STDEV"
        min = resampler.min()
        min.columns = min.columns + " MIN"
        max = resampler.max()
        max.columns = max.columns + " MAX"
        return mean.join([std, min, max]).stack()
    def readSimpleStats(self, hadmid, unittime=pd.Timedelta('1 hours')):
        '''
        combines read and simpleStats
        '''
        data = self.read(hadmid)
        stats = self.simpleStats(data)
        return stats

    def simpleStatsAllHelper(self, toRun, toReturn):
        for hadmid in iter(toRun.get, None):
            print(toRun.qsize())
            toReturn.put((hadmid, self.readSimpleStats(hadmid)))


    def simpleStatsAll(self, hadmids=None, Y=None):
        '''
        @param Y a vector categorizing sepsis, if None ignored, has index of hadmids
        @param hadmids, a list of hadmids to read, if None, use the info_df
        return dataframe that is (index=hadmid by columns= time-feature multiindex)
        '''
        if hadmids==None:
            hadmids = self.info_df.index

        manager = Manager()
        toRun = manager.Queue()
        toReturn = manager.Queue()
        [toRun.put(hadmid) for hadmid in hadmids]
        [toRun.put(None) for worker in range(self.num_workers)] #end signal for process while loop
        processes = [Process(target=self.simpleStatsAllHelper, args=(toRun, toReturn)) for worker in range(self.num_workers)]
        [process.start() for process in processes]
        [process.join() for process in processes]

        hadmidsInd = []
        allStats = []
        while not toReturn.empty():
            hadmid, stats = toReturn.get()
            hadmidsInd.append(hadmid)
            allStats.append(stats)


        allStats = pd.DataFrame(allStats, index=hadmidsInd)

        if Y is not None:
            raise BaseException("Not implemented yet!")
        return allStats
