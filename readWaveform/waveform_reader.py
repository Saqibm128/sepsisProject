### This is a reader for local waveform data such as the data stored on qdqc
### @author Mohammed Saqib

import pandas as pd
import numpy as np
import time
import datetime
import commonDB
import math
import os
import wfdb
from readWaveform.waveform_traverser import WaveformFileTraverser

class WaveformReader():

    def __init__(self, traverser = WaveformFileTraverser()):
        '''
        @param traverser is the object which provides paths and info about files
        '''
        self.traverser = traverser

    def getRecord(self, subject_id, record):
        '''
        Numerical records is a minute by minute summary of the data
        @param subject_id
        @param record
        @return the numerical record df
        '''
        path = self.traverser.getSubjectPath(subject_id, False)
        sig, fields = wfdb.rdsamp(path + "/" + record)
        sig = pd.DataFrame(sig)
        sig.columns = fields["sig_name"]
        # Convert datetime and date.date into timestamp for a timeseries
        baseDate = fields["base_date"]
        baseTime = fields["base_time"]
        baseTimestamp = pd.Timestamp(year=baseDate.year, month=baseDate.month, day=baseDate.day, hour=baseTime.hour, minute=baseTime.minute, second=baseTime.second, microsecond=baseTime.microsecond)
        ts = pd.date_range(baseTimestamp, periods=len(sig), freq=pd.Timedelta(seconds=60))
        sig.index = ts
        return sig, fields


    def getWaveform(self, subject_id, record):
        '''
        Waveform is 125 hz data
        @param subject_id
        @param record
        @return the waveform df
        '''
        path = self.traverser.getSubjectPath(subject_id, False)
        sig, fields = wfdb.rdsamp(path + "/" + record)
        sig = pd.DataFrame(sig)
        sig.columns = fields["sig_name"]
        # Convert datetime and date.date into timestamp for a timeseries
        baseDate = fields["base_date"]
        baseTime = fields["base_time"]
        baseTimestamp = pd.Timestamp(year=baseDate.year, month=baseDate.month, day=baseDate.day, hour=baseTime.hour, minute=baseTime.minute, second=baseTime.second, microsecond=baseTime.microsecond)
        ts = pd.date_range(baseTimestamp, periods=len(sig), freq=pd.Timedelta(seconds=1/125))
        sig.index = ts
        return sig, fields
