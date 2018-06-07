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

    def __init__(self, traverser = WaveformFileTraverser(), numericMapping=None):
        '''
        @param traverser is the object which provides paths and info about files
        @param numericMapping is the dataframe which maps the signal names to high level variables
                if None, don't use
        '''
        self.traverser = traverser
        if numericMapping is not None:
            numericMapping["numeric"] = numericMapping["numeric"].str.upper()
            numericMapping["high_level_var"] = numericMapping["high_level_var"].str.upper()
        self.numericMapping = numericMapping

    def getRecord(self, record, subject_id=None):
        '''
        Numerical records is a minute by minute summary of the data
        @param subject_id if None, will try to extract from record name
        @param record
        @return the numerical record df
        @return fields the dictionary of specific fields associated with the numeric record
        '''
        if subject_id == None:
            subject_id = record[1:7]
        path = self.traverser.getSubjectPath(subject_id, False)
        sig, fields = wfdb.rdsamp(path + "/" + record)
        sig = pd.DataFrame(sig)
        columns = fields["sig_name"]
        for i in range(len(columns)):
            columns[i] = columns[i].upper()
            if (self.numericMapping is not None):
                if (self.numericMapping["numeric"] == columns[i]).any():
                    columns[i] = self.numericMapping["high_level_var"][self.numericMapping["numeric"] == columns[i]].iloc[0]
        for col in sig.columns[sig.columns.duplicated()]:
            sig[col] = sig[col].iloc[:,0].fillna(sig[col].iloc[:,1]) #if two numerics signals exist with same name, use fillna to correctly fill in
        sig.columns = columns
        # Convert datetime and date.date into timestamp for a timeseries
        baseDate = fields["base_date"]
        baseTime = fields["base_time"]
        baseTimestamp = pd.Timestamp(year=baseDate.year, month=baseDate.month, day=baseDate.day, hour=baseTime.hour, minute=baseTime.minute, second=baseTime.second, microsecond=baseTime.microsecond)
        ts = pd.date_range(baseTimestamp, periods=len(sig), freq=pd.Timedelta(seconds=60))
        sig.index = ts
        sig = sig.loc[:,~sig.columns.duplicated()]
        return sig, fields


    def getWaveform(self, subject_id, record):
        '''
        Waveform is 125 hz data
        @param subject_id
        @param record
        @return the waveform df
        @return fields the dictionary of specific fields associated with the numeric record
        
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
