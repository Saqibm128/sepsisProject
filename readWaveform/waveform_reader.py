### This is a reader for local waveform data such as the data stored on qdqc
### @author Mohammed Saqib

import pandas as pd
import numpy as np
import time
import datetime
from commonDB import read_sql
import math
import os
import wfdb
from readWaveform.waveform_traverser import WaveformFileTraverser
from preprocessing.preprocessing import read_variable_ranges


class WaveformReader():

    def __init__(self, traverser = WaveformFileTraverser(), numericMapping=None, columnsToUse=None, variable_ranges=read_variable_ranges("preprocessing/resources/variable_ranges.csv")):
        '''
        @param traverser is the object which provides paths and info about files
        @param numericMapping is the dataframe which maps the signal names to high level variables
                if None, don't use
        @param columnsToUse instead of all columns, only output dfs with the selected columnsToUse
                if None then return the columns from the file
        @param variable_ranges - used primarily to remove outliers
                if columnsToUse is set, then this must be set
        '''
        self.traverser = traverser
        if numericMapping is not None:
            numericMapping["numeric"] = numericMapping["numeric"].str.upper()
            numericMapping["high_level_var"] = numericMapping["high_level_var"].str.upper()
        self.numericMapping = numericMapping
        self.columnsToUse = columnsToUse
        self.variable_ranges = variable_ranges

    def getRecordByHADMID(self, hadmid, subjectID=None):
        '''
        Records combined by HADMID to form a cohesive history over entire hospital admission
        '''
        # Apparently multiple records for same hadm_id
        if subjectID is None:
            subjectID = read_sql("SELECT SUBJECT_ID FROM ADMISSIONS WHERE HADM_ID = " + str(hadmid))['SUBJECT_ID'].iloc[0]

        matching = pd.DataFrame(self.traverser.matchWithHADMID(subjectID))
        matching = matching.T
        #find records from specific subjectID that have same hadmid and order by time of the record
        matching = matching[(matching['hadmid'].astype(str) == hadmid)]
        matching = matching.sort_values(by=['admittime']) #use earliest record as base
        currentDF, fields = self.getRecord(matching.index[0])
        if matching.shape[0] > 1:
            for recName in matching.index[1:]:
                data, fields = self.getRecord(recName)
                currentDF = currentDF.fillna(data)
        return currentDF, matching




    def getRecord(self, record, subject_id=None):
        '''
        Numerical records is a minute by minute summary of the data
        @param subject_id if None, will try to extract from record name
        @param record
        @return the numerical record df, fields the dictionary of specific fields associated with the numeric record
        '''
        if subject_id == None:
            subject_id = record[1:7]
        path = self.traverser.getSubjectPath(subject_id, False)
        try:
            sig, fields = wfdb.rdsamp(path + "/" + record)
        except:
            print("Could not read:", path + "/" + record)
            raise BaseException("Failure on record")
        sig = pd.DataFrame(sig)
        columns = fields["sig_name"]
        for i in range(len(columns)):
            columns[i] = columns[i].upper()
            if (self.numericMapping is not None):
                if (self.numericMapping["numeric"] == columns[i]).any():
                    columns[i] = self.numericMapping["high_level_var"][self.numericMapping["numeric"] == columns[i]].iloc[0]
        for col in sig.columns[sig.columns.duplicated()]:
            accumulated = sig[col].iloc[:,0]
            #if two numerics signals exist with same name, use fillna to correctly fill in
            for i in range(sig[col].shape[1] - 1):
                accumulated = accumulated.fillna(sig[col].iloc[:, i])
            sig[col] = accumulated #weird quirk overrides all columns, instead of replacing multiple columns with one
        sig.columns = columns
        sig = sig.loc[:,~sig.columns.duplicated()] #remove extraneous duplicates
        if self.columnsToUse is not None:
            #drop columns not in columnsToUse
            for column in sig.columns:
                toDrop = []
                if column not in self.columnsToUse:
                    toDrop.append(column)
                sig = sig.drop(toDrop, axis=1)
            #add nan columns not in df
            for column in self.columnsToUse:
                if column not in sig.columns:
                    sig[column] = pd.Series(index=sig.index)
                if self.variable_ranges is not None:
                    sig.loc[sig[column] < self.variable_ranges["OUTLIER_LOW"][column], column] = np.nan
                    sig.loc[sig[column] > self.variable_ranges["OUTLIER_HIGH"][column], column] = np.nan
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
