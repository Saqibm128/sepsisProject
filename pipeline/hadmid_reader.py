import pandas as pd
import numpy as np
import os
from preprocessing.preprocessing import read_variable_ranges
from addict import Dict

class Hadm_Id_Reader():

    def __init__(self, hadm_dir, file_name="episode_timeseries.csv", variable_ranges="preprocessing/resources/variable_ranges.csv"):
        '''
        :param hadm_dir the directory where each folder holding hadm_id data is located
        :param file_name the name of the file inside each directory that holds data
        :param variable_ranges the file which holds key info on variables and the mapping to mean values for imputing completely
                                missing data
        '''
        self.hadm_dir = hadm_dir
        self.file_name = file_name
        self.hadms = os.listdir(os.path.join(hadm_dir))
        self.__current_hadm = self.hadms[0] #to use when Hadm_Id_Reader is used like an iterator
        self.__index = 0 #to use when Hadm_Id_Reader is used like an iterator
        self.__ranges = read_variable_ranges(variable_ranges)
    def __convert_timeseries_to_features(self, timeseries):
        '''
        A simple method to convert timeseries data into a format which can be taken
            by traditional ML techniques
            i.e. turn feature time combo ("Heart Rate" at 6 hours) into 'pseudofeature'
            ('Heart Rate, 6')
        :param timeseries time indexed, feature columns data
        :return dataframe of 1 row, containing rearranged data
        '''
        toReturn = pd.DataFrame()
        for hour in timeseries.index:
            for col in timeseries.columns:
                toReturn.loc[0, str(col) + ", " + str(hour)] = timeseries.loc[hour, col]
        return toReturn
    def traditional_time_event_vector(self, hadm_id):
        '''
        Preprocess a single hadm_id for analysis with a traditional ML technique
        include some hand engineered features, such as standard deviation of features
        as well as change over time of features
        :param hadm_id
        :return a traditional feature vector
        '''
        ts = self.resample_fixed_length(hadm_id=hadm_id)
        numericCols = []
        for col in ts.columns:
            if np.issubdtype(ts[col].dtype, np.number):
                numericCols.append(col)
        ts = ts[numericCols]
        var_std = ts.std(axis=0)
        var_std_ind = pd.Series(var_std.index)
        for i in range(0, len(var_std_ind)):
            var_std_ind[i] = var_std_ind[i] + " STDEV"
        var_std = pd.DataFrame(var_std).set_index(var_std_ind).transpose()
        delta = ts.loc[0].subtract(ts.loc[ts.index[ts.shape[0] - 1]])
        deltaInd = pd.Series(delta.index).apply(lambda s: s + " DELTA")
        delta = pd.DataFrame(delta).set_index(deltaInd).transpose()
        vec = self.__convert_timeseries_to_features(ts)
        return pd.concat([vec, var_std, delta], axis=1)
    def traditional_time_event_matrix(self):
        toReturn = []
        for hadm_id in self.hadms:
            print(hadm_id)
            toAppend = (self.traditional_time_event_vector(hadm_id))
            toReturn.append(toAppend.set_index(pd.Series([int(hadm_id)])))
        toReturn = pd.concat(toReturn) #drop the nonnumeric columns due to inabilty to deal with mean()
        return toReturn
    def countEvents(self, hadm_id, endbound=None):
        '''
        This method provides the counts of events for a certain hadm_id for each variable
        :param hadm_id the hospital admission to apply this method to
        :param endbound the last time, in hours, to take into account data; if None
            all data from all time of the hospital admission is taken into account
        :return dataframe containing count of events for each feature
        '''
        data = pd.read_csv(os.path.join(self.hadm_dir, hadm_id, self.file_name))
        if endbound is not None:
            data = data.loc[data.index <= endbound] #Exclude any data after the last end
        eventCounts = Dict()
        totalLength = data.shape[0]
        for col in data.columns:
            eventCounts[col] = data[col].shape[0] - data[col].isnull().sum()
        return eventCounts
    def countEventsPerHospitalAdmission(self, endbound=None):
        toConcat = []
        i = 0
        for hadm_id in self.hadms:
            if i % 100 == 0:
                print(i)
            i+=1
            eventCounts = self.countEvents(hadm_id, endbound=endbound)
            toConcat.append(pd.DataFrame(eventCounts, index=[int(hadm_id)]))
        return (pd.concat(toConcat))
    def avg(self, hadm_id, endbound = None):
        '''
        This method provides no analysis over time and instead only provides the average
        for every variable
        :param endbound the last time, in hours, after first event to take into
            account data, if None dataframe is generated without end limit
        :param hadm_id hadm_id to apply this function to
        '''
        data = self.__get_data(hadm_id, endbound=endbound)
        return data.mean()
    def getFullAvg(self, endbound = None):
        toReturn = {}
        for hadm_id in self.hadms:
            toAppend = (self.avg(hadm_id, endbound=endbound))
            toReturn[int(hadm_id)] = (self.avg(hadm_id, endbound=endbound))
        toReturn = pd.DataFrame(toReturn).transpose().dropna(axis=1, how="any") #drop the nonnumeric columns due to inabilty to deal with mean()
        return toReturn
    def __get_data(self, hadm_id, endbound=None):
        '''
        Helper function to read and do a simple preprocessing of dataset
        to fill in completely missing data variables
        :param hadm_id which hospital admission to read
        :param endbound the last time, in hours, after first event to take into account data, if None no bound
        :return the preprocessed dataframe
        '''
        data = pd.read_csv(os.path.join(self.hadm_dir, hadm_id, self.file_name))
        if endbound is not None:
            data = data.loc[data.index <= endbound] #Exclude any data after the last end
        for var in data.columns:
            if var not in self.__ranges.index:
                data = data.drop(var, axis=1)
                continue
            if data[var].isnull().all():
                data[var] = self.__ranges["IMPUTE"][var]
        return data
    def resample(self, hadm_id, timeUnit = 6):
        '''
        This method provides the correct dataframe for an object which corresponds to the
        properly filled out df. The DF will be forwardfilled, unless if there is no value to use,
        in which case the value is backfilled. If the data is entirely missing, we should use
        physiologically appropriate data
        :param timeUnit hours to resample by. For example, doing every 6 hours will resample events as such
        :param hadm_id the events of the correct hadm_id to sample
        :return the properly resampled df
        '''
        data = pd.read_csv(os.path.join(self.hadm_dir, hadm_id, self.file_name))
        charttime = data["CHARTTIME"]
        data.set_index(pd.DatetimeIndex(charttime), inplace=True)
        data = data.resample(str(60 * timeUnit) + "T").mean() #https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.resample.html
        for var in data.columns:
            if var not in self.__ranges.index: #drop any index that isn't an actual variable in Ranges i.e. we cannot impute
                data = data.drop(var, axis=1)
                continue
            if data[var].isnull().all():
                data[var] = data[var].fillna(self.__ranges["IMPUTE"][var]) # for variables that are completely missing, just use the imputed variable
        data = data.fillna(method="ffill") # Forward fill
        data = data.fillna(method="bfill") # any remaining NaN, fill with backfilling

        # add back in the hours column and sort by it
        charttime = pd.Series(data.index)
        hours = charttime.apply(pd.Timestamp) - charttime.apply(pd.Timestamp).min()
        hours = hours.astype('timedelta64[s]')/60/60 # set it to hours
        return data.set_index(hours).sort_index()
    def resample_fixed_length(self, hadm_id, total_time = 24, timeUnit = 6):
        '''
        This method is similar to resample but forward fills such that the resulting DataFrame
        is as long as the total_time parameter given ie if an admission had events spanning
        only 12 hours, this method extends the last 12 using the last observed values of previous
        12 hours
        :param timeUnit hours to resample by. For example, doing every 6 hours will resample events as such
        :param hadm_id the events of the correct hadm_id to sample
        :param total_time total hours the final df should span
        :return the properly resampled df of correct size
        '''
        data = self.resample(timeUnit=timeUnit, hadm_id = hadm_id)
        while (data.index.max() < total_time):
            data.loc[data.index.max() + timeUnit] = data.loc[data.index.max()]
        return data.loc[data.index <= total_time]

    def next_hadm():
        '''
        This method goes to the next hadm_id if this reader is used as an iterator
        TODO: implement other iterator-like features or just plain remove it
        '''
        self.__current_hadm = hadms[self.__index + 1]
        self.__index += 1
