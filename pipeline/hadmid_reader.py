import pandas as pd
import numpy as np
import os
from preprocessing.preprocessing import read_variable_ranges
from addict import Dict
from multiprocessing import Process
from multiprocessing import Queue
from multiprocessing import Manager

class Hadm_Id_Reader():

    def __init__(self, hadm_dir, file_name="episode_timeseries.csv", variable_ranges="preprocessing/resources/variable_ranges.csv"):
        '''
        :param hadm_dir the directory where each folder holding hadm_id data is located
        :param file_name the name of the file inside each directory that holds data
        :param vars_to_keep list of all variables to keep, if None use all variables given
        :param variable_ranges the file which holds key info on variables and the mapping to mean values for imputing completely
                                missing data
        '''
        self.hadm_dir = hadm_dir
        self.file_name = file_name
        ## Because I save some results in hadm directory (bad decision), I check to see we don't add results into here
        self.hadms = [hadmid for hadmid in os.listdir(os.path.join(hadm_dir)) if os.path.exists(os.path.join(self.hadm_dir, hadmid, self.file_name))]
        self.__current_hadm = self.hadms[0] #to use when Hadm_Id_Reader is used like an iterator
        self.__index = 0 #to use when Hadm_Id_Reader is used like an iterator
        self.__ranges = read_variable_ranges(variable_ranges)
        self.__vars_to_keep = self.__ranges.index ## keep originally all variables that have imputed vals
        self.__n_workers = 1
        self.manager = Manager()
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
    def use_multiprocessing(self, num_process):
        '''
        :num_process how many processes to use to read and process data
        '''
        self.__n_workers = num_process
    def set_features(self, features):
        '''
        Sets the variables, in case of feature selection, to keep in processed dataset
        :precondition all feature in features has impute value available in variable_ranges
        :param features to keep in processed data returned by hadmid_reader
        :postcondition should include only features from this vector/arraylike passed in
        '''
        self.__vars_to_keep = features
    def traditional_time_event_vector(self, hadm_id, include_var=False, total_time=24, time_unit=6):
        '''
        Preprocess a single hadm_id for analysis with a traditional ML technique
        include some hand engineered features, such as standard deviation of features
        as well as change over time of features
        :param time_unit hours to resample by. For example, doing every 6 hours will resample events as such
        :param total_time total hours the final df should span
        :param hadm_id
        :param include_var include variance from within each 6 hour bin
        :return a traditional feature vector
        '''
        ts, _ = self.resample_fixed_length(hadm_id=hadm_id, include_var=include_var, total_time=24, time_unit=6)
        if ts is None:
            return None;
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
        toReturn = pd.concat([vec, var_std, delta], axis=1)
        toReturn.index=[int(hadm_id)]
        return toReturn

    def get_time_matrices_helper(self, toRun, toReturn, total_time=24, time_unit=6):
        '''
        '''
        for hadm_id in iter(toRun.get, None):
            #place a tuple with the data as well as the hospital admission identifying the data
            toReturn.put((int(hadm_id), self.resample_fixed_length(hadm_id=hadm_id, total_time=total_time, time_unit=time_unit)[0]))
    def get_time_matrices(self, hadm_ids=None, total_time=24, time_unit=6):
        '''
        '''
        toRun = self.manager.Queue()
        toReturn = self.manager.Queue()
        if hadm_ids is None:
            hadm_ids = self.hadms
        [toRun.put(hadm_id) for hadm_id in hadm_ids]
        [toRun.put(None) for i in range(self.__n_workers)]
        processes = [Process(target=self.get_time_matrices_helper, args=(toRun, toReturn, total_time, time_unit)) for i in range(self.__n_workers)]
        [process.start() for process in processes]
        [process.join() for process in processes]
        hadmToData = Dict()
        while not toReturn.empty():
            (hadm_id, data) = toReturn.get()
            hadmToData[hadm_id] = data
        return hadmToData
    def populate_all_hadms(self):
        '''
        This function goes through and writes a copy of the final data time by features matrix
        from the end of the pipeline to file for each hospital admission
        '''
        for hadm_id in self.hadms:
            if not os.path.exists(os.path.join(self.hadm_dir, hadm_id, self.file_name)):
                continue;
            (data, _) = self.resample_fixed_length(hadm_id)
            data.to_csv(os.path.join(self.hadm_dir, hadm_id, "processed.csv"))
    def traditional_time_event_matrix_helper(self, toRun, toReturn, include_var=False, total_time=24, time_unit=6):
        '''
        A helper method to run through all of the hadms, primarily for multiprocessing needs
        used for parallelism
        :param time_unit hours to resample by. For example, doing every 6 hours will resample events as such
        :param total_time total hours the final df should span
        :param toRun a queue of hadmid followed by None to stop this process
        :param to return a queue to post result

        '''
        for hadm in iter(toRun.get, None): #https://stackoverflow.com/questions/6672525/multiprocessing-queue-in-python
            toReturn.put(self.traditional_time_event_vector(hadm, include_var=include_var, total_time=total_time, time_unit=time_unit))
        return
    def traditional_time_event_matrix(self, include_var=False, total_time=24, time_unit=6):
        '''
        This function returns the time event matrix, resampled currently for 24 hours with 6 hour periods,
        in a format such that each row is one hadmid, and columns are 'pseudofeatures' ie features with time bins

        If use_multiprocessing was used, the hadmids will be read

        :param time_unit hours to resample by. For example, doing every 6 hours will resample events as such
        :param total_time total hours the final df should span
        :param include_var include variance from within each 6 hour bin
        :return a properly formatted a matrix
        '''
        toReturn = self.manager.Queue()
        toRun = self.manager.Queue()
        [toRun.put(hadm) for hadm in self.hadms]
        [toRun.put(None) for i in range(self.__n_workers)]
        running = [Process(target=self.traditional_time_event_matrix_helper, args=(toRun, toReturn, include_var, total_time, time_unit)) for i in range(self.__n_workers)]
        [runner.start() for runner in running]
        [runner.join() for runner in running]
        toConcat = []
        toConcatIndex = []
        while not toReturn.empty():
            df = toReturn.get()
            toConcat.append(df)
        return pd.concat(toConcat)


    def num_imputed_helper(self, toRun, toReturn, include_var=False, total_time=24, time_unit=6):
        '''
        A helper method to run through all of the hadms, primarily for multiprocessing needs
        used for parallelism
        :param time_unit hours to resample by. For example, doing every 6 hours will resample events as such
        :param total_time total hours the final df should span
        :param toRun a queue of hadmid followed by None to stop this process
        :param to return a queue to post result

        '''
        for hadm in iter(toRun.get, None): #https://stackoverflow.com/questions/6672525/multiprocessing-queue-in-python
            toReturn.put(self.findNumBackAndForwardFilled(hadm, include_var=include_var, total_time=total_time, time_unit=time_unit))
        return
    def num_imputed(self, include_var=False, total_time=24, time_unit=6):
        '''
        This function returns the number of backfilled and forwardfilled bins
        If use_multiprocessing was used, the hadmids will be read

        :param time_unit hours to resample by. For example, doing every 6 hours will resample events as such
        :param total_time total hours the final df should span
        :param include_var include variance from within each 6 hour bin
        :return a properly formatted a matrix
        '''
        toReturn = self.manager.Queue()
        toRun = self.manager.Queue()
        [toRun.put(hadm) for hadm in self.hadms]
        [toRun.put(None) for i in range(self.__n_workers)]
        running = [Process(target=self.num_imputed_helper, args=(toRun, toReturn, include_var, total_time, time_unit)) for i in range(self.__n_workers)]
        [runner.start() for runner in running]
        [runner.join() for runner in running]
        backFilledToConcat = []
        missingSomeDataToConcat = []
        forwardFilledToConcat = []
        while not toReturn.empty():
            _, missingSomeData, backFilled, forwardFilled = toReturn.get()
            missingSomeDataToConcat.append(missingSomeData)
            backFilledToConcat.append(backFilled)
            forwardFilledToConcat.append(forwardFilled)
        return pd.concat(missingSomeDataToConcat), pd.concat(backFilledToConcat), pd.concat(forwardFilledToConcat)
    def countEvents(self, hadm_id, endbound=None):
        '''
        This method provides the counts of events for a certain hadm_id for each variable
        :param hadm_id the hospital admission to apply this method to
        :param endbound the last time, in hours, to take into account data; if None
            all data from all time of the hospital admission is taken into account
        :return dataframe containing count of events for each feature or None if the hadm_id is not found
        '''
        if not os.path.exists(os.path.join(self.hadm_dir, hadm_id, self.file_name)):
            return None;
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
            if not os.path.exists(os.path.join(self.hadm_dir, hadm_id, self.file_name)):
                continue;
            if i % 100 == 0:
                print(i)
            i+=1
            eventCounts = self.countEvents(hadm_id, endbound=endbound)
            if eventCounts is not None:
                toConcat.append(pd.DataFrame(eventCounts, index=[int(hadm_id)]))
        return (pd.concat(toConcat))
    def countAllImputedWindows(self):
        '''
        This method counts the number of 6 hour windows imputed for each variable
        '''
        toConcat = []
        for hadm_id in self.hadms:
            if not os.path.exists(os.path.join(self.hadm_dir, hadm_id, self.file_name)):
                continue;
            (_, missingData) = self.resample_fixed_length(hadm_id)
            toConcat.append(missingData)
        return pd.concat(toConcat)
    def avg(self, hadm_id, endbound = None):
        '''
        This method provides no analysis over time and instead only provides the average
        for every variable
        :param endbound the last time, in hours, after first event to take into
            account data, if None dataframe is generated without end limit
        :param hadm_id hadm_id to apply this function to
        '''
        if self.__vars_to_keep is None:
            data = self.__get_data(hadm_id, endbound=endbound)
        else:
            data = self.__get_data(hadm_id, endbound=endbound)[self.__vars_to_keep]
        if data is None:
            return None;
        return data.mean()
    def getFullAvgHelper(self, toRun, toReturn, idnum):
        '''
        Helper method for use_multiprocessing
        :toRun queue that holds hadm, endbound, then None to signify end
        :toReturn to put tuple of hadmid, and average vector in
        '''
        for hadm_id, endbound in iter(toRun.get, None):
            toReturn.put((hadm_id, self.avg(hadm_id, endbound=endbound)))
    def getFullAvg(self, endbound = None):
        '''
        Returns a dataframe of hospital admission hadm_id by average features
        :param endbound how long to keep. If none, use as much of the record as possible
        '''
        toReturn = self.manager.Queue()
        toRun = self.manager.Queue()
        [toRun.put((hadm_id, endbound)) for hadm_id in self.hadms]
        [toRun.put(None) for i in range(self.__n_workers)]
        runners = [Process(target=self.getFullAvgHelper, args=(toRun, toReturn, i)) for i in range(self.__n_workers)]
        [runner.start() for runner in runners]
        [runner.join() for runner in runners]
        hadmDict = Dict()
        while not toReturn.empty():
            hadm_id, avgVect = (toReturn.get())
            hadmDict[int(hadm_id)] = avgVect
        toReturn = pd.DataFrame(hadmDict).transpose().dropna(axis=1, how="any") #drop the nonnumeric columns due to inabilty to deal with mean()
        return toReturn
    def __get_data(self, hadm_id, endbound=None):
        '''
        Helper function to read and do a simple preprocessing of dataset
        to fill in completely missing data variables
        :param hadm_id which hospital admission to read
        :param endbound the last time, in hours, after first event to take into account data, if None no bound
        :return the preprocessed dataframe
        '''
        if not os.path.exists(os.path.join(self.hadm_dir, hadm_id, self.file_name)):
            return None;
        data = pd.read_csv(os.path.join(self.hadm_dir, hadm_id, self.file_name))
        if endbound is not None:
            data = data.loc[data.index <= endbound] #Exclude any data after the last end
        for var in data.columns:
            if var not in self.__vars_to_keep:
                data = data.drop(var, axis=1)
                continue
            if data[var].isnull().all():
                if self.__vars_to_keep is None or var in self.__vars_to_keep:
                    data[var] = self.__ranges["IMPUTE"][var]
        return data

    def findNumBackAndForwardFilled(self, hadm_id, time_unit = 6, include_var = False, total_time=24):
        '''
        Hadm_Id_Reader.resample was original method to resample data, but relied on prepackaged pd.DF.fillna methods
            This keeps resample functionality, but allows for window into how imputation occurred
            :param time_unit hours to resample by. For example, doing every 6 hours will resample events as such
            :param hadm_id the events of the correct hadm_id to sample
            :param include_var whether or not to include the variance within the bin
            :return tuple of resampled df, df of missing variables originally, df of number forwardfilled, and df of number of backfilled
        '''
        if not os.path.exists(os.path.join(self.hadm_dir, hadm_id, self.file_name)):
            return None;

        data = pd.read_csv(os.path.join(self.hadm_dir, hadm_id, self.file_name))
        charttime = data["CHARTTIME"]
        data.set_index(pd.DatetimeIndex(charttime), inplace=True)
        data = data.resample(str(60 * time_unit) + "T").mean() #https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.resample.html
        missingEntirely = pd.DataFrame(index=[hadm_id], columns=data.columns)
        for var in self.__vars_to_keep:
            if var not in data.columns:
                missingEntirely.loc[hadm_id, var] = 1
                data[var] = pd.Series(index=data.index).apply(lambda a: self.__ranges["IMPUTE"][var])
            else:
                missingEntirely.loc[hadm_id, var] = 0
        missingSomeData = pd.DataFrame(columns=data.columns, index=[hadm_id])
        for var in data.columns:
            if var not in self.__vars_to_keep: #drop any columns that isn't an actual variable in Ranges i.e. we cannot impute
                data = data.drop(var, axis=1)
                try:
                    missingSomeData = missingSomeData.drop(var, axis=1)
                except:
                    print("Could not drop variable") #TODO: make a better way to count 6 hour windows imputed?
                continue
            time = pd.Series(data.index).apply(pd.Timestamp) - pd.Series(data.index).apply(pd.Timestamp).min()
            time.index = data.index
            missingSomeData.loc[hadm_id, var] = data.loc[(time.astype('timedelta64[s]')/60/60 < total_time), var].isnull().sum()
            if data[var].isnull().all():
                if self.__vars_to_keep is None or var in self.__vars_to_keep:
                    data[var] = data[var].fillna(self.__ranges["IMPUTE"][var]) # for variables that are completely missing, just use the imputed variable
        if include_var:
            data_stdev = data.resample(str(60 * time_unit) + "T").std()
            data_stdev = data_stdev.rename(columns=lambda s: str(s) + " stdev")
            data = pd.concat([data, data_stdev], axis=1, verify_integrity=True)


        numForwardFilled = pd.DataFrame(columns=data.columns, index=[hadm_id])
        numForwardFilled.loc[:,:] = 0
        numBackFilled = pd.DataFrame(columns=data.columns, index=[hadm_id])
        numBackFilled.loc[:,:] = 0
        for var in data.columns:
            foundVarToUseFF = False
            numHoursSinceFirst = pd.Series(data.index).apply(pd.Timestamp) - pd.Series(data.index).apply(pd.Timestamp).min()
            numHoursSinceFirst.index = data.index
            for binTimeIndex in data.index:
                if numHoursSinceFirst[binTimeIndex].total_seconds()/60/60 < total_time:
                    if not np.isnan(data[var][binTimeIndex]):
                        foundVarToUseFF = True
                    elif foundVarToUseFF:
                        numForwardFilled.loc[hadm_id, var] = numForwardFilled.loc[hadm_id, var] + 1
        data = data.fillna(method="ffill") # Forward fill
        data = data.fillna(method="bfill") # any remaining NaN, fill with backfilling

        # add back in the hours column and sort by it
        charttime = pd.Series(data.index)
        hours = charttime.apply(pd.Timestamp) - charttime.apply(pd.Timestamp).min()
        hours = hours.astype('timedelta64[s]')/60/60 # set it to hours
        data = data.set_index(hours).sort_index()
        if self.__vars_to_keep is not None:
            data = data[self.__vars_to_keep]
        while (data.index.max() < total_time):
            data.loc[data.index.max() + time_unit] = data.loc[data.index.max()]
        return (data, missingSomeData, numForwardFilled, numBackFilled)

    def resample(self, hadm_id, time_unit = 6, include_var = False):
        '''
        This method provides the correct dataframe for an object which corresponds to the
        properly filled out df. The DF will be forwardfilled, unless if there is no value to use,
        in which case the value is backfilled. If the data is entirely missing, we should use
        physiologically appropriate data
        :param time_unit hours to resample by. For example, doing every 6 hours will resample events as such
        :param hadm_id the events of the correct hadm_id to sample
        :param include_var whether or not to include the variance within the bin
        :return the properly resampled df
        '''
        data, missingData, _, _ = self.findNumBackAndForwardFilled(hadm_id, time_unit, include_var)
        return (data, missingData)
    def resample_fixed_length(self, hadm_id, total_time = 24, time_unit = 6, include_var=False):
        '''
        This method is similar to resample but forward fills such that the resulting DataFrame
        is as long as the total_time parameter given ie if an admission had events spanning
        only 12 hours, this method extends the last 12 using the last observed values of previous
        12 hours
        :param time_unit hours to resample by. For example, doing every 6 hours will resample events as such
        :param hadm_id the events of the correct hadm_id to sample
        :param total_time total hours the final df should span
        :param include_var include variance from within each 6 hour bin
        :return the properly resampled df of correct size
        '''
        if not os.path.exists(os.path.join(self.hadm_dir, hadm_id, self.file_name)):
            return None;
        (data, missingData) = self.resample(time_unit=time_unit, hadm_id = hadm_id, include_var=include_var)
        while (data.index.max() < total_time):
            data.loc[data.index.max() + time_unit] = data.loc[data.index.max()]
        return (data.loc[data.index < total_time], missingData)

    def next_hadm():
        '''
        This method goes to the next hadm_id if this reader is used as an iterator
        TODO: implement other iterator-like features or just plain remove it
        '''
        self.__current_hadm = hadms[self.__index + 1]
        self.__index += 1
