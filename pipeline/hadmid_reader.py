import pandas as pd
import os
from preprocessing import read_variable_ranges

class Hadm_Id_Reader():

    def __init__(hadm_dir, file_name="episode_timeseries.csv", variable_ranges="preprocessing/resources/variable_ranges.csv"):
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

    def avg(hadmid = self.__current_hadm):
        '''
        This method provides no analysis over time and instead only provides the average
        for every variable
        :param hadmid hadm_id to apply this function to
        '''
        data = self.__get_data()
        return data.mean(axis=1)
    def __get_data(hadmid = self.__current_hadm):
        '''
        Helper function to read and do a simple preprocessing of dataset
        to fill in completely missing data variables
        :param hadmid which hospital admission to read
        :return the preprocessed dataframe
        '''
        data = pd.read_csv(os.path.join(hadm_dir, hadmid))
        for var in data.columns:
            if var not in self.__range.index:
                data = data.drop(var, axis=1)
                continue
            if data[var].isnull().all():
                data[var] = self.__ranges["IMPUTE"][var]
        return data
    def resample(timeUnit = 6, hadmid = self.__current_hadm):
        '''
        This method provides the correct dataframe for an object which corresponds to the
        properly filled out df. The DF will be forwardfilled, unless if there is no value to use,
        in which case the value is backfilled. If the data is entirely missing, we should use
        physiologically appropriate data
        :param timeUnit hours to resample by. For example, doing every 6 hours will resample events as such
        :param hadmid the events of the correct hadmid to sample
        :return the properly resampled df
        '''
        data = pd.read_csv(os.path.join(hadm_dir, hadmid))
        data.set_index(data["CHARTTIME"], inplace=True)
        data.resample(str(60 * timeUnit) + "T").mean() #https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.resample.html
        for var in data.columns:
            if var not in self.__range.index: #drop any index that isn't an actual variable in Ranges i.e. we cannot impute
                data = data.drop(var, axis=1)
                continue
            if data[var].isnull().all():
                data[var] = data[var].fillna(self.__ranges["IMPUTE"][var]) # for variables that are completely missing, just use the imputed variable
        data = data.fillna(method="ffill") # Forward fill
        data = data.fillna(method="bfill") # any remaining NaN, fill with backfilling

        # add back in the hours column and sort by it
        data["HOURS"] = data["CHARTTIME"].astype(pd.Timestamp) - data["CHARTTIME"].astype(pd.Timestamp).min()
        data["HOURS"] = data["HOURS"].astype('timedelta64[s]')/60/60 # set it to hours
        return data
    def resample_fixed_length(total_time = 24, timeUnit = 6, hadmid = self.__current_hadm):
        '''
        This method is similar to resample but forward fills such that the resulting DataFrame
        is as long as the total_time parameter given ie if an admission had events spanning
        only 12 hours, this method extends the last 12 using the last observed values of previous
        12 hours
        :param timeUnit hours to resample by. For example, doing every 6 hours will resample events as such
        :param hadmid the events of the correct hadmid to sample
        :param total_time total hours the final df should span
        :return the properly resampled df of correct size
        '''
        data = resample(timeUnit=timeUnit, hadmid = hadmid)
        data.

    def next_hadm():
        '''
        This method goes to the next hadm_id if this reader is used as an iterator
        '''
        self.__current_hadm = hadms[self.__index + 1]
        self.__index += 1
