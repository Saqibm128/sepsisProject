import pandas as pd
import os


class Hadm_Id_Reader():

    def __init__(hadm_dir, file_name):
        self.hadm_dir = hadm_dir
        self.file_name = file_name
        self.hadms = os.listdir(os.path.join(hadm_dir))
        self.__current_hadm = self.hadms[0] #to use when Hadm_Id_Reader is used like an iterator
        self.__index = 0 #to use when Hadm_Id_Reader is used like an iterator

    def read(timeUnit = 6, hadmid = self.current_hadm):
        '''
        This method provides the correct dataframe for an object which corresponds to the
        properly filled out df. The DF will be backfilled, unless if there is no value to use,
        in which case the value is forward filled. If the data is entirely missing, we should use
        physiologically appropriate data
        '''
