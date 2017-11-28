### This is a reader for local waveform data such as the data stored on qdqc
### @author Mohammed Saqib

import pandas as pd
import numpy as np
import time
import datetime
import commonDB
import math
import os

class Waveform_Reader():

    def __init__(self, file_path):
        self.file_path = file_path;

    def list_all_subjects(self):
        '''
        Retrieve a list of all subjects who are contained within file path
        :precondition assume that the directory structure is p**/pSUBJECT_ID
        '''
        folders = [f for f in os.listdir(self.file_path) if f[0] == 'p' and f != 'p09']
        subjects = [os.listdir(os.path.join(self.file_path, folder)) for folder in folders]
        return subjects

    def access_subject(self, subject_id):
        '''
        Retrieve a subject by subject_id
        :param subject_id in format (p + subjectid padded to length 6)
        :return waveform record names
        '''
        files = os.listdir(os.path.join(self.file_path, subject_id[0:3], subject_id))
        return [f.split('.')[0] for f in files if f != "RECORDS" and "dat" in f]

    def get_record():
