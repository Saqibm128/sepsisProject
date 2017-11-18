import pandas as pd
import os
import numpy as np
from preprocessing import preprocessing
import commonDB
from multiprocessing import Process
'''
This script is influenced/copy pasta from extract_episodes_from_subject.py in the MIMIC3 benchmark repo
https://github.com/YerevaNN/mimic3-benchmarks/
It's job is to convert data I retrieved using extract_subjects.py from same side repo into an hadm_id episodic format
TODO: implement/copypasta extract_subjects.py functionality. Currently I do all data into one file in a very serial way
instead of the more effective multifolder way
'''

subjects_root_path = "data/rawdatafiles/benchmarkData"
var_map = preprocessing.read_itemid_to_variable_map(
    "preprocessing/resources/itemid_to_variable_map.csv")
ranges = preprocessing.read_variable_ranges("preprocessing/resources/variable_ranges.csv")
new_path = "data/rawdatafiles/byHadmID"
variables = var_map.VARIABLE.unique()

def read_events(subject_path, remove_null=True):
    events = pd.DataFrame.from_csv(os.path.join(subject_path, 'events.csv'), index_col=None)
    if remove_null:
        events = events.ix[events.VALUE.notnull()]
    events.CHARTTIME = pd.to_datetime(events.CHARTTIME)
    events.HADM_ID = events.HADM_ID.fillna(value=-1).astype(int)
    events.ICUSTAY_ID = events.ICUSTAY_ID.fillna(value=-1).astype(int)
    events.VALUEUOM = events.VALUEUOM.fillna('').astype(str)
    events.columns = events.columns.str.upper()
    return events

def get_first_valid_from_timeseries(timeseries, variable):
    '''
    Stolen from MIMIC3 benchmark repo even though it seems weird
    '''
    if variable in timeseries:
        idx = timeseries[variable].notnull()
        if idx.any():
            loc = np.where(idx)[0][0]
            return timeseries[variable].iloc[loc]
    return np.nan


def get_events_for_stay(events, hadm_id, intime=None, outtime=None):
    '''
    Stolen from MIMIC3 benchmark repo except it does hadm_id now
    '''
    idx = (events["HADM_ID"] == hadm_id)
    if intime is not None and outtime is not None:
        idx = idx | ((events.CHARTTIME >= intime) & (events.CHARTTIME <= outtime))
    events = events.ix[idx]
    return events

def read_stays(subject_path):
    '''
    Stolen from mimic3 benchmark repo
    :param subject_path where the subject folder is located
    :return a cleaned representation of the staysDF
    '''
    stays = pd.DataFrame.from_csv(os.path.join(subject_path, 'stays.csv'), index_col=None)
    stays.INTIME = pd.to_datetime(stays.INTIME)
    stays.OUTTIME = pd.to_datetime(stays.OUTTIME)
    stays.DOB = pd.to_datetime(stays.DOB)
    stays.DOD = pd.to_datetime(stays.DOD)
    stays.DEATHTIME = pd.to_datetime(stays.DEATHTIME)
    stays.sort_values(by=['INTIME', 'OUTTIME'], inplace=True)
    return stays

def convert_events_to_timeseries(events, variable_column='VARIABLE', variables=[]):
    '''
    Stolen from MIMIC3 benchmark repo
    '''
    metadata = events[['CHARTTIME', 'ICUSTAY_ID']].sort_values(by=['CHARTTIME', 'ICUSTAY_ID'])\
                    .drop_duplicates(keep='first').set_index('CHARTTIME')
    timeseries = events \
                    .sort_values(by=['CHARTTIME', variable_column, 'VALUE'], axis=0)\
                    .drop_duplicates(subset=['CHARTTIME', variable_column], keep='last')
    timeseries = timeseries.pivot(index='CHARTTIME', columns=variable_column, values='VALUE').merge(metadata, left_index=True, right_index=True)\
                    .sort_index(axis=0).reset_index()
    for v in variables:
        if v not in timeseries:
            timeseries[v] = np.nan
    return timeseries
def add_hours_elapsed_to_events(events):
    '''
    Taken from the MIMIC3 benchmark repo
    '''
    events["CHARTTIME"] = events["CHARTTIME"].astype(pd.Timestamp)
    dt = events["CHARTTIME"].min()
    events['HOURS'] = (events.CHARTTIME - dt).astype('timedelta64[s]')/60/60
    return events

def extract_multiple_subjects(subjects):
    for subject in subjects:
        dn = os.path.join(subjects_root_path, subject)
        try:
            subject_id = int(subject)
        except:
            return
        events = read_events(dn)
        stays = read_stays(dn)
        events = events.merge(stays[["ICUSTAY_ID", "HADM_ID"]], left_on=["ICUSTAY_ID"], right_on=["ICUSTAY_ID"], how="left", suffixes=["_l", ""])
        events = events.dropna(axis=0, subset=["HADM_ID"], how="any")
        events = preprocessing.map_itemids_to_variables(events, var_map)
        events = preprocessing.clean_events(events, ranges=ranges)
        for hadm_id in events["HADM_ID"].unique():
            # For every hadm_id we want to separate, clean, find variable ranges, and get key constants out for each hadm
            episode = get_events_for_stay(events, hadm_id)
            timeseries = convert_events_to_timeseries(episode)
            timeseries = add_hours_elapsed_to_events(timeseries)
            if timeseries.shape[0] == 0:
                print(' (no data!)')
                continue
            if not os.path.isdir(os.path.join(new_path, str(int(hadm_id)))):
                os.mkdir(os.path.join(new_path, str(int(hadm_id))))
            timeseries.set_index(["HOURS"], inplace=True)
            timeseries.to_csv(os.path.join(new_path, str(int(hadm_id)), 'episode_timeseries.csv'), index_label='HOURS')
            print("finished:", hadm_id)




if __name__ == '__main__':
    subjects = os.listdir(subjects_root_path)
    print("beginning to process", len(subjects))
    num_process = 10
    for i in range(0, num_process):
        portion = int(len(subjects) / num_process)
        if i != num_process - 1:
            to_process = subjects[portion * i: portion * (i + 1)]
        else:
            to_process = subjects[portion * i:]
        p = Process(target=extract_multiple_subjects, args=(to_process,))
        p.start()
