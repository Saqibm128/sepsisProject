from readWaveform.waveform_reader import Waveform_Reader
from readWaveform import waveformUtil
from multiprocessing import Process
from multiprocessing import Queue
from addict import Dict
import pandas as pd


reader = Waveform_Reader("/data/mimic3wdb/matched")
# icuComp = waveformUtil.preliminaryCompareTimesICU()
# icuComp.to_csv("data/rawdatafiles/icustay_waveform_info.csv")
icuComp = pd.DataFrame.from_csv("data/rawdatafiles/icustay_waveform_info.csv")
careunits = icuComp["first_careunit"].unique()
careunitFreq = pd.DataFrame()
for unit in careunits:
    print((icuComp["first_careunit"]==unit))
    careunitFreq[unit] = (icuComp["first_careunit"]==unit).value_counts()
careunitFreq.to_csv("data/rawdatafiles/icustay_waveform_freq.csv")
# inQueue = queue()
# outQueue = queue()
# subjects = reader.list_all_subjects()
# for subject in subjects:
#     q.put(subject)
# print(reader.access_subject("p010013"))
# sig, fields = reader.get_record("p010013", reader.access_subject("p010013")[0])
# print(fields)
# print(len(sig))
