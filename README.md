# SepsisProject
This project provides key tools for predicting the onset of sepsis in ICU patients from the MIMIC3 db.

# Key resources
I found the MIMIC3 benchmark repo by the YerevaNN group to be very helpful.
In fact, I used significant portions of their scripts for my project.

## Dependencies
Use
```
pip install -r requirements.txt
```
for any dependencies.

If you are using conda instead, do
```
conda install -f requirements.txt
```

## Structure

### Waveform analysis
analyze_waveforms.py is a giant grabbag of scripted code for waveform analysis, matching all with HADMIDs, analyzing segments of numeric records, etc...

### Clinical Dataset
commonDB.py holds multiple functions to access mimic3 sql db (general queries, etc.)

event_count.py is responsible for generating statistics for comparison of preprocessed and postprocessed input data

categorization.py holds code for Angus criterion (sql script taken from MIMIC3 code repo https://github.com/MIT-LCP/mimic-code)

dataCollect holds the scripts I used to develop the data collection process before I found
the MIMIC3 benchmark repo (now deprecated)

pipeline holds files to read timeseries that was segmented by hadmid into a directory structure

learning holds machine learning gridsearch modules that depend on being passed dataframes with an "angus" columns
the "angus" column is set as the Y while X is all other columns (TODO: generalize to other forms of data?)

data holds sql scripts, some of which are preprocessed by python scripts before being send to mimic3 database

readWaveform holds scripts in order to read waveform data.
waveformUtil.py holds primarily scripts for reading data from the online server (very spotty service btw)
waveform_reader.py is a work in progress to read data from qdqc:/data/mimic3wdb/matched

## Building the dataset locally
```
python ./preprocessing/extract_subjects.py csvFiles data/rawdatafiles/benchmarkData
#or if using a db
python ./preprocessing/extract_subjects.py doesntMatter data/rawdatafiles/benchmarkData --use_db
```
Use extract_subjects.py to gain access to the data partitioned by subjects, so that subjects below age of 18 at time of icu stay are excluded.
(Code taken from mimic3 benchmark group https://github.com/YerevaNN/mimic3-benchmarks)
Afterwards, I partition events during a hospital admission using split_subjects_into_events.py.
```
python ./split_subjects_into_events
```
This cleans the data and fixes outliers.

Use hadmid_reader object to access timeseries of data.

main.py runs LogisticRegression and random_forest classifiers.
runTrainpad.sh runs the NN.
runTrainpad2.sh runs more configurations of the NN.
