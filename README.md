# SepsisProject
This project provides key tools for predicting the onset of sepsis in ICU patients from the MIMIC3 db.

# Key resources
I found the MIMIC3 benchmark repo by the YerevaNN group to be very helpful.
In fact, I used Significant portions of their scripts for my project.

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
dataCollect holds the scripts I used to develop the data collection process before I found
the MIMIC3 benchmark repo

## Building the dataset
Use extract_subjects.py to gain access to the data partitioned by subjects.
Afterwards, I partition events during a hospital admission using split_subjects_into_events.py.
This cleans the data and fixes outliers.
