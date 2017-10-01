from matplotlib import pyplot as plt
import pickle
import pandas as pd
import pandas.core.indexes
import waveformUtil as wfutil
import featureSelection.frequencyCount.featureFrequencySQLCount as freq
import categorization as cat
## @author Mohammed Saqib
## This file is responsible for creating histograms to represent the distribution
##      of the frequency of each labitem and chartitem, as well as providing some other statistics
##      from the imported pickle file (i.e. the reporting/underreporting of features)


freq.countFeatures(ids=wfutil.listAllSubjects())
frequencies = pandas.DataFrame.from_csv("data/rawDataFiles/counts.csv")


distinctEvents = frequencies["countperadmission"]
plt.hist(distinctEvents, bins=10)
plt.title("Distribution of Frequency of Items, Removing Repeated Measure per Admission")
plt.xlabel("Frequency Bins of Each Item")
plt.ylabel("Count of Items in Each Frequency Bin")
plt.show()

print(frequencies.describe())

cat.getCategorizations(writeToCSV = True)
angusCounts = pandas.DataFrame.from_csv("data/rawdatafiles/classifiedAngusSepsis.csv")
