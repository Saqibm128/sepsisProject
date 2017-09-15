from matplotlib import pyplot as plt
import pickle
import pandas as pd
import pandas.core.indexes

## @author Mohammed Saqib
## This file is responsible for creating histograms to represent the distribution
##      of the frequency of each labitem and chartitem, as well as providing some other statistics
##      from the imported pickle file (i.e. the reporting/underreporting of features)

with open("data/rawdatafiles/labEventCounts.p", "rb") as dataFile:
    frequencies = pickle.load(dataFile)

nonDistinctLabEvents = frequencies.values[:, 2]
distinctAdmissionLabEvents = frequencies.values[:,3]
distinctPatientsLabEvents = frequencies.values[:,4]

plt.hist(nonDistinctLabEvents, bins=10)
plt.title("Distribution of Frequency of Lab Items")
plt.xlabel("Frequency Bins of Each Item")
plt.ylabel("Count of Lab Items in Each Frequency Bin")
plt.show()

plt.hist(distinctAdmissionLabEvents, bins=10)
plt.title("Distribution of Frequency of Lab Items, Removing Repeated Measure per Admission")
plt.xlabel("Frequency Bins of Each Item")
plt.ylabel("Count of Lab Items in Each Frequency Bin")
plt.show()

plt.hist(distinctPatientsLabEvents, bins=10)
plt.title("Distribution of Frequency of Lab Items, Removing Repeated Measure per Patient")
plt.xlabel("Frequency Bins of Each Item")
plt.ylabel("Count of Lab Items in Each Frequency Bin")
plt.show()


print(frequencies.describe())


## now for chartsEvents

with open("data/rawdatafiles/chartEventCounts.p", "rb") as dataFile:
    frequencies = pickle.load(dataFile, encoding='latin1')

nonDistinctChartEvents = frequencies.values[:, 2]
distinctAdmissionChartEvents = frequencies.values[:,3]
distinctPatientsChartEvents = frequencies.values[:,4]

plt.hist(nonDistinctChartEvents, bins=10)
plt.title("Distribution of Frequency of Chart Items")
plt.xlabel("Frequency Bins of Each Item")
plt.ylabel("Count of Chart Items in Each Frequency Bin")
plt.show()

plt.hist(distinctAdmissionChartEvents, bins=10)
plt.title("Distribution of Frequency of Chart Items, Removing Repeated Measure per Admission")
plt.xlabel("Frequency Bins of Each Item")
plt.ylabel("Count of Chart Items in Each Frequency Bin")
plt.show()

plt.hist(distinctPatientsChartEvents, bins=10)
plt.title("Distribution of Frequency of Chart Items, Removing Repeated Measure per Patient")
plt.xlabel("Frequency Bins of Each Item")
plt.ylabel("Count of Chart Items in Each Frequency Bin")
plt.show()


print(frequencies.describe())
