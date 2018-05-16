from pipeline.hadmid_reader import Hadm_Id_Reader
import commonDB
from preprocessing import preprocessing
import os

from addict import Dict

import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, show, savefig, xlim, figure, \
                hold, ylim, legend, boxplot, setp, axes

# function for setting the colors of the box plots pairs
# https://stackoverflow.com/questions/16592222/matplotlib-group-boxplots
def setBoxColors(bp):
    setp(bp['boxes'][0], color='blue')
    setp(bp['caps'][0], color='blue')
    setp(bp['caps'][1], color='blue')
    setp(bp['whiskers'][0], color='blue')
    setp(bp['whiskers'][1], color='blue')
    setp(bp['fliers'][0], color='blue')
    setp(bp['fliers'][1], color='blue')
    setp(bp['medians'][0], color='blue')

    setp(bp['boxes'][1], color='red')
    setp(bp['caps'][2], color='red')
    setp(bp['caps'][3], color='red')
    setp(bp['whiskers'][2], color='red')
    setp(bp['whiskers'][3], color='red')
    # setp(bp['fliers'][2], color='red')
    # setp(bp['fliers'][3], color='red')
    # setp(bp['medians'][1], color='red')
#
# print("beginning to read all files in")
# reader = Hadm_Id_Reader("./data/rawdatafiles/byHadmID0/")

# reader.countEventsPerHospitalAdmission().to_csv("data/rawdatafiles/byHadmID0/countsByHadmid.csv")
# reader.traditional_time_event_matrix().to_csv("data/rawdatafiles/byHadmID0/full_data_matrix.csv")
# reader.countAllImputedWindows().to_csv("data/rawdatafiles/byHadmID0/impute_count.csv")
# reader.populate_all_hadms();

vars_to_use = ["DIASTOLIC BLOOD PRESSURE", "SYSTOLIC BLOOD PRESSURE", "MEAN BLOOD PRESSURE"]

# reader.set_features(vars_to_use)
#
# mapping = preprocessing.read_itemid_to_variable_map("preprocessing/resources/itemid_to_variable_map.csv")
# high_level_vars = mapping.VARIABLE.unique()
# quantiles = pd.DataFrame()
# for var in vars_to_use:
#     itemids = mapping.loc[mapping["VARIABLE"] == var]["ITEMID"]
#     events = commonDB.read_sql("SELECT VALUE FROM CHARTEVENTS WHERE ITEMID in " + commonDB.convertListToSQL(itemids) + " LIMIT 100")
#     events.columns = events.columns.str.upper()
#     print(var)
#     todrop = []
#     for irow in range(events.shape[0]):
#         try:
#             float(events.loc[irow, "VALUE"])
#         except:
#             todrop.append(irow)
#     events = events.drop(todrop)
#     events["VALUE"] = events["VALUE"].astype(np.number)
#     data[var] = []
#     data[var].append(events["VALUE"])
# quantiles.to_csv("data/rawdatafiles/byHadmID0/rawdata_distribution.csv")
# toConcat = []
#
# mat = reader.traditional_time_event_matrix()
# for var in vars_to_use:
#     processed = mat[var]
#     data[var].append(processed)
#

rawQuantiles = pd.DataFrame.from_csv("data/rawdatafiles/rawdata_distribution.csv")
rawQuantiles = rawQuantiles[rawQuantiles.columns[0:11]] # Remove num events item
rawQuantiles.columns = rawQuantiles.columns.astype(np.number)
processedQuantiles = pd.DataFrame.from_csv("data/rawdatafiles/postprocessQuantiles.csv").transpose()
processedQuantiles.columns = processedQuantiles.columns.astype(np.number)


i = 0;
places = []
for var in vars_to_use:
    data = []
    data.append(rawQuantiles.loc[var, [0, .25, .5, .75, .9999]])
    print(rawQuantiles.columns)
    data.append(processedQuantiles.loc[var, [0, .25, .5, .75, 1]])
    bp = boxplot(data, vert=False, positions=[1 + i, 2 + i], widths=.6, whis = 100)
    setBoxColors(bp)
    places.append(1.5 + i)
    i += 3
ylim(0, 9)
xlim(0, 300)
hB, = plot([1,1],'b-')
hR, = plot([1,1],'r-')
legend((hB, hR),('Raw', 'Processed (First 24 Hours)'))
hB.set_visible(False)
hR.set_visible(False)
ax = axes()
ax.set_yticklabels(vars_to_use)
ax.set_yticks(places)
plt.yticks(fontsize=7)
plt.xlabel("mmHg")
savefig('boxcompare.png', dpi=600)
