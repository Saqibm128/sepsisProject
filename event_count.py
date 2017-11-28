from pipeline.hadmid_reader import Hadm_Id_Reader
import commonDB
from preprocessing import preprocessing
import pandas as pd
import os
import numpy as np

print("beginning to read all files in")
reader = Hadm_Id_Reader("./data/rawdatafiles/byHadmID/")

reader.countEventsPerHospitalAdmission().to_csv("data/rawdatafiles/byHadmID/countsByHadmid.csv")
reader.traditional_time_event_matrix().to_csv("data/rawdatafiles/full_data_matrix.csv")
reader.countAllImputedWindows().to_csv("data/rawdatafiles/byHadmID/impute_count.csv")
reader.populate_all_hadms();

mapping = preprocessing.read_itemid_to_variable_map("preprocessing/resources/itemid_to_variable_map.csv")
high_level_vars = mapping.VARIABLE.unique()
quantiles = pd.DataFrame()
quantiles["VARIABLE"] = high_level_vars
quantiles.set_index(["VARIABLE"], inplace=True)
for var in high_level_vars:
    itemids = mapping.loc[mapping["VARIABLE"] == var]["ITEMID"]
    events = commonDB.read_sql("SELECT VALUE FROM CHARTEVENTS WHERE ITEMID in " + commonDB.convertListToSQL(itemids))
    events.columns = events.columns.str.upper()
    print(var)
    for irow in range(events.shape[0]):
        try:
            float(events.loc[irow, "VALUE"])
        except:
            events = events.drop(irow)
    events["VALUE"] = events["VALUE"].astype(np.number)
    for i in [0, .25, .5, .75, 1]:
        quantiles.loc[var, str(i)] = events["VALUE"].quantile(i)
    quantiles.loc[var, "number_of_events"] = events.shape[0]
quantiles.to_csv("data/rawdatafiles/rawdata_distribution.csv")

toConcat = []
for hadm_id in reader.hadms:
    if not os.path.exists(os.path.join(reader.hadm_dir, hadm_id, "processed.csv")):
        continue;
    print(hadm_id)
    processed = pd.DataFrame.from_csv(os.path.join(reader.hadm_dir, hadm_id, "processed.csv"))
    toConcat.append(processed)
fullProcessed = pd.concat(toConcat)
variable_quantiles = pd.DataFrame()
for col in fullProcessed:
    print(col)
    for i in [0, .25, .5, .75, 1]:
        variable_quantiles.loc[i, col] = fullProcessed[col].quantile(i)
variable_quantiles.to_csv("data/rawdatafiles/postprocessQuantiles.csv")
