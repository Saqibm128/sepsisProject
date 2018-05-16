from pipeline.hadmid_reader import Hadm_Id_Reader
import pandas as pd

reader = Hadm_Id_Reader("./data/rawdatafiles/byHadmID0/")
reader.use_multiprocessing(15)

features = pd.read_csv("data/rawdatafiles/features.csv")
features.index = features['Unnamed: 0']
toKeep = features['Unnamed: 0'][features['pval'] < 0.05]

reader.set_features(toKeep)
numImputed = reader.num_imputed()

missingSomeData = numImputed[0]
missingSomeData.to_csv("data/rawdatafiles/numMissingWindows.csv")
backFilled = numImputed[1]
backFilled.to_csv("data/rawdatafiles/numBackFilledWindows.csv")
forwardFilled = numImputed[2]
forwardFilled.to_csv("data/rawdatafiles/numForwardFilledWindows.csv")


numImputed = reader.num_imputed(total_time=36)

missingSomeData = numImputed[0]
missingSomeData.to_csv("data/rawdatafiles/numMissingWindows36.csv")
backFilled = numImputed[1]
backFilled.to_csv("data/rawdatafiles/numBackFilledWindows36.csv")
forwardFilled = numImputed[2]
forwardFilled.to_csv("data/rawdatafiles/numForwardFilledWindows36.csv")
