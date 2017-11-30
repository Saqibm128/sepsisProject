import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

rawdata = pd.DataFrame.from_csv("data/rawdatafiles/rawdata_distribution.csv")
rawdata = rawdata.transpose().to_csv("data/rawdatafiles/rawdata_distribution.csv")

plt.boxplot(rawdata)
plt.show()
