# from pipeline.hadmid_reader import Hadm_Id_Reader
# import commonDB
# from preprocessing import preprocessing
# import os

from addict import Dict

import pandas as pd
import numpy as np
import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, show, savefig, xlim, figure, \
                hold, ylim, legend, boxplot, setp, axes

vars_to_use = ['BLOOD\nUREA\nNITROGEN', 'RESPIRATORY\nRATE', 'HEMATOCRIT', 'GLASCOW\nCOMA\nSCALE', 'HEMOGLOBIN', 'URINE\nOUTPUT']
hardcodedVals = [0.445302, 0.322941, 0.27626, 0.37822, 0.3728, 0.29538]

plt.bar([1.5, 3, 4.5, 7, 8.5, 10], hardcodedVals, align='center', color=['g', 'g', 'g', 'r', 'r', 'r']);
ax = axes()
ax.set_xticklabels(vars_to_use)
ax.set_xticks([1.5, 3, 4.5, 7, 8.5, 10])
for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(7.2)
plt.xticks(fontsize=7.4)
plt.ylabel("Weights")
plt.title("Top LR Feature Weights")
hB, = plot([10,10],'g-')
hR, = plot([10,10],'r-')
xlim(0, 11)
ylim(0, .5)
legend((hB, hR),('Positive', 'Negative'))
print("saving fig")
plt.show()
# savefig('lrfeatweight.png', dpi=3000)
plt.gcf().clear()

vars_to_use = ['BLOOD\nUREA\nNITROGEN', 'RESPIRATORY\nRATE', 'URINE\nOUTPUT', 'SYSTOLIC\nBLOOD\nPRESSURE','CREATININE','HEART\nRATE']
hardcodedVals = [0.095779, 0.065888, 0.05061, 0.048572, 0.044741, 0.043825]

plt.bar([1.5, 3, 4.5, 6, 7.5, 9], hardcodedVals, align='center', color=['g', 'g', 'g', 'g', 'g', 'g']);
ax = axes()
ax.set_xticklabels(vars_to_use)
ax.set_xticks([1.5, 3, 4.5, 6, 7.5, 9])
for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(7.2)
plt.xticks(fontsize=7.4)
plt.ylabel("Relative Importance")
plt.title("Top RF Feature Importances")
# hB, = plot([10,10],'g-')
# hR, = plot([10,10],'r-')
xlim(0, 10)
ylim(0, .1)
# legend((hB, hR),('Positive', 'Negative'))
print("saving fig")
plt.show()
# savefig('rffeatweight.png', dpi=3000)
