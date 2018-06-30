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

# vars_to_use = ['BLOOD\nUREA\nNITROGEN', 'RESPIRATORY\nRATE', 'HEMATOCRIT', 'HEART\nRATE', 'GLASCOW\nCOMA\nSCALE', 'HEMOGLOBIN', 'PARTIAL\nPRESSURE\nOXYGEN', 'URINE\nOUTPUT']
# hardcodedVals = [0.440755928, 0.329634442, 0.290656426, 0.186756642, 0.374459459, 0.373960596, 0.287535659, 0.275592957]
#
# plt.bar([1.5, 3, 4.5, 6, 7.5, 9, 10.5, 12], hardcodedVals, align='center', color=['g', 'g', 'g', 'g', 'r', 'r', 'r', 'r']);
# ax = axes()
# ax.set_xticklabels(vars_to_use)
# ax.set_xticks([1.5, 3, 4.5, 6, 7.5, 9, 10.5, 12])
# for tick in ax.xaxis.get_major_ticks():
#             tick.label.set_fontsize(14)
# plt.xticks(fontsize=14)
# plt.ylabel("Weights")
# title = plt.title("Top LR Feature Weights")
# title.set_fontsize(24)
# hB, = plot([10,10],'g-')
# hR, = plot([10,10],'r-')
# xlim(0, 13)
# ylim(0, .5)
# legend((hB, hR),('Positive', 'Negative'))
# print("saving fig")
# plt.show()
# # savefig('lrfeatweight.png', dpi=3000)
# plt.gcf().clear()

vars_to_use = ['BLOOD\nUREA\nNITROGEN', 'RESPIRATORY\nRATE', 'CREATININE', 'URINE\nOUTPUT', 'SYSTOLIC\nBLOOD\nPRESSURE', 'HEART\nRATE', 'WHITE\nBLOOD\nCELLS', 'PLATELETS']
hardcodedVals = [0.084397627, 0.065827449, 0.056655529, 0.053245396, 0.047173729, 0.045381528, 0.042435394, 0.041639452]

plt.bar([1.5, 3, 4.5, 6, 7.5, 9, 10.5, 12], hardcodedVals, align='center', color=['g', 'g', 'g', 'g', 'g', 'g', 'g', 'g']);
ax = axes()
ax.set_xticklabels(vars_to_use)
ax.set_xticks([1.5, 3, 4.5, 6, 7.5, 9, 10.5, 12])
for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(14)
plt.xticks(fontsize=14)
plt.ylabel("Relative Importance")
title = plt.title("Top RF Feature Importances")
title.set_fontsize(24)
# hB, = plot([10,10],'g-')
# hR, = plot([10,10],'r-')
xlim(0, 10)
ylim(0, .1)
# legend((hB, hR),('Positive', 'Negative'))
print("saving fig")
plt.show()
# savefig('rffeatweight.png', dpi=3000)
