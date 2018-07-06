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
axis_label = 16
places = [.3, .6, .9, 1.2, 1.5, 1.8, 2.1, 2.4]
# vars_to_use = ['BLOOD\nUREA\nNITROGEN', 'RESP.\nRATE', 'HEMATO-\nCRIT', 'HEART\nRATE', 'GLASCOW\nCOMA\nSCALE', 'HEMO-\nGLOBIN', 'PARTIAL\nPRESS.\nOXYGEN', 'URINE\nOUTPUT']
# hardcodedVals = [0.440755928, 0.329634442, 0.290656426, 0.186756642, 0.374459459, 0.373960596, 0.287535659, 0.275592957]
#
# plt.bar(places, hardcodedVals, width=.2, align='center', color=['g', 'g', 'g', 'g', 'r', 'r', 'r', 'r']);
# ax = axes()
# ax.set_xticklabels(vars_to_use)
# ax.set_xticks(places)
# for tick in ax.xaxis.get_major_ticks():
#             tick.label.set_fontsize(axis_label)
# plt.xticks(fontsize=axis_label)
# plt.ylabel("Weights",  size = axis_label)
# title = plt.title("Top LR Feature Weights")
# title.set_fontsize(24)
# hB, = plot([10,10],'g-')
# hR, = plot([10,10],'r-')
# xlim(0.15, 2.55)
# ylim(0, .5)
# legend((hB, hR),('Positive', 'Negative'), prop={'size': 15})
# print("saving fig")
# plt.show()
# # savefig('lrfeatweight.png', dpi=3000)
# plt.gcf().clear()
#
# vars_to_use = ['BLOOD\nUREA\nNITROGEN', 'RESP.\nRATE', 'CREAT-\nININE', 'URINE\nOUT-\nPUT', 'SYSTOLIC\nBLOOD\nPRESSURE', 'HEART\nRATE', 'WHITE\nBLOOD\nCELLS', 'PLATE-\nLETS']
# hardcodedVals = [0.084397627, 0.065827449, 0.056655529, 0.053245396, 0.047173729, 0.045381528, 0.042435394, 0.041639452]
#
# plt.bar(places, hardcodedVals, align='center',  width=.2, color=['g', 'g', 'g', 'g', 'g', 'g', 'g', 'g']);
# ax = axes()
# ax.set_xticklabels(vars_to_use)
# ax.set_xticks(places)
# for tick in ax.xaxis.get_major_ticks():
#             tick.label.set_fontsize(axis_label)
# plt.xticks(fontsize=axis_label)
# plt.ylabel("Relative Importance", size=axis_label)
# title = plt.title("Top RF Feature Importances")
# title.set_fontsize(24)
# # hB, = plot([10,10],'g-')
# # hR, = plot([10,10],'r-')
# xlim(0.15, 2.55)
# ylim(0, .1)
# # legend((hB, hR),('Positive', 'Negative'))
# print("saving fig")
# plt.show()
# # savefig('rffeatweight.png', dpi=3000)
# plt.gcf().clear()



vars_to_use = ["Explicit\nSepsis\n(Septicemia)", "Organ\nFailure + \n Infection", " Mech.\nVent. +\nInfection", "Total\nSepsis"]
hardcodedVals=[4085, 12180, 8174, 15254]
plt.bar([1,2,3,4], hardcodedVals, color=['g', 'g', 'g', 'r'])
ax = axes()
ax.set_xticklabels(vars_to_use, size=15)
ax.set_xticks([1,2,3,4])
plt.ylabel("Num. Admissions", size=15)
plt.title("Preliminary Analysis of Dataset", size=20)


plt.show()
