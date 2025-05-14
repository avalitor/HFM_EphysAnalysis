# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 21:09:45 2024

@author: student
"""
# import pickle
import scipy.io as sio
import numpy as np
# import mat73
import matplotlib.pyplot as plt
# from pylab import *
# from scipy.stats import gamma
# from scipy import io
# from IPython import get_ipython
# get_ipython().run_line_magic('matplotlib', 'qt')
# import pandas as pd
# import seaborn as sns
# import os 
# import glob 
# from tqdm.notebook import tqdm
# import scipy.io as sio
# from elephant import kernels
# from elephant.statistics import instantaneous_rate
# from neo.core import SpikeTrain
# from quantities import ms, s, Hz
#import xlwt
# from xlwt import Workbook
# from collections import defaultdict
# from scipy.signal import find_peaks
# from scipy.ndimage import gaussian_filter1d
# from pathlib import Path

#%% Load CellExplorer info
# waveform = sio.loadmat("J:\Spike_train_analysis\mice multi\A1 dec 10\\recording.cell_metrics.cellinfo.mat")['cell_metrics']['waveforms'][0][0]['filt'][0][0]
# waveform[0].flatten().shape

# labels = sio.loadmat("J:\Spike_train_analysis\mice multi\A1, Dec 10\\recorDing.cell_metrics.cellinfo.mat")['cell_metrics']['putativeCellType'][0][0][0]
# labels

# cell_id = sio.loadmat("J:\Spike_train_analysis\mice multi\A1, Dec 10\\recording.cell_metrics.cellinfo.mat")['cell_metrics'][0][0][0]
# cell_id

# cell_metrics = sio.loadmat(os.path.join(folder_path, "recording.cell_metrics.cellinfo.mat"))['cell_metrics']
# cell_metrics[0][0][0][0]

cell_metrics = sio.loadmat(r'H:\Phy\2024-02-15_M105\Day6_T14-19\M105\recording\recording.cell_metrics.cellinfo.mat')['cell_metrics']
neuron = 5
waveform = cell_metrics['waveforms'][0][0]['filt'][0][0]
labels = cell_metrics['putativeCellType'][0][0][0]
# cell_id = cell_metrics[0][0][0][0]

plt.plot(waveform[0][neuron][0], label='waveform')
#plot 2nd derivative
plt.plot(np.diff(waveform[0][neuron][0]), label='2nd derivative')
plt.title(f'Cell {neuron}')
plt.legend()
print(labels[neuron][0])


