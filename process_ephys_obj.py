# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 18:28:52 2023

create raster plot

@author: Kelly
"""

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
import pickle
import lib_ephys_obj as elib
import modules.lib_process_data_to_mat as plib

#%%
epoch = 2
path_m = r'H:\Phy\2023-12-18_M103\Day5_T13-18\recording'

TTL_in = sio.loadmat(path_m + '\TTL_in-T13-15.mat')['v'] #TTL Pulse

spikes = sio.loadmat(path_m + '\\recording.spikes.cellinfo')['spikes'] #loads the spikes!
cell_metrics = sio.loadmat(path_m + '\\recording.cell_metrics.cellinfo')['cell_metrics'][0]
cell_labels  = cell_metrics['putativeCellType'][0][0] #gets lables of each neuron
cell_firingRate = cell_metrics['firingRate'][0][0] #gets lables of each neuron
session = sio.loadmat(path_m + '\\recording.session.mat')['session']
interval = [session['epochs'][0][0][0][1][0][0][1][0][0], session['epochs'][0][0][0][1][0][0][2][0][0]] #load epoch array

#%%
exp = '2023-12-18'
mouse = 103
trials = 'T13-15'

edata = elib.EphysEpoch().Load(exp, mouse, trials)
edata.firingRate = cell_firingRate
edata.Update(exp, mouse, trials)

#%%
def fix_spike_formats(Spike_sample_raw): #converts input into numpy array
    Spike_sample = [] #create empty list
    for i in range(len(Spike_sample_raw)):
        Spike_sample.extend(Spike_sample_raw[i]) #add each item of the input to list
    return np.array(Spike_sample) #convert to numpy array

# def take_behav_out(spikes,interval,sr):
#     spike_train = []
#     for st in range(len(spikes['ts'][0][0][0])): #iterates over all the neurons (the same as the cell lables)
#         spike_sample = fix_spike_formats(spikes['ts'][0][0][0][st])
#         inter = np.where((spike_sample >= interval[0]*sr ) & (spike_sample <= interval[1]*sr))
#         assert len(spike_sample) > len(spike_sample[inter])
#         spike_train.append(spike_sample[inter] - interval[0]*sr)
        
#     return spike_train


def take_first_trials_out(spikes, interval, sr): #gets spikes from first epoch
    spike_train = []
    for st in range(len(spikes['ts'][0][0][0])):
        spike_sample = fix_spike_formats(spikes['ts'][0][0][0][st])
        inter = np.where(spike_sample <= interval[0]*sr) #get first interval
        assert len(spike_sample) > len(spike_sample[inter])
        spike_train.append(spike_sample[inter])
    return spike_train

def take_last_trials_out(spikes, interval, sr): #get spikes from last epoch
    spike_train = []
    for st in range(len(spikes['ts'][0][0][0])):
        spike_sample = fix_spike_formats(spikes['ts'][0][0][0][st])
        inter = np.where(spike_sample >= interval[-1]*sr) #get start of last interval
        assert len(spike_sample) > len(spike_sample[inter])
        spike_train.append(spike_sample[inter] - interval[-1]*sr)
    return spike_train

def TTL_extr(input): #for digital TTL
    start_tray = []
    end_tray   = []
    output = []
    for j,i in enumerate(input):
        if i == 1 and input[j-1]==0:
            start_tray.append(j)
        if i == 1 and input[j+1]==0:
            end_tray.append(j)
            
    for i in range(len(start_tray)):
        output.append([start_tray[i], end_tray[i]])
    return output

def extract_analog_TTL(raw_TTL): #for analog TTL
    start = []
    end = []
    output = []
    for i, v in enumerate(raw_TTL[0,:]):
        if v > 0 and raw_TTL[0,i-1] <=0: #threshold is set at zero but this can be changed based on TTL preview
            start.append(i)
        elif v <= 0 and raw_TTL[0,i-1] >0:
            end.append(i)
    for i in range(len(start)):
        if start[i] != 0: #skips if the TTL is on at the beginning
            output.append(np.array([start[i], end[i]]))
    return np.array(output)

#%%
sr = 30000

if epoch == 1: 
    spike_train = take_first_trials_out(spikes,interval,sr) #get the first epoch spikes
    TTL = extract_analog_TTL(TTL_in)

if epoch == 2: 
    spike_train = take_last_trials_out(spikes,interval,sr) #get the last epoch spikes
    TTL = extract_analog_TTL(TTL_in)
    
t_spike_train = [st/sr for st in spike_train]
TTL_time = TTL/sr #gets TTL for first epoch
#%%
'''Preview the raw TTL'''
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 4))
plt.plot(TTL_in[0,:])
plt.show()

#%%
# spike_sample = spike_train[3]/sr #choose a particular neuron

# '''Raster plot of single neuron'''
# fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 8))
# # create a horizontal raster plot
# ax.eventplot(spike_sample, colors='k', lineoffsets=0, linelengths=5)
# ax.eventplot(TTL_time, colors='r', lineoffsets=0, linelengths=7)
# plt.show()

#%%
'''plot all the neurons in a raster plot'''

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 8))
lineoffsets1 = range(len(spike_train))
ax.eventplot(t_spike_train,lineoffsets=lineoffsets1)
plt.vlines(TTL_time, min(lineoffsets1), max(lineoffsets1), color='r')
# plt.plot(spike_all, (tri) * np.ones(spike_sample.shape),'|')
plt.show()
    
#%% 

trial_name = 'T13-15'    
Epoch = elib.EphysEpoch(trial=trial_name, t_TTL = TTL_time, t_spike_train = t_spike_train, 
                        spike_labels=cell_labels, sample_rate=sr, firingRate = cell_firingRate)

#save the processed data
path_data = r'F:\Spike Sorting\Data\3_Raster\2023-12-18_M103\\'
file=open(path_data+f'{trial_name}.EphysEpoch', 'wb')
pickle.dump(Epoch, file)
file.close()

# test if it opens
# file = open(path_data+r'\Habit1.EphysEpoch', 'rb')
# test = pickle.load(file)
# file.close()

#%% add TTL data to Ethovision mat files, only done once per mouse per experiment
file = open(path_data+r'\offset_dict_M103.pydict', 'rb')
vid_offset = pickle.load(file)
file.close()

for trial in vid_offset.keys():
    tdata = plib.TrialData()
    tdata.Load('2023-12-18',103,trial)
    
    if hasattr(tdata, 'time_ttl'): #checks if ttl synch has already been calculated
        pass
    else:
        tdata.time_ttl = tdata.time - vid_offset[tdata.trial] #synch video with TTL
        tdata.Update()
    
