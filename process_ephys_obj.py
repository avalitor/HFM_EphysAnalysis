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
epoch = 3
path_m = r'H:\Phy\2024-02-15_M105\Day7_Probe-T20-24\M105\recording'

# TTL_in = sio.loadmat(path_m + '\TTL_in-T22-24.mat')['v'] #TTL Pulse

spikes = sio.loadmat(path_m + '\\recording.spikes.cellinfo')['spikes'] #loads the spikes!
cell_metrics = sio.loadmat(path_m + '\\recording.cell_metrics.cellinfo')['cell_metrics'][0]
cell_labels  = cell_metrics['putativeCellType'][0][0] #gets lables of each neuron
cell_firingRate = cell_metrics['firingRate'][0][0] #gets lables of each neuron
session = sio.loadmat(path_m + '\\recording.session.mat')['session']
interval = [session['epochs'][0][0][0][1][0][0][1][0][0], session['epochs'][0][0][0][1][0][0][2][0][0]] #load epoch array

#%% update existing files with firing rate
# exp = '2023-12-18'
# mouse = 103
# trials = 'T13-15'

# edata = elib.EphysEpoch().Load(exp, mouse, trials)
# edata.firingRate = cell_firingRate
# edata.Update(exp, mouse, trials)

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

def take_last_trials_out_trial7(spikes, interval, sr): #get spikes from last epoch
    spike_train = []
    for st in range(len(spikes['ts'][0][0][0])):
        spike_sample = fix_spike_formats(spikes['ts'][0][0][0][st])
        inter = np.where(spike_sample >= 8566.285*sr) #get start of last interval
        assert len(spike_sample) > len(spike_sample[inter])
        spike_train.append(spike_sample[inter] - 8566.285*sr) #8566.285*sr
    return spike_train

def take_baseline_out(spikes, interval, sr): #get spikes from baseline epoch, assuming it is the second epoch
    spike_train = []
    for st in range(len(spikes['ts'][0][0][0])):
        spike_sample = fix_spike_formats(spikes['ts'][0][0][0][st])
        inter = np.where((spike_sample  >= interval[0]*sr) & (spike_sample  <= interval[1]*sr)) #get second interval
        assert len(spike_sample) > len(spike_sample[inter])
        spike_train.append(spike_sample[inter] - interval[0]*sr)
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


sr = 30000

if epoch == 1: 
    spike_train = take_first_trials_out(spikes,interval,sr) #get the first epoch spikes
    TTL = extract_analog_TTL(TTL_in)

if epoch == 2: 
    spike_train = take_last_trials_out(spikes,interval,sr) #get the last epoch spikes
    TTL = extract_analog_TTL(TTL_in)
    
if epoch == 3:
    spike_train = take_baseline_out(spikes,interval,sr) #get the second epoch spikes
    
t_spike_train = [st/sr for st in spike_train]
# TTL_time = TTL/sr #gets TTL for first epoch
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
# plt.vlines(TTL_time, min(lineoffsets1), max(lineoffsets1), color='r')
# plt.plot(spike_all, (tri) * np.ones(spike_sample.shape),'|')
plt.show()

#%% Get firing rate of all cells during the epoch
epoch_time = interval[1] - interval[0]
fr_list = []
for c in range(len(t_spike_train)):
    fr = len(t_spike_train[c])/epoch_time
    fr_list.append(fr)
    
    
#%% 
trial_name = 'T22-24'    
Epoch = elib.EphysEpoch(trial=trial_name, t_TTL = TTL_time, t_spike_train = t_spike_train, 
                        spike_labels=cell_labels, sample_rate=sr, firingRate = cell_firingRate)

#save the processed data
path_data = r'F:\Spike Sorting\Data\3_Raster\2024-02-15_M105\\'
file=open(path_data+f'{trial_name}.EphysEpoch', 'wb')
pickle.dump(Epoch, file)
file.close()

# test if it opens
# file = open(path_data+r'\Habit1.EphysEpoch', 'rb')
# test = pickle.load(file)
# file.close()

#%% 
exp = '2024-02-15'
mouse = "105"
trial = '24'

path_data = rf'F:\Spike Sorting\Data\3_Raster\{exp}_M{mouse}\\'
file = open(path_data+rf'\offset_dict_M{mouse}.pydict', 'rb')
vid_offset = pickle.load(file)
file.close()

# add TTL time to mat files. only need to be done once per experiment
# for trial in vid_offset.keys():
#     tedata = elib.EphysTrial()
#     tedata.Load('2023-12-18', 103, trial)
    
#     if hasattr(tdata, 'time_ttl'): #checks if ttl synch has already been calculated
#         pass
#     else:
#         tdata.time_ttl = tdata.time - vid_offset[tdata.trial] #synch video with TTL
#         tdata.Update()
    
# create EphysTrial Mat files
t = plib.TrialData()
t.Load(exp, mouse, trial)

edata = elib.EphysEpoch().Load(exp, mouse, trial_name)
tr = 2 #is it trial 0, 1, 2 in the epoch?

spike_sample_all = [edata.t_spike_train[i] - edata.t_TTL[tr][0] for i in range(len(edata.t_spike_train))] #synch all neuorns to ttl

def crop_trains_to_trial(spike_trains, trialdata): #given synched spike trains & trial
    spike_sample_trialcrop = []
    for n in spike_trains:
        k_spike_event = np.where((n > trialdata.time_ttl[0]) & (n < trialdata.time_ttl[-1])) #get spikes in between trial start and end time
        spike_sample_trialcrop.append(n[k_spike_event]) 
    return spike_sample_trialcrop

e = elib.EphysTrial()
e.exp = exp
e.protocol_name=t.protocol_name
e.protocol_description=t.protocol_description
e.eth_file = t.eth_file
e.bkgd_img=t.bkgd_img
e.filename = ('hfmE_%s_M%s_%s.mat' %(t.exp, t.mouse_number, t.trial))

e.img_extent=t.img_extent
e.experimenter=t.experimenter
e.mouse_number=t.mouse_number
e.mouse_sex=t.mouse_sex
e.day=t.day
e.trial=t.trial
e.entrance=t.entrance
e.target=t.target
if hasattr(t, 'target_reverse'): e.target_reverse=t.target_reverse
e.time=t.time
e.r_nose=t.r_nose
e.r_center=t.r_center
e.r_tail=t.r_tail

if hasattr(t, 'velocity'): #checks if file has velocity info
    e.velocity = t.velocity
if hasattr(t, 'head_direction'): #checks if file has HD info
    e.head_direction = t.head_direction
e.heading = t.heading
e.r_arena_holes = t.r_arena_holes
e.arena_circle = t.arena_circle
e.k_reward = t.k_reward
e.k_hole_checks = t.k_hole_checks

e.time_ttl = t.time - vid_offset[t.trial] #synch video with TTL

e.t_spikeTrains = crop_trains_to_trial(spike_sample_all, e)

test = crop_trains_to_trial(spike_sample_all, e)

# e.spikeLabels = cell_labels
# e.firingRate = cell_firingRate
# e.sampleRate = sr

e.cellLabels = np.asarray(edata.spike_labels)
e.firingRate = np.asarray(edata.firingRate)
e.sampleRate = edata.sample_rate

e.Store()
#%%
exp = '2024-02-15'
mouse = "105"
trial = '24'

t = plib.TrialData()
t.Load(exp, mouse, trial)

e2 = elib.EphysTrial()
e2.Load(exp, mouse, trial)