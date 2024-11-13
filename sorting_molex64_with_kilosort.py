# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 17:47:58 2023

Load files from intan and run through kilosort

@author: Kelly
"""
import os
# import glob
import numpy as np
# import matplotlib.pyplot as plt
from pprint import pprint

import spikeinterface as si
import spikeinterface.extractors as se
# import spikeinterface.exporters as ex
# import spikeinterface.widgets as sw
import spikeinterface.sorters as ss
# from spikeinterface.exporters import export_to_phy

# from probeinterface import Probe
# import probeinterface.io as pio
from probeinterface import get_probe
# from probeinterface.plotting import plot_probe
# from probeinterface import generate_linear_probe, generate_multi_shank

from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt') #display figures as separate windows
#ss.Kilosort2_5Sorter.set_kilosort2_5_path(r'C:\Users\fmora103\Downloads\Kilosort-2.5')
ss.Kilosort2Sorter.set_kilosort2_path(r'F:\Spike Sorting\toolboxes\Kilosort-2.0\Kilosort-2.0')

pprint(ss.installed_sorters()) #confirms that the sorter is loaded

#%%
'''load a single file'''
multirecording = se.BinaryRecordingExtractor(r'F:\Spike Sorting\Data\1_RawDats\M110_loweringelectrode\M110_Day10_Turn15_241107_171851\amplifier.dat', 30000,64,'int16')
multirecording

#%%
'''splitting two ports'''
multirecording = se.BinaryRecordingExtractor(r'F:\Spike Sorting\Data\1_RawDats\2023-10-16_M101\Day6_Probe-T19-20\N12 Probe1_Base NoTurn_231021_134626\amplifier.dat', 30000,128,'int16')
multirecording

channel_ids = np.arange(128)
recording1 = multirecording.channel_slice(channel_ids=channel_ids[:64])
recording2 = multirecording.channel_slice(channel_ids=channel_ids[64:])

#%%
'''plot raw data'''
fs = 30000
rec_f = si.bandpass_filter(multirecording, freq_min=300, freq_max=10000)
w_ts = sw.plot_timeseries(rec_f, time_range=(300, 400))

#%%
'''concat a day's files together'''
recording_list = []
sampling_frequency = 30000
num_channels = 64
rawDatPath = r'F:\Spike Sorting\Data\1_RawDats'
dayPath = r'\2024-02-15_M105\Day4_T1-7'
path = rawDatPath + dayPath
c=0
for (root, dirs, files) in os.walk(path):
    if len(files) > 1:
        print(root)
        if files[0]== 'amplifier.dat':
            print(files[0])
            recording_list.append(se.BinaryRecordingExtractor(root +'\\'+ files[0], sampling_frequency,num_channels,'int16'))
            c+=1
multirecording = si.concatenate_recordings(recording_list)
print(c)
multirecording 

#%%
'''concat files based on mouse and split ports'''
rawDatPath = r'F:\Spike Sorting\Data\1_RawDats'
dayPath = r'\2024-02-12_M104\Day6_T13-18'
mouse = 'M105'
port = 'B'

recording_list = []
sampling_frequency = 30000
path = rawDatPath + dayPath
channel_ids = np.arange(128)

trial_list = []
trial_order= []

for (root, dirs, files) in os.walk(path):
    for dirname in dirs:
        if dirname.split('_')[1] == 'baseline':
            print(f'Baseline: {dirname}')
            baseline_recording = se.BinaryRecordingExtractor(os.path.join(path, dirname, 'amplifier.dat'), sampling_frequency,128,'int16')
            if port == 'A':
                baseline_recording = baseline_recording.channel_slice(channel_ids=channel_ids[:64])
            elif port == 'B':
                baseline_recording = baseline_recording.channel_slice(channel_ids=channel_ids[64:], renamed_channel_ids= channel_ids[:64])
        elif dirname.startswith(mouse):
            print(f'Trials: {dirname}')
            trial_list.append(se.BinaryRecordingExtractor(os.path.join(path, dirname, 'amplifier.dat'), sampling_frequency,64,'int16'))
            trial_order.append(dirname.split('_')[-1])
        
if len(trial_order) < 2:
    recording_list = [trial_list[0], baseline_recording]
elif trial_order[0] > trial_order[1]: #if the order is backwards
    recording_list = [trial_list[1], baseline_recording, trial_list[0]]
else:
    recording_list = [trial_list[0], baseline_recording, trial_list[1]]
    
# recording_list


#%% Manual file selection
channel_ids = np.arange(128)
sampling_frequency = 30000

# path1 = r'F:\Spike Sorting\Data\1_RawDats\2024-02-12_M104\Day6_T13-18\M104_T13-15_240217_140548\amplifier.dat'
# path_base = r'F:\Spike Sorting\Data\1_RawDats\2024-02-15_M105\Day3_Habit1-Habit2\M104-105_baseline_afterT15-Habit1_240217_151422\amplifier.dat'
# path2 = r'F:\Spike Sorting\Data\1_RawDats\2024-02-12_M104\Day6_T13-18\M104_T16-18_240217_193947\amplifier.dat'

path1 = r'F:\Spike Sorting\Data\1_RawDats\2024-02-15_M105\Day4_T1-7\M105_T1-3_240218_144827\amplifier.dat'
path_base = r'F:\Spike Sorting\Data\1_RawDats\2024-02-15_M105\Day4_T1-7\M105_baseline_240218_151918\amplifier.dat'
path2 = r'F:\Spike Sorting\Data\1_RawDats\2024-02-15_M105\Day4_T1-7\M105_T4-6_240218_183212\amplifier.dat'
path3 = r'F:\Spike Sorting\Data\1_RawDats\2024-02-15_M105\Day4_T1-7\M105_T7probe_240218_185448\amplifier.dat'

trial_list1 = se.BinaryRecordingExtractor(path1,sampling_frequency,64,'int16')
baseline_recording = se.BinaryRecordingExtractor(path_base,sampling_frequency,128,'int16')
baseline_recording = baseline_recording.channel_slice(channel_ids=channel_ids[64:], renamed_channel_ids= channel_ids[:64])
trial_list2 = se.BinaryRecordingExtractor(path2,sampling_frequency,64,'int16')
trial3 = se.BinaryRecordingExtractor(path3,sampling_frequency,64,'int16')

recording_list = [trial_list1, baseline_recording, trial_list2, trial3]

multirecording = si.concatenate_recordings(recording_list)
multirecording

#%%
'''get epochs from files, need to run the previous cell first'''
times = [round(t.get_num_frames()/sampling_frequency, 3) for t in recording_list]
time_ranges = []
cumulative_time = 0

for i, point in enumerate(times):
        epoch_start = cumulative_time
        cumulative_time += point
        epoch_end = cumulative_time

        time_ranges.append(f"Epoch {i + 1}: {epoch_start} - {epoch_end}")
print(time_ranges)

#%%

#MATLAB ch = [42, 40, 39, 38, 36, 35, 34, 33, 30, 31, 29, 27, 26, 25, 23, 21,47, 46, 45, 44, 43, 41, 37, 32, 28, 24, 22, 19, 20, 18, 17, 15,
# 56, 54, 55, 53, 52, 51, 50, 49, 48, 16, 14, 13, 12, 10, 11, 9,64, 63, 62, 61, 60, 59, 58, 57, 8, 7, 6, 5, 4, 3, 2, 1]
ch = [41,39,38,37,35,34,33,32,29,30,28,26,25,24,22,20,46,45,44,43,42,40,36,31,27,23
      ,21,18,19,17,16,14,55,53,54,52,51,50,49,48,47,15,13,12,11,9,10,8,63,62,61,60,59,58,57,56,7,6,5,4,3,2,1,0]
manufacturer = 'cambridgeneurotech'
probe_name = 'ASSY-236-H10'
probe = get_probe(manufacturer, probe_name)
probe.set_device_channel_indices(ch)
multirecording = multirecording.set_probe(probe)
# print(probe)
# plot_probe(probe, with_contact_id=False, with_device_index=True)


'''change parameters'''
default_TDC_params = ss.Kilosort2Sorter.default_params()
#default_TDC_params['detect_threshold']= 6
default_TDC_params['projection_threshold'] = [9, 9]
print(default_TDC_params)

sorting_TDC_5 = ss.run_sorter("kilosort2", recording=multirecording, output_folder=r'F:\Spike Sorting\Data\2_Kilosorted2.5\\M110_loweringelectrode\M110_Day10_Turn15_241107_171851\recording')
# sorting_TDC_5 = ss.run_sorter("kilosort2", recording=multirecording, output_folder=r'F:\Spike Sorting\Data\2_Kilosorted2.5'+dayPath+'\\'+mouse+'\\recording')
sorting_TDC_5.get_unit_ids()