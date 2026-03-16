# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 17:47:58 2023

Load files from intan and run through kilosort

@author: Kelly
"""
import os
# import glob
import numpy as np
import matplotlib.pyplot as plt
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
multirecording = se.BinaryRecordingExtractor(
    r'F:\Spike Sorting\Data\1_RawDats\2025-01-21_Ephys\Day12_T35-39-Probe4\M112_T35-37_noTurn_250201_170835\amplifier.dat',
    30000,64,'int16')
multirecording

#%%
'''splitting two ports'''
multirecording = se.BinaryRecordingExtractor(r'F:\Spike Sorting\Data\1_RawDats\2025-01-21_Ephys\Day12_T35-39-Probe4\M112_T35-37_baseline_250201_173726\amplifier.dat', 30000,128,'int16')
multirecording

channel_ids = np.arange(128)
recording1 = multirecording.channel_slice(channel_ids=channel_ids[:64])
recording2 = multirecording.channel_slice(channel_ids=channel_ids[64:])


#%%
'''plot raw data'''
multirecording = se.BinaryRecordingExtractor(r'F:\Spike Sorting\Data\2_Kilosorted2.5\2024-11-09_Ephys\Day3_T1-3\M110\recording\recording.dat', 30000,64,'int16')
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
dayPath = r'\2023-12-18_Ephys\Day3_T1-6'
mouse = 'M102'
port = 'A'

recording_list = []
sampling_frequency = 30000
path = rawDatPath + dayPath
channel_ids = np.arange(128)

trial_list = []
trial_order= []

for (root, dirs, files) in os.walk(path):
    for dirname in dirs:
        if dirname.split('_')[2] == 'baseline':
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
    
recording_list
multirecording = si.concatenate_recordings(recording_list)

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

#%% plot raw data using matplotlib, need to have recordinglist to plot epochs

path=r'F:\Spike Sorting\Data\2_Kilosorted2.5'+dayPath+'\\'+mouse+'\\recording'
recording = se.BinaryRecordingExtractor(path+r'\recording.dat', 30000,64,'int16')

# Get recording info
sampling_rate = recording.get_sampling_frequency()
duration = recording.get_num_frames() / sampling_rate
n_channels = min(8, recording.get_num_channels())
channel_ids = recording.channel_ids[:n_channels]

print(f"Recording duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")

# Downsample to ~1 sample per second for overview
downsample_factor = int(sampling_rate)  # 30000 -> 1 Hz
chunk_size = 30 * sampling_rate  # Process 30 seconds at a time

# Collect data in a list first (more flexible)
traces_list = []
time_list = []

print("Extracting downsampled data in chunks...")

for start_frame in range(0, recording.get_num_frames(), chunk_size):
    end_frame = min(start_frame + chunk_size, recording.get_num_frames())
    
    # Get chunk
    chunk = recording.get_traces(start_frame=start_frame, end_frame=end_frame, 
                                  channel_ids=channel_ids)
    
    # Downsample this chunk
    chunk_ds = chunk[::downsample_factor, :]
    
    # Create time vector for this chunk
    chunk_frames = np.arange(0, chunk.shape[0], downsample_factor)[:chunk_ds.shape[0]]
    chunk_times = (start_frame + chunk_frames) / sampling_rate
    
    # Append to lists
    traces_list.append(chunk_ds)
    time_list.append(chunk_times)
    
    if start_frame % (300 * sampling_rate) == 0:  # Progress every 5 minutes
        print(f"  Processed {start_frame/sampling_rate:.0f}s / {duration:.0f}s")

# Concatenate all chunks
traces_ds = np.vstack(traces_list)
time_vector = np.concatenate(time_list)

print(f"Plotting {len(time_vector)} points...")

# Plot
fig, ax = plt.subplots(figsize=(24, 10))
offset = 5000

for i in range(n_channels):
    ax.plot(time_vector, traces_ds[:, i] + i * offset, linewidth=0.3, alpha=0.8)

# Mark key timepoints
times = [t.get_num_frames() / sampling_frequency for t in recording_list]

seam_times = []
cumulative_time = 0.0

for t in times[:-1]:          # exclude final endpoint
    cumulative_time += t
    seam_times.append(cumulative_time)
    
for seam in seam_times:
    plt.axvline(seam, color="r", linestyle="--", alpha=0.7)

# Add vertical lines every 1000s for reference
for t in range(0, int(duration), 1000):
    ax.axvline(t, color='gray', linestyle=':', linewidth=0.5, alpha=0.3)

ax.set_xlabel('Time (s)', fontsize=14)
ax.set_ylabel('Channel + offset', fontsize=14)
ax.set_title('Full recording raw traces (1 Hz downsampled)', fontsize=16)
ax.set_xlim(0, duration)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.2)
plt.tight_layout()

plt.savefig(path+r'\fullRecording.png', dpi=150, bbox_inches='tight')
print("Plot saved to full_recording.png")
plt.show()

#%% runs kilosort2.4

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

# sorting_TDC_5 = ss.run_sorter("kilosort2", recording=multirecording, output_folder=r'F:\Spike Sorting\Data\2_Kilosorted2.5\\M110_loweringelectrode\M110_Day10_Turn15_241107_171851\recording')
sorting_TDC_5 = ss.run_sorter("kilosort2", recording=multirecording, output_folder=r'F:\Spike Sorting\Data\2_Kilosorted2.5'+dayPath+'\\'+mouse+'\\recording')
sorting_TDC_5.get_unit_ids()