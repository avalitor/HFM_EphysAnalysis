# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 00:12:01 2024

Make aligned raster plots

@author: student
"""
import matplotlib.pyplot as plt
import numpy as np
import lib_ephys_obj as elib
import modules.lib_process_data_to_mat as plib
import config as cg


def get_trial_event_crop(spike_trains, trialdata): #given synched spike trains & trial
    spike_sample_eventcrop = []
    t_reward = trialdata.time_ttl[trialdata.k_reward]
    for n in spike_trains:
        k_spike_event = np.where((n > t_reward - 10) & (n < t_reward + 10)) #get 10 seconds before and after event
        spike_sample_eventcrop.append(n[k_spike_event]-t_reward) #align event to 0 on the x axis
    return spike_sample_eventcrop

def get_intertrial_event_crop(spike_train, event_list, tedata):
    
    spike_sample_eventcrop = []
    for e in event_list:
        k_spike_event = np.where((spike_train > tedata.time_ttl[e] - 30) & (spike_train < tedata.time_ttl[e] + 30)) #get 30 seconds before and after event
        spike_sample_eventcrop.append(spike_train[k_spike_event]-tedata.time_ttl[e]) #align event to 0 on the x axis
    return spike_sample_eventcrop

def get_epoch_event_crop(edata, tdata_list, neuron='all'):
    spike_sample_eventcrop = []
    for tr in range(3): #iterate through the 3 trials
        spike_sample_all = [edata.t_spike_train[i] - edata.t_TTL[tr][0] for i in range(len(edata.t_spike_train))] #synch all neuorns to ttl
        # k_cellID = np.where(edata.spike_labels == 'Pyramidal Cell')[0]
        # spike_sample_all = [spike_sample_all[i] for i in k_cellID]
        spike_sample_eventcrop = spike_sample_eventcrop + get_trial_event_crop(spike_sample_all, tdata_list[tr])

    return spike_sample_eventcrop

def get_single_cell_trial_spikes(edata, tdata_list, neuron_no):
    spike_sample_eventcrop = []
    for tr in range(3): #iterate through the 3 trials
        spike_sample = edata.t_spike_train[neuron_no] - edata.t_TTL[tr][0] #synch single neuron to ttl
        t_reward = tdata_list[tr].time_ttl[tdata_list[tr].k_reward]
        spike_sample_eventcrop.append(spike_sample-t_reward) #align event to 0 on the x axis
        # spike_sample_eventcrop = spike_sample_eventcrop + get_trial_event_crop(spike_sample, tdata_list[tr])
    return spike_sample_eventcrop

#%%
'''get spike sample of single trial'''
edata = elib.EphysEpoch().Load('2023-12-18', '102','T13-15')
(tdata := plib.TrialData()).Load('2023-12-18', '102', '18')

tr = 0 #is it trial 0, 1, 2 in the epoch?
spike_sample_all = [edata.t_spike_train[i] - edata.t_TTL[tr][0] for i in range(len(edata.t_spike_train))] #synch all neuorns to ttl
spike_sample_eventcrop = get_trial_event_crop(spike_sample_all, tdata)

#%%
'''get spike samples of multi trials'''

exp = '2023-12-18'
mouse = '102'

edata_dict = {
    'T13-15': elib.EphysEpoch().Load(exp, mouse,'T13-15'),
    'T16-18': elib.EphysEpoch().Load(exp, mouse,'T16-18'),
    }

tdata = [plib.TrialData() for i in range(6)]
tdata[0].Load(exp, mouse, '13')
tdata[1].Load(exp, mouse, '14')
tdata[2].Load(exp, mouse, '15')
tdata[3].Load(exp, mouse, '16')
tdata[4].Load(exp, mouse, '17')
tdata[5].Load(exp, mouse, '18')

# edata_dict = {
#     'T13-15': EphysEpoch().Load('2023-10-16', '101','T13-15'),
#     'T16-18': EphysEpoch().Load('2023-10-16', '101','T16-18'),
#     }

# tdata = [plib.TrialData() for i in range(6)]
# tdata[0].Load('2023-10-16', '101', '13')
# tdata[1].Load('2023-10-16', '101', '14')
# tdata[2].Load('2023-10-16', '101', '15')
# tdata[3].Load('2023-10-16', '101', '16')
# tdata[4].Load('2023-10-16', '101', '17')
# tdata[5].Load('2023-10-16', '101', '18')
#%%
'''get spike sample of single neuron on a single trial'''
exp = '2023-12-18'
mouse = '102'
edata = elib.EphysEpoch().Load(exp, mouse,'T16-18')
(tdata := plib.TrialData()).Load(exp, mouse, '18')

tr = 2 #is it trial 0, 1, 2 in the epoch?
neuron = 14
print(edata.spike_labels[neuron])
print(edata.firingRate[neuron])
spike_sample = edata.t_spike_train[neuron] - edata.t_TTL[tr][0]

(e := elib.EphysTrial()).Load(exp, mouse, '18')
t_hole_checks = np.sort(e.time_ttl[tdata.k_hole_checks[:,1]])

# test = np.where((tdata.k_hole_checks[:,1] > 0) & (tdata.k_hole_checks[:,1] < tdata.k_reward)) 

'''plot raster of whole trial'''
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 2))
lineoffsets1 = range(1)
ax.eventplot(spike_sample, lineoffsets=lineoffsets1, color='#5f0f40')
# ax.eventplot(t_hole_checks, lineoffsets=lineoffsets1, color = '#5f0f40', linestyles='--')
# plt.plot(t_hole_checks , np.zeros(t_hole_checks.shape),'o', color = '#F18701')

plt.vlines(e.time_ttl[tdata.k_reward], min(lineoffsets1)-1, max(lineoffsets1)+1, color='g') #when did mouse find reward
t_start = e.time_ttl[np.isfinite(tdata.r_center)[:,0]][0] #returns the first non-nan coordinate
plt.vlines(t_start, min(lineoffsets1)-1, max(lineoffsets1)+1, color='k') #when did trial start
# plt.vlines(tdata.time_ttl[923], min(lineoffsets1)-1, max(lineoffsets1)+1, color='b') #custom


# neuron_no = len(edata_dict['T16-18'].t_spike_train)
# plt.hlines([neuron_no, neuron_no*2, neuron_no*3, neuron_no*4, neuron_no*5], -10, 10, colors='k')
# pad = neuron_no/2
# plt.yticks([0+pad, neuron_no+pad, neuron_no*2+pad, neuron_no*3+pad, neuron_no*4+pad, neuron_no*5+pad], 
#            ['Trial 13', 'Trial 14', 'Trial 15', 'Trial 16', 'Trial 17', 'Trial 18'], fontsize=18)

plt.xticks(fontsize=18)
plt.xlabel('Time (s)' , fontsize=20)
# plt.ylabel('Neurons', fontsize=20)
plt.title(f'Cell {neuron}')
ax.set_yticks([])
ax.margins(0)
plt.xlim((t_start-10,e.time_ttl[tdata.k_reward]+10))

# path_data = cg.ROOT_DIR
# plt.savefig(path_data+f'/figs/Raster_M{tdata.mouse_number}_T{tdata.trial}_Cell{neuron}.png', dpi=600, bbox_inches='tight', pad_inches = 0)

plt.show()

#%%
'''NEW MAT FILE get spike sample of single neuron on a single trial'''
exp = '2024-02-15'
mouse = '105'
trial = '19'
neuron = 13
edata = elib.EphysTrial.Load(exp, mouse,trial)

print(edata.cellLabels[neuron])
print(edata.firingRate[neuron])

t_start = edata.time_ttl[np.isfinite(edata.r_center)[:,0]][0] #returns the first non-nan coordinate
t_reward = edata.time_ttl[edata.k_reward]-t_start #this is relative to a t_start=0

spike_sample = edata.t_spikeTrains[neuron]-t_start

# test = np.where((edata.k_hole_checks[:,1] > 0) & (edata.k_hole_checks[:,1] < edata.k_reward)) 

'''plot raster of whole trial'''
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 2))
lineoffsets1 = range(1)

ax.eventplot(spike_sample, lineoffsets=lineoffsets1, color='#5f0f40')

#not callebrated to t_start yet
# t_hole_checks = np.sort(edata.time_ttl[edata.k_hole_checks[:,1]])
# plt.plot(t_hole_checks , np.zeros(t_hole_checks.shape),'o', color = '#F18701')

plt.vlines(0, min(lineoffsets1)-1, max(lineoffsets1)+1, color='k') #when did trial start
plt.vlines(t_reward, min(lineoffsets1)-1, max(lineoffsets1)+1, color='g') #when did mouse find reward

# plt.vlines(edata.time_ttl[923], min(lineoffsets1)-1, max(lineoffsets1)+1, color='b') #custom marker

# neuron_no = len(edata_dict['T16-18'].t_spike_train)
# plt.hlines([neuron_no, neuron_no*2, neuron_no*3, neuron_no*4, neuron_no*5], -10, 10, colors='k')
# pad = neuron_no/2
# plt.yticks([0+pad, neuron_no+pad, neuron_no*2+pad, neuron_no*3+pad, neuron_no*4+pad, neuron_no*5+pad], 
#            ['Trial 13', 'Trial 14', 'Trial 15', 'Trial 16', 'Trial 17', 'Trial 18'], fontsize=18)

plt.xticks(fontsize=18)
plt.xlabel('Time (s)' , fontsize=20)
# plt.ylabel('Neurons', fontsize=20)
plt.title(f'Cell {neuron}')
ax.set_yticks([])
ax.margins(0)
plt.xlim((-20, t_reward+10))

# path_data = rf'F:\Spike Sorting\Data\3_Raster\{exp}_M{mouse}'
# plt.savefig(path_data+f'/figs/Raster_M{edata.mouse_number}_T{edata.trial}_Cell{neuron}.png', dpi=600, bbox_inches='tight', pad_inches = 0)

plt.show()

#%% Count Firing rate before and after events
time_interval = 20
# event = edata.time_ttl[21937]-t_start #event at a certain index
event = t_reward
numSpike_pre = np.sum((spike_sample >= event-time_interval) & (spike_sample <= event))
numSpike_post = np.sum((spike_sample >= event) & (spike_sample <= event+time_interval))
print(f"Firing rate pre reward is {numSpike_pre/time_interval} Hz")
print(f"Firing rate post reward is {numSpike_post/time_interval} Hz")

#%% Plot ISI and ISIH
ISI = np.diff(spike_sample) #calculate ISI

#show raw ISI
plt.plot(ISI)
plt.title("Raw ISI")
plt.show()

#show histogram
count, bins, ignored = plt.hist(ISI, 40, density=True, log=True)
plt.title("ISI Histogram")
plt.show() #if it is a poisson process, graph is exponential

#%% Plot firing rate histogram over trajectory
bin_width = 0.1
min_time = t_start
# max_time = t_reward+10
max_time = np.max(spike_sample)
bins = np.arange(min_time, max_time + bin_width, bin_width)

# Plot the histogram
plt.figure(figsize=(8, 5))
plt.hist(spike_sample, bins=bins, edgecolor='black', alpha=0.75)
plt.title('Firing Rate Histogram')
plt.xlabel('Time (s)')
plt.ylabel('Spike Count')

# Show the plot
plt.show()

#%% plot correlation between firing rate and speed
speed = edata.velocity

# Define 100 ms time bins (0.1 seconds)
bin_size = 0.1  # 100 ms in seconds
min_time = min(edata.time_ttl)
max_time = max(edata.time_ttl)
time_bins = np.arange(min_time, max_time + bin_size, bin_size)

# Calculate firing rate: count spikes in each 100ms time bin
firing_rate, _ = np.histogram(edata.t_spikeTrains[neuron], bins=time_bins)

# Resample velocity to match 100 ms bins by averaging velocity values in each bin
binned_velocity = []
for i in range(len(time_bins) - 1):
    # Find velocity values within the current bin
    indices_in_bin = np.where((edata.time_ttl >= time_bins[i]) & (edata.time_ttl < time_bins[i + 1]))[0]
    if len(indices_in_bin) > 0:
        binned_velocity.append(np.mean(speed[indices_in_bin]))
    else:
        binned_velocity.append(np.nan)  # Use NaN if no data points in this bin

binned_velocity = np.array(binned_velocity)

# Remove NaN values from both binned_velocity and firing_rate for correlation and plotting
valid_indices = ~np.isnan(binned_velocity)
binned_velocity = binned_velocity[valid_indices]
firing_rate = firing_rate[valid_indices]

# Plot velocity vs. firing rate
plt.figure(figsize=(8, 6))
plt.scatter(binned_velocity, firing_rate, color='blue', label='Data points')
plt.title('Velocity vs. Firing Rate (100ms bins)')
plt.xlabel('Average Velocity (cm/s)')
plt.ylabel('Firing Rate (spikes/100ms)')
plt.grid(True)
plt.legend()
#%%
'''NEW mat file plot aligned MULTIPLE inter-trial events'''
exp = '2024-02-15'
mouse = '105'
trial = '19'
neuron = 13

edata = elib.EphysTrial().Load(exp, mouse, trial)

print(edata.cellLabels[neuron])
print(edata.firingRate[neuron])
spike_train = edata.t_spikeTrains[neuron]
event_list = edata.k_hole_checks[np.where(edata.k_hole_checks[:,0] == 51)][:,1] #get list of hole checks at reward hole

event_list = np.delete(event_list, np.argwhere(np.ediff1d(event_list) <= 50) + 1) # gets rid of values too close to each other

spike_sample_eventcrop = get_intertrial_event_crop(spike_train, event_list, edata)


'''plot 10 sec before and after reward'''
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 8))
lineoffsets1 = range(len(spike_sample_eventcrop))
ax.eventplot(spike_sample_eventcrop, lineoffsets=lineoffsets1, color='#5f0f40')
# plt.vlines(tdata[0].time_ttl[tdata[0].k_reward], min(lineoffsets1)-1, max(lineoffsets1)+1, color='k') #when did mouse find reward
plt.vlines(0, min(lineoffsets1)-1, max(lineoffsets1)+1, color='g') #when did mouse find reward


plt.xticks(fontsize=18)
plt.xlabel('Time (s)' , fontsize=20)

ax.margins(0)
# plt.xlim((-30,30))

# path_data = rf'F:\Spike Sorting\Data\3_Raster\{exp}_M{mouse}'
# plt.savefig(path_data+f'/figs/Raster_HoleCheck_M{edata.mouse_number}_T{edata.trial}_Cell{neuron}.png', dpi=600, bbox_inches='tight', pad_inches = 0)

plt.show()


#%%
'''plot raster of whole trial over multiple trials'''
neuron = 2
spike_sample_eventcrop = []
for x, k_edata in enumerate(edata_dict):
    spike_sample_eventcrop = spike_sample_eventcrop + get_single_cell_trial_spikes(edata_dict[k_edata], tdata[x*3:(x+1)*3], neuron) #select the epoch and the trials that go with the epoch

'''plot raster of whole trial'''
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 8))
lineoffsets1 = range(len(spike_sample_eventcrop))
ax.eventplot(spike_sample_eventcrop, lineoffsets=lineoffsets1)
# plt.vlines(tdata[0].time_ttl[tdata[0].k_reward], min(lineoffsets1)-1, max(lineoffsets1)+1, color='k') #when did mouse find reward
plt.vlines(0, min(lineoffsets1)-1, max(lineoffsets1)+1, color='red') #when did mouse find reward

t_start_list = [x.time_ttl[0] for x in tdata]
t_reward_list = [x.time_ttl[x.k_reward] for x in tdata]
# plt.vlines(tdata.time_ttl[0], min(lineoffsets1)-1, max(lineoffsets1)+1, ) 
# ax.eventplot(t_start_list, lineoffsets=lineoffsets1, color='g') #when did trial start
# neuron_no = len(edata_dict['T16-18'].t_spike_train)
# plt.hlines([neuron_no, neuron_no*2, neuron_no*3, neuron_no*4, neuron_no*5], -10, 10, colors='k')
# pad = neuron_no/2
# plt.yticks([0+pad, neuron_no+pad, neuron_no*2+pad, neuron_no*3+pad, neuron_no*4+pad, neuron_no*5+pad], 
#            ['Trial 13', 'Trial 14', 'Trial 15', 'Trial 16', 'Trial 17', 'Trial 18'], fontsize=18)

plt.xticks(fontsize=18)
plt.xlabel('Time (s)' , fontsize=20)
# plt.ylabel('Neurons', fontsize=20)
ax.set_yticks([])
ax.margins(0)
plt.xlim((min(t_start_list)-max(t_reward_list)-5,50))

# path_data = rf'F:\Spike Sorting\Data\3_Raster\{exp}_M{mouse}'
# plt.savefig(path_data+f'/figs/Raster_M{tdata[0].mouse_number}_T{tdata[0].trial}-{tdata[-1].trial}_Cell{neuron}.png', dpi=600, bbox_inches='tight', pad_inches = 0)

plt.show()

#%%
'''NEW MAT FILE plot 20 sec before and after event'''
exp = '2024-02-15'
mouse = 105
trial = '19'
neuron = 13
edata = elib.EphysTrial.Load(exp, mouse,trial)

print(edata.cellLabels[neuron])
print(edata.firingRate[neuron])

t_start = edata.time_ttl[np.isfinite(edata.r_center)[:,0]][0] #returns the first non-nan coordinate

spike_sample = edata.t_spikeTrains[neuron]-t_start
if len(spike_sample) <=0: raise ValueError('there are no spikes in this trial')

k_event = 13049#edata.k_reward
# test = np.where((edata.k_hole_checks[:,1] > 0) & (edata.k_hole_checks[:,1] < edata.k_reward)) 

'''plot raster of trial'''
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 2))
lineoffsets1 = range(1)

ax.eventplot(spike_sample, lineoffsets=lineoffsets1, color='#5f0f40')

#not callebrated to t_start yet
# t_hole_checks = np.sort(edata.time_ttl[edata.k_hole_checks[:,1]])
# plt.plot(t_hole_checks , np.zeros(t_hole_checks.shape),'o', color = '#F18701')

plt.vlines(0, min(lineoffsets1)-1, max(lineoffsets1)+1, color='k') #when did trial start
plt.vlines(edata.time_ttl[k_event]-t_start, min(lineoffsets1)-1, max(lineoffsets1)+1, color='g') #when did mouse find reward

# plt.vlines(edata.time_ttl[923], min(lineoffsets1)-1, max(lineoffsets1)+1, color='b') #custom marker

# neuron_no = len(edata_dict['T16-18'].t_spike_train)
# plt.hlines([neuron_no, neuron_no*2, neuron_no*3, neuron_no*4, neuron_no*5], -10, 10, colors='k')
# pad = neuron_no/2
# plt.yticks([0+pad, neuron_no+pad, neuron_no*2+pad, neuron_no*3+pad, neuron_no*4+pad, neuron_no*5+pad], 
#            ['Trial 13', 'Trial 14', 'Trial 15', 'Trial 16', 'Trial 17', 'Trial 18'], fontsize=18)

plt.xticks(fontsize=18)
plt.xlabel('Time (s)' , fontsize=20)
# plt.ylabel('Neurons', fontsize=20)
plt.title(f'Cell {neuron}')
ax.set_yticks([])
ax.margins(0)
plt.xlim((edata.time_ttl[k_event]-t_start-20,edata.time_ttl[k_event]-t_start+120))

# path_data = cg.ROOT_DIR
# plt.savefig(path_data+f'/figs/Raster_M{edata.mouse_number}_T{edata.trial}_Cell{neuron}.png', dpi=600, bbox_inches='tight', pad_inches = 0)

plt.show()

#%%
spike_sample_eventcrop = []
for x, k_edata in enumerate(edata_dict):
    spike_sample_eventcrop = spike_sample_eventcrop + get_epoch_event_crop(edata_dict[k_edata], tdata[x*3:(x+1)*3]) #select the epoch and the trials that go with the epoch

'''plot 10 sec before and after reward'''
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 8))
lineoffsets1 = range(len(spike_sample_eventcrop))
ax.eventplot(spike_sample_eventcrop, lineoffsets=lineoffsets1)
# plt.vlines(tdata[0].time_ttl[tdata[0].k_reward], min(lineoffsets1)-1, max(lineoffsets1)+1, color='k') #when did mouse find reward
plt.vlines(0, min(lineoffsets1)-1, max(lineoffsets1)+1, color='red') #when did mouse find reward
plt.vlines(tdata.time_ttl[0], min(lineoffsets1)-1, max(lineoffsets1)+1, color='g') #when did trial start

# neuron_no = len(edata_dict['T16-18'].t_spike_train)
# plt.hlines([neuron_no, neuron_no*2, neuron_no*3, neuron_no*4, neuron_no*5], -10, 10, colors='k')
# pad = neuron_no/2
# plt.yticks([0+pad, neuron_no+pad, neuron_no*2+pad, neuron_no*3+pad, neuron_no*4+pad, neuron_no*5+pad], 
#            ['Trial 13', 'Trial 14', 'Trial 15', 'Trial 16', 'Trial 17', 'Trial 18'], fontsize=18)

plt.xticks(fontsize=18)
plt.xlabel('Time (s)' , fontsize=20)
# plt.ylabel('Neurons', fontsize=20)
ax.margins(0)
plt.xlim((-10,10))

# path_data = r'F:\Spike Sorting\Data\3_Raster\2023-12-18_M102'
# plt.savefig(path_data+f'/figs/Raster_M{tdata[0].mouse_number}_T{tdata[0].trial}-{tdata[-1].trial}.png', dpi=600, bbox_inches='tight', pad_inches = 0)

plt.show()


#%%
'''
Pyramidal Cell = #cc3333
Narrow Interneuron = #3333cc
Wide Interneuron = #33cccc
'''
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 4))
test = np.concatenate( spike_sample_eventcrop, axis=0)
plt.hist(test, bins=15)
plt.vlines(0, ymin=0, ymax=450., color='r') #when did mouse find reward
# ax.set_yticks([])

# path_data = r'F:\Spike Sorting\Data\3_Raster\2023-12-18_M102'
# plt.savefig(path_data+f'/figs/hist_M{tdata[0].mouse_number}_T{tdata[0].trial}-{tdata[-1].trial}.png', dpi=600, bbox_inches='tight', pad_inches = 0)

plt.show()

#%% get cell info
for i in edata.cellLabels: print(i)
for i in edata.firingRate: print(i)
#%% get index when mouse enters an area of interest
def is_inside_circle_np(coordinates, circle):
    """
    Check if points (x, y) are inside a circle using NumPy.
    
    coordinates: NumPy array of shape (n, 2) where each row is an (x, y) point.
    circle: tuple defining the circle (x_center, y_center, radius).
    
    Returns a boolean array where True indicates the point is inside the circle.
    """
    x_center, y_center, radius = circle
    distances = np.sqrt((coordinates[:, 0] - x_center) ** 2 + (coordinates[:, 1] - y_center) ** 2)
    return distances <= radius

def find_first_entry_indices_np(coordinates, circle):
    """
    Given a NumPy array of (x, y) coordinates and a circular area,
    find the indices where the agent first enters the circle in each continuous trajectory.
    
    coordinates: NumPy array of shape (n, 2) where each row is an (x, y) point.
    circle: tuple defining the circle (x_center, y_center, radius).
    
    Returns a list of indices corresponding to the first entry into the circle in each trajectory.
    """
    inside_circle = is_inside_circle_np(coordinates, circle)
    
    # Detect the first entry point by checking transitions from False to True
    entry_indices = np.where((~inside_circle[:-1]) & inside_circle[1:])[0] + 1
    
    # Check if the first point is inside the circle (edge case)
    if inside_circle[0]:
        entry_indices = np.insert(entry_indices, 0, 0)
    
    return entry_indices.tolist()

exp = '2024-02-15'
mouse = 105
trial = '19'
neuron = 11
edata = elib.EphysTrial.Load(exp, mouse,trial)
coordinates = edata.r_center
circle = (edata.r_arena_holes[34][0], edata.r_arena_holes[34][1], 6)  # circle with center at hole34 and radius 2
k_entry_points = find_first_entry_indices_np(coordinates, circle)