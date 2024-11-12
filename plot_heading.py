# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 03:49:17 2024

@author: Kelly
"""
import numpy as np
import matplotlib.pyplot as plt
import lib_ephys_obj as elib
import plot_place_fields as ppf

def heading(exp):
    #calculate mouse-direction vector
    mouse_vector = np.diff(exp.r_center, axis=0)
    mouse_heading = np.arctan2(mouse_vector[:,1],mouse_vector[:,0]) * (180 / np.pi)
    mouse_heading = np.insert(mouse_heading, 0, np.nan) #add a nan value at the beginning so its the same size
    return mouse_heading

def calc_spike_idx(tdata, spike_sample, idx_end):
    '''
    for each spike in spike sample, return the index in coordinate data
    '''
    idx_spike = [] #list of idx times when spikes occured
    for sp in spike_sample: #np.searchsorted(tdata.time_ttl, sp) != len(tdata.time)
        if np.searchsorted(tdata.time_ttl, sp) != 0   : #if it didn't happen before the ethovision recording started
            idx_spike.append(np.searchsorted(tdata.time_ttl, sp))
    
    idx_spike = [x for x in idx_spike if x <= idx_end]
    return idx_spike

def heading_from_idx(tdata, idx_spike):
    '''depreciated. this gets heading from each spike idx'''
    spike_heading = [] #use time index to get heading
    # spike_coords = [] #use time index to get coords
    for i in idx_spike:
        if tdata.velocity[i-1]>4:
            if i == len(tdata.r_center): pass #avoids an out of index error
            else:
                spike_heading.append(tdata.heading[i]) #don't offset this i
                # spike_coords.append(tdata.r_center[i])
                # print(tdata.heading[i])
    spike_heading = np.array(spike_heading)
    # spike_coords = np.array(spike_coords)
    if len(spike_heading) <1: raise ValueError('No spikes detected in this trial')
    return spike_heading
    

def calc_normalized_heading_preference(edata, bins=36):
    ''' bins heading data and normalizes output to between 0 and 1'''
    # Change heading data to be within 0 to 360 degrees
    heading360 = np.mod(edata.heading, 360.0) #changes the axis -180 to 180 > 0 to 360

    # Initialize arrays to store spike counts 
    spike_counts = np.zeros(bins)
    occupancy_counts = np.zeros(bins)

    # Determine bin size in degrees
    bin_size = 360 / bins

    #iterate over each heading
    for i in range(len(heading360)):
        if np.isnan(heading360[i]): 
            pass
        elif edata.velocity[i-1]<4: pass #filters out when mouse is not moving fast
        else:
            bin_index = int(heading360[i] // bin_size)
            occupancy_counts[bin_index] += 1 #increase count if mouse is headed in that direction
            if i in idx_spike:
                spike_counts[bin_index] += idx_spike.count(i) #increase count if neuron spikes in that direction
                # print(i, idx_spike.count(i))
    # return occupancy_counts, spike_counts

    # Calculate normalized preference
    if sum(spike_counts) <= 0: raise ValueError('No spikes detected in this trial')
    
    weighted_spikes = spike_counts / occupancy_counts
    # norm = np.linalg.norm(weighted_spikes)
    # norm = weighted_spikes
    norm = (weighted_spikes - np.min(weighted_spikes)) / (np.max(weighted_spikes) - np.min(weighted_spikes))
    return norm

def calculate_average_vector(angles, magnitudes):
    # # Convert angles to radians
    # angles_rad = np.deg2rad(angles)

    # Convert to Cartesian coordinates
    x_components = magnitudes * np.cos(angles)
    y_components = magnitudes * np.sin(angles)

    # Calculate sum of Cartesian components
    sum_x = np.sum(x_components)
    sum_y = np.sum(y_components)

    # Calculate average vector
    average_x = sum_x / np.count_nonzero(magnitudes)
    average_y = sum_y / np.count_nonzero(magnitudes)

    # Calculate magnitude and angle of the average vector
    magnitude_average = round(np.sqrt(average_x**2 + average_y**2), 10)
    angle_average = np.arctan2(average_y, average_x)

    return magnitude_average, angle_average

def sum_vector(angles, magnitudes):
    # # Convert angles to radians
    # angles_rad = np.deg2rad(angles)

    # Convert to Cartesian coordinates
    x_components = magnitudes * np.cos(angles)
    y_components = magnitudes * np.sin(angles)

    # Calculate sum of Cartesian components
    sum_x = np.sum(x_components)
    sum_y = np.sum(y_components)

    # Calculate magnitude and angle of the average vector
    magnitude_sum = round(np.sqrt(sum_x**2 + sum_y**2), 10)
    angle_sum = np.arctan2(sum_y, sum_x)

    return magnitude_sum, angle_sum

#%%
data = elib.EphysTrial.Load('2024-02-15', '105', '24')
neuron = 13

if hasattr(data, 'heading'): pass
else: 
    data.heading = heading(data)


spike_train = data.t_spikeTrains[neuron]
idx_end = len(data.time)

idx_spike = calc_spike_idx(data, spike_train, idx_end)
# headings = heading_from_idx(data, idx_spike)

# test_ang = np.deg2rad(np.array([0, 10, 20, 30, 40]))
# test_mag = np.array([0.658228, 0.573889, 0.444333 ,0.438756, 0])
# vector_avg = calculate_average_vector(test_ang, test_mag)

bins = 36
heading_norm = calc_normalized_heading_preference(data, bins)
x = np.linspace(0.0, 2*np.pi, num=bins, endpoint=False)
vector_avg = calculate_average_vector(x, heading_norm)
#plot polar bar graph
fig, ax = plt.subplots(1, 1, subplot_kw={'projection': 'polar'})

width = 2 * np.pi / bins

ax.bar(x, heading_norm, zorder=1, align='edge', width=width, edgecolor='C0', fill=True, linewidth=1)
# Add an arrow
# ax.annotate('', xy=(vector_sum[0], vector_sum[1]), xytext=(0,0), arrowprops=dict(facecolor='green', edgecolor='green', arrowstyle='->'), zorder=0)

# arr2 = plt.arrow(0, 0.1, vector_avg[1], vector_avg[0], alpha = 1, width = 0.05, edgecolor = 'none', facecolor = 'red', lw = 2, zorder = 5)
# plt.arrow(vector_avg[1], 0.0, 0, 0.45396262233400994,  width = 0.015, edgecolor = 'red',  lw = 3,head_width=0.1, head_length=0.05)
plt.arrow(vector_avg[1], 0.0, 0, vector_avg[0],  width = 0.015, edgecolor = 'red',  lw = 3,head_width=0.1, head_length=0.05)


# ax.annotate('', xy=(1, np.deg2rad(90)), xytext=(0,0), arrowprops=dict(facecolor='red', edgecolor='red', arrowstyle='->'), zorder=5)

ax.set_theta_zero_location('E')
# ax.set_ylim(0, 1) 
plt.show()

#%% plot straight bar graph
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))
plt.bar(x, heading_norm, width=width)
# path_data = r'F:\Spike Sorting\Data\3_Raster\2023-12-18_M102'
# plt.savefig(path_data+f'/figs/hist_M{tdata[0].mouse_number}_T{tdata[0].trial}-{tdata[-1].trial}.png', dpi=600, bbox_inches='tight', pad_inches = 0)
plt.xlim((0,2*np.pi))
plt.xlabel('angle (rad)')
plt.ylabel('spikes/angle normalized')
plt.show()

#%%
'''Heading per spatial bin'''
edata = elib.EphysTrial.Load('2024-02-15', '105', '24')
neuron = 13
if hasattr(edata, 'heading'): pass
else: 
    edata.heading = heading(edata)

#data
coordinates = edata.r_center  # x and y coordinates between 0 and 30
neural_spikes = edata.t_spikeTrains[neuron]  # neural spiking data
idx_end = len(edata.time)
idx_spike = np.array(calc_spike_idx(edata, neural_spikes, idx_end))
headings_rad = np.deg2rad(np.mod(edata.heading, 360.0)) #changes the axis -180 to 180 > 0 to 360 and convert to radians

# Remove rows with NaN values in coordinates
# valid_indices = ~np.isnan(coordinates).any(axis=1)
# coordinates = coordinates[valid_indices]
# headings_rad = headings_rad[valid_indices]
# idx_spike2 = np.array([idx for idx in idx_spike if valid_indices[idx]]) # Filter the spike indices to only keep valid ones

#velocity filter
velocity_mask = edata.velocity >= 4
coordinates = coordinates[velocity_mask] # Apply the velocity mask to coordinates, head directions, and velocity
headings_rad = headings_rad[velocity_mask]
# idx_spike_valid = np.array([idx for idx in idx_spike if velocity_mask[idx]]) # Filter the spike indices again to only keep those that correspond to high-speed coordinates
# Create a mapping from original indices to filtered indices
original_to_filtered_index = np.where(velocity_mask)[0]
# Adjust the spike indices to match the filtered data
idx_spike_valid = np.array([np.where(original_to_filtered_index == idx)[0][0] for idx in idx_spike if velocity_mask[idx]])



# Define grid size
grid_size = 30

# Create bins from coordinates between xmin and max
# Get the range of coordinates
x_min, x_max = np.nanmin(coordinates[:, 0]), np.nanmax(coordinates[:, 0])
y_min, y_max = np.nanmin(coordinates[:, 1]), np.nanmax(coordinates[:, 1])

x_bins = np.linspace(x_min, x_max, grid_size + 1)
y_bins = np.linspace(y_min, y_max, grid_size + 1)

# Digitize the coordinates into bins
x_indices = np.digitize(coordinates[:, 0], x_bins) - 1
y_indices = np.digitize(coordinates[:, 1], y_bins) - 1


# Count visits to each bin
visit_counts = np.zeros((grid_size, grid_size))
for i in range(grid_size):
    for j in range(grid_size):
        visit_counts[i, j] = np.sum((x_indices == i) & (y_indices == j))

# Initialize arrays to store results
preferred_directions = np.zeros((grid_size, grid_size))
spike_counts = np.zeros((grid_size, grid_size))

#mas away nan values by comparing them to original array 
#masked = digiti_arr[~np.isnan(orig_arr)]       
            
# Calculate the preferred direction in each bin
# for i in range(grid_size):
#     for j in range(grid_size):
#         mask = (x_indices == i) & (y_indices == j)
#         bin_spike_indices = [idx for idx in idx_spike if mask[idx]]
#         if np.sum(mask) > 0:
#             weighted_directions = np.sum(idx_spike[mask] * np.exp(1j * headings_rad[mask]))
#             preferred_directions[i, j] = np.angle(weighted_directions)
#             spike_counts[i, j] = len(bin_spike_indices)
            
            
# Calculate the preferred direction in each bin
for i in range(grid_size):
    for j in range(grid_size):
        if visit_counts[i, j] >= 4:  # Ensure at least 4 visits
            # Find spikes that fall into the current bin
            bin_spike_indices = [idx for idx in idx_spike_valid if (x_indices[idx] == i) & (y_indices[idx] == j)]
            if len(bin_spike_indices) > 0:
                weighted_directions = np.sum(np.exp(1j * headings_rad[bin_spike_indices]))
                preferred_directions[i, j] = np.angle(weighted_directions)
                spike_counts[i, j] = len(bin_spike_indices)


# Plot the preferred directions
fig, ax = plt.subplots(figsize=(10, 10))
nonzero_spike_indices = np.nonzero(spike_counts)

x_centers = x_bins[:-1] + (x_bins[1] - x_bins[0]) / 2
y_centers = y_bins[:-1] + (y_bins[1] - y_bins[0]) / 2

quiver_plot = ax.quiver(
    x_centers[nonzero_spike_indices[1]],
    y_centers[nonzero_spike_indices[0]],
    np.cos(preferred_directions[nonzero_spike_indices]),
    np.sin(preferred_directions[nonzero_spike_indices]),
    spike_counts[nonzero_spike_indices],
    scale=20, headwidth=4, headlength=6
)
ppf.draw_arena(edata, ax, color='k')
# draw target
target = plt.Circle((edata.target), 2.5 , color='g', alpha=1)
ax.add_artist(target)

#draw entrance
for i, _ in enumerate(edata.r_nose):
    if np.isnan(edata.r_nose[i][0]): continue
    else:
        first_coord = edata.r_nose[i]
        break
entrance = plt.Rectangle((first_coord-3.5), 7, 7, fill=False, color='k', alpha=0.8, lw=3)
ax.add_artist(entrance)

ax.set_xlabel('X coordinate')
ax.set_ylabel('Y coordinate')
ax.set_title('Preferred Heading Direction Spiking (Filtered by Speed)', fontsize=18)
cbar = plt.colorbar(quiver_plot, ax=ax)
cbar.set_label('Spike Count', fontsize=15)
cbar.ax.tick_params(labelsize=12)  # Set font size of tick labels