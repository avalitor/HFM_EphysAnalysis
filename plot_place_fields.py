# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 21:51:42 2023

process data that has gone through cell explorer into an object

@author: student
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors #for heatmap colours
import pickle
import numpy as np
import lib_ephys_obj as elib
import sys
sys.path.append("F:\Spike Sorting\EthovisionPathAnalysis_HDF5")
# import modules.lib_process_data_to_mat as plib
from scipy.ndimage import gaussian_filter
# from scipy.stats import binned_statistic_2d

#%%

# def get_vid_TTL_start_idx(tdata, vid_offset):
#     idx_start = np.searchsorted(tdata.time, vid_offset[tdata.trial])
#     return idx_start

def calc_spike_coords(tdata, spike_sample, idx_end, velocity_cutoff = 2):
    '''
    for each spike in spike sample, return the coordinate where the mouse was spatially located
    '''
    idx_spike = [] #list of idx times when spikes occured
    for sp in spike_sample: #np.searchsorted(tdata.time_ttl, sp) != len(tdata.time)
        if np.searchsorted(tdata.time_ttl, sp) != 0   : #if it didn't happen before the ethovision recording started
            idx_spike.append(np.searchsorted(tdata.time_ttl, sp))
    
    
    idx_spike = [x for x in idx_spike if x <= idx_end]
    
    # print(idx_spike)
    
    spike_coords = [] #use time index to get coords
    for i in idx_spike:
        if tdata.velocity[i-1]>velocity_cutoff:
            if i == len(tdata.r_center): pass #avoids an out of index error
            else:
                spike_coords.append(tdata.r_center[i]) #don't offset this i
                # print(tdata.r_center[i])
    spike_coords = np.array(spike_coords)
    if len(spike_coords) <1: raise ValueError('No spikes detected in this trial')
    return spike_coords

def draw_spikes(tdata, spike_sample, idx_end, ax):
    spike_coords = calc_spike_coords(tdata, spike_sample, idx_end)
    # print(spike_coords)
    ax.scatter(spike_coords[:,0], spike_coords[:,1], 
                s=50, marker = 'o', facecolors='#5F0F40', edgecolors='none', alpha=0.3, 
                linewidths=2. ,zorder=10)
    return ax

def make_heatmap(x, y, s=32, bins=1000):
    #gets rid of the NaN values
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins, range=[[-65, 65], [-65, 65]])
    heatmap = gaussian_filter(heatmap, sigma=s)
    
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    
    # x_bins = np.linspace(-65, 65, bins)
    # y_bins = np.linspace(-65, 65, bins)
    # extent = min(x_bins), max(x_bins), min(y_bins), max(y_bins)
    # heatmap = np.zeros((bins, bins)) #create empty 2d array
    
    # for i in range(len(x)):
    #     x_bin_index = np.digitize(x[i], x_bins) - 1 #Return the indices of the bins to which each value in input array belongs.
    #     y_bin_index = np.digitize(y[i], y_bins) - 1 #minus one so it fits the original shape
    #     heatmap[x_bin_index, y_bin_index] += 1
    
    return heatmap.T, extent #the T transposes the array so it is correctly oriented


def occupancy_bins(edata, bins = 60):
    '''occupancy (2D numpy array): time spent of the animal in each spatial bin.'''
    
    #first, filter out low velocity coordinates
    filterd_coords = [] #use time index to get coords
    for i, xs in enumerate(edata.time):
        if edata.velocity[i-1]>2:
            if i == len(edata.r_center): pass #avoids an out of index error
            else:
                filterd_coords.append([edata.r_center[i,0], edata.r_center[i,1]]) #don't offset this i
    filterd_coords = np.array(filterd_coords)
    x, y =filterd_coords[:,0],filterd_coords[:,1]
    
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    
    
    occupancy, xedges, yedges = np.histogram2d(x, y, bins=bins, range=[[-65, 65], [-65, 65]]) #this bins the data
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    
    #alternate way to bin, but it takes slower
    # samplingRate = edata.time[1] #how long each sample is
    # # x_bins = np.linspace(np.nanmin(x), np.nanmax(x), bins) #get evenly spaced range of numbers
    # # y_bins = np.linspace(np.nanmin(y), np.nanmax(y), bins)
    # x_bins = np.linspace(-65, 65, bins)
    # y_bins = np.linspace(-65, 65, bins)
    # occupancy = np.zeros((bins, bins)) #create empty 2d array
    # extent = min(x_bins), max(x_bins), min(y_bins), max(y_bins)
    # for i in range(len(x)):
    #     x_bin_index = np.digitize(x[i], x_bins) - 1 #Return the indices of the bins to which each value in input array belongs.
    #     y_bin_index = np.digitize(y[i], y_bins) - 1 #minus one so it fits the original shape
    #     occupancy[x_bin_index, y_bin_index] += samplingRate

    return occupancy.T, extent

def draw_spike_heatmap(spike_coords, trial_coords, idx_end, ax, weighted = True):
    
    x = spike_coords[:idx_end, 0]
    y = spike_coords[:idx_end, 1]
    #plot heatmap
    
    bins = 60
    
    hm_spike, extent = make_heatmap(x, y, s=0, bins=bins)
    
    x_t = trial_coords[:, 0]
    y_t = trial_coords[:, 1]
    hm_trial, extent = make_heatmap(x_t, y_t, s=0, bins=bins)
       
    
    if weighted: #divides spike heatmap by occupancy heatmap
        occupancy, _ = occupancy_bins(edata, bins)
        hm_weighted = np.divide(hm_spike, occupancy, out=np.zeros_like(hm_spike), where=occupancy!=0)
        img = gaussian_filter(hm_weighted, sigma=2)
    else: img = gaussian_filter(hm_spike, sigma=2)
    
    colors = [(1,0,0,c) for c in np.linspace(0,1,100)]
    cmapred = mcolors.LinearSegmentedColormap.from_list('mycmap', colors, N=20)
    
    hm = ax.imshow(img, extent=extent, origin='lower', cmap='inferno') #othor colours: rainbow_alpha, cm.jet
    # plt.colorbar(mappable=hm) #add colorbar
    
    #scales the heatmap to match the background picture
    # plt.xlim(tdata.img_extent[0], tdata.img_extent[1]) 
    # plt.ylim(tdata.img_extent[2], tdata.img_extent[3]
    return ax


def draw_traj_heatmap(coords, idx_end, ax):
    x = coords[:idx_end, 0]
    y = coords[:idx_end, 1]
    
    #plot heatmap
    img, extent = make_heatmap(x, y, s=2, bins=60)
    # img, extent = occupancy_bins(edata)
    colors = [(1,0,0,c) for c in np.linspace(0,1,100)]
    cmapred = mcolors.LinearSegmentedColormap.from_list('mycmap', colors, N=20)
    
    hm = ax.imshow(img, extent=extent, origin='lower', cmap="inferno") #other colours: inferno, cm.jet, cmapred
    return ax
    
    
def draw_arena(data, ax, color='k'):
    #draws arena
    Drawing_arena_circle = plt.Circle( (data.arena_circle[0], data.arena_circle[1]), 
                                          data.arena_circle[2] , fill = False, color=color )
    ax.add_artist( Drawing_arena_circle )
    
    #labels holes with numbers
    # n = np.arange(0,100)
    # for i, txt in enumerate(n):
    #     ax.annotate(txt, (data.r_arena_holes[i]))
    
    for c in data.r_arena_holes:
        small_hole = plt.Circle( (c[0], c[1] ), 0.5 , fill = False ,alpha=0.5, color = color)
        ax.add_artist( small_hole )

    ax.set_aspect('equal','box')
    ax.set_xlim([data.img_extent[2],data.img_extent[3]])
    ax.set_ylim([data.img_extent[2],data.img_extent[3]])
    ax.axis('off')
    return ax

def draw_hole_checks(data, idx_end, ax):
    
    k_times = data.k_hole_checks[data.k_hole_checks[:,1]<= idx_end] #crop at target
    # k_times = data.k_hole_checks
    
    colors_time_course = plt.get_cmap('cool') # plt.get_cmap('cool') #jet_r
    t_seq_hole = data.time[k_times[:,1]]/data.time[data.k_reward-1]
    # t_seq_traj = data.time/data.time[data.k_reward-1]
    
    #plots hole checks
    ax.scatter(data.r_arena_holes[k_times[:,0]][:,0], data.r_arena_holes[k_times[:,0]][:,1], 
               s=50, marker = 'o', facecolors='none', edgecolors=colors_time_course(t_seq_hole), 
               linewidths=2.)
    return ax

def plot_place_field(data, spike_sample, crop_at_target = True, savefig=False):

    fig, ax = plt.subplots()
    
    if crop_at_target: idx_end = data.k_reward + 10 #75 = extra 3 seconds
    else: idx_end = len(data.time)
    
    draw_arena(data, ax, color='white')
    # draw_hole_checks(data, idx_end, ax)
    # draw_spikes(data, spike_sample, idx_end, ax)
    
    
    plt.plot(data.r_center[:idx_end,0], data.r_center[:idx_end,1], color='k', alpha=0.3) #plot path, alpha 0.3
    # ax.scatter(data.r_nose[:idx_target,0], data.r_nose[:idx_target,1], s=1.5, facecolors=colors_time_course(t_seq_traj[:idx_target])) #plot path with colours
    
    spike_coords = calc_spike_coords(data, spike_sample, idx_end, 2)
    draw_spike_heatmap(spike_coords, data.r_center, idx_end, ax, weighted=True)
    # draw_traj_heatmap(data.r_center, idx_end, ax)
    
    # draw target
    target = plt.Circle((data.target), 2.5 , color='g', alpha=1)
    ax.add_artist(target)

    #draw entrance
    for i, _ in enumerate(data.r_nose):
        if np.isnan(data.r_nose[i][0]): continue
        else:
            first_coord = data.r_nose[i]
            break
    entrance = plt.Rectangle((first_coord-3.5), 7, 7, fill=False, color='white', alpha=0.8, lw=3)
    ax.add_artist(entrance)        
    
    if savefig == True:
        plt.savefig(path_data+f'/figs/Placefield_M{data.mouse_number}_T{data.trial}_Cell{neuron}.png', dpi=600, bbox_inches='tight', pad_inches = 0)
    
    plt.show()

#%%



if __name__ == '__main__':
       
    exp = '2023-12-18'
    mouse = 102
    trial = 'Probe'
    neuron = 13
    
    path_data =  rf'F:\Spike Sorting\Data\3_Raster\{exp}_M{mouse}'
    
    # (tdata := plib.TrialData()).Load(exp, mouse,'13')
    # edata = EphysEpoch().Load(exp, mouse,'T13-15')
    
    (edata := elib.EphysTrial()).Load(exp, mouse,trial)
    # spike_sample = edata.t_spikeTrains[neuron]
    # spike_coords = calc_spike_coords(edata, spike_sample, len(edata.time), 2)

    # x = spike_coords[:, 0]
    # y = spike_coords[:, 1]
    # #plot heatmap
    # bins = 60
    # hm_spike2, extent2 = make_heatmap(x, y, s=0, bins=bins)
    # occupancy, extent3 = occupancy_bins(edata, bins)
    
    # spike_sample = edata.t_spike_train[neuron] - edata.t_TTL[0][0] #synch spike train with TTL REMEMBER TO CHOOSE CORRECT TTL
    # spikesample_crop = []#crop according to time_ttl
    print(edata.cellLabels[neuron])
    print(edata.firingRates[neuron])
    # assert len(spike_sample) > 0
    
    # spikecoords = calc_spike_coords(tedata, tedata.t_spikeTrains[14], tedata.k_reward)
    
    # k_times = tedata.k_hole_checks[tedata.k_hole_checks[:,1]<= tedata.k_reward+10]
    # plot_place_field(tdata, spike_sample, crop_at_target=True, savefig=False)
    
    plot_place_field(edata, edata.t_spikeTrains[neuron], crop_at_target=False, savefig=False)
