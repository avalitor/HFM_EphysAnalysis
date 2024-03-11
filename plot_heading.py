# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 03:49:17 2024

@author: Kelly
"""
import numpy as np
import matplotlib.pyplot as plt
import lib_ephys_obj as elib
import plot_place_fields as ppf

def calc_spike_idx(tdata, spike_sample, idx_end):
    '''
    for each spike in spike sample, return the index in tdata
    '''
    idx_spike = [] #list of idx times when spikes occured
    for sp in spike_sample: #np.searchsorted(tdata.time_ttl, sp) != len(tdata.time)
        if np.searchsorted(tdata.time_ttl, sp) != 0   : #if it didn't happen before the ethovision recording started
            idx_spike.append(np.searchsorted(tdata.time_ttl, sp))
    
    idx_spike = [x for x in idx_spike if x <= idx_end]
    return idx_spike

def heading_from_idx(tdata, idx_spike):
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
    

(data := elib.EphysTrial()).Load('2024-02-15', '105', 'Probe')

neuron = 11

spike_train = data.t_spikeTrains[neuron]
idx_end = len(data.time)

idx_spike = calc_spike_idx(data, spike_train, idx_end)
headings = heading_from_idx(data, idx_spike)

#%%
def circular_hist(ax, x, bins=16, density=True, offset=0, gaps=True):
    """
    Produce a circular histogram of angles on ax.

    Parameters
    ----------
    ax : matplotlib.axes._subplots.PolarAxesSubplot
        axis instance created with subplot_kw=dict(projection='polar').

    x : array
        Angles to plot, expected in units of radians.

    bins : int, optional
        Defines the number of equal-width bins in the range. The default is 16.

    density : bool, optional
        If True plot frequency proportional to area. If False plot frequency
        proportional to radius. The default is True.

    offset : float, optional
        Sets the offset for the location of the 0 direction in units of
        radians. The default is 0.

    gaps : bool, optional
        Whether to allow gaps between bins. When gaps = False the bins are
        forced to partition the entire [-pi, pi] range. The default is True.

    Returns
    -------
    n : array or list of arrays
        The number of values in each bin.

    bins : array
        The edges of the bins.

    patches : `.BarContainer` or list of a single `.Polygon`
        Container of individual artists used to create the histogram
        or list of such containers if there are multiple input datasets.
    """
    # Wrap angles to [-pi, pi)
    # x = (x+np.pi) % (2*np.pi) - np.pi
    x = x*(np.pi/180.)

    # Force bins to partition entire circle
    if not gaps:
        bins = np.linspace(-np.pi, np.pi, num=bins+1)

    # Bin data and record counts
    n, bins = np.histogram(x, bins=bins)

    # Compute width of each bin
    widths = np.diff(bins)

    # By default plot frequency proportional to area
    if density:
        # Area to assign each bin
        area = n / x.size
        # Calculate corresponding bin radius
        radius = (area/np.pi) ** .5
    # Otherwise plot frequency proportional to radius
    else:
        radius = n

    # Plot data on ax
    patches = ax.bar(bins[:-1], radius, zorder=1, align='edge', width=widths,
                     edgecolor='C0', fill=True, linewidth=1)

    # Set the direction of the zero angle
    ax.set_theta_offset(offset)

    # Remove ylabels for area plots (they are mostly obstructive)
    if density:
        ax.set_yticks([])

    return n, bins, patches
    


angles0 = headings[~np.isnan(headings)]


# Construct figure and axis to plot on
fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection='polar'))

# Visualise by area of bins
circular_hist(ax, angles0, density=False)
# Visualise by radius of bins
# circular_hist(ax[1], angles1, offset=np.pi/2, density=False)
#%%
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))
plt.hist(headings, bins=10)
# plt.vlines(0, ymin=0, ymax=450., color='r') #when did mouse find reward
# ax.set_yticks([])

# path_data = r'F:\Spike Sorting\Data\3_Raster\2023-12-18_M102'
# plt.savefig(path_data+f'/figs/hist_M{tdata[0].mouse_number}_T{tdata[0].trial}-{tdata[-1].trial}.png', dpi=600, bbox_inches='tight', pad_inches = 0)
plt.xlim((-180,180))
plt.show()

'''This data still needs to be normalized aginst total time going in each direction'''
