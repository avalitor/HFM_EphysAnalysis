# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 10:49:28 2026

@author: student

Single-trial spatial firing rate map plotting.
Plots occupancy-normalized rate maps with arena overlay for individual neurons.
"""

import matplotlib.pyplot as plt
import numpy as np
import lib_ephys_obj as elib
import sys
sys.path.append("F:\Spike Sorting\EthovisionPathAnalysis_HDF5")
from scipy.ndimage import gaussian_filter
import config as cg
import os


# ============================================================================
# Core computation functions
# ============================================================================

def calc_spike_coords(tdata, spike_sample, idx_end, velocity_cutoff=2):
    '''
    For each spike in spike_sample, return the coordinate where the mouse
    was spatially located.
    '''
    spike_sample = np.atleast_1d(spike_sample)
    idx_spike = np.searchsorted(tdata.time_ttl, spike_sample)

    # Filter: within valid range
    valid = (idx_spike > 0) & (idx_spike < min(idx_end, len(tdata.r_center)))
    idx_spike = idx_spike[valid]

    vel_at_spike   = tdata.velocity[idx_spike]
    coord_at_spike = tdata.r_center[idx_spike]

    keep = ((vel_at_spike > velocity_cutoff)
            & ~np.isnan(coord_at_spike[:, 0])
            & ~np.isnan(coord_at_spike[:, 1]))

    spike_coords = coord_at_spike[keep]
    if len(spike_coords) < 1:
        raise ValueError('No spikes detected in this trial')
    return spike_coords


def occupancy_bins(edata, idx_end, bins=60, velocity_cutoff=2):
    '''
    Occupancy (2D numpy array): time spent of the animal in each spatial bin,
    in seconds.
    '''
    coords = edata.r_center[:idx_end]
    vel    = edata.velocity[:idx_end]

    valid = (np.ones(len(coords), dtype=bool)
             & ~np.isnan(coords[:, 0])
             & ~np.isnan(coords[:, 1])
             & ~np.isnan(vel)
             & (vel > velocity_cutoff))

    x, y = coords[valid, 0], coords[valid, 1]

    occ_counts, xedges, yedges = np.histogram2d(
        x, y, bins=bins, range=[[-65, 65], [-65, 65]])
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    dt = np.median(np.diff(edata.time_ttl))
    occ_sec = occ_counts.T * dt

    return occ_sec, extent


def compute_ratemap(edata, spike_sample, idx_end, bins=60, velocity_cutoff=2,
                    sigma=2, min_occ=0.4):
    '''
    Compute occupancy-normalized spatial firing rate map for a single trial.
    Returns:
        ratemap      : 2D array of firing rates in Hz, NaN for unvisited bins
        extent       : [xmin, xmax, ymin, ymax]
        occupancy_sec: 2D array of time in seconds per bin
    '''
    # Spike histogram
    spike_coords = calc_spike_coords(edata, spike_sample, idx_end, velocity_cutoff)
    spike_counts, xedges, yedges = np.histogram2d(
        spike_coords[:, 0], spike_coords[:, 1],
        bins=bins, range=[[-65, 65], [-65, 65]])
    spike_counts = spike_counts.T

    # Occupancy in seconds
    occ_sec, extent = occupancy_bins(edata, idx_end, bins, velocity_cutoff)

    visited = occ_sec > 0

    if sigma > 0:
        spike_smooth = gaussian_filter(
            np.where(visited, spike_counts, 0).astype(float), sigma=sigma)
        occ_smooth = gaussian_filter(
            np.where(visited, occ_sec, 0).astype(float), sigma=sigma)
        ratemap = np.full_like(spike_smooth, np.nan)
        good = occ_smooth > min_occ
        ratemap[good] = spike_smooth[good] / occ_smooth[good]
    else:
        ratemap = np.full_like(spike_counts, np.nan, dtype=float)
        good = occ_sec > min_occ
        ratemap[good] = spike_counts[good] / occ_sec[good]

    return ratemap, extent, occ_sec


# ============================================================================
# Drawing functions
# ============================================================================

def draw_arena(data, ax, color='k'):
    '''Draws arena circle and holes.'''
    circle = plt.Circle((data.arena_circle[0], data.arena_circle[1]),
                          data.arena_circle[2], fill=False, color=color)
    ax.add_artist(circle)

    for c in data.r_arena_holes:
        hole = plt.Circle((c[0], c[1]), 0.5, fill=False, alpha=0.5, color=color)
        ax.add_artist(hole)

    ax.set_aspect('equal', 'box')
    ax.set_xlim([data.img_extent[2], data.img_extent[3]])
    ax.set_ylim([data.img_extent[2], data.img_extent[3]])
    ax.axis('off')
    return ax


def draw_spikes(tdata, spike_sample, idx_end, ax):
    spike_coords = calc_spike_coords(tdata, spike_sample, idx_end)
    ax.scatter(spike_coords[:, 0], spike_coords[:, 1],
               s=50, marker='o', facecolors='#5F0F40', edgecolors='none',
               alpha=0.3, linewidths=2., zorder=10)
    return ax


def draw_hole_checks(data, idx_end, ax):
    k_times = data.k_hole_checks[data.k_hole_checks[:, 1] <= idx_end]

    colors_time_course = plt.get_cmap('cool')
    t_seq_hole = data.time[k_times[:, 1]] / data.time[idx_end - 1]

    ax.scatter(data.r_arena_holes[k_times[:, 0]][:, 0],
               data.r_arena_holes[k_times[:, 0]][:, 1],
               s=50, marker='o', facecolors='none',
               edgecolors=colors_time_course(t_seq_hole),
               linewidths=2.)
    return ax


# ============================================================================
# Plot functions
# ============================================================================

def plot_place_field(data, spike_sample, neuron, crop_at_target=True,
                     savefig=False, sigma=2, min_occ=0.15, velocity_cutoff=2):

    fig, ax = plt.subplots()

    if crop_at_target and data.k_reward is not None:
        idx_end = data.k_reward + 10
    else:
        idx_end = len(data.time_ttl)

    try:
        ratemap, extent, occ_sec = compute_ratemap(
            data, spike_sample, idx_end, bins=60, velocity_cutoff=velocity_cutoff, sigma=sigma, min_occ=min_occ)
    except ValueError as e:
        print(f"  Cell {neuron}: {e}")
        plt.close(fig)
        return

    rm_display = np.where(np.isnan(ratemap), 0, ratemap)
    ax.imshow(rm_display, extent=extent, origin='lower', cmap='inferno')

    draw_arena(data, ax, color='white')

    # Draw target
    target = plt.Circle((data.target), 2.5, color='g', alpha=1)
    ax.add_artist(target)

    # Draw entrance
    for i, _ in enumerate(data.r_nose):
        if np.isnan(data.r_nose[i][0]):
            continue
        else:
            first_coord = data.r_nose[i]
            break
    entrance = plt.Rectangle((first_coord - 3.5), 7, 7,
                              fill=False, color='white', alpha=0.8, lw=3)
    ax.add_artist(entrance)

    label = data.cellLabels_BITP[neuron][0]
    plt.title(f'Cell {neuron} ({label})\nFR: {round(data.firingRate[neuron], 2)} Hz')

    if savefig:
        plt.savefig(os.path.join(cg.ROOT_DIR, 'figs',
                    f'Placefield_M{data.mouse_number}_T{data.trial}_Cell{neuron}.png'),
                    dpi=600, bbox_inches='tight', pad_inches=0)

    plt.show()


def plot_traj_hole_checks(data, spike_sample, neuron, crop_at_target=True,
                          savefig=False):

    fig, ax = plt.subplots()

    if crop_at_target and data.k_reward is not None:
        idx_end = data.k_reward + 75
    else:
        idx_end = len(data.time_ttl)

    draw_arena(data, ax, color='k')
    draw_spikes(data, spike_sample, idx_end, ax)

    plt.plot(data.r_center[:idx_end, 0], data.r_center[:idx_end, 1],
             color='k', alpha=0.3)

    # Draw target
    target = plt.Circle((data.target), 2.5, color='g', alpha=1)
    ax.add_artist(target)

    # Draw entrance
    for i, _ in enumerate(data.r_nose):
        if np.isnan(data.r_nose[i][0]):
            continue
        else:
            first_coord = data.r_nose[i]
            break
    entrance = plt.Rectangle((first_coord - 3.5), 7, 7,
                              fill=False, color='white', alpha=0.8, lw=3)
    ax.add_artist(entrance)

    label = data.cellLabels_BITP[neuron][0]
    plt.title(f'Cell {neuron} ({label})\nFR: {round(data.firingRate[neuron], 2)} Hz')

    if savefig:
        plt.savefig(os.path.join(cg.ROOT_DIR, 'figs',
                    f'Placefield_M{data.mouse_number}_T{data.trial}_Cell{neuron}.png'),
                    dpi=600, bbox_inches='tight', pad_inches=0)

    plt.show()


# ============================================================================
# Main
# ============================================================================

#%%

if __name__ == '__main__':

    exp = '2025-01-21'
    mouse = 112
    trial = '2'
    neuron = 0

    edata = elib.EphysTrial.Load(exp, mouse, trial)
    spike_sample = edata.t_spikeTrains[neuron]

    print(edata.cellLabels_BITP[neuron][0])
    print(f"FR: {edata.firingRate[neuron]:.2f} Hz")

    plot_place_field(edata, edata.t_spikeTrains[neuron], neuron,
                     crop_at_target=False, savefig=False)
    # plot_traj_hole_checks(edata, edata.t_spikeTrains[neuron], neuron,
    #                       crop_at_target=True, savefig=False)