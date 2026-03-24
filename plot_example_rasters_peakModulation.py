# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 15:44:30 2026

@author: student
"""
"""
Example raster plots for neurons with high peak-aligned reward modulation.
For each cell type, shows the top 3 neurons, each with multiple trial rows
aligned to reward at t=0.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
import glob
import os
import re
import lib_ephys_obj as elib
import config as cg

# %% === PARAMETERS ===
experiments = ['2024-02-15', '2024-11-09', '2025-01-21']
exclude_prefixes = ('Habituation', 'Probe', 'Tape', 'Dark')

bin_width = 0.25
baseline_window = 10
min_post_duration = 10
max_post_duration = 300
min_firing_rate = 0.2
n_example_neurons = 3    # top neurons per cell type

window_pre = 5            # seconds before reward for raster display
window_post = 5           # seconds after reward for raster display

CELL_COLORS = {
    "Granule Cell": "#c00021",
    "Mossy Cell": "#358940",
    "Narrow Interneuron": "#0050a0",
    "Wide Interneuron": "#00b8cc",
    "Excitatory Principal Cell": "#e36414",
    "Bursty Narrow Interneuron": "#8d165f",
}

path = cg.PROCESSED_FILE_DIR
save_dir = os.path.join(cg.ROOT_DIR, 'figs', 'example_rasters')
os.makedirs(save_dir, exist_ok=True)

# %% === LOAD DATA ===
trial_data = []

for exp in experiments:
    exp_dir = os.path.join(path, exp)
    mat_files = glob.glob(os.path.join(exp_dir, 'hfmE_*.mat'))

    if not mat_files:
        continue

    for fpath in mat_files:
        fname = os.path.basename(fpath)
        match = re.match(r'hfmE_\d{4}-\d{2}-\d{2}_M(\d+)_(.+)\.mat', fname)
        if not match:
            continue

        mouse = match.group(1)
        trial = match.group(2)

        if trial.startswith(exclude_prefixes):
            continue

        try:
            edata = elib.EphysTrial.Load(exp, mouse, trial)
        except Exception as e:
            print(f"  Failed to load {fname}: {e}")
            continue

        if not hasattr(edata, 'k_reward') or edata.k_reward is None:
            continue

        t_reward = edata.time_ttl[edata.k_reward]
        finite_mask = np.isfinite(edata.r_center)[:, 0]
        if not np.any(finite_mask):
            continue
        t_start = edata.time_ttl[finite_mask][0]
        t_end = edata.time_ttl[finite_mask][-1]

        if t_reward > t_end or t_reward < t_start:
            continue

        post_available = t_end - t_reward
        if post_available < min_post_duration:
            continue
        post_available = min(post_available, max_post_duration)

        day = str(edata.day)
        n_neurons = len(edata.t_spikeTrains)

        trial_data.append({
            'exp': exp,
            'mouse': mouse,
            'trial': trial,
            'day': day,
            't_reward': t_reward,
            't_start': t_start,
            't_end': t_end,
            'n_neurons': n_neurons,
            'spike_trains': [edata.t_spikeTrains[i].copy() for i in range(n_neurons)],
            'labels': [],
        })

        # Assign labels — subdivide Excitatory Principal Cells via PCA
        for i in range(n_neurons):
            bitp = edata.cellLabels_BITP[i][0] if hasattr(edata, 'cellLabels_BITP') else 'unknown'
            if bitp == 'Excitatory Principal Cell' and hasattr(edata, 'cellLabels_pca'):
                pca_entry = edata.cellLabels_pca[i]
                if pca_entry.size > 0 and pca_entry[0] != '':
                    bitp = pca_entry[0]
            trial_data[-1]['labels'].append(bitp)

print(f"Loaded {len(trial_data)} valid trials")

# %% === COMPUTE PEAK MODULATION PER NEURON ===
# Use same bins/baseline as the main script
crop_pre = min(td['t_reward'] - td['t_start'] for td in trial_data)
crop_post = min(min(td['t_end'] - td['t_reward'], max_post_duration) for td in trial_data)

bins = np.arange(-crop_pre, crop_post + bin_width, bin_width)
bin_centers = bins[:-1] + bin_width / 2

baseline_start = min(baseline_window, crop_pre)
baseline_mask = (bin_centers >= -baseline_start) & (bin_centers < -1.0)

# Distant baseline for peak modulation
distant_bl_end = -crop_pre + (crop_pre * 0.25)
distant_bl_mask = (bin_centers >= -crop_pre) & (bin_centers < distant_bl_end)
if np.sum(distant_bl_mask) < 2:
    distant_bl_mask = baseline_mask

peak_half_width = 2

# Build neuron data with trial-level spike trains preserved
neuron_trials = defaultdict(lambda: {
    'fr_list': [], 'spike_trials': [], 'label': None, 'trial_info': []
})

for td in trial_data:
    for i in range(td['n_neurons']):
        nid = (td['exp'], td['mouse'], td['day'], i)
        spikes = td['spike_trains'][i]

        # Aligned spikes for firing rate
        aligned = spikes - td['t_reward']
        aligned_w = aligned[(aligned >= bins[0]) & (aligned <= bins[-1])]
        counts, _ = np.histogram(aligned_w, bins=bins)
        fr = counts / bin_width

        neuron_trials[nid]['fr_list'].append(fr)
        neuron_trials[nid]['spike_trials'].append(spikes)
        neuron_trials[nid]['label'] = td['labels'][i]
        neuron_trials[nid]['trial_info'].append({
            'exp': td['exp'],
            'mouse': td['mouse'],
            'trial': td['trial'],
            't_reward': td['t_reward'],
        })

# Compute peak modulation for each neuron
neurons = []
for nid, data in neuron_trials.items():
    avg_fr = np.mean(data['fr_list'], axis=0)
    overall_rate = np.mean(avg_fr)

    if overall_rate < min_firing_rate:
        continue

    peak_idx = np.argmax(avg_fr)
    lo = max(0, peak_idx - peak_half_width)
    hi = min(len(avg_fr), peak_idx + peak_half_width + 1)
    peak_rate = np.mean(avg_fr[lo:hi])
    bl_rate = np.mean(avg_fr[distant_bl_mask])
    denom = peak_rate + bl_rate
    if denom > 0:
        peak_mod = (peak_rate - bl_rate) / denom
    else:
        peak_mod = 0.0

    neurons.append({
        'neuron_id': nid,
        'label': data['label'],
        'peak_mod': peak_mod,
        'peak_time': bin_centers[peak_idx],
        'overall_rate': overall_rate,
        'spike_trials': data['spike_trials'],
        'trial_info': data['trial_info'],
        'avg_fr': avg_fr,
    })

print(f"Neurons passing rate filter: {len(neurons)}")

# Filter out extreme modulation indices (unless high firing rate) and require minimum trials
min_trials = 3
mi_rate_threshold = 2.0  # Hz — neurons above this can keep extreme MI values
neurons = [n for n in neurons
           if (abs(n['peak_mod']) < 1.0 or n['overall_rate'] >= mi_rate_threshold)
           and len(n['spike_trials']) >= min_trials]
print(f"Neurons after filtering (±1.0 MI relaxed if ≥{mi_rate_threshold} Hz, "
      f"≥{min_trials} trials): {len(neurons)}")

# %% === SELECT TOP NEURONS PER CELL TYPE ===
type_groups = defaultdict(list)
for n in neurons:
    type_groups[n['label']].append(n)

# Sort each type by peak modulation (descending) and pick top N
selected = {}
for label in sorted(type_groups.keys()):
    group = sorted(type_groups[label], key=lambda x: x['peak_mod'], reverse=True)
    selected[label] = group[:n_example_neurons]
    print(f"\n{label} — top {len(selected[label])} neurons:")
    for n in selected[label]:
        exp, mouse, day, idx = n['neuron_id']
        print(f"  {exp} M{mouse} day{day} neuron{idx}: "
              f"peak_mod={n['peak_mod']:.3f}, peak_time={n['peak_time']:.2f}s, "
              f"rate={n['overall_rate']:.1f}Hz, n_trials={len(n['spike_trials'])}")

# %% === PLOT RASTERS ===
print("\n=== Plotting rasters ===")

for label, group in sorted(selected.items()):
    color = CELL_COLORS.get(label, '#333333')
    n_neurons_plot = len(group)

    for ni, neuron in enumerate(group):
        nid = neuron['neuron_id']
        exp, mouse, day, idx = nid
        n_trials = len(neuron['spike_trials'])

        fig, ax = plt.subplots(figsize=(12, max(2, n_trials * 0.4 + 1)))

        spike_trains_aligned = []
        trial_labels = []

        for ti in range(n_trials):
            spikes = neuron['spike_trials'][ti]
            t_reward = neuron['trial_info'][ti]['t_reward']
            trial_name = neuron['trial_info'][ti]['trial']

            aligned = spikes - t_reward
            windowed = aligned[(aligned >= -window_pre) & (aligned <= window_post)]
            spike_trains_aligned.append(windowed)
            trial_labels.append(f"T{trial_name}")

        ax.eventplot(spike_trains_aligned,
                     lineoffsets=range(n_trials),
                     linelengths=0.8,
                     colors=color)

        ax.axvline(0, color='green', linewidth=2, label='Reward')

        ax.set_xlim(-window_pre, window_post)
        ax.set_ylim(-0.5, n_trials - 0.5)
        ax.set_yticks(range(n_trials))
        ax.set_yticklabels(trial_labels, fontsize=9)
        ax.set_xlabel('Time relative to reward (s)', fontsize=14)
        ax.set_ylabel('Trial', fontsize=14)
        ax.set_title(f'{label} — {exp} M{mouse} day{day} neuron{idx}\n'
                     f'Peak mod={neuron["peak_mod"]:.3f}, '
                     f'peak time={neuron["peak_time"]:.2f}s, '
                     f'rate={neuron["overall_rate"]:.1f} Hz',
                     fontsize=13)
        ax.legend(loc='upper right', fontsize=10)
        ax.tick_params(labelsize=11)

        plt.tight_layout()
        safe_label = label.replace(' ', '_')
        fname = f'raster_{safe_label}_top{ni+1}_{exp}_M{mouse}_d{day}_n{idx}.png'
        fpath = os.path.join(save_dir, fname)
        fig.savefig(fpath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {fpath}")

print(f"\nDone! All rasters saved to: {save_dir}")