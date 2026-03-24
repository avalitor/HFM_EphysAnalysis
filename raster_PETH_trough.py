# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 16:06:12 2026

@author: student
"""

"""
Trough-aligned analysis: identify neurons that go quiet around reward.
- Smoothed firing rate trough detection
- Trough modulation index and timing distributions
- Grouped PETHs by cell type
- Speed PETH comparison
- Example raster plots for most suppressed neurons
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.ndimage import uniform_filter1d

import glob
import os
import re
import lib_ephys_obj as elib
import config as cg

# %% === PARAMETERS ===
experiments = ['2024-02-15', '2024-11-09', '2025-01-21']
exclude_prefixes = ('Habituation', 'Probe', 'Tape', 'Dark')

bin_width = 0.25         # seconds per bin
baseline_window = 10     # baseline: -baseline_window to -1 s
min_post_duration = 10
max_post_duration = 300
min_firing_rate = 0.2    # Hz
min_trials = 2           # require more than 1 trial
smooth_bins = 5          # number of bins for smoothing before trough detection
trough_half_width = 2    # bins on each side of trough for averaging
n_example_neurons = 3    # top suppressed neurons per cell type to plot

raster_window_pre = 5    # seconds before reward for raster display
raster_window_post = 5   # seconds after reward for raster display

CELL_COLORS = {
    "Granule Cell": "#c00021",
    "Mossy Cell": "#358940",
    "Narrow Interneuron": "#0050a0",
    "Wide Interneuron": "#00b8cc",
    "Excitatory Principal Cell": "#e36414",
    "Bursty Narrow Interneuron": "#8d165f",
}

path = cg.PROCESSED_FILE_DIR
save_dir = os.path.join(cg.ROOT_DIR, 'figs', 'trough_analysis')
os.makedirs(save_dir, exist_ok=True)

# %% === LOAD DATA ===
trial_data = []

for exp in experiments:
    exp_dir = os.path.join(path, exp)
    mat_files = glob.glob(os.path.join(exp_dir, 'hfmE_*.mat'))

    if not mat_files:
        print(f"No .mat files found in {exp_dir}, skipping.")
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
            'pre_available': t_reward - t_start,
            'post_available': post_available,
            'n_neurons': n_neurons,
            'spike_trains': [edata.t_spikeTrains[i].copy() for i in range(n_neurons)],
            'labels': [],
            'velocity': edata.velocity.copy(),
            'time_ttl': edata.time_ttl.copy(),
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

if len(trial_data) == 0:
    raise ValueError("No valid trials found.")

# %% === CROP WINDOW AND BINS ===
window_pre = min(td['pre_available'] for td in trial_data)
window_post = min(td['post_available'] for td in trial_data)
print(f"Crop window: -{window_pre:.2f} to +{window_post:.2f} s")

baseline_start = min(baseline_window, window_pre)
baseline_mask_start = -baseline_start
baseline_mask_end = -1.0
print(f"Baseline: {baseline_mask_start:.2f} to {baseline_mask_end:.2f} s")

bins = np.arange(-window_pre, window_post + bin_width, bin_width)
bin_centers = bins[:-1] + bin_width / 2
n_bins = len(bin_centers)

baseline_mask = (bin_centers >= baseline_mask_start) & (bin_centers < baseline_mask_end)

# Distant baseline for trough modulation: earliest quarter of pre-reward
distant_bl_end = -window_pre + (window_pre * 0.25)
distant_bl_mask = (bin_centers >= -window_pre) & (bin_centers < distant_bl_end)
if np.sum(distant_bl_mask) < 2:
    distant_bl_mask = baseline_mask

# %% === BUILD PER-NEURON DATA ===
neuron_data = defaultdict(lambda: {
    'fr_list': [], 'z_list': [], 'spike_trials': [],
    'label': None, 'trial_info': []
})

for td in trial_data:
    for i in range(td['n_neurons']):
        nid = (td['exp'], td['mouse'], td['day'], i)
        spikes = td['spike_trains'][i]

        aligned = spikes - td['t_reward']
        aligned_w = aligned[(aligned >= bins[0]) & (aligned <= bins[-1])]
        counts, _ = np.histogram(aligned_w, bins=bins)
        fr = counts / bin_width

        bl_mean = np.mean(fr[baseline_mask])
        bl_std = np.std(fr[baseline_mask])
        if bl_std == 0:
            bl_std = 1.0
        z = (fr - bl_mean) / bl_std

        neuron_data[nid]['fr_list'].append(fr)
        neuron_data[nid]['z_list'].append(z)
        neuron_data[nid]['spike_trials'].append(spikes)
        neuron_data[nid]['label'] = td['labels'][i]
        neuron_data[nid]['trial_info'].append({
            'exp': td['exp'],
            'mouse': td['mouse'],
            'trial': td['trial'],
            't_reward': td['t_reward'],
        })

# Average across trials, compute trough modulation
neurons = []
for nid, data in neuron_data.items():
    n_trials = len(data['fr_list'])
    if n_trials < min_trials:
        continue

    avg_fr = np.mean(data['fr_list'], axis=0)
    avg_z = np.mean(data['z_list'], axis=0)
    overall_rate = np.mean(avg_fr)

    if overall_rate < min_firing_rate:
        continue

    # Smooth before finding trough to avoid single empty bins
    smoothed_fr = uniform_filter1d(avg_fr, size=smooth_bins)

    # Find trough
    trough_idx = np.argmin(smoothed_fr)
    lo = max(0, trough_idx - trough_half_width)
    hi = min(n_bins, trough_idx + trough_half_width + 1)
    trough_rate = np.mean(avg_fr[lo:hi])

    bl_rate = np.mean(avg_fr[distant_bl_mask])
    denom = trough_rate + bl_rate
    if denom > 0:
        trough_mod = (trough_rate - bl_rate) / denom
    else:
        trough_mod = 0.0

    neurons.append({
        'neuron_id': nid,
        'label': data['label'],
        'avg_fr': avg_fr,
        'avg_z': avg_z,
        'smoothed_fr': smoothed_fr,
        'overall_rate': overall_rate,
        'trough_mod': trough_mod,
        'trough_time': bin_centers[trough_idx],
        'n_trials': n_trials,
        'spike_trials': data['spike_trials'],
        'trial_info': data['trial_info'],
    })

# Filter out extreme ±1.0 values (unless high firing rate)
mi_rate_threshold = 2.0  # Hz — neurons above this can keep extreme MI values
neurons = [n for n in neurons
           if abs(n['trough_mod']) < 1.0 or n['overall_rate'] >= mi_rate_threshold]

print(f"Neurons after filters (rate ≥ {min_firing_rate} Hz, "
      f"trials ≥ {min_trials}, ±1.0 MI relaxed if ≥{mi_rate_threshold} Hz): {len(neurons)}")

# %% === GROUP BY CELL TYPE ===
type_groups = defaultdict(list)
for n in neurons:
    type_groups[n['label']].append(n)

label_colors = {label: CELL_COLORS.get(label, '#333333') for label in type_groups.keys()}
n_types = len(type_groups)

# Print summary
print("\nTrough modulation summary:")
for label in sorted(type_groups.keys()):
    group = type_groups[label]
    mods = [n['trough_mod'] for n in group]
    n_suppressed = sum(1 for m in mods if m < -0.1)
    print(f"  {label} (n={len(group)}): mean MI={np.mean(mods):.3f}, "
          f"{n_suppressed} suppressed (MI < -0.1)")


# %% =====================================================================
# ANALYSIS 1: GROUPED PETHs
# =====================================================================
print("\n=== Grouped PETHs ===")

# --- Z-scored, separate panels ---
fig, axes = plt.subplots(n_types, 1, figsize=(14, 3.5 * n_types), sharex=True)
if n_types == 1:
    axes = [axes]

for ax, (label, group) in zip(axes, sorted(type_groups.items())):
    z_mat = np.array([n['avg_z'] for n in group])
    mean_z = np.mean(z_mat, axis=0)
    sem_z = np.std(z_mat, axis=0) / np.sqrt(len(group))

    color = label_colors[label]
    ax.fill_between(bin_centers, mean_z - sem_z, mean_z + sem_z,
                    alpha=0.3, color=color)
    ax.plot(bin_centers, mean_z, color=color, linewidth=2)
    ax.axvline(0, color='black', linewidth=1.5, linestyle='--')
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.set_ylabel('Z-score', fontsize=14)
    ax.set_title(f'{label}  (n={len(group)})', fontsize=15)
    ax.tick_params(labelsize=12)

axes[-1].set_xlabel('Time relative to reward (s)', fontsize=14)
fig.suptitle('Grouped PETHs by cell type (z-scored, mean ± SEM)',
             fontsize=16, y=1.01)
plt.tight_layout()
fpath = os.path.join(save_dir, 'trough_peth_zscore.png')
fig.savefig(fpath, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  Saved: {fpath}")

# --- All types overlaid ---
fig, ax = plt.subplots(figsize=(14, 5))

for label in sorted(type_groups.keys()):
    group = type_groups[label]
    z_mat = np.array([n['avg_z'] for n in group])
    mean_z = np.mean(z_mat, axis=0)
    sem_z = np.std(z_mat, axis=0) / np.sqrt(len(group))
    color = label_colors[label]

    ax.fill_between(bin_centers, mean_z - sem_z, mean_z + sem_z,
                    alpha=0.2, color=color)
    ax.plot(bin_centers, mean_z, color=color, linewidth=2,
            label=f'{label} (n={len(group)})')

ax.axvline(0, color='black', linewidth=1.5, linestyle='--', label='Reward')
ax.axhline(0, color='gray', linewidth=0.5)
ax.set_xlabel('Time relative to reward (s)', fontsize=14)
ax.set_ylabel('Z-score', fontsize=14)
ax.set_title('Grouped PETHs — all cell types overlaid', fontsize=16)
ax.legend(fontsize=12, loc='upper right')
ax.tick_params(labelsize=12)
plt.tight_layout()
fpath = os.path.join(save_dir, 'trough_peth_overlaid.png')
fig.savefig(fpath, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  Saved: {fpath}")


# %% =====================================================================
# ANALYSIS 2: SPEED PETH
# =====================================================================
print("\n=== Speed PETH ===")

speed_traces = []
for td in trial_data:
    vel = np.ravel(td['velocity'].copy())
    t = td['time_ttl']
    t_aligned = t - td['t_reward']

    binned_speed = np.full(n_bins, np.nan)
    for b in range(n_bins):
        mask = (t_aligned >= bins[b]) & (t_aligned < bins[b + 1])
        valid = vel[mask]
        valid = valid[np.isfinite(valid)]
        if len(valid) > 0:
            binned_speed[b] = np.mean(valid)
    speed_traces.append(binned_speed)

speed_matrix = np.array(speed_traces)
mean_speed = np.nanmean(speed_matrix, axis=0)
n_valid = np.sum(np.isfinite(speed_matrix), axis=0)
sem_speed = np.nanstd(speed_matrix, axis=0) / np.sqrt(np.maximum(n_valid, 1))

# Speed stacked with neural PETHs
fig, axes = plt.subplots(n_types + 1, 1, figsize=(14, 3.5 * (n_types + 1)), sharex=True)

ax = axes[0]
ax.fill_between(bin_centers, mean_speed - sem_speed, mean_speed + sem_speed,
                alpha=0.3, color='gray')
ax.plot(bin_centers, mean_speed, color='black', linewidth=2)
ax.axvline(0, color='black', linewidth=1.5, linestyle='--')
ax.set_ylabel('Speed (cm/s)', fontsize=14)
ax.set_title(f'Speed ({len(speed_traces)} trials)', fontsize=15)
ax.tick_params(labelsize=12)

for ax, (label, group) in zip(axes[1:], sorted(type_groups.items())):
    z_mat = np.array([n['avg_z'] for n in group])
    mean_z = np.mean(z_mat, axis=0)
    sem_z = np.std(z_mat, axis=0) / np.sqrt(len(group))

    color = label_colors[label]
    ax.fill_between(bin_centers, mean_z - sem_z, mean_z + sem_z,
                    alpha=0.3, color=color)
    ax.plot(bin_centers, mean_z, color=color, linewidth=2)
    ax.axvline(0, color='black', linewidth=1.5, linestyle='--')
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.set_ylabel('Z-score', fontsize=14)
    ax.set_title(f'{label}  (n={len(group)})', fontsize=15)
    ax.tick_params(labelsize=12)

axes[-1].set_xlabel('Time relative to reward (s)', fontsize=14)
fig.suptitle('Speed vs neural PETHs aligned to reward', fontsize=16, y=1.01)
plt.tight_layout()
fpath = os.path.join(save_dir, 'trough_speed_vs_neural_peth.png')
fig.savefig(fpath, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  Saved: {fpath}")


# %% =====================================================================
# ANALYSIS 3: TROUGH MODULATION INDEX
# =====================================================================
print("\n=== Trough modulation index ===")

# --- Histogram per cell type ---
sorted_labels = sorted(type_groups.keys())
fig, axes = plt.subplots(n_types, 1, figsize=(10, 3.5 * n_types), sharex=True)
if n_types == 1:
    axes = [axes]

for ax, label in zip(axes, sorted_labels):
    group = type_groups[label]
    mi_vals = [n['trough_mod'] for n in group]
    color = label_colors[label]
    ax.hist(mi_vals, bins=20, range=(-1, 1), alpha=0.7, color=color,
            edgecolor='black')
    ax.axvline(0, color='black', linewidth=1.5, linestyle='--')
    mean_mi = np.mean(mi_vals)
    ax.axvline(mean_mi, color=color, linewidth=2, label=f'Mean = {mean_mi:.3f}')
    ax.set_ylabel('Count', fontsize=14)
    ax.set_title(f'{label}  (n={len(group)})', fontsize=15)
    ax.legend(fontsize=12)
    ax.tick_params(labelsize=12)

axes[-1].set_xlabel('Trough modulation index', fontsize=14)
fig.suptitle(f'Trough modulation by cell type (neurons ≥ {min_firing_rate} Hz, '
             f'≥ {min_trials} trials)', fontsize=16, y=1.01)
plt.tight_layout()
fpath = os.path.join(save_dir, 'trough_modulation_histogram.png')
fig.savefig(fpath, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  Saved: {fpath}")

# --- Strip plot ---
fig, ax = plt.subplots(figsize=(10, 5))

for i, label in enumerate(sorted_labels):
    group = type_groups[label]
    mi_vals = np.array([n['trough_mod'] for n in group])
    color = label_colors[label]
    jitter = np.random.uniform(-0.2, 0.2, size=len(mi_vals))
    ax.scatter(np.full_like(mi_vals, i) + jitter, mi_vals,
               color=color, alpha=0.6, s=40, edgecolors='k', linewidths=0.5)
    mean_mi = np.mean(mi_vals)
    sem_mi = np.std(mi_vals) / np.sqrt(len(mi_vals))
    ax.errorbar(i + 0.35, mean_mi, yerr=sem_mi, fmt='o', color='black',
                markersize=8, capsize=5, linewidth=2)

ax.axhline(0, color='gray', linewidth=0.5)
ax.set_xticks(range(len(sorted_labels)))
ax.set_xticklabels(sorted_labels, fontsize=12, rotation=15, ha='right')
ax.set_ylabel('Trough modulation index', fontsize=14)
ax.set_title(f'Trough modulation by cell type (dots = neurons, black = mean ± SEM)',
             fontsize=14)
ax.tick_params(labelsize=12)
plt.tight_layout()
fpath = os.path.join(save_dir, 'trough_modulation_strip.png')
fig.savefig(fpath, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  Saved: {fpath}")


# %% =====================================================================
# ANALYSIS 4: TROUGH TIME DISTRIBUTION
# =====================================================================
print("\n=== Trough time distribution ===")

fig, axes = plt.subplots(n_types, 1, figsize=(10, 3.5 * n_types), sharex=True)
if n_types == 1:
    axes = [axes]

for ax, label in zip(axes, sorted_labels):
    group = type_groups[label]
    trough_times = [n['trough_time'] for n in group]
    color = label_colors[label]
    ax.hist(trough_times, bins=20, alpha=0.7, color=color, edgecolor='black')
    ax.axvline(0, color='black', linewidth=1.5, linestyle='--')
    ax.set_ylabel('Count', fontsize=14)
    ax.set_title(f'{label}  (n={len(group)})', fontsize=15)
    ax.tick_params(labelsize=12)

axes[-1].set_xlabel('Trough time relative to reward (s)', fontsize=14)
fig.suptitle(f'Distribution of trough firing times (neurons ≥ {min_firing_rate} Hz, '
             f'≥ {min_trials} trials)', fontsize=16, y=1.01)
plt.tight_layout()
fpath = os.path.join(save_dir, 'trough_time_distribution.png')
fig.savefig(fpath, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  Saved: {fpath}")


# %% =====================================================================
# ANALYSIS 5: EXAMPLE RASTERS — MOST SUPPRESSED NEURONS
# =====================================================================
print("\n=== Example rasters ===")

# Select top N most suppressed (most negative MI) per cell type
selected = {}
for label in sorted(type_groups.keys()):
    group = sorted(type_groups[label], key=lambda x: x['trough_mod'])
    selected[label] = group[:n_example_neurons]
    print(f"\n{label} — top {len(selected[label])} suppressed neurons:")
    for n in selected[label]:
        exp, mouse, day, idx = n['neuron_id']
        print(f"  {exp} M{mouse} day{day} neuron{idx}: "
              f"trough_mod={n['trough_mod']:.3f}, trough_time={n['trough_time']:.2f}s, "
              f"rate={n['overall_rate']:.1f}Hz, n_trials={n['n_trials']}")

for label, group in sorted(selected.items()):
    color = CELL_COLORS.get(label, '#333333')

    for ni, neuron in enumerate(group):
        nid = neuron['neuron_id']
        exp, mouse, day, idx = nid
        n_trials = neuron['n_trials']

        # --- Raster plot ---
        fig, axes = plt.subplots(2, 1, figsize=(12, max(3, n_trials * 0.4 + 1) + 3),
                                 gridspec_kw={'height_ratios': [1, max(1, n_trials * 0.3)]},
                                 sharex=True)

        # Top panel: smoothed firing rate
        ax_fr = axes[0]
        ax_fr.fill_between(bin_centers,
                           neuron['avg_fr'] - neuron['avg_fr'] * 0,  # placeholder
                           neuron['smoothed_fr'],
                           alpha=0.3, color=color)
        ax_fr.plot(bin_centers, neuron['avg_fr'], color=color, alpha=0.4,
                   linewidth=1, label='Raw')
        ax_fr.plot(bin_centers, neuron['smoothed_fr'], color=color,
                   linewidth=2, label='Smoothed')
        ax_fr.axvline(0, color='green', linewidth=2)
        ax_fr.axvline(neuron['trough_time'], color='blue', linewidth=1.5,
                      linestyle=':', label=f'Trough ({neuron["trough_time"]:.2f}s)')
        ax_fr.set_ylabel('Firing rate (Hz)', fontsize=13)
        ax_fr.set_title(f'{label} — {exp} M{mouse} day{day} neuron{idx}\n'
                        f'Trough MI={neuron["trough_mod"]:.3f}, '
                        f'trough time={neuron["trough_time"]:.2f}s, '
                        f'rate={neuron["overall_rate"]:.1f} Hz',
                        fontsize=13)
        ax_fr.legend(fontsize=10, loc='upper right')
        ax_fr.tick_params(labelsize=11)

        # Bottom panel: raster
        ax_rast = axes[1]
        spike_trains_aligned = []
        trial_labels = []

        for ti in range(n_trials):
            spikes = neuron['spike_trials'][ti]
            t_reward = neuron['trial_info'][ti]['t_reward']
            trial_name = neuron['trial_info'][ti]['trial']

            aligned = spikes - t_reward
            windowed = aligned[(aligned >= -raster_window_pre) &
                               (aligned <= raster_window_post)]
            spike_trains_aligned.append(windowed)
            trial_labels.append(f"T{trial_name}")

        ax_rast.eventplot(spike_trains_aligned,
                          lineoffsets=range(n_trials),
                          linelengths=0.8,
                          colors=color)

        ax_rast.axvline(0, color='green', linewidth=2, label='Reward')
        ax_rast.axvline(neuron['trough_time'], color='blue', linewidth=1.5,
                        linestyle=':', label='Trough')

        ax_rast.set_xlim(-raster_window_pre, raster_window_post)
        ax_rast.set_ylim(-0.5, n_trials - 0.5)
        ax_rast.set_yticks(range(n_trials))
        ax_rast.set_yticklabels(trial_labels, fontsize=9)
        ax_rast.set_xlabel('Time relative to reward (s)', fontsize=14)
        ax_rast.set_ylabel('Trial', fontsize=14)
        ax_rast.legend(loc='upper right', fontsize=10)
        ax_rast.tick_params(labelsize=11)

        plt.tight_layout()
        safe_label = label.replace(' ', '_')
        fname = (f'trough_raster_{safe_label}_top{ni+1}_'
                 f'{exp}_M{mouse}_d{day}_n{idx}.png')
        fpath_out = os.path.join(save_dir, fname)
        fig.savefig(fpath_out, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {fpath_out}")

print(f"\nDone! All figures saved to: {save_dir}")