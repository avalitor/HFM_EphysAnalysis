"""
1. Grouped PETHs: average peri-event time histogram per cell type
2. Reward modulation index: per-neuron comparison of post-reward vs baseline
   firing rate, visualized as distributions and overlaid on BI/TP scatter
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from collections import defaultdict
import glob
import os
import re
import lib_ephys_obj as elib
import config as cg

# %% === PARAMETERS ===
experiments = ['2024-11-09', '2025-01-21']
exclude_prefixes = ('Habituation', 'Probe', 'Tape', 'Dark')

bin_width = 0.25         # seconds per bin (finer for PETHs)
baseline_window = 10     # seconds before reward for baseline (-baseline to -1 s)
post_window = 3          # seconds after reward for modulation index (0 to post_window)
min_post_duration = 10   # seconds — exclude trials with less post-reward tracking
max_post_duration = 300  # seconds — cap post-reward data to avoid long recording tails

path = cg.PROCESSED_FILE_DIR
save_dir = os.path.join(cg.ROOT_DIR, 'figs')
os.makedirs(save_dir, exist_ok=True)

# %% === LOAD DATA ===
trial_data = []
bitp_cache = {}  # cache BI/TP values per (exp, mouse, day)

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
            print(f"  Reward outside tracking window in {fname}, skipping.")
            continue

        post_available = t_end - t_reward

        # Skip trials with too little post-reward data
        if post_available < min_post_duration:
            print(f"  Only {post_available:.1f}s post-reward in {fname}, skipping.")
            continue

        # Cap post-reward duration to avoid long recording tails
        post_available = min(post_available, max_post_duration)

        day = str(edata.day)
        n_neurons = len(edata.t_spikeTrains)

        # Load BI/TP values once per (exp, mouse, day)
        day_key = (exp, mouse, day)
        if day_key not in bitp_cache:
            try:
                edata_cm = elib.EphysTrial.Load(exp, mouse, trial, load_cell_metrics=True)
                cm = edata_cm.cell_metrics
                bitp_cache[day_key] = {
                    'bi': np.array(cm['burstIndex_Royer2012']).flatten(),
                    'tp': np.array(cm['troughToPeak']).flatten(),
                }
                print(f"  Loaded cell_metrics for {day_key}")
            except Exception as e:
                print(f"  Could not load cell_metrics for {day_key}: {e}")
                bitp_cache[day_key] = {'bi': None, 'tp': None}

        bi_vals = bitp_cache[day_key]['bi']
        tp_vals = bitp_cache[day_key]['tp']

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
            'labels': [edata.cellLabels_BITP[i][0] if hasattr(edata, 'cellLabels_BITP')
                       else 'unknown' for i in range(n_neurons)],
            'bi_vals': bi_vals,
            'tp_vals': tp_vals,
            'velocity': edata.velocity.copy(),
            'time_ttl': edata.time_ttl.copy(),
        })

print(f"Loaded {len(trial_data)} valid trials")

# %% === CROP WINDOW ===
window_pre = min(td['pre_available'] for td in trial_data)
window_post_crop = min(td['post_available'] for td in trial_data)
print(f"Crop window: -{window_pre:.2f} to +{window_post_crop:.2f} s")

baseline_start = min(baseline_window, window_pre)
print(f"Baseline: -{baseline_start:.2f} to -1.0 s")
print(f"Post-reward window for modulation index: 0 to +{post_window} s")

bins = np.arange(-window_pre, window_post_crop + bin_width, bin_width)
bin_centers = bins[:-1] + bin_width / 2
n_bins = len(bin_centers)

baseline_mask = (bin_centers >= -baseline_start) & (bin_centers < -1.0)
post_mask = (bin_centers >= 0) & (bin_centers < post_window)

# %% === BUILD PER-NEURON DATA (trial-averaged) ===
neuron_groups = defaultdict(lambda: {
    'z_list': [], 'fr_list': [], 'label': None, 'bi': None, 'tp': None
})

for td in trial_data:
    for i in range(td['n_neurons']):
        nid = (td['exp'], td['mouse'], td['day'], i)
        spikes = td['spike_trains'][i] - td['t_reward']
        spikes_w = spikes[(spikes >= bins[0]) & (spikes <= bins[-1])]
        counts, _ = np.histogram(spikes_w, bins=bins)
        fr = counts / bin_width

        bl_mean = np.mean(fr[baseline_mask])
        bl_std = np.std(fr[baseline_mask])
        if bl_std == 0:
            bl_std = 1.0
        z = (fr - bl_mean) / bl_std

        neuron_groups[nid]['z_list'].append(z)
        neuron_groups[nid]['fr_list'].append(fr)
        neuron_groups[nid]['label'] = td['labels'][i]
        if td['bi_vals'] is not None and i < len(td['bi_vals']):
            neuron_groups[nid]['bi'] = td['bi_vals'][i]
        if td['tp_vals'] is not None and i < len(td['tp_vals']):
            neuron_groups[nid]['tp'] = td['tp_vals'][i]

# Average across trials for each unique neuron
neurons = []
for nid, data in neuron_groups.items():
    avg_z = np.mean(data['z_list'], axis=0)
    avg_fr = np.mean(data['fr_list'], axis=0)
    neurons.append({
        'neuron_id': nid,
        'label': data['label'],
        'avg_z': avg_z,
        'avg_fr': avg_fr,
        'bi': data['bi'],
        'tp': data['tp'],
        'n_trials': len(data['z_list']),
    })

print(f"Unique neurons: {len(neurons)}")

# %% =====================================================================
# ANALYSIS 1: GROUPED PETHs BY CELL TYPE
# =====================================================================
print("\n=== Grouped PETHs ===")

# Group neurons by label
type_groups = defaultdict(list)
for n in neurons:
    type_groups[n['label']].append(n)

label_colors = {}
default_palette = ['#5f0f40', '#e36414', '#0f4c5c', '#9a031e', '#fb8b24']
for i, label in enumerate(sorted(type_groups.keys())):
    label_colors[label] = default_palette[i % len(default_palette)]

# --- Plot 1a: Separate panel per cell type (z-scored) ---
n_types = len(type_groups)
fig, axes = plt.subplots(n_types, 1, figsize=(14, 3.5 * n_types), sharex=True)
if n_types == 1:
    axes = [axes]

for ax, (label, group) in zip(axes, sorted(type_groups.items())):
    z_matrix = np.array([n['avg_z'] for n in group])
    mean_z = np.mean(z_matrix, axis=0)
    sem_z = np.std(z_matrix, axis=0) / np.sqrt(len(group))

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
fpath = os.path.join(save_dir, 'peth_by_celltype_panels.png')
fig.savefig(fpath, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  Saved: {fpath}")

# --- Plot 1b: All cell types overlaid ---
fig, ax = plt.subplots(figsize=(14, 5))

for label in sorted(type_groups.keys()):
    group = type_groups[label]
    z_matrix = np.array([n['avg_z'] for n in group])
    mean_z = np.mean(z_matrix, axis=0)
    sem_z = np.std(z_matrix, axis=0) / np.sqrt(len(group))
    color = label_colors[label]

    ax.fill_between(bin_centers, mean_z - sem_z, mean_z + sem_z,
                    alpha=0.2, color=color)
    ax.plot(bin_centers, mean_z, color=color, linewidth=2, label=f'{label} (n={len(group)})')

ax.axvline(0, color='black', linewidth=1.5, linestyle='--', label='Reward')
ax.axhline(0, color='gray', linewidth=0.5)
ax.set_xlabel('Time relative to reward (s)', fontsize=14)
ax.set_ylabel('Z-score', fontsize=14)
ax.set_title('Grouped PETHs — all cell types overlaid', fontsize=16)
ax.legend(fontsize=12, loc='upper right')
ax.tick_params(labelsize=12)
plt.tight_layout()
fpath = os.path.join(save_dir, 'peth_by_celltype_overlaid.png')
fig.savefig(fpath, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  Saved: {fpath}")

# --- Plot 1c: Raw firing rate version (not z-scored) ---
fig, axes = plt.subplots(n_types, 1, figsize=(14, 3.5 * n_types), sharex=True)
if n_types == 1:
    axes = [axes]

for ax, (label, group) in zip(axes, sorted(type_groups.items())):
    fr_matrix = np.array([n['avg_fr'] for n in group])
    mean_fr = np.mean(fr_matrix, axis=0)
    sem_fr = np.std(fr_matrix, axis=0) / np.sqrt(len(group))

    color = label_colors[label]
    ax.fill_between(bin_centers, mean_fr - sem_fr, mean_fr + sem_fr,
                    alpha=0.3, color=color)
    ax.plot(bin_centers, mean_fr, color=color, linewidth=2)
    ax.axvline(0, color='black', linewidth=1.5, linestyle='--')
    ax.set_ylabel('Firing rate (Hz)', fontsize=14)
    ax.set_title(f'{label}  (n={len(group)})', fontsize=15)
    ax.tick_params(labelsize=12)

axes[-1].set_xlabel('Time relative to reward (s)', fontsize=14)
fig.suptitle('Grouped PETHs by cell type (raw firing rate, mean ± SEM)',
             fontsize=16, y=1.01)
plt.tight_layout()
fpath = os.path.join(save_dir, 'peth_by_celltype_raw.png')
fig.savefig(fpath, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  Saved: {fpath}")


# %% =====================================================================
# ANALYSIS 1d: SPEED PETH — check if neural bumps track speed changes
# =====================================================================
print("\n=== Speed PETH ===")

# For each trial, interpolate velocity onto the same time bins used for spikes
# velocity maps onto time_ttl, so we align to reward the same way
speed_traces = []

for td in trial_data:
    vel = td['velocity'].copy()
    t = td['time_ttl'] - td['t_reward']  # align to reward

    # Velocity might be (n,) or (n,1) — flatten
    vel = np.ravel(vel)

    # Bin velocity into the same bins as firing rate
    binned_speed = np.full(n_bins, np.nan)
    for b in range(n_bins):
        mask = (t >= bins[b]) & (t < bins[b + 1])
        valid = vel[mask]
        valid = valid[np.isfinite(valid)]
        if len(valid) > 0:
            binned_speed[b] = np.mean(valid)

    speed_traces.append(binned_speed)

speed_matrix = np.array(speed_traces)

# Compute mean and SEM ignoring NaN
mean_speed = np.nanmean(speed_matrix, axis=0)
n_valid = np.sum(np.isfinite(speed_matrix), axis=0)
sem_speed = np.nanstd(speed_matrix, axis=0) / np.sqrt(np.maximum(n_valid, 1))

# --- Plot: Speed PETH alone ---
fig, ax = plt.subplots(figsize=(14, 4))
ax.fill_between(bin_centers, mean_speed - sem_speed, mean_speed + sem_speed,
                alpha=0.3, color='gray')
ax.plot(bin_centers, mean_speed, color='black', linewidth=2)
ax.axvline(0, color='black', linewidth=1.5, linestyle='--')
ax.set_xlabel('Time relative to reward (s)', fontsize=14)
ax.set_ylabel('Speed', fontsize=14)
ax.set_title(f'Speed PETH aligned to reward (n={len(trial_data)} trials, mean ± SEM)',
             fontsize=16)
ax.tick_params(labelsize=12)
plt.tight_layout()
fpath = os.path.join(save_dir, 'speed_peth.png')
fig.savefig(fpath, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  Saved: {fpath}")

# --- Plot: Speed PETH stacked with neural PETHs for direct comparison ---
fig, axes = plt.subplots(n_types + 1, 1, figsize=(14, 3.5 * (n_types + 1)), sharex=True)

# Top panel: speed
ax = axes[0]
ax.fill_between(bin_centers, mean_speed - sem_speed, mean_speed + sem_speed,
                alpha=0.3, color='gray')
ax.plot(bin_centers, mean_speed, color='black', linewidth=2)
ax.axvline(0, color='black', linewidth=1.5, linestyle='--')
ax.set_ylabel('Speed', fontsize=14)
ax.set_title(f'Speed (n={len(trial_data)} trials)', fontsize=15)
ax.tick_params(labelsize=12)

# Remaining panels: z-scored PETHs per cell type
for ax, (label, group) in zip(axes[1:], sorted(type_groups.items())):
    z_matrix = np.array([n['avg_z'] for n in group])
    mean_z = np.mean(z_matrix, axis=0)
    sem_z = np.std(z_matrix, axis=0) / np.sqrt(len(group))

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
fpath = os.path.join(save_dir, 'speed_vs_neural_peth.png')
fig.savefig(fpath, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  Saved: {fpath}")


# %% =====================================================================
# ANALYSIS 2: REWARD MODULATION INDEX
# =====================================================================
print("\n=== Reward modulation index ===")

# Compute modulation index per neuron: (post - baseline) / (post + baseline)
# Ranges from -1 (complete suppression) to +1 (complete activation)
min_firing_rate = 0.5  # Hz — exclude neurons below this to avoid ±1 artifacts
mod_indices = []
for n in neurons:
    fr = n['avg_fr']
    bl_rate = np.mean(fr[baseline_mask])
    post_rate = np.mean(fr[post_mask])
    overall_rate = np.mean(fr)
    denom = post_rate + bl_rate
    if denom > 0:
        mi = (post_rate - bl_rate) / denom
    else:
        mi = 0.0
    n['mod_index'] = mi
    n['overall_rate'] = overall_rate

# Filter out low-firing neurons
neurons_filtered = [n for n in neurons if n['overall_rate'] >= min_firing_rate]
n_excluded = len(neurons) - len(neurons_filtered)
print(f"Excluding {n_excluded} neurons with firing rate < {min_firing_rate} Hz")
print(f"Neurons remaining: {len(neurons_filtered)}")

# Rebuild type groups with filtered neurons
type_groups_filt = defaultdict(list)
for n in neurons_filtered:
    type_groups_filt[n['label']].append(n)

mod_indices = np.array([n['mod_index'] for n in neurons_filtered])

# --- Plot 2a: Separate panels per cell type ---
sorted_labels = sorted(type_groups_filt.keys())
n_types_filt = len(sorted_labels)
fig, axes = plt.subplots(n_types_filt, 1, figsize=(10, 3.5 * n_types_filt), sharex=True)
if n_types_filt == 1:
    axes = [axes]

for ax, label in zip(axes, sorted_labels):
    group = type_groups_filt[label]
    mi_vals = [n['mod_index'] for n in group]
    color = label_colors[label]
    ax.hist(mi_vals, bins=20, range=(-1, 1), alpha=0.7, color=color,
            edgecolor='black')
    ax.axvline(0, color='black', linewidth=1.5, linestyle='--')
    mean_mi = np.mean(mi_vals)
    ax.axvline(mean_mi, color=color, linewidth=2, linestyle='-',
               label=f'Mean = {mean_mi:.3f}')
    ax.set_ylabel('Count', fontsize=14)
    ax.set_title(f'{label}  (n={len(group)})', fontsize=15)
    ax.legend(fontsize=12)
    ax.tick_params(labelsize=12)

axes[-1].set_xlabel('Reward modulation index', fontsize=14)
fig.suptitle(f'Reward modulation index by cell type (neurons ≥ {min_firing_rate} Hz)',
             fontsize=16, y=1.01)
plt.tight_layout()
fpath = os.path.join(save_dir, 'reward_modulation_histogram.png')
fig.savefig(fpath, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  Saved: {fpath}")

# --- Plot 2b: Strip/swarm plot of modulation index by cell type ---
fig, ax = plt.subplots(figsize=(10, 5))

for i, label in enumerate(sorted_labels):
    group = type_groups_filt[label]
    mi_vals = np.array([n['mod_index'] for n in group])
    color = label_colors[label]
    jitter = np.random.uniform(-0.2, 0.2, size=len(mi_vals))
    ax.scatter(np.full_like(mi_vals, i) + jitter, mi_vals,
               color=color, alpha=0.6, s=40, edgecolors='k', linewidths=0.5)
    # Mean and SEM bar
    mean_mi = np.mean(mi_vals)
    sem_mi = np.std(mi_vals) / np.sqrt(len(mi_vals))
    ax.errorbar(i + 0.35, mean_mi, yerr=sem_mi, fmt='o', color='black',
                markersize=8, capsize=5, linewidth=2)

ax.axhline(0, color='gray', linewidth=0.5)
ax.set_xticks(range(len(sorted_labels)))
ax.set_xticklabels(sorted_labels, fontsize=12, rotation=15, ha='right')
ax.set_ylabel('Reward modulation index', fontsize=14)
ax.set_title(f'Reward modulation by cell type — neurons ≥ {min_firing_rate} Hz '
             f'(dots = neurons, black = mean ± SEM)', fontsize=14)
ax.tick_params(labelsize=12)
plt.tight_layout()
fpath = os.path.join(save_dir, 'reward_modulation_strip.png')
fig.savefig(fpath, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  Saved: {fpath}")

# --- Plot 2c: Overlay on BI/TP scatter plot ---
# Only for neurons where BI and TP values are available
neurons_with_bitp = [n for n in neurons_filtered if n['bi'] is not None and n['tp'] is not None]
print(f"Neurons with BI/TP values for scatter: {len(neurons_with_bitp)}")

if len(neurons_with_bitp) > 0:
    fig, ax = plt.subplots(figsize=(10, 8))

    bi = np.array([n['bi'] for n in neurons_with_bitp])
    tp = np.array([n['tp'] for n in neurons_with_bitp])
    mi = np.array([n['mod_index'] for n in neurons_with_bitp])

    vmax_scatter = np.percentile(np.abs(mi), 95)
    sc = ax.scatter(tp, bi, c=mi, cmap='RdBu_r', vmin=-vmax_scatter, vmax=vmax_scatter,
                    s=60, edgecolors='k', linewidths=0.5, alpha=0.8)

    ax.set_yscale('log')
    ax.set_xlabel('Trough-to-Peak (ms)', fontsize=14)
    ax.set_ylabel('Burst Index (Royer 2012)', fontsize=14)
    ax.set_title(f'BI/TP scatter colored by reward modulation (neurons ≥ {min_firing_rate} Hz)',
                 fontsize=16)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Modulation index', fontsize=13)
    cbar.ax.tick_params(labelsize=12)
    ax.tick_params(labelsize=12)
    plt.tight_layout()
    fpath = os.path.join(save_dir, 'bitp_scatter_modulation.png')
    fig.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {fpath}")

print("\nDone! All figures saved to:", save_dir)