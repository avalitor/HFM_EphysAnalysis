# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 15:46:35 2026

@author: student
"""

"""
Z-scored firing rate heatmaps aligned to reward.
- Each neuron-row is z-scored against its own pre-reward baseline
- Multiple sorting options: by peak time, peak magnitude, cell type
- Excludes non-task trials and trials with reward outside tracking window
Saves figures as PNG files.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# from matplotlib.lines import Line2D
import glob
import os
import re
import lib_ephys_obj as elib
import config as cg

# %% === PARAMETERS ===
experiments = ['2024-11-09', '2025-01-21']

exclude_prefixes = ('Habituation', 'Probe', 'Tape', 'Dark')

bin_width = 0.5          # seconds per bin
baseline_window = 10     # seconds before reward used for z-score baseline
                         # (uses the period from -baseline_window to -1 s)

path = cg.PROCESSED_FILE_DIR
save_dir = os.path.join(cg.ROOT_DIR, 'figs')
os.makedirs(save_dir, exist_ok=True)

# %% === DISCOVER AND LOAD ALL TRIALS ===
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
            print(f"  Skipping unrecognized filename: {fname}")
            continue
        
        mouse = match.group(1)
        trial = match.group(2)
        
        if trial.startswith(exclude_prefixes):
            print(f"  Excluding {fname} (filtered trial type)")
            continue
        
        try:
            edata = elib.EphysTrial.Load(exp, mouse, trial)
        except Exception as e:
            print(f"  Failed to load {fname}: {e}")
            continue
        
        if not hasattr(edata, 'k_reward') or edata.k_reward is None:
            print(f"  No reward event in {fname}, skipping.")
            continue
        
        t_reward = edata.time_ttl[edata.k_reward]
        
        finite_mask = np.isfinite(edata.r_center)[:, 0]
        if not np.any(finite_mask):
            print(f"  No valid positions in {fname}, skipping.")
            continue
        t_start = edata.time_ttl[finite_mask][0]
        t_end = edata.time_ttl[finite_mask][-1]
        
        # Skip if reward falls outside tracking window
        if t_reward > t_end or t_reward < t_start:
            print(f"  Reward outside tracking window in {fname}, skipping.")
            continue
        
        pre_available = t_reward - t_start
        post_available = t_end - t_reward
        n_neurons = len(edata.t_spikeTrains)
        
        trial_data.append({
            'exp': exp,
            'mouse': mouse,
            'trial': trial,
            't_reward': t_reward,
            't_start': t_start,
            't_end': t_end,
            'pre_available': pre_available,
            'post_available': post_available,
            'n_neurons': n_neurons,
            'spike_trains': [edata.t_spikeTrains[i].copy() for i in range(n_neurons)],
            'labels': [edata.cellLabels_BITP[i][0] if hasattr(edata, 'cellLabels_BITP')
                       else 'unknown' for i in range(n_neurons)],
        })

print(f"\nLoaded {len(trial_data)} valid trials")

if len(trial_data) == 0:
    raise ValueError("No valid trials found.")

# %% === CROP TO SHORTEST TRIAL ===
window_pre = min(td['pre_available'] for td in trial_data)
window_post = min(td['post_available'] for td in trial_data)
print(f"Crop window: -{window_pre:.2f} to +{window_post:.2f} s relative to reward")

# Make sure baseline window fits within available pre-reward data
baseline_start = min(baseline_window, window_pre)
print(f"Baseline for z-score: -{baseline_start:.2f} to -1.0 s")

# %% === BUILD FIRING RATE MATRIX ===
bins = np.arange(-window_pre, window_post + bin_width, bin_width)
bin_centers = bins[:-1] + bin_width / 2
n_bins = len(bin_centers)

# Identify which bins fall in the baseline window (-baseline_start to -1 s)
baseline_mask = (bin_centers >= -baseline_start) & (bin_centers < -1.0)
if np.sum(baseline_mask) < 2:
    print("WARNING: Very few baseline bins. Consider a wider baseline or smaller bin_width.")

all_rows = []
for td in trial_data:
    for i in range(td['n_neurons']):
        spikes = td['spike_trains'][i] - td['t_reward']
        spikes_windowed = spikes[(spikes >= -window_pre) & (spikes <= window_post)]
        
        counts, _ = np.histogram(spikes_windowed, bins=bins)
        firing_rate = counts / bin_width
        
        all_rows.append({
            'exp': td['exp'],
            'mouse': td['mouse'],
            'trial': td['trial'],
            'neuron_idx': i,
            'label': td['labels'][i],
            'firing_rate': firing_rate,
        })

n_rows = len(all_rows)
print(f"Total neuron-rows: {n_rows}")

# Assemble into matrix
rate_matrix = np.array([r['firing_rate'] for r in all_rows])  # (n_rows, n_bins)

# %% === Z-SCORE EACH ROW ===
baseline_rates = rate_matrix[:, baseline_mask]
baseline_mean = np.mean(baseline_rates, axis=1, keepdims=True)
baseline_std = np.std(baseline_rates, axis=1, keepdims=True)

# Avoid division by zero for silent neurons
baseline_std[baseline_std == 0] = 1.0

z_matrix = (rate_matrix - baseline_mean) / baseline_std

# %% === SORTING STRATEGIES ===

# 1. Sort by peak time (when does the neuron's max z-score occur?)
peak_time_idx = np.argmax(z_matrix, axis=1)
sort_by_peak_time = np.argsort(peak_time_idx)

# 2. Sort by peak magnitude (how strongly does it respond?)
peak_magnitude = np.max(z_matrix, axis=1)
sort_by_magnitude = np.argsort(peak_magnitude)[::-1]  # strongest first

# 3. Sort by cell type, then peak time within each type
labels_array = np.array([r['label'] for r in all_rows])
unique_labels = sorted(set(labels_array))
sort_by_type = []
for lbl in unique_labels:
    type_mask = np.where(labels_array == lbl)[0]
    # Within this type, sort by peak time
    type_peak_times = peak_time_idx[type_mask]
    type_sorted = type_mask[np.argsort(type_peak_times)]
    sort_by_type.extend(type_sorted)
sort_by_type = np.array(sort_by_type)


# %% === PLOTTING HELPER ===
def save_zscore_heatmap(z_mat, sort_order, row_data, title_suffix, filename,
                        bin_centers, window_pre, window_post, label_color_map,
                        show_type_boundaries=False):
    """Plot and save a z-scored heatmap with the given sort order."""
    sorted_mat = z_mat[sort_order]
    n = len(sort_order)
    
    fig, ax = plt.subplots(figsize=(14, max(4, n * 0.08)))
    
    # Clip z-scores for better color contrast
    vmax = np.percentile(np.abs(z_mat), 97)
    im = ax.imshow(sorted_mat, aspect='auto', cmap='RdBu_r',
                   vmin=-vmax, vmax=vmax,
                   extent=[-window_pre, window_post, n - 0.5, -0.5],
                   interpolation='nearest')
    
    ax.axvline(0, color='black', linewidth=2)
    
    # Type boundaries if sorting by cell type
    if show_type_boundaries:
        sorted_labels = [row_data[i]['label'] for i in sort_order]
        prev_label = sorted_labels[0]
        boundary_positions = []
        for idx, lbl in enumerate(sorted_labels):
            if lbl != prev_label:
                ax.axhline(idx - 0.5, color='white', linewidth=1.5)
                boundary_positions.append((idx, prev_label))
                prev_label = lbl
            if idx == len(sorted_labels) - 1:
                boundary_positions.append((idx + 1, lbl))
        
        # Label each type block on the right side
        block_start = 0
        for boundary_idx, lbl in boundary_positions:
            mid = (block_start + boundary_idx - 1) / 2
            ax.text(window_post + 0.5, mid, lbl, fontsize=8, va='center',
                    color=label_color_map.get(lbl, 'black'), fontweight='bold',
                    clip_on=False)
            block_start = boundary_idx
    
    ax.set_xlabel('Time relative to reward (s)', fontsize=14)
    ax.set_ylabel('Neuron (sorted)', fontsize=14)
    ax.set_title(f'Z-scored firing rate — {title_suffix}', fontsize=13)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Z-score (vs baseline)', fontsize=12)
    ax.tick_params(labelsize=11)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, filename)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


# %% === COLOR MAP FOR LABELS ===
label_color_map = {}
default_palette = ['#5f0f40', '#e36414', '#0f4c5c', '#9a031e', '#fb8b24']
for row in all_rows:
    if row['label'] not in label_color_map:
        label_color_map[row['label']] = default_palette[len(label_color_map) % len(default_palette)]

# %% === SAVE ALL THREE SORTED HEATMAPS ===
print("Saving heatmaps...")

save_zscore_heatmap(
    z_matrix, sort_by_peak_time, all_rows,
    f'sorted by peak time ({n_rows} rows)',
    'zscore_heatmap_sort_peaktime.png',
    bin_centers, window_pre, window_post, label_color_map
)

save_zscore_heatmap(
    z_matrix, sort_by_magnitude, all_rows,
    f'sorted by peak magnitude ({n_rows} rows)',
    'zscore_heatmap_sort_magnitude.png',
    bin_centers, window_pre, window_post, label_color_map
)

save_zscore_heatmap(
    z_matrix, sort_by_type, all_rows,
    f'sorted by cell type → peak time ({n_rows} rows)',
    'zscore_heatmap_sort_celltype.png',
    bin_centers, window_pre, window_post, label_color_map,
    show_type_boundaries=True
)

print("\nDone! All heatmaps saved to:", save_dir)

#%%
"""
Test whether the diagonal pattern in z-scored heatmaps is real or noise.

Two approaches:
1. Trial-averaged heatmap: average each unique neuron across its trials,
   then sort by peak time. Real sequences survive averaging.
2. Circular shuffle test: shift each neuron's spike train by a random
   amount (preserving ISI structure), recompute z-scores, sort, and
   compare. Repeated N times to build a null distribution of "sequence
   sharpness" vs the real data.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob
import os
import re
import lib_ephys_obj as elib
import config as cg

# %% === PARAMETERS ===
experiments = ['2024-11-09', '2025-01-21']
exclude_prefixes = ('Habituation', 'Probe', 'Tape', 'Dark')

bin_width = 0.5
baseline_window = 10
n_shuffles = 500  # number of shuffle iterations for null distribution

path = cg.PROCESSED_FILE_DIR
save_dir = os.path.join(cg.ROOT_DIR, 'figs')
os.makedirs(save_dir, exist_ok=True)

# %% === LOAD DATA (same as before) ===
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
            print(f"  Reward outside tracking window in {fname}, skipping.")
            continue

        n_neurons = len(edata.t_spikeTrains)

        day = str(edata.day)

        trial_data.append({
            'exp': exp,
            'mouse': mouse,
            'trial': trial,
            'day': day,
            't_reward': t_reward,
            't_start': t_start,
            't_end': t_end,
            'pre_available': t_reward - t_start,
            'post_available': t_end - t_reward,
            'n_neurons': n_neurons,
            'spike_trains': [edata.t_spikeTrains[i].copy() for i in range(n_neurons)],
            'labels': [edata.cellLabels_BITP[i][0] if hasattr(edata, 'cellLabels_BITP')
                       else 'unknown' for i in range(n_neurons)],
        })

print(f"Loaded {len(trial_data)} valid trials")

# %% === CROP WINDOW ===
window_pre = min(td['pre_available'] for td in trial_data)
window_post = min(td['post_available'] for td in trial_data)
print(f"Crop window: -{window_pre:.2f} to +{window_post:.2f} s")

bins = np.arange(-window_pre, window_post + bin_width, bin_width)
bin_centers = bins[:-1] + bin_width / 2
n_bins = len(bin_centers)

baseline_start = min(baseline_window, window_pre)
baseline_mask = (bin_centers >= -baseline_start) & (bin_centers < -1.0)


# %% === HELPER: spikes -> z-scored firing rate ===
def spikes_to_zscore(spike_times, t_reward, bins, bin_width, baseline_mask):
    """Convert a spike train to a z-scored firing rate vector."""
    spikes = spike_times - t_reward
    spikes_w = spikes[(spikes >= bins[0]) & (spikes <= bins[-1])]
    counts, _ = np.histogram(spikes_w, bins=bins)
    fr = counts / bin_width

    bl_mean = np.mean(fr[baseline_mask])
    bl_std = np.std(fr[baseline_mask])
    if bl_std == 0:
        bl_std = 1.0
    return (fr - bl_mean) / bl_std


# %% === BUILD PER-ROW Z-SCORES (all neuron-trial rows) ===
all_rows = []
for td in trial_data:
    for i in range(td['n_neurons']):
        z = spikes_to_zscore(td['spike_trains'][i], td['t_reward'],
                             bins, bin_width, baseline_mask)
        # Unique neuron ID: same mouse on same day = same neurons
        neuron_id = (td['exp'], td['mouse'], td['day'], i)

        all_rows.append({
            'neuron_id': neuron_id,
            'label': td['labels'][i],
            'z': z,
            'spike_train': td['spike_trains'][i],
            't_reward': td['t_reward'],
            't_start': td['t_start'],
            't_end': td['t_end'],
        })

print(f"Total neuron-rows: {len(all_rows)}")

# %% =====================================================================
# TEST 1: TRIAL-AVERAGED HEATMAP
# =====================================================================
print("\n=== Test 1: Trial-averaged heatmap ===")

# Group rows by unique neuron and average z-scores
from collections import defaultdict
neuron_groups = defaultdict(list)
neuron_labels = {}
for row in all_rows:
    neuron_groups[row['neuron_id']].append(row['z'])
    neuron_labels[row['neuron_id']] = row['label']

avg_neurons = []
for nid, z_list in neuron_groups.items():
    avg_z = np.mean(z_list, axis=0)
    avg_neurons.append({
        'neuron_id': nid,
        'label': neuron_labels[nid],
        'z': avg_z,
        'n_trials': len(z_list),
    })

print(f"Unique neurons: {len(avg_neurons)}")
print(f"Trials per neuron: {[n['n_trials'] for n in avg_neurons]}")

avg_z_matrix = np.array([n['z'] for n in avg_neurons])

# Sort by peak time
peak_time_idx = np.argmax(avg_z_matrix, axis=1)
sort_order = np.argsort(peak_time_idx)

sorted_mat = avg_z_matrix[sort_order]
vmax = np.percentile(np.abs(avg_z_matrix), 97)
n_unique = len(avg_neurons)

fig, ax = plt.subplots(figsize=(14, max(4, n_unique * 0.2)))
im = ax.imshow(sorted_mat, aspect='auto', cmap='RdBu_r',
               vmin=-vmax, vmax=vmax,
               extent=[-window_pre, window_post, n_unique - 0.5, -0.5],
               interpolation='nearest')
ax.axvline(0, color='black', linewidth=2)
ax.set_xlabel('Time relative to reward (s)', fontsize=14)
ax.set_ylabel('Neuron (trial-averaged, sorted)', fontsize=14)
ax.set_title(f'Trial-averaged z-scored heatmap — {n_unique} unique neurons, '
             f'sorted by peak time', fontsize=13)
plt.colorbar(im, ax=ax, label='Z-score')
ax.tick_params(labelsize=11)
plt.tight_layout()
fpath = os.path.join(save_dir, 'zscore_trial_averaged.png')
fig.savefig(fpath, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  Saved: {fpath}")


# %% =====================================================================
# TEST 2: CIRCULAR SHUFFLE TEST
# =====================================================================
print(f"\n=== Test 2: Circular shuffle ({n_shuffles} iterations) ===")

def sequence_score(z_matrix):
    """
    Quantify how 'sharp' the diagonal is after sorting by peak time.
    For each row, measure the concentration of activity around the peak bin.
    Returns the mean peak z-score across neurons — higher = sharper sequence.
    A more robust metric: for each row, compute the fraction of total
    absolute z-score that falls within +/-2 bins of the peak.
    """
    n_rows, n_bins = z_matrix.shape
    peak_idx = np.argmax(z_matrix, axis=1)
    concentration = np.zeros(n_rows)
    half_width = 2  # bins on each side of peak

    for i in range(n_rows):
        lo = max(0, peak_idx[i] - half_width)
        hi = min(n_bins, peak_idx[i] + half_width + 1)
        total = np.sum(np.abs(z_matrix[i]))
        if total > 0:
            concentration[i] = np.sum(np.abs(z_matrix[i, lo:hi])) / total
    return np.mean(concentration)


# Real score (using trial-averaged data for a cleaner comparison)
real_score = sequence_score(avg_z_matrix)
print(f"Real sequence score: {real_score:.4f}")

# Shuffle: for each neuron-trial, circularly shift spikes within the
# valid recording window, recompute z-scores, trial-average, then score
shuffle_scores = []
for shuf_i in range(n_shuffles):
    if (shuf_i + 1) % 100 == 0:
        print(f"  Shuffle {shuf_i + 1}/{n_shuffles}...")

    # Recompute z-scores with circularly shifted spikes
    shuf_groups = defaultdict(list)
    for row in all_rows:
        # Circular shift within the recording window
        t_range = row['t_end'] - row['t_start']
        if t_range <= 0:
            shuf_groups[row['neuron_id']].append(np.zeros(n_bins))
            continue

        shift = np.random.uniform(0, t_range)
        shifted_spikes = row['spike_train'] - row['t_start']  # zero to recording
        shifted_spikes = (shifted_spikes + shift) % t_range    # circular wrap
        shifted_spikes = shifted_spikes + row['t_start']       # back to TTL time

        z = spikes_to_zscore(shifted_spikes, row['t_reward'],
                             bins, bin_width, baseline_mask)
        shuf_groups[row['neuron_id']].append(z)

    # Trial-average the shuffled data
    shuf_avg = []
    for nid in neuron_groups.keys():
        shuf_avg.append(np.mean(shuf_groups[nid], axis=0))
    shuf_avg_matrix = np.array(shuf_avg)

    shuffle_scores.append(sequence_score(shuf_avg_matrix))

shuffle_scores = np.array(shuffle_scores)
p_value = np.mean(shuffle_scores >= real_score)
print(f"Shuffle mean: {np.mean(shuffle_scores):.4f} ± {np.std(shuffle_scores):.4f}")
print(f"P-value: {p_value:.4f}")

# %% === PLOT: SHUFFLE DISTRIBUTION VS REAL ===
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(shuffle_scores, bins=30, color='gray', alpha=0.7, edgecolor='black',
        label='Shuffle distribution')
ax.axvline(real_score, color='red', linewidth=2.5, label=f'Real data (p={p_value:.3f})')
ax.set_xlabel('Sequence score (peak concentration)', fontsize=13)
ax.set_ylabel('Count', fontsize=13)
ax.set_title(f'Circular shuffle test — {n_shuffles} iterations', fontsize=13)
ax.legend(fontsize=11)
ax.tick_params(labelsize=11)
plt.tight_layout()
fpath = os.path.join(save_dir, 'shuffle_test_sequence.png')
fig.savefig(fpath, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  Saved: {fpath}")

# %% === PLOT: EXAMPLE SHUFFLE HEATMAP FOR COMPARISON ===
# Show one shuffled heatmap side-by-side with the real one
shuf_groups_example = defaultdict(list)
for row in all_rows:
    t_range = row['t_end'] - row['t_start']
    if t_range <= 0:
        shuf_groups_example[row['neuron_id']].append(np.zeros(n_bins))
        continue
    shift = np.random.uniform(0, t_range)
    shifted_spikes = row['spike_train'] - row['t_start']
    shifted_spikes = (shifted_spikes + shift) % t_range
    shifted_spikes = shifted_spikes + row['t_start']
    z = spikes_to_zscore(shifted_spikes, row['t_reward'], bins, bin_width, baseline_mask)
    shuf_groups_example[row['neuron_id']].append(z)

shuf_avg_example = []
for nid in neuron_groups.keys():
    shuf_avg_example.append(np.mean(shuf_groups_example[nid], axis=0))
shuf_avg_example = np.array(shuf_avg_example)

# Sort both by their own peak times
real_sort = np.argsort(np.argmax(avg_z_matrix, axis=1))
shuf_sort = np.argsort(np.argmax(shuf_avg_example, axis=1))

fig, axes = plt.subplots(1, 2, figsize=(20, max(4, n_unique * 0.15)))
fig.subplots_adjust(right=0.88, wspace=0.3)

vmax_both = max(np.percentile(np.abs(avg_z_matrix), 97),
                np.percentile(np.abs(shuf_avg_example), 97))

for ax, mat, order, title in zip(
    axes,
    [avg_z_matrix, shuf_avg_example],
    [real_sort, shuf_sort],
    ['Real data', 'Shuffled control']
):
    im = ax.imshow(mat[order], aspect='auto', cmap='RdBu_r',
                   vmin=-vmax_both, vmax=vmax_both,
                   extent=[-window_pre, window_post, n_unique - 0.5, -0.5],
                   interpolation='nearest')
    ax.axvline(0, color='black', linewidth=2)
    ax.set_xlabel('Time relative to reward (s)', fontsize=16)
    ax.set_ylabel('Neuron (sorted)', fontsize=16)
    ax.set_title(title, fontsize=18)
    ax.tick_params(labelsize=13)

cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label('Z-score', fontsize=14)
cbar.ax.tick_params(labelsize=12)
fig.suptitle('Trial-averaged heatmaps: real vs shuffled', fontsize=18, y=0.98)
fpath = os.path.join(save_dir, 'zscore_real_vs_shuffle.png')
fig.savefig(fpath, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  Saved: {fpath}")

print("\nDone!")