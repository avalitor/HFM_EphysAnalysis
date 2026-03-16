# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 14:56:12 2026

@author: student
"""

"""
Neural and speed analysis aligned to hole check events.
Same analyses as reward-aligned script:
  - Z-scored sorted heatmaps (peak time, magnitude, cell type)
  - Grouped PETHs by cell type
  - Speed PETH comparison
  - Modulation index
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

bin_width = 0.1          # seconds per bin (finer for short windows)
window_pre = 2.0         # seconds before hole check
window_post = 2.0        # seconds after hole check
baseline_pre = 2.0       # baseline: -baseline_pre to -baseline_buffer
baseline_buffer = 0.5    # seconds before event excluded from baseline
min_firing_rate = 0.5    # Hz — exclude low-rate neurons from modulation index
post_mod_window = 0.5    # seconds after event for modulation index

path = cg.PROCESSED_FILE_DIR
save_dir = os.path.join(cg.ROOT_DIR, 'figs', 'hole_checks')
os.makedirs(save_dir, exist_ok=True)

# %% === LOAD DATA ===
trial_data = []
bitp_cache = {}

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

        # Check for hole check data
        if not hasattr(edata, 'k_hole_checks') or edata.k_hole_checks is None:
            print(f"  No hole check data in {fname}, skipping.")
            continue

        if edata.k_hole_checks.size == 0:
            print(f"  Empty hole check array in {fname}, skipping.")
            continue

        # Get valid hole check time indices (column 1 > 0)
        hc_indices = edata.k_hole_checks[:, 1]
        valid_mask = hc_indices > 0
        hc_indices = hc_indices[valid_mask].astype(int)

        if len(hc_indices) == 0:
            print(f"  No valid hole checks in {fname}, skipping.")
            continue

        # Convert to times
        hc_times = edata.time_ttl[hc_indices]

        # Trial boundaries from tracking
        finite_mask = np.isfinite(edata.r_center)[:, 0]
        if not np.any(finite_mask):
            continue
        t_start = edata.time_ttl[finite_mask][0]
        t_end = edata.time_ttl[finite_mask][-1]

        # Only keep hole checks within the tracking window with enough margin
        hc_times = hc_times[(hc_times - window_pre >= t_start) &
                            (hc_times + window_post <= t_end)]

        if len(hc_times) == 0:
            print(f"  No hole checks with sufficient margin in {fname}, skipping.")
            continue

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
            'hc_times': hc_times,
            'n_neurons': n_neurons,
            'spike_trains': [edata.t_spikeTrains[i].copy() for i in range(n_neurons)],
            'labels': [edata.cellLabels_BITP[i][0] if hasattr(edata, 'cellLabels_BITP')
                       else 'unknown' for i in range(n_neurons)],
            'bi_vals': bi_vals,
            'tp_vals': tp_vals,
            'velocity': edata.velocity.copy(),
            'time_ttl': edata.time_ttl.copy(),
        })

total_hc = sum(len(td['hc_times']) for td in trial_data)
print(f"\nLoaded {len(trial_data)} valid trials")
print(f"Total hole check events: {total_hc}")

if len(trial_data) == 0:
    raise ValueError("No valid trials found.")

# %% === BINS AND BASELINE MASK ===
bins = np.arange(-window_pre, window_post + bin_width, bin_width)
bin_centers = bins[:-1] + bin_width / 2
n_bins = len(bin_centers)

baseline_mask = (bin_centers >= -baseline_pre) & (bin_centers < -baseline_buffer)
post_mask = (bin_centers >= 0) & (bin_centers < post_mod_window)

print(f"Baseline: -{baseline_pre:.1f} to -{baseline_buffer:.1f} s")
print(f"Post-event window for modulation index: 0 to +{post_mod_window} s")

# %% === BUILD EVENT-ALIGNED FIRING RATES ===
# For each neuron, average across all hole check events across all its trials
neuron_groups = defaultdict(lambda: {
    'fr_events': [], 'label': None, 'bi': None, 'tp': None
})

for td in trial_data:
    for i in range(td['n_neurons']):
        nid = (td['exp'], td['mouse'], td['day'], i)
        spikes = td['spike_trains'][i]

        # Accumulate firing rate for each hole check event
        for hc_t in td['hc_times']:
            aligned = spikes - hc_t
            aligned_w = aligned[(aligned >= -window_pre) & (aligned <= window_post)]
            counts, _ = np.histogram(aligned_w, bins=bins)
            fr = counts / bin_width
            neuron_groups[nid]['fr_events'].append(fr)

        neuron_groups[nid]['label'] = td['labels'][i]
        if td['bi_vals'] is not None and i < len(td['bi_vals']):
            neuron_groups[nid]['bi'] = td['bi_vals'][i]
        if td['tp_vals'] is not None and i < len(td['tp_vals']):
            neuron_groups[nid]['tp'] = td['tp_vals'][i]

# Average across events for each neuron
neurons = []
for nid, data in neuron_groups.items():
    avg_fr = np.mean(data['fr_events'], axis=0)

    bl_mean = np.mean(avg_fr[baseline_mask])
    bl_std = np.std(avg_fr[baseline_mask])
    if bl_std == 0:
        bl_std = 1.0
    avg_z = (avg_fr - bl_mean) / bl_std

    neurons.append({
        'neuron_id': nid,
        'label': data['label'],
        'avg_fr': avg_fr,
        'avg_z': avg_z,
        'bi': data['bi'],
        'tp': data['tp'],
        'n_events': len(data['fr_events']),
    })

print(f"Unique neurons: {len(neurons)}")
print(f"Events per neuron: min={min(n['n_events'] for n in neurons)}, "
      f"max={max(n['n_events'] for n in neurons)}, "
      f"mean={np.mean([n['n_events'] for n in neurons]):.0f}")

# %% === GROUP BY CELL TYPE ===
type_groups = defaultdict(list)
for n in neurons:
    type_groups[n['label']].append(n)

label_colors = {}
default_palette = ['#5f0f40', '#e36414', '#0f4c5c', '#9a031e', '#fb8b24']
for i, label in enumerate(sorted(type_groups.keys())):
    label_colors[label] = default_palette[i % len(default_palette)]

n_types = len(type_groups)

# %% =====================================================================
# Z-SCORED HEATMAPS
# =====================================================================
print("\n=== Z-scored heatmaps ===")

z_matrix = np.array([n['avg_z'] for n in neurons])
n_neurons_total = len(neurons)

def save_zscore_heatmap(z_mat, sort_order, neurons_list, title_suffix, filename,
                        show_type_boundaries=False):
    sorted_mat = z_mat[sort_order]
    n = len(sort_order)

    fig, ax = plt.subplots(figsize=(14, max(4, n * 0.08)))
    vmax = np.percentile(np.abs(z_mat), 97)
    im = ax.imshow(sorted_mat, aspect='auto', cmap='RdBu_r',
                   vmin=-vmax, vmax=vmax,
                   extent=[-window_pre, window_post, n - 0.5, -0.5],
                   interpolation='nearest')
    ax.axvline(0, color='black', linewidth=2)

    if show_type_boundaries:
        sorted_labels = [neurons_list[i]['label'] for i in sort_order]
        prev_label = sorted_labels[0]
        block_start = 0
        for idx, lbl in enumerate(sorted_labels):
            if lbl != prev_label:
                ax.axhline(idx - 0.5, color='white', linewidth=1.5)
                mid = (block_start + idx - 1) / 2
                ax.text(window_post + 0.05, mid, prev_label, fontsize=8, va='center',
                        color=label_colors.get(prev_label, 'black'), fontweight='bold',
                        clip_on=False)
                block_start = idx
                prev_label = lbl
        mid = (block_start + n - 1) / 2
        ax.text(window_post + 0.05, mid, prev_label, fontsize=8, va='center',
                color=label_colors.get(prev_label, 'black'), fontweight='bold',
                clip_on=False)

    ax.set_xlabel('Time relative to hole check (s)', fontsize=14)
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

# Sort by peak time
peak_time_idx = np.argmax(z_matrix, axis=1)
sort_peak_time = np.argsort(peak_time_idx)

# Sort by peak magnitude
peak_mag = np.max(z_matrix, axis=1)
sort_magnitude = np.argsort(peak_mag)[::-1]

# Sort by cell type, then peak time
labels_array = np.array([n['label'] for n in neurons])
sort_by_type = []
for lbl in sorted(set(labels_array)):
    type_mask = np.where(labels_array == lbl)[0]
    type_sorted = type_mask[np.argsort(peak_time_idx[type_mask])]
    sort_by_type.extend(type_sorted)
sort_by_type = np.array(sort_by_type)

save_zscore_heatmap(z_matrix, sort_peak_time, neurons,
                    f'hole checks, sorted by peak time ({n_neurons_total} neurons)',
                    'hc_zscore_sort_peaktime.png')

save_zscore_heatmap(z_matrix, sort_magnitude, neurons,
                    f'hole checks, sorted by peak magnitude ({n_neurons_total} neurons)',
                    'hc_zscore_sort_magnitude.png')

save_zscore_heatmap(z_matrix, sort_by_type, neurons,
                    f'hole checks, sorted by cell type ({n_neurons_total} neurons)',
                    'hc_zscore_sort_celltype.png',
                    show_type_boundaries=True)

# %% =====================================================================
# GROUPED PETHs BY CELL TYPE
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

axes[-1].set_xlabel('Time relative to hole check (s)', fontsize=14)
fig.suptitle('Grouped PETHs by cell type — hole checks (z-scored, mean ± SEM)',
             fontsize=16, y=1.01)
plt.tight_layout()
fpath = os.path.join(save_dir, 'hc_peth_by_celltype_panels.png')
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

ax.axvline(0, color='black', linewidth=1.5, linestyle='--', label='Hole check')
ax.axhline(0, color='gray', linewidth=0.5)
ax.set_xlabel('Time relative to hole check (s)', fontsize=14)
ax.set_ylabel('Z-score', fontsize=14)
ax.set_title('Grouped PETHs — all cell types overlaid (hole checks)', fontsize=16)
ax.legend(fontsize=12, loc='upper right')
ax.tick_params(labelsize=12)
plt.tight_layout()
fpath = os.path.join(save_dir, 'hc_peth_by_celltype_overlaid.png')
fig.savefig(fpath, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  Saved: {fpath}")

# --- Raw firing rate panels ---
fig, axes = plt.subplots(n_types, 1, figsize=(14, 3.5 * n_types), sharex=True)
if n_types == 1:
    axes = [axes]

for ax, (label, group) in zip(axes, sorted(type_groups.items())):
    fr_mat = np.array([n['avg_fr'] for n in group])
    mean_fr = np.mean(fr_mat, axis=0)
    sem_fr = np.std(fr_mat, axis=0) / np.sqrt(len(group))

    color = label_colors[label]
    ax.fill_between(bin_centers, mean_fr - sem_fr, mean_fr + sem_fr,
                    alpha=0.3, color=color)
    ax.plot(bin_centers, mean_fr, color=color, linewidth=2)
    ax.axvline(0, color='black', linewidth=1.5, linestyle='--')
    ax.set_ylabel('Firing rate (Hz)', fontsize=14)
    ax.set_title(f'{label}  (n={len(group)})', fontsize=15)
    ax.tick_params(labelsize=12)

axes[-1].set_xlabel('Time relative to hole check (s)', fontsize=14)
fig.suptitle('Grouped PETHs by cell type — hole checks (raw firing rate, mean ± SEM)',
             fontsize=16, y=1.01)
plt.tight_layout()
fpath = os.path.join(save_dir, 'hc_peth_by_celltype_raw.png')
fig.savefig(fpath, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  Saved: {fpath}")


# %% =====================================================================
# SPEED PETH
# =====================================================================
print("\n=== Speed PETH ===")

speed_traces = []
for td in trial_data:
    vel = np.ravel(td['velocity'].copy())
    t = td['time_ttl']

    for hc_t in td['hc_times']:
        t_aligned = t - hc_t
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

# --- Speed PETH alone ---
fig, ax = plt.subplots(figsize=(14, 4))
ax.fill_between(bin_centers, mean_speed - sem_speed, mean_speed + sem_speed,
                alpha=0.3, color='gray')
ax.plot(bin_centers, mean_speed, color='black', linewidth=2)
ax.axvline(0, color='black', linewidth=1.5, linestyle='--')
ax.set_xlabel('Time relative to hole check (s)', fontsize=14)
ax.set_ylabel('Speed (cm/s)', fontsize=14)
ax.set_title(f'Speed PETH aligned to hole checks ({len(speed_traces)} events, mean ± SEM)',
             fontsize=16)
ax.tick_params(labelsize=12)
plt.tight_layout()
fpath = os.path.join(save_dir, 'hc_speed_peth.png')
fig.savefig(fpath, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  Saved: {fpath}")

# --- Speed stacked with neural PETHs ---
fig, axes = plt.subplots(n_types + 1, 1, figsize=(14, 3.5 * (n_types + 1)), sharex=True)

ax = axes[0]
ax.fill_between(bin_centers, mean_speed - sem_speed, mean_speed + sem_speed,
                alpha=0.3, color='gray')
ax.plot(bin_centers, mean_speed, color='black', linewidth=2)
ax.axvline(0, color='black', linewidth=1.5, linestyle='--')
ax.set_ylabel('Speed (cm/s)', fontsize=14)
ax.set_title(f'Speed ({len(speed_traces)} events)', fontsize=15)
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

axes[-1].set_xlabel('Time relative to hole check (s)', fontsize=14)
fig.suptitle('Speed vs neural PETHs aligned to hole checks', fontsize=16, y=1.01)
plt.tight_layout()
fpath = os.path.join(save_dir, 'hc_speed_vs_neural_peth.png')
fig.savefig(fpath, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  Saved: {fpath}")


# %% =====================================================================
# MODULATION INDEX
# =====================================================================
print("\n=== Hole check modulation index ===")

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

# Filter low-firing neurons
neurons_filtered = [n for n in neurons if n['overall_rate'] >= min_firing_rate]
n_excluded = len(neurons) - len(neurons_filtered)
print(f"Excluding {n_excluded} neurons with firing rate < {min_firing_rate} Hz")
print(f"Neurons remaining: {len(neurons_filtered)}")

type_groups_filt = defaultdict(list)
for n in neurons_filtered:
    type_groups_filt[n['label']].append(n)

# --- Separate panels per cell type ---
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

axes[-1].set_xlabel('Hole check modulation index', fontsize=14)
fig.suptitle(f'Hole check modulation index by cell type (neurons ≥ {min_firing_rate} Hz)',
             fontsize=16, y=1.01)
plt.tight_layout()
fpath = os.path.join(save_dir, 'hc_modulation_histogram.png')
fig.savefig(fpath, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  Saved: {fpath}")

# --- Strip plot ---
fig, ax = plt.subplots(figsize=(10, 5))

for i, label in enumerate(sorted_labels):
    group = type_groups_filt[label]
    mi_vals = np.array([n['mod_index'] for n in group])
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
ax.set_ylabel('Hole check modulation index', fontsize=14)
ax.set_title(f'Hole check modulation by cell type — neurons ≥ {min_firing_rate} Hz '
             f'(dots = neurons, black = mean ± SEM)', fontsize=14)
ax.tick_params(labelsize=12)
plt.tight_layout()
fpath = os.path.join(save_dir, 'hc_modulation_strip.png')
fig.savefig(fpath, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  Saved: {fpath}")

# --- BI/TP scatter colored by modulation ---
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
    ax.set_title(f'BI/TP scatter colored by hole check modulation (neurons ≥ {min_firing_rate} Hz)',
                 fontsize=14)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Modulation index', fontsize=13)
    cbar.ax.tick_params(labelsize=12)
    ax.tick_params(labelsize=12)
    plt.tight_layout()
    fpath = os.path.join(save_dir, 'hc_bitp_scatter_modulation.png')
    fig.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {fpath}")

print("\nDone! All figures saved to:", save_dir)