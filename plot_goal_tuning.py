# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 13:05:00 2026

@author: student

Goal-Distance Tuning Curves
============================
Computes firing rate as a function of Euclidean distance to the reward
location during the approach phase (pre-reward) of goal-directed trials.

Compares Narrow vs Wide hilar interneurons, and compares r_nose vs r_center
as the position variable.

Analyses:
  1. Population-average goal-distance tuning curves by cell type
  2. Goal-distance modulation index per neuron, compared across types
  3. Learning effect: early vs late training trials
  4. Individual neuron tuning curve examples

Restricted to initial training trials (1–18, before Probe).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
import os
import glob
import re
import lib_ephys_obj as elib
import sys
sys.path.append("F:\Spike Sorting\EthovisionPathAnalysis_HDF5")
import config as cg

# ============================================================================
# Parameters
# ============================================================================
EXPERIMENTS    = ['2024-11-09', '2025-01-21']
TRIAL_RANGE    = range(1, 19)           # trials 1–18 (pre-Probe training)
VEL_CUTOFF     = 2                      # cm/s — exclude immobility
K_REWARD_PAD   = 50                     # samples past k_reward to include (~2s at 25Hz)
DIST_BIN_EDGES = np.arange(0, 70, 5)   # 5 cm bins from 0 to 65 cm
DIST_BIN_CENTERS = (DIST_BIN_EDGES[:-1] + DIST_BIN_EDGES[1:]) / 2

# Split trials for learning analysis
EARLY_TRIALS = range(1, 7)    # trials 1–6
LATE_TRIALS  = range(13, 19)  # trials 13–18

SAVE_DIR = os.path.join(cg.ROOT_DIR, 'figs')
os.makedirs(SAVE_DIR, exist_ok=True)

TYPE_COLORS = {
    'Narrow Interneuron':           '#E63946',
    'Wide Interneuron':             '#457B9D',
    'Excitatory Principal Cell':    '#2A9D8F',
    'Bursty Narrow Interneuron':    '#E9C46A',
}

POS_VARS = ['r_center', 'r_nose']  # compare both


# ============================================================================
# Core computation
# ============================================================================

def compute_distance_to_goal(coords, target):
    """
    Euclidean distance from each position sample to goal.
    Returns array of distances (same length as coords), NaN where coords are NaN.
    """
    dist = np.full(len(coords), np.nan)
    valid = ~np.isnan(coords[:, 0]) & ~np.isnan(coords[:, 1])
    dist[valid] = np.sqrt((coords[valid, 0] - target[0])**2 +
                          (coords[valid, 1] - target[1])**2)
    return dist


def compute_tuning_curve(edata, spike_times, idx_end, target, pos_var='r_center',
                         velocity_cutoff=VEL_CUTOFF):
    """
    Compute firing rate as a function of distance to goal for a single trial.

    Returns:
        spike_counts : (n_bins,) spike counts per distance bin
        occ_sec      : (n_bins,) occupancy in seconds per distance bin
    """
    coords = getattr(edata, pos_var)[:idx_end]
    vel    = edata.velocity[:idx_end]
    dist   = compute_distance_to_goal(coords, target)

    # Valid mask for occupancy: not NaN, velocity above cutoff
    valid_occ = (~np.isnan(dist) & ~np.isnan(vel) & (vel > velocity_cutoff))

    dt = np.median(np.diff(edata.time_ttl))

    # Occupancy per distance bin (in seconds)
    occ_counts, _ = np.histogram(dist[valid_occ], bins=DIST_BIN_EDGES)
    occ_sec = occ_counts * dt

    # Spike positions and distances
    spike_times = np.atleast_1d(spike_times)
    if spike_times.size == 0:
        return np.zeros(len(DIST_BIN_CENTERS)), occ_sec

    idx_spike = np.searchsorted(edata.time_ttl, spike_times)
    n_samples = min(idx_end, len(coords))
    valid_sp = (idx_spike > 0) & (idx_spike < n_samples)
    idx_spike = idx_spike[valid_sp]

    # Filter spikes by velocity and valid coordinates
    spike_vel  = edata.velocity[idx_spike]
    spike_dist = dist[idx_spike]
    keep = ((spike_vel > velocity_cutoff)
            & ~np.isnan(spike_dist))

    spike_counts, _ = np.histogram(spike_dist[keep], bins=DIST_BIN_EDGES)

    return spike_counts, occ_sec


def goal_distance_modulation_index(tuning_curve_hz):
    """
    Modulation index: (FR_near - FR_far) / (FR_near + FR_far).
    near = first 3 bins (0–15 cm), far = last 3 bins (50–65 cm).
    Returns NaN if either zone has no valid data.
    """
    near_bins = tuning_curve_hz[:3]
    far_bins  = tuning_curve_hz[-3:]

    near_valid = near_bins[~np.isnan(near_bins)]
    far_valid  = far_bins[~np.isnan(far_bins)]

    if len(near_valid) == 0 or len(far_valid) == 0:
        return np.nan

    fr_near = np.mean(near_valid)
    fr_far  = np.mean(far_valid)

    if (fr_near + fr_far) == 0:
        return np.nan

    return (fr_near - fr_far) / (fr_near + fr_far)

def shuffle_modulation_index(edata_list, trial_nums, neuron_idx, target,
                             pos_var='r_center', n_shuffles=500,
                             shuffle_min=20, shuffle_max_frac=0.9):
    """
    Circular time-shift shuffle test for goal-distance modulation index.
    Returns:
        shuffle_dist : (n_shuffles,) array of shuffled MI values
    """
    shuffle_dist = np.full(n_shuffles, np.nan)

    for si in range(n_shuffles):
        spike_pooled = np.zeros(len(DIST_BIN_CENTERS))
        occ_pooled   = np.zeros(len(DIST_BIN_CENTERS))

        for trial_num, edata in edata_list:
            if (edata.k_reward is None
                    or edata.k_reward >= len(edata.time_ttl)):
                continue
            idx_end = min(edata.k_reward + K_REWARD_PAD, len(edata.time_ttl))

            spike_times = np.atleast_1d(edata.t_spikeTrains[neuron_idx])
            if spike_times.size == 0:
                continue

            t_start  = edata.time_ttl[0]
            t_end    = edata.time_ttl[idx_end - 1]
            duration = t_end - t_start
            shift_max = duration * shuffle_max_frac

            if shift_max < shuffle_min:
                continue

            shift = np.random.uniform(shuffle_min, shift_max)
            shifted = t_start + np.mod(spike_times + shift - t_start, duration)

            sp_counts, occ = compute_tuning_curve(
                edata, shifted, idx_end, target, pos_var=pos_var)
            spike_pooled += sp_counts
            occ_pooled   += occ

        tc = np.full(len(DIST_BIN_CENTERS), np.nan)
        good = occ_pooled > 0
        tc[good] = spike_pooled[good] / occ_pooled[good]
        shuffle_dist[si] = goal_distance_modulation_index(tc)

    return shuffle_dist

# ============================================================================
# Trial discovery
# ============================================================================

def discover_training_trials(experiments, trial_range=TRIAL_RANGE):
    """
    Discover training trials, grouped by (exp, mouse, day).
    Also verifies target location stability within each group.
    """
    recordings = {}

    for exp in experiments:
        exp_dir = os.path.join(cg.PROCESSED_FILE_DIR, exp)
        if not os.path.isdir(exp_dir):
            print(f"  [WARN] Experiment folder not found: {exp_dir}")
            continue

        mat_files = glob.glob(os.path.join(exp_dir, 'hfmE_*.mat'))
        pattern = re.compile(r'hfmE_[\d-]+_M(\d+)_(.+)\.mat')

        for fpath in sorted(mat_files):
            fname = os.path.basename(fpath)
            m = pattern.match(fname)
            if not m:
                continue
            mouse_str, trial_label = m.group(1), m.group(2)
            mouse = int(mouse_str)

            # Only include numbered trials in the specified range
            try:
                trial_num = int(trial_label)
            except ValueError:
                continue
            if trial_num not in trial_range:
                continue

            try:
                edata = elib.EphysTrial.Load(exp, mouse, trial_label)
            except Exception as e:
                print(f"  [WARN] Could not load {fname}: {e}")
                continue

            day = str(edata.day)
            key = (exp, mouse, day)

            if key not in recordings:
                recordings[key] = []
            recordings[key].append((trial_num, trial_label, edata))

    # Sort trials by number within each group
    for key in recordings:
        recordings[key].sort(key=lambda x: x[0])

    # Verify target stability
    for key, trial_list in recordings.items():
        targets = [edata.target for _, _, edata in trial_list]
        target_array = np.array([np.array(t).flatten() for t in targets])
        if not np.allclose(target_array, target_array[0]):
            print(f"  [WARN] Target shifted within {key}!")
            print(f"         Unique targets: {np.unique(target_array, axis=0)}")
        else:
            print(f"  {key}: {len(trial_list)} trials, "
                  f"target=({target_array[0][0]:.1f}, {target_array[0][1]:.1f})")

    print(f"\nDiscovered {len(recordings)} groups, "
          f"{sum(len(v) for v in recordings.values())} total trials")
    return recordings


# ============================================================================
# Main analysis
# ============================================================================

def run_analysis():
    print("Discovering training trials (1–18)...")
    recordings = discover_training_trials(EXPERIMENTS)

    all_results = []

    for (exp, mouse, day), trial_list in recordings.items():
        print(f"\nProcessing {exp} M{mouse} day={day}  ({len(trial_list)} trials)")

        first_edata = trial_list[0][2]
        n_neurons = len(first_edata.t_spikeTrains)
        labels = [first_edata.cellLabels_BITP[i][0] for i in range(n_neurons)]
        target = np.array(first_edata.target).flatten()

        for neuron_idx in range(n_neurons):
            # For each position variable
            for pos_var in POS_VARS:

                # Pool across all training trials
                spike_pooled_all = np.zeros(len(DIST_BIN_CENTERS))
                occ_pooled_all   = np.zeros(len(DIST_BIN_CENTERS))

                # Also track early vs late
                spike_pooled_early = np.zeros(len(DIST_BIN_CENTERS))
                occ_pooled_early   = np.zeros(len(DIST_BIN_CENTERS))
                spike_pooled_late  = np.zeros(len(DIST_BIN_CENTERS))
                occ_pooled_late    = np.zeros(len(DIST_BIN_CENTERS))

                for trial_num, trial_label, edata in trial_list:
                    # Crop at k_reward + pad
                    if (edata.k_reward is not None
                            and edata.k_reward < len(edata.time_ttl)):
                        idx_end = min(edata.k_reward + K_REWARD_PAD,
                                      len(edata.time_ttl))
                    else:
                        # Mouse never reached reward — skip this trial
                        continue

                    spike_times = np.atleast_1d(edata.t_spikeTrains[neuron_idx])
                    sp_counts, occ = compute_tuning_curve(
                        edata, spike_times, idx_end, target,
                        pos_var=pos_var)

                    spike_pooled_all += sp_counts
                    occ_pooled_all   += occ

                    if trial_num in EARLY_TRIALS:
                        spike_pooled_early += sp_counts
                        occ_pooled_early   += occ
                    elif trial_num in LATE_TRIALS:
                        spike_pooled_late += sp_counts
                        occ_pooled_late   += occ

                # Compute tuning curves (Hz) from pooled counts
                def counts_to_hz(sp, occ):
                    tc = np.full(len(DIST_BIN_CENTERS), np.nan)
                    good = occ > 0
                    tc[good] = sp[good] / occ[good]
                    return tc

                tc_all   = counts_to_hz(spike_pooled_all, occ_pooled_all)
                tc_early = counts_to_hz(spike_pooled_early, occ_pooled_early)
                tc_late  = counts_to_hz(spike_pooled_late, occ_pooled_late)

                mod_idx = goal_distance_modulation_index(tc_all)
                mean_rate = np.nanmean(tc_all)

                # Exclude very low-rate neurons where MI is unreliable
                if mean_rate < 1.0:
                    mod_idx = np.nan

                all_results.append({
                    'exp':       exp,
                    'mouse':     mouse,
                    'day':       day,
                    'neuron':    neuron_idx,
                    'label':     labels[neuron_idx],
                    'pos_var':   pos_var,
                    'tc_all':    tc_all,
                    'tc_early':  tc_early,
                    'tc_late':   tc_late,
                    'mod_idx':   mod_idx,
                    'mean_rate': mean_rate,
                })

    print(f"\nTotal entries: {len(all_results)} "
          f"({len(all_results)//2} neurons × 2 pos vars)")
    return all_results

def run_goal_shuffle_test(all_results, recordings, min_rate=1.0,
                          n_shuffles=500):
    """
    Shuffle test for goal-distance MI + population-level one-sample test.
    """
    # Build lookup: (exp, mouse, day) -> [(trial_num, edata), ...]
    edata_lookup = {}
    for key, trial_list in recordings.items():
        edata_lookup[key] = [(tn, ed) for tn, _, ed in trial_list]

    for r in all_results:
        if r['pos_var'] != 'r_center':  # only run on one pos_var to save time
            r['shuffle_p'] = np.nan
            continue
        if np.isnan(r['mod_idx']) or r['mean_rate'] < min_rate:
            r['shuffle_p'] = np.nan
            continue

        key = (r['exp'], r['mouse'], r['day'])
        target = np.array(edata_lookup[key][0][1].target).flatten()

        shuf_dist = shuffle_modulation_index(
            edata_lookup[key], None, r['neuron'], target,
            pos_var='r_center', n_shuffles=n_shuffles)

        valid_shuf = shuf_dist[~np.isnan(shuf_dist)]
        if len(valid_shuf) == 0:
            r['shuffle_p'] = np.nan
        else:
            # Two-tailed: fraction of shuffles with |MI| >= |observed|
            r['shuffle_p'] = np.mean(np.abs(valid_shuf) >= abs(r['mod_idx']))
        r['shuffle_dist'] = shuf_dist

    # Population-level tests: is MI distribution ≠ 0?
    print(f"\n  Population-level goal-distance modulation (Wilcoxon signed-rank):")
    for label in ['Narrow Interneuron', 'Wide Interneuron']:
        vals = [r['mod_idx'] for r in all_results
                if r['label'] == label and r['pos_var'] == 'r_center'
                and not np.isnan(r['mod_idx'])]
        if len(vals) < 5:
            print(f"    {label}: n={len(vals)}, too few for test")
            continue
        stat, p = stats.wilcoxon(vals, alternative='two-sided')
        n_sig = sum(1 for r in all_results
                    if r['label'] == label and r['pos_var'] == 'r_center'
                    and not np.isnan(r.get('shuffle_p', np.nan))
                    and r['shuffle_p'] < 0.05)
        n_tested = sum(1 for r in all_results
                       if r['label'] == label and r['pos_var'] == 'r_center'
                       and not np.isnan(r.get('shuffle_p', np.nan)))
        print(f"    {label}: median MI={np.median(vals):.4f}, "
              f"Wilcoxon p={p:.6f}, n={len(vals)}")
        print(f"      Per-neuron shuffle: {n_sig}/{n_tested} significant at p<0.05")

    return all_results
# ============================================================================
# Plotting
# ============================================================================

def plot_population_tuning(all_results):
    """Plot 1: Population-average tuning curves, Narrow vs Wide, for each pos_var."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    for ax_idx, pos_var in enumerate(POS_VARS):
        ax = axes[ax_idx]

        for cell_type in ['Narrow Interneuron', 'Wide Interneuron']:
            tcs = [r['tc_all'] for r in all_results
                   if r['label'] == cell_type and r['pos_var'] == pos_var]
            if len(tcs) == 0:
                continue

            tc_stack = np.array(tcs)

            # Normalize each neuron to its own mean rate before averaging
            # so high-rate neurons don't dominate the population curve
            row_means = np.nanmean(tc_stack, axis=1, keepdims=True)
            row_means[row_means == 0] = np.nan
            tc_norm = tc_stack / row_means

            mean_tc = np.nanmean(tc_norm, axis=0)
            sem_tc  = np.nanstd(tc_norm, axis=0) / np.sqrt(np.sum(~np.isnan(tc_norm), axis=0))

            ax.plot(DIST_BIN_CENTERS, mean_tc,
                    color=TYPE_COLORS[cell_type], lw=2,
                    label=f'{cell_type} (n={len(tcs)})')
            ax.fill_between(DIST_BIN_CENTERS,
                            mean_tc - sem_tc, mean_tc + sem_tc,
                            color=TYPE_COLORS[cell_type], alpha=0.2)

        ax.set_xlabel('Distance to goal (cm)')
        ax.set_ylabel('Normalized firing rate')
        ax.set_title(f'Position variable: {pos_var}')
        ax.legend(fontsize=9)
        ax.axhline(1.0, color='gray', ls='--', alpha=0.5)

    fig.suptitle('Goal-Distance Tuning Curves (all training trials)', fontsize=13)
    plt.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, 'goal_distance_tuning_population.png'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Saved: goal_distance_tuning_population.png")


def plot_modulation_index(all_results):
    """Plot 2: Modulation index comparison, Narrow vs Wide, for each pos_var."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax_idx, pos_var in enumerate(POS_VARS):
        ax = axes[ax_idx]
        compare_types = ['Narrow Interneuron', 'Wide Interneuron']
        data_by_type = {}

        for ct in compare_types:
            data_by_type[ct] = [r['mod_idx'] for r in all_results
                                if r['label'] == ct and r['pos_var'] == pos_var
                                and not np.isnan(r['mod_idx'])]

        positions = [0, 1]
        for i, ct in enumerate(compare_types):
            vals = data_by_type[ct]
            if len(vals) == 0:
                continue

            bp = ax.boxplot([vals], positions=[positions[i]], widths=0.4,
                            patch_artist=True, showfliers=False)
            bp['boxes'][0].set_facecolor(TYPE_COLORS[ct])
            bp['boxes'][0].set_alpha(0.4)
            bp['medians'][0].set_color('black')

            jitter = np.random.normal(0, 0.06, size=len(vals))
            ax.scatter(np.full(len(vals), positions[i]) + jitter, vals,
                       c=TYPE_COLORS[ct], alpha=0.5, s=20,
                       edgecolors='white', linewidths=0.5, zorder=5)

        # Mann-Whitney U test
        narrow_vals = data_by_type['Narrow Interneuron']
        wide_vals   = data_by_type['Wide Interneuron']
        if len(narrow_vals) > 1 and len(wide_vals) > 1:
            u, p = stats.mannwhitneyu(narrow_vals, wide_vals,
                                      alternative='two-sided')
            sig_str = ('***' if p < 0.001 else '**' if p < 0.01
                       else '*' if p < 0.05 else 'n.s.')
            ymax = max(max(narrow_vals), max(wide_vals))
            ymin = min(min(narrow_vals), min(wide_vals))
            ax.plot([0, 1], [ymax * 1.05, ymax * 1.05], 'k-', lw=1)
            ax.text(0.5, ymax * 1.08, f'{sig_str}\np={p:.4f}',
                    ha='center', fontsize=9)

        ax.axhline(0, color='gray', ls='--', alpha=0.5)
        ax.set_xticks(positions)
        ax.set_xticklabels(['Narrow\nInterneuron', 'Wide\nInterneuron'])
        ax.set_ylabel('Modulation Index\n(near−far)/(near+far)')
        ax.set_title(f'Position: {pos_var}')

    fig.suptitle('Goal-Distance Modulation Index', fontsize=13)
    plt.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, 'goal_distance_modulation_index.png'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Saved: goal_distance_modulation_index.png")


def plot_learning_effect(all_results):
    """Plot 3: Early vs late training tuning curves by cell type."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), sharey='row')

    for col_idx, pos_var in enumerate(POS_VARS):
        for row_idx, cell_type in enumerate(['Narrow Interneuron', 'Wide Interneuron']):
            ax = axes[row_idx, col_idx]

            results_sub = [r for r in all_results
                           if r['label'] == cell_type and r['pos_var'] == pos_var]
            if len(results_sub) == 0:
                continue

            for phase, tc_key, color, ls in [
                    ('Early (1–6)',  'tc_early', TYPE_COLORS[cell_type], '--'),
                    ('Late (13–18)', 'tc_late',  TYPE_COLORS[cell_type], '-')]:

                tcs = [r[tc_key] for r in results_sub]
                tc_stack = np.array(tcs)

                row_means = np.nanmean(tc_stack, axis=1, keepdims=True)
                row_means[row_means == 0] = np.nan
                tc_norm = tc_stack / row_means

                mean_tc = np.nanmean(tc_norm, axis=0)
                sem_tc  = np.nanstd(tc_norm, axis=0) / np.sqrt(
                    np.sum(~np.isnan(tc_norm), axis=0))

                ax.plot(DIST_BIN_CENTERS, mean_tc, color=color, lw=2,
                        ls=ls, label=phase)
                ax.fill_between(DIST_BIN_CENTERS,
                                mean_tc - sem_tc, mean_tc + sem_tc,
                                color=color, alpha=0.15)

            ax.set_xlabel('Distance to goal (cm)')
            ax.set_ylabel('Normalized firing rate')
            ax.set_title(f'{cell_type} — {pos_var}')
            ax.legend(fontsize=9)
            ax.axhline(1.0, color='gray', ls='--', alpha=0.5)

    fig.suptitle('Learning Effect: Early vs Late Training', fontsize=13)
    plt.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, 'goal_distance_learning_effect.png'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Saved: goal_distance_learning_effect.png")


def plot_example_neurons(all_results, pos_var='r_center', n_examples=4):
    """Plot 4: Individual neuron tuning curves — top modulated per type."""
    fig, axes = plt.subplots(2, n_examples, figsize=(4 * n_examples, 7))

    for row_idx, cell_type in enumerate(['Narrow Interneuron', 'Wide Interneuron']):
        candidates = [r for r in all_results
                      if r['label'] == cell_type and r['pos_var'] == pos_var
                      and not np.isnan(r['mod_idx'])]
        # Sort by absolute modulation index
        candidates.sort(key=lambda r: abs(r['mod_idx']), reverse=True)
        top = candidates[:n_examples]

        for col_idx, r in enumerate(top):
            ax = axes[row_idx, col_idx]
            ax.plot(DIST_BIN_CENTERS, r['tc_all'],
                    color=TYPE_COLORS[cell_type], lw=2)
            ax.fill_between(DIST_BIN_CENTERS, 0, r['tc_all'],
                            color=TYPE_COLORS[cell_type], alpha=0.15)
            ax.set_xlabel('Dist to goal (cm)')
            if col_idx == 0:
                ax.set_ylabel('Firing rate (Hz)')
            ax.set_title(
                f"Cell {r['neuron']} ({r['exp']} M{r['mouse']})\n"
                f"MI={r['mod_idx']:.3f}  FR={r['mean_rate']:.1f}Hz",
                fontsize=8)

        # Label row
        axes[row_idx, 0].annotate(
            cell_type, xy=(-0.35, 0.5), xycoords='axes fraction',
            fontsize=11, fontweight='bold', rotation=90,
            ha='center', va='center')

    fig.suptitle(f'Example Goal-Distance Tuning Curves ({pos_var})', fontsize=13)
    plt.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, f'goal_distance_examples_{pos_var}.png'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: goal_distance_examples_{pos_var}.png")


def plot_nose_vs_center(all_results):
    """Plot 5: Scatter of modulation index — r_nose vs r_center per neuron."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax_idx, cell_type in enumerate(['Narrow Interneuron', 'Wide Interneuron']):
        ax = axes[ax_idx]

        # Match neurons across pos_vars
        center_results = {(r['exp'], r['mouse'], r['day'], r['neuron']): r['mod_idx']
                          for r in all_results
                          if r['label'] == cell_type and r['pos_var'] == 'r_center'
                          and not np.isnan(r['mod_idx'])}
        nose_results = {(r['exp'], r['mouse'], r['day'], r['neuron']): r['mod_idx']
                        for r in all_results
                        if r['label'] == cell_type and r['pos_var'] == 'r_nose'
                        and not np.isnan(r['mod_idx'])}

        common_keys = set(center_results.keys()) & set(nose_results.keys())
        if len(common_keys) == 0:
            continue

        mi_center = [center_results[k] for k in common_keys]
        mi_nose   = [nose_results[k]   for k in common_keys]

        ax.scatter(mi_center, mi_nose,
                   c=TYPE_COLORS[cell_type], alpha=0.5, s=25,
                   edgecolors='white', linewidths=0.5)

        # Unity line
        lims = [min(min(mi_center), min(mi_nose)) - 0.05,
                max(max(mi_center), max(mi_nose)) + 0.05]
        ax.plot(lims, lims, 'k--', alpha=0.4)

        r_corr, p_corr = stats.pearsonr(mi_center, mi_nose)
        ax.set_xlabel('Modulation Index (r_center)')
        ax.set_ylabel('Modulation Index (r_nose)')
        ax.set_title(f'{cell_type}\nr={r_corr:.3f}, p={p_corr:.4f}, n={len(common_keys)}')

    fig.suptitle('Position Variable Comparison: r_center vs r_nose', fontsize=13)
    plt.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, 'goal_distance_nose_vs_center.png'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Saved: goal_distance_nose_vs_center.png")


# ============================================================================
# Summary
# ============================================================================

def print_summary(all_results):
    print(f"\n{'='*70}")
    print(f"  Goal-Distance Modulation — Summary")
    print(f"{'='*70}")

    for pos_var in POS_VARS:
        print(f"\n  Position variable: {pos_var}")
        print(f"  {'-'*50}")

        for label in ['Narrow Interneuron', 'Wide Interneuron']:
            vals = [r['mod_idx'] for r in all_results
                    if r['label'] == label and r['pos_var'] == pos_var
                    and not np.isnan(r['mod_idx'])]
            if len(vals) == 0:
                print(f"    {label:35s}  n=0")
                continue

            n_pos = sum(1 for v in vals if v > 0)
            n_neg = sum(1 for v in vals if v < 0)

            print(f"    {label:35s}  n={len(vals):4d}  "
                  f"median={np.median(vals):.4f}  mean={np.mean(vals):.4f}  "
                  f"range=[{np.min(vals):.4f}, {np.max(vals):.4f}]  "
                  f"pos/neg={n_pos}/{n_neg}")

        # Narrow vs Wide comparison
        narrow = [r['mod_idx'] for r in all_results
                  if r['label'] == 'Narrow Interneuron' and r['pos_var'] == pos_var
                  and not np.isnan(r['mod_idx'])]
        wide = [r['mod_idx'] for r in all_results
                if r['label'] == 'Wide Interneuron' and r['pos_var'] == pos_var
                and not np.isnan(r['mod_idx'])]
        if len(narrow) > 1 and len(wide) > 1:
            u, p = stats.mannwhitneyu(narrow, wide, alternative='two-sided')
            print(f"\n    Mann-Whitney U (Narrow vs Wide):  U={u:.1f}, p={p:.6f}")

    print(f"\n{'='*70}\n")


# ============================================================================
# Entry point
# ============================================================================

if __name__ == '__main__':

    all_results = run_analysis()
    
    # Run goal-distance shuffle test (r_center only to save time)
    recordings = discover_training_trials(EXPERIMENTS)
    print("\nRunning goal-distance shuffle tests...")
    all_results = run_goal_shuffle_test(all_results, recordings)

    print("\nGenerating figures...")
    plot_population_tuning(all_results)
    plot_modulation_index(all_results)
    plot_learning_effect(all_results)
    plot_example_neurons(all_results, pos_var='r_center')
    plot_example_neurons(all_results, pos_var='r_nose')
    plot_nose_vs_center(all_results)

    print_summary(all_results)

    print("Done.")