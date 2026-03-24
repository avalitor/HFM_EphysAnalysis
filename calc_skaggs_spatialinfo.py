# -*- coding: utf-8 -*-
"""
Skaggs Spatial Information Analysis
====================================
Computes occupancy-normalized firing rate maps and Skaggs spatial information
(bits/spike) for each neuron, pooling spikes and occupancy across trials 
within each day. Compares distributions across BI/TP cell types.

Produces:
  1. Histogram of Skaggs info (bits/spike) by cell type
  2. Box/strip plot comparing Narrow vs Wide interneurons
  3. Example rate maps for top spatially-informative neurons per type
  4. Shuffle test: observed Skaggs info vs spike-time-shifted null distribution
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter
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
EXPERIMENTS = ['2024-02-15', '2024-11-09', '2025-01-21']
BINS        = 60                    # spatial bins per axis
ARENA_RANGE = [[-65, 65], [-65, 65]]
VEL_CUTOFF  = 2                     # cm/s — exclude low-speed samples
SIGMA       = 2                     # Gaussian smoothing (bins)
MIN_OCC     = 0.15                   # seconds — min occupancy per bin to avoid edge artifacts
N_SHUFFLES  = 500                   # for shuffle significance test
SHUFFLE_MIN = 20                    # seconds — min circular shift
SHUFFLE_MAX_FRAC = 0.9              # max shift as fraction of recording

EXCLUDED_PREFIXES = ()              # Include all trial types for spatial coverage

SAVE_DIR = os.path.join(cg.ROOT_DIR, 'figs', 'skaggs')
os.makedirs(SAVE_DIR, exist_ok=True)


TYPE_COLORS = {
    "Granule Cell": "#c00021",
    "Mossy Cell": "#358940",
    "Narrow Interneuron": "#0050a0",
    "Wide Interneuron": "#07CCC3",
    "Excitatory Principal Cell": "#e36414",
    "Bursty Narrow Interneuron": "#87147A",
}

# ============================================================================
# Core computation functions
# ============================================================================

def occupancy_histogram(edata, idx_end, bins=BINS, velocity_cutoff=VEL_CUTOFF):
    """
    Compute 2D occupancy histogram (in seconds) for a single trial.
    Returns:
        occ_sec : (bins, bins) array — time spent per spatial bin in seconds
        extent  : [xmin, xmax, ymin, ymax]
    """
    coords = edata.r_center[:idx_end]
    vel    = edata.velocity[:idx_end]

    valid = (np.ones(len(coords), dtype=bool)
             & ~np.isnan(coords[:, 0])
             & ~np.isnan(coords[:, 1])
             & ~np.isnan(vel)
             & (vel > velocity_cutoff))

    x, y = coords[valid, 0], coords[valid, 1]

    occ_counts, xedges, yedges = np.histogram2d(
        x, y, bins=bins, range=ARENA_RANGE)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    dt = np.median(np.diff(edata.time_ttl))
    occ_sec = occ_counts.T * dt          # .T for imshow orientation

    return occ_sec, extent


def spike_histogram(edata, spike_times, idx_end, bins=BINS, velocity_cutoff=VEL_CUTOFF):
    """
    Compute 2D spike-count histogram for a single trial.
    Returns:
        spike_counts : (bins, bins) array
    """
    idx_spike = np.searchsorted(edata.time_ttl, spike_times)
    n_samples = min(idx_end, len(edata.r_center))
    valid = (idx_spike > 0) & (idx_spike < n_samples)
    idx_spike = idx_spike[valid]

    vel_at_spike   = edata.velocity[idx_spike]
    coord_at_spike = edata.r_center[idx_spike]

    keep = ((vel_at_spike > velocity_cutoff)
            & ~np.isnan(coord_at_spike[:, 0])
            & ~np.isnan(coord_at_spike[:, 1]))

    sx, sy = coord_at_spike[keep, 0], coord_at_spike[keep, 1]

    spike_counts, _, _ = np.histogram2d(
        sx, sy, bins=bins, range=ARENA_RANGE)

    return spike_counts.T   # .T for imshow orientation


def compute_ratemap(spike_counts_pooled, occ_sec_pooled, sigma=SIGMA,
                    min_occ=MIN_OCC):
    """
    Compute smoothed, occupancy-normalized firing rate map from
    already-pooled spike counts and occupancy.
    Returns:
        ratemap : (bins, bins) float array, NaN for unvisited/low-occupancy bins
    """
    visited = occ_sec_pooled > 0

    if sigma > 0:
        spike_smooth = gaussian_filter(
            np.where(visited, spike_counts_pooled, 0).astype(float), sigma=sigma)
        occ_smooth = gaussian_filter(
            np.where(visited, occ_sec_pooled, 0).astype(float), sigma=sigma)
        ratemap = np.full_like(spike_smooth, np.nan)
        # Require minimum occupancy to avoid edge artifacts
        good = occ_smooth > min_occ
        ratemap[good] = spike_smooth[good] / occ_smooth[good]
    else:
        ratemap = np.full_like(spike_counts_pooled, np.nan, dtype=float)
        good = occ_sec_pooled > min_occ
        ratemap[good] = spike_counts_pooled[good] / occ_sec_pooled[good]

    return ratemap


def skaggs_spatial_info(ratemap, occ_sec):
    """
    Skaggs et al. (1993) spatial information.
    Returns:
        bits_per_spike, bits_per_sec, mean_rate
    """
    visited = ~np.isnan(ratemap) & (occ_sec > 0)
    if np.sum(visited) == 0:
        return np.nan, np.nan, np.nan

    rates      = ratemap[visited]
    occ        = occ_sec[visited]
    total_time = np.sum(occ)
    p_i        = occ / total_time
    mean_rate  = np.sum(p_i * rates)

    if mean_rate <= 0:
        return np.nan, np.nan, mean_rate

    nonzero = rates > 0
    ratio   = rates[nonzero] / mean_rate

    bits_per_sec   = np.sum(p_i[nonzero] * rates[nonzero] * np.log2(ratio))
    bits_per_spike = bits_per_sec / mean_rate

    return bits_per_spike, bits_per_sec, mean_rate


def shuffle_skaggs(edata_list, idx_end_list, neuron_idx, occ_sec_pooled,
                   n_shuffles=N_SHUFFLES):
    """
    Circular time-shift shuffle test for Skaggs spatial info.
    For each shuffle iteration, shift the spike train by a random offset
    relative to the position data, recompute the spike histogram, and 
    derive Skaggs info from the shifted data.
    
    Returns:
        shuffle_distribution : (n_shuffles,) array of shuffled bits/spike
    """
    shuffle_dist = np.full(n_shuffles, np.nan)

    for si in range(n_shuffles):
        spike_counts_shuf = np.zeros((BINS, BINS))

        for edata, idx_end in zip(edata_list, idx_end_list):
            spike_times = np.atleast_1d(edata.t_spikeTrains[neuron_idx])
            if spike_times.size == 0:
                continue

            # Determine shift bounds for this trial
            t_start = edata.time_ttl[0]
            t_end   = edata.time_ttl[min(idx_end, len(edata.time_ttl)) - 1]
            duration = t_end - t_start
            shift_max = duration * SHUFFLE_MAX_FRAC

            if shift_max < SHUFFLE_MIN:
                # Trial too short for a meaningful shift
                continue

            shift = np.random.uniform(SHUFFLE_MIN, shift_max)
            shifted_spikes = spike_times + shift

            # Wrap around
            shifted_spikes = t_start + np.mod(shifted_spikes - t_start, duration)

            spike_counts_shuf += spike_histogram(
                edata, shifted_spikes, idx_end)

        rm_shuf = compute_ratemap(spike_counts_shuf, occ_sec_pooled, sigma=SIGMA)
        bps, _, _ = skaggs_spatial_info(rm_shuf, occ_sec_pooled)
        shuffle_dist[si] = bps

    return shuffle_dist


# ============================================================================
# File discovery and trial filtering
# ============================================================================

def discover_trials(experiments):
    """
    Auto-discover recordings, group by (experiment, mouse, day).
    Returns dict: (exp, mouse, day) -> list of (trial_label, edata)
    """
    recordings = {}   # key: (exp, mouse, day) -> list of trial edatas

    for exp in experiments:
        exp_dir = os.path.join(cg.PROCESSED_FILE_DIR, exp)
        if not os.path.isdir(exp_dir):
            print(f"  [WARN] Experiment folder not found: {exp_dir}")
            continue

        mat_files = glob.glob(os.path.join(exp_dir, 'hfmE_*.mat'))
        # Parse filenames: hfmE_{date}_M{mouse}_{trial}.mat
        pattern = re.compile(r'hfmE_[\d-]+_M(\d+)_(.+)\.mat')

        for fpath in sorted(mat_files):
            fname = os.path.basename(fpath)
            m = pattern.match(fname)
            if not m:
                continue
            mouse_str, trial_label = m.group(1), m.group(2)
            mouse = int(mouse_str)

            # Filter excluded trial types (empty tuple = include all)
            if any(trial_label.startswith(prefix) for prefix in EXCLUDED_PREFIXES):
                continue

            try:
                edata = elib.EphysTrial.Load(exp, mouse, trial_label,
                                             load_cell_metrics=True)
            except Exception as e:
                print(f"  [WARN] Could not load {fname}: {e}")
                continue

            day = str(edata.day)
            key = (exp, mouse, day)

            if key not in recordings:
                recordings[key] = []
            recordings[key].append((trial_label, edata))

    print(f"Discovered {len(recordings)} unique (exp, mouse, day) groups "
          f"across {sum(len(v) for v in recordings.values())} trials")
    return recordings


# ============================================================================
# Main analysis
# ============================================================================

def run_analysis():
    print("Discovering trials...")
    recordings = discover_trials(EXPERIMENTS)

    all_results = []

    for (exp, mouse, day), trial_list in recordings.items():
        print(f"\nProcessing {exp} M{mouse} day={day}  ({len(trial_list)} trials)")

        # Use the first trial's metadata for cell labels and counts
        first_edata = trial_list[0][1]
        n_neurons = len(first_edata.t_spikeTrains)
        labels = [first_edata.cellLabels_BITP[i][0] for i in range(n_neurons)]

        # ----- Pool occupancy and spike counts across trials -----
        edata_list   = []
        idx_end_list = []
        trial_labels = []
        occ_pooled   = np.zeros((BINS, BINS))

        for trial_label, edata in trial_list:
            # Use full trial length for spatial info to maximize coverage
            idx_end = len(edata.time_ttl)

            occ_trial, extent = occupancy_histogram(edata, idx_end)
            occ_pooled += occ_trial
            edata_list.append(edata)
            idx_end_list.append(idx_end)
            trial_labels.append(trial_label)

        # ----- Per-neuron: pool spike counts, compute ratemap & Skaggs -----
        for neuron_idx in range(n_neurons):
            spike_pooled = np.zeros((BINS, BINS))

            for edata, idx_end in zip(edata_list, idx_end_list):
                spike_times = np.atleast_1d(edata.t_spikeTrains[neuron_idx])
                if spike_times.size == 0:
                    continue
                spike_pooled += spike_histogram(edata, spike_times, idx_end)

            ratemap = compute_ratemap(spike_pooled, occ_pooled, sigma=SIGMA)
            bps, bps_sec, mean_rate = skaggs_spatial_info(ratemap, occ_pooled)

            all_results.append({
                'exp':            exp,
                'mouse':          mouse,
                'day':            day,
                'trial_labels':   trial_labels,
                'neuron':         neuron_idx,
                'label':          labels[neuron_idx],
                'bits_per_spike': bps,
                'bits_per_sec':   bps_sec,
                'mean_rate':      mean_rate,
                'ratemap':        ratemap,
                'extent':         extent,
                # Store references for shuffle test later
                '_edata_list':    edata_list,
                '_idx_end_list':  idx_end_list,
                '_occ_pooled':    occ_pooled,
            })

    print(f"\nTotal neurons analysed: {len(all_results)}")
    return all_results


def run_shuffle_test(all_results, types_to_test=None):
    """
    Run circular-shift shuffle test on each neuron to get a p-value
    for observed spatial information.
    """
    if types_to_test is None:
        types_to_test = ['Narrow Interneuron', 'Wide Interneuron']

    for r in all_results:
        if r['label'] not in types_to_test:
            r['shuffle_p'] = np.nan
            continue
        if np.isnan(r['bits_per_spike']):
            r['shuffle_p'] = np.nan
            continue

        shuf_dist = shuffle_skaggs(
            r['_edata_list'], r['_idx_end_list'],
            r['neuron'], r['_occ_pooled'],
            n_shuffles=N_SHUFFLES)

        # p-value: fraction of shuffles >= observed
        n_valid = np.sum(~np.isnan(shuf_dist))
        if n_valid == 0:
            r['shuffle_p'] = np.nan
        else:
            r['shuffle_p'] = np.nansum(shuf_dist >= r['bits_per_spike']) / n_valid
        r['shuffle_dist'] = shuf_dist

    n_sig = sum(1 for r in all_results
                if r.get('shuffle_p', np.nan) is not np.nan
                and not np.isnan(r.get('shuffle_p', np.nan))
                and r['shuffle_p'] < 0.05)
    n_tested = sum(1 for r in all_results
                   if not np.isnan(r.get('shuffle_p', np.nan)))
    print(f"Shuffle test: {n_sig}/{n_tested} neurons significant at p<0.05")
    return all_results


# ============================================================================
# Plotting
# ============================================================================

def plot_skaggs_histograms(all_results):
    """Plot 1: Overlapping histograms of bits/spike by cell type."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for label, color in TYPE_COLORS.items():
        vals_spike = [r['bits_per_spike'] for r in all_results
                      if r['label'] == label and not np.isnan(r['bits_per_spike'])]
        vals_sec   = [r['bits_per_sec'] for r in all_results
                      if r['label'] == label and not np.isnan(r['bits_per_sec'])]
        if len(vals_spike) == 0:
            continue

        axes[0].hist(vals_spike, bins=20, alpha=0.5,
                     label=f'{label} (n={len(vals_spike)})',
                     color=color, edgecolor='white')
        axes[1].hist(vals_sec, bins=20, alpha=0.5,
                     label=label, color=color, edgecolor='white')

    axes[0].set_xlabel('Spatial Info (bits/spike)')
    axes[0].set_ylabel('Neuron count')
    axes[0].set_title('Skaggs Spatial Information')
    axes[0].legend(fontsize=8)

    axes[1].set_xlabel('Spatial Info (bits/sec)')
    axes[1].set_ylabel('Neuron count')
    axes[1].set_title('Spatial Information Rate')
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, 'skaggs_histograms_all.png'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Saved: skaggs_histograms_all.png")


def plot_box_comparison(all_results):
    """Plot 2: Box + strip plot comparing Narrow vs Wide interneurons."""
    compare_types = ['Narrow Interneuron', 'Wide Interneuron']
    data_by_type = {}
    for ct in compare_types:
        data_by_type[ct] = [r['bits_per_spike'] for r in all_results
                            if r['label'] == ct and not np.isnan(r['bits_per_spike'])]

    if all(len(v) == 0 for v in data_by_type.values()):
        print("No data for box comparison — skipping.")
        return

    fig, ax = plt.subplots(figsize=(6, 5))

    positions = [0, 1]
    for i, ct in enumerate(compare_types):
        vals = data_by_type[ct]
        if len(vals) == 0:
            continue

        # Box
        bp = ax.boxplot([vals], positions=[positions[i]], widths=0.4,
                        patch_artist=True, showfliers=False)
        bp['boxes'][0].set_facecolor(TYPE_COLORS[ct])
        bp['boxes'][0].set_alpha(0.4)
        bp['medians'][0].set_color('black')

        # Jittered individual points
        jitter = np.random.normal(0, 0.06, size=len(vals))
        ax.scatter(np.full(len(vals), positions[i]) + jitter, vals,
                   c=TYPE_COLORS[ct], alpha=0.6, s=25, edgecolors='white',
                   linewidths=0.5, zorder=5)

    # Mann-Whitney U test
    narrow_vals = data_by_type['Narrow Interneuron']
    wide_vals   = data_by_type['Wide Interneuron']
    if len(narrow_vals) > 1 and len(wide_vals) > 1:
        u_stat, p_val = stats.mannwhitneyu(narrow_vals, wide_vals,
                                           alternative='two-sided')
        sig_str = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'n.s.'
        ymax = max(max(narrow_vals), max(wide_vals))
        ax.plot([0, 1], [ymax * 1.05, ymax * 1.05], 'k-', lw=1)
        ax.text(0.5, ymax * 1.08, f'{sig_str}\np={p_val:.4f}',
                ha='center', fontsize=9)

    ax.set_xticks(positions)
    ax.set_xticklabels(['Narrow\nInterneuron', 'Wide\nInterneuron'])
    ax.set_ylabel('Spatial Info (bits/spike)')
    ax.set_title('Spatial Information: Narrow vs Wide Interneurons')

    # Print medians in subtitle
    med_n = np.median(narrow_vals) if len(narrow_vals) > 0 else float('nan')
    med_w = np.median(wide_vals)   if len(wide_vals)   > 0 else float('nan')
    ax.text(0.5, -0.12,
            f'Narrow: median={med_n:.3f}, n={len(narrow_vals)}    '
            f'Wide: median={med_w:.3f}, n={len(wide_vals)}',
            transform=ax.transAxes, ha='center', fontsize=8, color='gray')

    plt.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, 'skaggs_narrow_vs_wide_box.png'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Saved: skaggs_narrow_vs_wide_box.png")


def plot_top_ratemaps(all_results, edata_lookup):
    """Plot 3: Top spatially-informative neurons per cell type (rate maps)."""
    for cell_type in ['Narrow Interneuron', 'Wide Interneuron']:
        type_results = [r for r in all_results
                        if r['label'] == cell_type
                        and not np.isnan(r['bits_per_spike'])
                        and r['ratemap'] is not None]
        type_results.sort(key=lambda r: r['bits_per_spike'], reverse=True)
        top_n = type_results[:min(8, len(type_results))]

        if len(top_n) == 0:
            continue

        ncols = 4
        nrows = int(np.ceil(len(top_n) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.5 * nrows))
        axes = np.atleast_2d(axes)

        for idx, r in enumerate(top_n):
            row, col = divmod(idx, ncols)
            ax = axes[row, col]
            rm = r['ratemap'].copy()
            rm_display = np.where(np.isnan(rm), 0, rm)
            ax.imshow(rm_display, extent=[-65, 65, -65, 65],
                      origin='lower', cmap='inferno')

            # Overlay arena from first trial of this day
            key = (r['exp'], r['mouse'], r['day'])
            if key in edata_lookup:
                draw_arena(edata_lookup[key], ax, color='white')

            p_str = ''
            if 'shuffle_p' in r and not np.isnan(r.get('shuffle_p', np.nan)):
                p_str = f'\np={r["shuffle_p"]:.3f}'
            trials_str = ', '.join(r.get('trial_labels', []))
            ax.set_title(
                f"Cell {r['neuron']}  ({r['exp']} M{r['mouse']})\n"
                f"{r['bits_per_spike']:.3f} bits/spk{p_str}\n"
                f"Trials: {trials_str}",
                fontsize=7)

        for idx in range(len(top_n), nrows * ncols):
            row, col = divmod(idx, ncols)
            axes[row, col].axis('off')

        safe = cell_type.replace(' ', '_')
        fig.suptitle(f'{cell_type} — Top rate maps by spatial info', fontsize=12)
        plt.tight_layout()
        fig.savefig(os.path.join(SAVE_DIR, f'skaggs_ratemaps_{safe}.png'),
                    dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: skaggs_ratemaps_{safe}.png")


def plot_shuffle_summary(all_results):
    """Plot 4: Observed vs shuffle distribution for one example neuron per type."""
    for cell_type in ['Narrow Interneuron', 'Wide Interneuron']:
        candidates = [r for r in all_results
                      if r['label'] == cell_type
                      and 'shuffle_dist' in r
                      and not np.isnan(r['bits_per_spike'])]
        if not candidates:
            continue

        # Pick the most spatially informative neuron as the example
        candidates.sort(key=lambda r: r['bits_per_spike'], reverse=True)
        r = candidates[0]

        fig, ax = plt.subplots(figsize=(6, 4))
        shuf = r['shuffle_dist'][~np.isnan(r['shuffle_dist'])]
        ax.hist(shuf, bins=30, color='gray', alpha=0.6, edgecolor='white',
                label='Shuffle distribution')
        ax.axvline(r['bits_per_spike'], color=TYPE_COLORS[cell_type],
                   lw=2, label=f'Observed ({r["bits_per_spike"]:.3f} bits/spk)')
        ax.set_xlabel('Spatial Info (bits/spike)')
        ax.set_ylabel('Count')
        ax.set_title(
            f'{cell_type} — Cell {r["neuron"]} ({r["exp"]} M{r["mouse"]})\n'
            f'Shuffle p = {r["shuffle_p"]:.3f}')
        ax.legend(fontsize=9)

        plt.tight_layout()
        safe = cell_type.replace(' ', '_')
        fig.savefig(os.path.join(SAVE_DIR, f'skaggs_shuffle_example_{safe}.png'),
                    dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: skaggs_shuffle_example_{safe}.png")


def draw_arena(data, ax, color='k'):
    """Draw arena circle and holes (copied from spatial plotting code)."""
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


# ============================================================================
# Summary statistics
# ============================================================================

def print_summary(all_results):
    print(f"\n{'='*70}")
    print("  Skaggs Spatial Information — Summary Across All Recordings")
    print(f"{'='*70}")

    for label in TYPE_COLORS:
        vals = [r['bits_per_spike'] for r in all_results
                if r['label'] == label and not np.isnan(r['bits_per_spike'])]
        if len(vals) == 0:
            print(f"  {label:35s}  n=0")
            continue

        sig = [r for r in all_results
               if r['label'] == label
               and not np.isnan(r.get('shuffle_p', np.nan))
               and r['shuffle_p'] < 0.05]

        print(f"  {label:35s}  n={len(vals):4d}  "
              f"median={np.median(vals):.4f}  mean={np.mean(vals):.4f}  "
              f"range=[{np.min(vals):.4f}, {np.max(vals):.4f}]  "
              f"sig(p<.05)={len(sig)}")

    # Narrow vs Wide test
    narrow = [r['bits_per_spike'] for r in all_results
              if r['label'] == 'Narrow Interneuron' and not np.isnan(r['bits_per_spike'])]
    wide   = [r['bits_per_spike'] for r in all_results
              if r['label'] == 'Wide Interneuron' and not np.isnan(r['bits_per_spike'])]
    if len(narrow) > 1 and len(wide) > 1:
        u, p = stats.mannwhitneyu(narrow, wide, alternative='two-sided')
        print(f"\n  Mann-Whitney U (Narrow vs Wide):  U={u:.1f}, p={p:.6f}")
    print(f"{'='*70}\n")


# ============================================================================
# Entry point
# ============================================================================

if __name__ == '__main__':

    # 1. Compute Skaggs info for all neurons
    all_results = run_analysis()

    # Build lookup for arena drawing: (exp, mouse, day) -> first edata
    edata_lookup = {}
    for r in all_results:
        key = (r['exp'], r['mouse'], r['day'])
        if key not in edata_lookup:
            edata_lookup[key] = r['_edata_list'][0]

    # 2. Run shuffle test (this takes a while — ~500 shuffles per neuron)
    print("\nRunning shuffle tests (Narrow & Wide interneurons)...")
    all_results = run_shuffle_test(all_results)

    # 3. Generate figures
    print("\nGenerating figures...")
    plot_skaggs_histograms(all_results)
    plot_box_comparison(all_results)
    plot_top_ratemaps(all_results, edata_lookup)
    plot_shuffle_summary(all_results)

    # 4. Print summary
    print_summary(all_results)

    # Clean up stored references to free memory
    for r in all_results:
        r.pop('_edata_list', None)
        r.pop('_idx_end_list', None)
        r.pop('_occ_pooled', None)

    print("Done.")