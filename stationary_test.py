# -*- coding: utf-8 -*-
"""
Stationarity Diagnostics for Spike Trains

Investigates why most neurons fail the stationarity test (Kruskal-Wallis on
segment means + Levene's on segment variances). Identifies which parameters
are changing, when they change relative to the reward, and whether trimming
the pre-reward search + eating epoch recovers stationarity.

Behavioral timeline per trial:
  1. Search phase:      trial start  →  t_reward
  2. Eating phase:      t_reward     →  t_reward + 60 s
  3. Free exploration:  t_reward + 60 s  →  trial end

Figures saved to: figs/stationarity_diagnostics/
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from scipy import stats
import glob, os, re, warnings
import pandas as pd

import lib_ephys_obj as elib
import config as cg

warnings.filterwarnings('ignore', category=RuntimeWarning)

# %% ============================================================
# PARAMETERS
# ===============================================================
EXPERIMENTS = ['2024-02-15', '2024-11-09', '2025-01-21']

MIN_TRIAL_DURATION = 1200   # seconds (20 min)
MIN_ISI_COUNT      = 100    # minimum ISIs for analysis
MIN_RATE_HZ        = 0.5    # minimum mean rate (Hz)
EATING_DURATION    = 60.0   # seconds after reward for eating phase
N_TIME_SEGMENTS    = 10     # segments for trajectory / heatmap plots
N_STAT_SEGMENTS    = 4      # segments for stationarity test (matches original)

EXCLUDED_PREFIXES = ()      # use all trial types (same as burst_analysis)

CELL_COLORS = {
    'Narrow Interneuron':         '#0050a0',
    'Wide Interneuron':           '#07CCC3',
    'Granule Cell':               '#c00021',
    'Mossy Cell':                 '#358940',
    'Bursty Narrow Interneuron':  '#87147A',
}
CELL_ORDER = ['Narrow Interneuron', 'Wide Interneuron',
              'Granule Cell', 'Mossy Cell', 'Bursty Narrow Interneuron']

FIG_DIR = os.path.join(cg.ROOT_DIR, 'figs', 'stationarity_diagnostics')
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(os.path.join(FIG_DIR, 'gallery'), exist_ok=True)


# %% ============================================================
# DATA DISCOVERY AND LOADING
# ===============================================================

def discover_trials(experiments, path=cg.PROCESSED_FILE_DIR):
    """Find all .mat files for specified experiments."""
    recordings = []
    for exp in experiments:
        exp_dir = os.path.join(path, exp)
        mat_files = glob.glob(os.path.join(exp_dir, 'hfmE_*.mat'))
        for fpath in mat_files:
            fname = os.path.basename(fpath)
            match = re.match(r'hfmE_\d{4}-\d{2}-\d{2}_M(\d+)_(.+)\.mat', fname)
            if match:
                mouse = match.group(1)
                trial = match.group(2)
                skip = any(trial.startswith(p) for p in EXCLUDED_PREFIXES)
                if not skip:
                    recordings.append((exp, mouse, trial))
    print(f"Discovered {len(recordings)} recordings across "
          f"{len(experiments)} experiments")
    return recordings


def _get_cell_label(edata, neuron_idx):
    """Get cell type label; returns None if cell should be excluded."""
    try:
        bitp = str(edata.cellLabels_BITP[neuron_idx][0])
    except (IndexError, TypeError, AttributeError):
        return None
    if bitp == 'Excitatory Principal Cell':
        if hasattr(edata, 'cellLabels_pca'):
            try:
                pca_entry = edata.cellLabels_pca[neuron_idx]
                if pca_entry.size > 0 and str(pca_entry[0]) != '':
                    return str(pca_entry[0])
                else:
                    return None
            except (IndexError, TypeError):
                return None
        else:
            return None
    return bitp


def _get_reward_time(edata):
    """
    Extract reward time in seconds (relative to trial start).
    Returns None if k_reward is missing or invalid.
    """
    if not hasattr(edata, 'k_reward') or edata.k_reward is None:
        return None
    try:
        k = int(edata.k_reward)
    except (ValueError, TypeError):
        return None
    if k <= 0 or k >= len(edata.time_ttl):
        return None
    t_start = edata.time_ttl[0]
    t_reward = edata.time_ttl[k] - t_start
    if t_reward <= 0:
        return None
    return float(t_reward)


def load_and_filter_trials(recordings):
    """
    Load trials > MIN_TRIAL_DURATION. Extract spike trains, cell labels,
    and reward time for each qualifying neuron.
    """
    trial_data = []
    loaded_days = {}

    for i, (exp, mouse, trial) in enumerate(recordings):
        if (i + 1) % 20 == 0:
            print(f"  Loading {i+1}/{len(recordings)}...")
        try:
            edata = elib.EphysTrial.Load(exp, mouse, trial)
        except Exception as e:
            print(f"  Failed to load {exp} M{mouse} T{trial}: {e}")
            continue

        if edata.time_ttl is None or len(edata.time_ttl) < 2:
            continue
        t_start = edata.time_ttl[0]
        t_end = edata.time_ttl[-1]
        duration = t_end - t_start
        if duration < MIN_TRIAL_DURATION:
            continue

        day = str(edata.day)
        day_key = (exp, mouse, day)

        # Get reward time
        t_reward = _get_reward_time(edata)

        # Load cell labels once per (exp, mouse, day)
        if day_key not in loaded_days:
            try:
                edata_cm = elib.EphysTrial.Load(exp, mouse, trial,
                                                 load_cell_metrics=True)
                n_neurons = len(edata_cm.t_spikeTrains)
                labels = []
                for ni in range(n_neurons):
                    lbl = _get_cell_label(edata_cm, ni)
                    labels.append(lbl)
                loaded_days[day_key] = labels
            except Exception as e:
                print(f"  Failed to load cell_metrics for {day_key}: {e}")
                n_neurons = len(edata.t_spikeTrains)
                loaded_days[day_key] = [None] * n_neurons

        labels = loaded_days[day_key]
        n_neurons = len(edata.t_spikeTrains)

        if len(labels) != n_neurons:
            print(f"  Label mismatch for {exp} M{mouse} T{trial}: "
                  f"{len(labels)} labels vs {n_neurons} neurons")
            if len(labels) > n_neurons:
                labels = labels[:n_neurons]
            else:
                labels = labels + [None] * (n_neurons - len(labels))

        for ni in range(n_neurons):
            if labels[ni] is None:
                continue

            spk = np.atleast_1d(edata.t_spikeTrains[ni])
            if spk.size == 0:
                continue

            # Make spike times relative to trial start
            spk_rel = spk - t_start

            isis = np.diff(spk_rel)
            if isis.size < MIN_ISI_COUNT:
                continue

            mean_rate = len(spk_rel) / duration
            if mean_rate < MIN_RATE_HZ:
                continue

            trial_data.append({
                'exp': exp, 'mouse': mouse, 'day': day, 'trial': trial,
                'neuron_idx': ni,
                'neuron_id': f"{exp}_M{mouse}_d{day}_n{ni}",
                'cell_type': labels[ni],
                'spike_times': spk_rel,
                'isis': isis,
                'duration': duration,
                'mean_rate': mean_rate,
                'n_spikes': len(spk_rel),
                'n_isis': len(isis),
                't_reward': t_reward,
            })

    print(f"\nQualifying neurons (trials>20min, >=100 ISIs, >=0.5Hz): "
          f"{len(trial_data)}")
    ct_counts = {}
    for td in trial_data:
        ct = td['cell_type']
        ct_counts[ct] = ct_counts.get(ct, 0) + 1
    for ct, n in sorted(ct_counts.items()):
        print(f"  {ct}: {n}")

    n_with_reward = sum(1 for td in trial_data if td['t_reward'] is not None)
    print(f"  Neurons with valid t_reward: {n_with_reward}/{len(trial_data)}")

    return trial_data


# %% ============================================================
# STATIONARITY TEST FUNCTIONS
# ===============================================================

def stationarity_test(spike_times, n_segments=N_STAT_SEGMENTS):
    """
    Test for wide-sense stationarity (Turner et al. 1996).
    Divides ISIs into n_segments equal parts, tests for equal means
    (Kruskal-Wallis) and equal variances (Levene's).
    Returns Python bool for 'stationary' (not numpy bool).
    """
    isis = np.diff(spike_times)
    n = len(isis)
    seg_size = n // n_segments
    if seg_size < 20:
        return {'stationary': None, 'p_mean': None, 'p_var': None,
                'seg_means': [], 'seg_vars': []}
    segments, seg_means, seg_vars = [], [], []
    for s in range(n_segments):
        start = s * seg_size
        end = start + seg_size if s < n_segments - 1 else n
        seg = isis[start:end]
        seg_means.append(np.mean(seg))
        seg_vars.append(np.var(seg))
        segments.append(seg)
    try:
        _, p_mean = stats.kruskal(*segments)
        p_mean = float(p_mean)
    except Exception:
        p_mean = np.nan
    try:
        _, p_var = stats.levene(*segments)
        p_var = float(p_var)
    except Exception:
        p_var = np.nan
    stationary = bool((p_mean > 0.05) and (p_var > 0.05))
    return {'stationary': stationary, 'p_mean': p_mean, 'p_var': p_var,
            'seg_means': seg_means, 'seg_vars': seg_vars}


def stationarity_test_on_isis(isis, n_segments=N_STAT_SEGMENTS):
    """Same as stationarity_test but takes ISI array directly.
    Returns Python bool for 'stationary'."""
    n = len(isis)
    seg_size = n // n_segments
    if seg_size < 20:
        return {'stationary': None, 'p_mean': None, 'p_var': None}
    segments = []
    for s in range(n_segments):
        start = s * seg_size
        end = start + seg_size if s < n_segments - 1 else n
        segments.append(isis[start:end])
    try:
        _, p_mean = stats.kruskal(*segments)
        p_mean = float(p_mean)
    except Exception:
        p_mean = np.nan
    try:
        _, p_var = stats.levene(*segments)
        p_var = float(p_var)
    except Exception:
        p_var = np.nan
    stationary = bool((p_mean > 0.05) and (p_var > 0.05))
    return {'stationary': stationary, 'p_mean': p_mean, 'p_var': p_var}


def compute_time_segment_rates(spike_times, duration, n_segments=N_TIME_SEGMENTS):
    """
    Divide the trial into n_segments equal-time bins and compute
    the firing rate in each bin.
    Returns: seg_centers (time of bin center), seg_rates (Hz)
    """
    edges = np.linspace(0, duration, n_segments + 1)
    seg_rates = np.zeros(n_segments)
    seg_centers = np.zeros(n_segments)
    for s in range(n_segments):
        t0, t1 = edges[s], edges[s + 1]
        n_spk = np.sum((spike_times >= t0) & (spike_times < t1))
        seg_rates[s] = n_spk / (t1 - t0) if (t1 - t0) > 0 else 0
        seg_centers[s] = 0.5 * (t0 + t1)
    return seg_centers, seg_rates, edges


# %% ============================================================
# ANALYSIS: Run all stationarity tests
# ===============================================================

def run_all_analyses(trial_data):
    """Run stationarity tests on full trial and free-exploration epoch."""
    results = []

    for i, td in enumerate(trial_data):
        if (i + 1) % 50 == 0:
            print(f"  Analyzing {i+1}/{len(trial_data)}...")

        spk = td['spike_times']
        isis = td['isis']
        t_reward = td['t_reward']
        duration = td['duration']

        # --- Full-trial stationarity ---
        stat_full = stationarity_test(spk)

        # --- Time-segment rates (10 bins) ---
        seg_centers, seg_rates, seg_edges = compute_time_segment_rates(
            spk, duration, N_TIME_SEGMENTS)

        # Which segment contains the reward?
        reward_segment = None
        if t_reward is not None:
            for s in range(N_TIME_SEGMENTS):
                if seg_edges[s] <= t_reward < seg_edges[s + 1]:
                    reward_segment = s
                    break

        # --- Rate deviation: max deviation from mean as percentage ---
        mean_rate = td['mean_rate']
        if mean_rate > 0:
            max_dev_pct = np.max(np.abs(seg_rates - mean_rate)) / mean_rate * 100
        else:
            max_dev_pct = np.nan

        # Find which segment has the maximum deviation
        max_dev_seg = int(np.argmax(np.abs(seg_rates - mean_rate)))
        max_dev_is_pre_reward = None
        if t_reward is not None:
            t_max_dev_seg_end = seg_edges[max_dev_seg + 1]
            max_dev_is_pre_reward = t_max_dev_seg_end <= t_reward

        # --- Leave-one-out recovery (4 segments) ---
        loo_recovers = [False] * N_STAT_SEGMENTS
        if stat_full['stationary'] == False:
            seg_size_4 = len(isis) // N_STAT_SEGMENTS
            if seg_size_4 >= 20:
                segments_4 = []
                for s in range(N_STAT_SEGMENTS):
                    start = s * seg_size_4
                    end = start + seg_size_4 if s < N_STAT_SEGMENTS - 1 else len(isis)
                    segments_4.append(isis[start:end])
                for drop in range(N_STAT_SEGMENTS):
                    remaining = [segments_4[s] for s in range(N_STAT_SEGMENTS)
                                 if s != drop]
                    remaining_isis = np.concatenate(remaining)
                    loo_stat = stationarity_test_on_isis(remaining_isis,
                                                         n_segments=N_STAT_SEGMENTS - 1)
                    if loo_stat['stationary'] == True:
                        loo_recovers[drop] = True

        # Which 4-segment bin contains the reward?
        reward_segment_4 = None
        if t_reward is not None and len(isis) >= N_STAT_SEGMENTS * 20:
            seg_size_4 = len(isis) // N_STAT_SEGMENTS
            # Approximate: find which ISI-index corresponds to t_reward
            # by finding the spike closest to t_reward
            k_reward_spk = np.searchsorted(spk, t_reward)
            if k_reward_spk > 0:
                k_reward_isi = k_reward_spk - 1  # ISI index
                reward_segment_4 = min(k_reward_isi // seg_size_4,
                                       N_STAT_SEGMENTS - 1)

        # --- Free-exploration stationarity (post reward + eating) ---
        stat_explore = {'stationary': None, 'p_mean': None, 'p_var': None}
        explore_n_isis = 0
        explore_skipped = False
        if t_reward is not None:
            t_explore_start = t_reward + EATING_DURATION
            spk_explore = spk[spk >= t_explore_start]
            if len(spk_explore) > 1:
                isis_explore = np.diff(spk_explore)
                explore_n_isis = len(isis_explore)
                if explore_n_isis >= MIN_ISI_COUNT:
                    stat_explore = stationarity_test_on_isis(isis_explore)
                else:
                    explore_skipped = True
            else:
                explore_skipped = True
        else:
            explore_skipped = True

        results.append({
            'neuron_id': td['neuron_id'], 'exp': td['exp'],
            'mouse': td['mouse'], 'day': td['day'], 'trial': td['trial'],
            'neuron_idx': td['neuron_idx'], 'cell_type': td['cell_type'],
            'n_spikes': td['n_spikes'], 'n_isis': td['n_isis'],
            'duration_s': duration, 'mean_rate_hz': mean_rate,
            't_reward': t_reward,

            # Full-trial stationarity
            'stationary_full': stat_full['stationary'],
            'p_mean_full': stat_full['p_mean'],
            'p_var_full': stat_full['p_var'],
            'seg_means_full': stat_full.get('seg_means', []),
            'seg_vars_full': stat_full.get('seg_vars', []),

            # Time-segment rates
            'seg_centers': seg_centers,
            'seg_rates': seg_rates,
            'seg_edges': seg_edges,
            'reward_segment_10': reward_segment,

            # Rate deviation
            'max_dev_pct': max_dev_pct,
            'max_dev_seg': max_dev_seg,
            'max_dev_is_pre_reward': max_dev_is_pre_reward,

            # Leave-one-out
            'loo_recovers': loo_recovers,
            'reward_segment_4': reward_segment_4,

            # Free-exploration stationarity
            'stationary_explore': stat_explore['stationary'],
            'p_mean_explore': stat_explore['p_mean'],
            'p_var_explore': stat_explore['p_var'],
            'explore_n_isis': explore_n_isis,
            'explore_skipped': explore_skipped,
        })

    return results


# %% ============================================================
# PLOTTING FUNCTIONS
# ===============================================================

def plot_fig1_pvalue_scatter(results, fig_dir):
    """Fig 1: Scatter of p_mean vs p_var, colored by cell type."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: scatter
    ax = axes[0]
    for ct in CELL_ORDER:
        sub = [r for r in results if r['cell_type'] == ct
               and r['p_mean_full'] is not None
               and r['p_var_full'] is not None]
        if not sub:
            continue
        p_means = [r['p_mean_full'] for r in sub]
        p_vars = [r['p_var_full'] for r in sub]
        # Clamp zeros for log scale
        p_means = np.clip(p_means, 1e-20, 1)
        p_vars = np.clip(p_vars, 1e-20, 1)
        ax.scatter(p_means, p_vars, c=CELL_COLORS.get(ct, 'gray'),
                   label=ct, alpha=0.6, s=20, edgecolors='none')
    ax.axvline(0.05, color='red', ls='--', lw=1, alpha=0.7, label='p = 0.05')
    ax.axhline(0.05, color='red', ls='--', lw=1, alpha=0.7)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('p_mean (Kruskal-Wallis)')
    ax.set_ylabel('p_var (Levene\'s)')
    ax.set_title('Stationarity Test p-values')
    ax.legend(fontsize=7, loc='lower right')

    # Quadrant counts
    valid = [r for r in results if r['stationary_full'] is not None]
    n_pass_both = sum(1 for r in valid if r['p_mean_full'] > 0.05
                      and r['p_var_full'] > 0.05)
    n_fail_mean = sum(1 for r in valid if r['p_mean_full'] <= 0.05
                      and r['p_var_full'] > 0.05)
    n_fail_var = sum(1 for r in valid if r['p_mean_full'] > 0.05
                     and r['p_var_full'] <= 0.05)
    n_fail_both = sum(1 for r in valid if r['p_mean_full'] <= 0.05
                      and r['p_var_full'] <= 0.05)
    n_total = len(valid)

    ax.text(0.03, 0.97,
            f'Pass both: {n_pass_both}/{n_total} ({100*n_pass_both/n_total:.0f}%)',
            transform=ax.transAxes, fontsize=8, va='top', color='green')
    ax.text(0.03, 0.91,
            f'Fail mean only: {n_fail_mean}/{n_total} ({100*n_fail_mean/n_total:.0f}%)',
            transform=ax.transAxes, fontsize=8, va='top', color='darkorange')
    ax.text(0.03, 0.85,
            f'Fail var only: {n_fail_var}/{n_total} ({100*n_fail_var/n_total:.0f}%)',
            transform=ax.transAxes, fontsize=8, va='top', color='purple')
    ax.text(0.03, 0.79,
            f'Fail both: {n_fail_both}/{n_total} ({100*n_fail_both/n_total:.0f}%)',
            transform=ax.transAxes, fontsize=8, va='top', color='red')

    # Right: per cell type stationarity rates
    ax = axes[1]
    ct_stats = []
    for ct in CELL_ORDER:
        sub = [r for r in valid if r['cell_type'] == ct]
        if len(sub) == 0:
            continue
        n_stat = sum(1 for r in sub if r['stationary_full'] == True)
        ct_stats.append((ct, n_stat, len(sub)))
    if ct_stats:
        x_pos = np.arange(len(ct_stats))
        fracs = [s[1] / s[2] for s in ct_stats]
        bar_colors = [CELL_COLORS.get(s[0], 'gray') for s in ct_stats]
        bars = ax.bar(x_pos, fracs, color=bar_colors, alpha=0.7, edgecolor='none')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([s[0].replace(' ', '\n') for s in ct_stats],
                           fontsize=8)
        for bar, sc_item in zip(bars, ct_stats):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f'{sc_item[1]}/{sc_item[2]}', ha='center', fontsize=8)
        ax.set_ylabel('Fraction Stationary')
        ax.set_title('Stationarity Rate by Cell Type')
        ax.set_ylim(0, 1.1)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'fig1_pvalue_scatter.png'),
                dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved fig1_pvalue_scatter.png")


def plot_fig2_rate_trajectories(results, fig_dir):
    """Fig 2: Segment-level rate trajectories with reward marker."""
    valid_types = [ct for ct in CELL_ORDER
                   if any(r['cell_type'] == ct for r in results)]
    n_types = len(valid_types)
    if n_types == 0:
        return

    fig, axes = plt.subplots(1, n_types, figsize=(4 * n_types, 4.5),
                             sharey=True)
    if n_types == 1:
        axes = [axes]

    for ax, ct in zip(axes, valid_types):
        sub = [r for r in results if r['cell_type'] == ct]
        color = CELL_COLORS.get(ct, 'gray')

        for r in sub:
            mean_rate = r['mean_rate_hz']
            if mean_rate <= 0:
                continue
            # Normalized rate per segment
            norm_rates = r['seg_rates'] / mean_rate
            seg_frac = r['seg_centers'] / r['duration_s']
            ax.plot(seg_frac, norm_rates, '-', color=color, alpha=0.15, lw=0.8)

            # Mark reward segment
            if r['reward_segment_10'] is not None:
                rs = r['reward_segment_10']
                ax.plot(seg_frac[rs], norm_rates[rs], 'v', color='black',
                        markersize=3, alpha=0.3, zorder=5)

        # Compute and plot median trajectory
        all_norm = []
        for r in sub:
            if r['mean_rate_hz'] > 0:
                all_norm.append(r['seg_rates'] / r['mean_rate_hz'])
        if all_norm:
            median_traj = np.median(np.array(all_norm), axis=0)
            seg_frac_common = np.linspace(0.05, 0.95, N_TIME_SEGMENTS)
            ax.plot(seg_frac_common, median_traj, '-', color='black',
                    lw=2.5, label='Median', zorder=10)

        ax.axhline(1.0, color='gray', ls='--', lw=0.8, alpha=0.5)
        ax.set_xlabel('Fraction of trial')
        ax.set_title(f'{ct}\n(n={len(sub)})', fontsize=9)
        if ax == axes[0]:
            ax.set_ylabel('Rate / mean rate')

    # Add legend for reward marker
    axes[-1].plot([], [], 'v', color='black', markersize=5,
                  label='Reward segment')
    axes[-1].legend(fontsize=7, loc='upper right')

    fig.suptitle('Firing Rate Trajectories Across Trial\n'
                 '(each line = one neuron, normalized to its mean rate)',
                 fontsize=11, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'fig2_rate_trajectories.png'),
                dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved fig2_rate_trajectories.png")


def plot_fig3_cumulative_spikes(results, trial_data, fig_dir):
    """Fig 3: Cumulative spike count vs time, gallery of examples."""
    # Select examples: 2 stationary, 2 non-stationary per cell type (if avail)
    examples = []
    for ct in CELL_ORDER:
        sub_stat = [r for r in results
                    if r['cell_type'] == ct and r['stationary_full'] == True]
        sub_nonstat = [r for r in results
                       if r['cell_type'] == ct and r['stationary_full'] == False]
        # Pick up to 2 of each
        for lst, label in [(sub_stat, 'stationary'), (sub_nonstat, 'non-stationary')]:
            np.random.seed(42)
            if len(lst) > 2:
                picks = np.random.choice(len(lst), 2, replace=False)
                for p in picks:
                    examples.append((lst[p], label))
            else:
                for item in lst:
                    examples.append((item, label))

    if not examples:
        print("  No examples for cumulative spike plot.")
        return

    n_examples = len(examples)
    n_cols = min(4, n_examples)
    n_rows = int(np.ceil(n_examples / n_cols))

    fig, axes = plt.subplots(n_rows * 2, n_cols,
                             figsize=(4 * n_cols, 3 * n_rows * 2))
    # Ensure axes is always 2D
    if n_rows * 2 == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows * 2 == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]
    axes = np.atleast_2d(axes)

    for idx, (r, stat_label) in enumerate(examples):
        row_pair = (idx // n_cols) * 2
        col = idx % n_cols

        # Find spike times from trial_data
        td_match = [td for td in trial_data
                    if td['neuron_id'] == r['neuron_id']
                    and td['trial'] == r['trial']]
        if not td_match:
            continue
        td = td_match[0]
        spk = td['spike_times']

        # Cumulative spike count
        try:
            ax_cum = axes[row_pair, col]
        except IndexError:
            continue
        cum_count = np.arange(1, len(spk) + 1)
        ax_cum.plot(spk, cum_count, 'k-', lw=0.5)

        # Linear fit
        slope, intercept = np.polyfit(spk, cum_count, 1)
        fit_line = slope * spk + intercept
        ax_cum.plot(spk, fit_line, 'r--', lw=1, alpha=0.7)

        # Shade search and eating phases
        t_rew = r['t_reward']
        if t_rew is not None:
            ax_cum.axvspan(0, t_rew, color='gold', alpha=0.15, label='Search')
            ax_cum.axvspan(t_rew, t_rew + EATING_DURATION,
                           color='orange', alpha=0.15, label='Eating')
            ax_cum.axvline(t_rew, color='darkorange', ls='-', lw=1, alpha=0.7)
            ax_cum.axvline(t_rew + EATING_DURATION, color='darkorange',
                           ls=':', lw=1, alpha=0.5)

        ct = r['cell_type']
        color = CELL_COLORS.get(ct, 'gray')
        ax_cum.set_title(f'{ct}\n{stat_label}', fontsize=8, color=color,
                         fontweight='bold')
        ax_cum.set_ylabel('Spike count', fontsize=7)
        ax_cum.tick_params(labelsize=7)

        # Residuals
        try:
            ax_res = axes[row_pair + 1, col]
        except IndexError:
            continue
        residuals = cum_count - fit_line
        ax_res.plot(spk, residuals, 'k-', lw=0.5)
        ax_res.axhline(0, color='red', ls='--', lw=0.8, alpha=0.5)
        if t_rew is not None:
            ax_res.axvspan(0, t_rew, color='gold', alpha=0.15)
            ax_res.axvspan(t_rew, t_rew + EATING_DURATION,
                           color='orange', alpha=0.15)
            ax_res.axvline(t_rew, color='darkorange', ls='-', lw=1, alpha=0.7)
            ax_res.axvline(t_rew + EATING_DURATION, color='darkorange',
                           ls=':', lw=1, alpha=0.5)
        ax_res.set_xlabel('Time (s)', fontsize=7)
        ax_res.set_ylabel('Residual', fontsize=7)
        ax_res.tick_params(labelsize=7)

    # Turn off unused axes
    for idx in range(n_examples, n_rows * n_cols):
        row_pair = (idx // n_cols) * 2
        col = idx % n_cols
        try:
            axes[row_pair, col].axis('off')
            axes[row_pair + 1, col].axis('off')
        except IndexError:
            pass

    # Legend
    legend_elements = [
        Patch(facecolor='gold', alpha=0.3, label='Search phase'),
        Patch(facecolor='orange', alpha=0.3, label='Eating phase'),
    ]
    fig.legend(handles=legend_elements, loc='lower center',
               ncol=2, fontsize=8, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle('Cumulative Spike Count vs Time\n'
                 '(top: cumulative + linear fit, bottom: residuals)',
                 fontsize=11, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'fig3_cumulative_spikes.png'),
                dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved fig3_cumulative_spikes.png")


def plot_fig4_rate_heatmap(results, fig_dir):
    """Fig 4: Segment-by-segment rate heatmap, all neurons as rows."""
    # Sort by cell type then mean rate
    sorted_results = sorted(results,
                            key=lambda r: (CELL_ORDER.index(r['cell_type'])
                                           if r['cell_type'] in CELL_ORDER
                                           else 99,
                                           r['mean_rate_hz']))

    n_neurons = len(sorted_results)
    rate_matrix = np.full((n_neurons, N_TIME_SEGMENTS), np.nan)
    reward_col = np.full(n_neurons, np.nan)
    cell_type_labels = []

    for i, r in enumerate(sorted_results):
        mean_rate = r['mean_rate_hz']
        if mean_rate > 0:
            rate_matrix[i, :] = r['seg_rates'] / mean_rate
        if r['reward_segment_10'] is not None:
            reward_col[i] = r['reward_segment_10']
        cell_type_labels.append(r['cell_type'])

    fig, axes = plt.subplots(1, 2, figsize=(10, max(6, n_neurons * 0.04)),
                             gridspec_kw={'width_ratios': [20, 1]},
                             sharey=True)

    # Main heatmap
    ax = axes[0]
    vmin, vmax = 0.3, 1.7
    im = ax.imshow(rate_matrix, aspect='auto', cmap='RdBu_r',
                   vmin=vmin, vmax=vmax, interpolation='none')
    ax.set_xlabel('Time segment (1-10)')
    ax.set_ylabel('Neuron (sorted by cell type, then rate)')
    ax.set_title('Firing Rate Heatmap\n(normalized to each neuron\'s mean)')
    ax.set_xticks(np.arange(N_TIME_SEGMENTS))
    ax.set_xticklabels(np.arange(1, N_TIME_SEGMENTS + 1))
    plt.colorbar(im, ax=ax, shrink=0.5, label='Rate / mean')

    # Cell type boundary lines
    prev_ct = None
    for i, ct in enumerate(cell_type_labels):
        if ct != prev_ct and prev_ct is not None:
            ax.axhline(i - 0.5, color='white', lw=1.5)
        prev_ct = ct

    # Add cell type labels on the right side
    ct_positions = {}
    for i, ct in enumerate(cell_type_labels):
        if ct not in ct_positions:
            ct_positions[ct] = []
        ct_positions[ct].append(i)
    for ct, positions in ct_positions.items():
        mid = np.mean(positions)
        ax.text(N_TIME_SEGMENTS + 0.3, mid, ct.replace(' ', '\n'),
                fontsize=6, va='center', ha='left',
                color=CELL_COLORS.get(ct, 'gray'), fontweight='bold',
                clip_on=False)

    # Reward segment indicator
    ax_rew = axes[1]
    reward_img = np.full((n_neurons, 1), np.nan)
    for i in range(n_neurons):
        if not np.isnan(reward_col[i]):
            reward_img[i, 0] = reward_col[i]
    ax_rew.imshow(reward_img, aspect='auto', cmap='YlOrRd',
                  vmin=0, vmax=N_TIME_SEGMENTS - 1, interpolation='none')
    ax_rew.set_xlabel('Rew\nseg', fontsize=7)
    ax_rew.set_xticks([])
    ax_rew.set_title('Rew', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'fig4_rate_heatmap.png'),
                dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved fig4_rate_heatmap.png")


def plot_fig5_leave_one_out(results, fig_dir):
    """Fig 5: Leave-one-out recovery analysis."""
    non_stat = [r for r in results if r['stationary_full'] == False]
    if not non_stat:
        print("  All neurons are stationary — no leave-one-out needed.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: overall which segment recovers stationarity
    ax = axes[0]
    recovery_counts = np.zeros(N_STAT_SEGMENTS)
    for r in non_stat:
        for s in range(N_STAT_SEGMENTS):
            if r['loo_recovers'][s]:
                recovery_counts[s] += 1

    n_any_recovery = sum(1 for r in non_stat if any(r['loo_recovers']))
    bars = ax.bar(np.arange(1, N_STAT_SEGMENTS + 1), recovery_counts,
                  color='steelblue', alpha=0.7, edgecolor='none')
    ax.set_xlabel('Segment dropped')
    ax.set_ylabel('# neurons recovering stationarity')
    ax.set_title(f'Leave-One-Out Recovery\n'
                 f'({n_any_recovery}/{len(non_stat)} non-stationary neurons '
                 f'recover by dropping 1 segment)')
    ax.set_xticks(np.arange(1, N_STAT_SEGMENTS + 1))

    # Annotate: does the recovery segment tend to match the reward segment?
    reward_match_counts = np.zeros(N_STAT_SEGMENTS)
    for r in non_stat:
        for s in range(N_STAT_SEGMENTS):
            if r['loo_recovers'][s] and r['reward_segment_4'] == s:
                reward_match_counts[s] += 1
    for s in range(N_STAT_SEGMENTS):
        if recovery_counts[s] > 0:
            ax.text(s + 1, recovery_counts[s] + 0.5,
                    f'({int(reward_match_counts[s])} rew)',
                    ha='center', fontsize=7, color='darkorange')

    # Right: per cell type
    ax = axes[1]
    ct_recovery = {}
    for ct in CELL_ORDER:
        sub = [r for r in non_stat if r['cell_type'] == ct]
        if not sub:
            continue
        counts = np.zeros(N_STAT_SEGMENTS)
        for r in sub:
            for s in range(N_STAT_SEGMENTS):
                if r['loo_recovers'][s]:
                    counts[s] += 1
        ct_recovery[ct] = (counts, len(sub))

    if ct_recovery:
        n_ct = len(ct_recovery)
        width = 0.8 / n_ct
        for ci, ct in enumerate(CELL_ORDER):
            if ct not in ct_recovery:
                continue
            counts, n_total = ct_recovery[ct]
            x_pos = np.arange(1, N_STAT_SEGMENTS + 1) + (ci - n_ct / 2) * width
            ax.bar(x_pos, counts, width=width,
                   color=CELL_COLORS.get(ct, 'gray'), alpha=0.7,
                   label=f'{ct} (n={n_total})', edgecolor='none')
        ax.set_xlabel('Segment dropped')
        ax.set_ylabel('# neurons recovering')
        ax.set_title('Recovery by Cell Type')
        ax.set_xticks(np.arange(1, N_STAT_SEGMENTS + 1))
        ax.legend(fontsize=7, loc='upper right')

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'fig5_leave_one_out.png'),
                dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved fig5_leave_one_out.png")


def plot_fig6_explore_vs_full(results, fig_dir):
    """Fig 6: Full-trial vs free-exploration stationarity comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: grouped bar chart
    ax = axes[0]
    ct_data = []
    for ct in CELL_ORDER:
        sub_full = [r for r in results
                    if r['cell_type'] == ct and r['stationary_full'] is not None]
        sub_expl = [r for r in results
                    if r['cell_type'] == ct
                    and r['stationary_explore'] is not None
                    and not r['explore_skipped']]
        if not sub_full:
            continue
        n_stat_full = sum(1 for r in sub_full if r['stationary_full'] == True)
        n_stat_expl = sum(1 for r in sub_expl if r['stationary_explore'] == True)
        ct_data.append({
            'ct': ct,
            'frac_full': n_stat_full / len(sub_full),
            'n_full': f'{n_stat_full}/{len(sub_full)}',
            'frac_expl': n_stat_expl / len(sub_expl) if sub_expl else 0,
            'n_expl': f'{n_stat_expl}/{len(sub_expl)}' if sub_expl else '0/0',
        })

    if ct_data:
        x_pos = np.arange(len(ct_data))
        width = 0.35
        bars1 = ax.bar(x_pos - width / 2,
                        [d['frac_full'] for d in ct_data],
                        width, label='Full trial', color='gray', alpha=0.6)
        bars2 = ax.bar(x_pos + width / 2,
                        [d['frac_expl'] for d in ct_data],
                        width, label='Free exploration only', color='seagreen',
                        alpha=0.7)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([d['ct'].replace(' ', '\n') for d in ct_data],
                           fontsize=8)
        for bar, d in zip(bars1, ct_data):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02, d['n_full'],
                    ha='center', fontsize=7)
        for bar, d in zip(bars2, ct_data):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02, d['n_expl'],
                    ha='center', fontsize=7, color='seagreen')
        ax.set_ylabel('Fraction Stationary')
        ax.set_title('Full Trial vs Free Exploration\n'
                     '(post-reward + 60 s eating excluded)')
        ax.set_ylim(0, 1.15)
        ax.legend(fontsize=8)

    # Right: paired scatter showing improvement
    ax = axes[1]
    for ct in CELL_ORDER:
        sub = [r for r in results
               if r['cell_type'] == ct
               and r['p_mean_full'] is not None
               and r['p_mean_explore'] is not None
               and not r['explore_skipped']]
        if not sub:
            continue
        p_full = np.clip([r['p_mean_full'] for r in sub], 1e-20, 1)
        p_expl = np.clip([r['p_mean_explore'] for r in sub], 1e-20, 1)
        ax.scatter(p_full, p_expl, c=CELL_COLORS.get(ct, 'gray'),
                   label=ct, alpha=0.5, s=15, edgecolors='none')
    ax.plot([1e-20, 1], [1e-20, 1], 'k--', lw=0.8, alpha=0.3)
    ax.axvline(0.05, color='red', ls=':', lw=0.8, alpha=0.5)
    ax.axhline(0.05, color='red', ls=':', lw=0.8, alpha=0.5)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('p_mean (full trial)')
    ax.set_ylabel('p_mean (free exploration)')
    ax.set_title('Kruskal-Wallis p-value:\nFull Trial vs Free Exploration')
    ax.legend(fontsize=7, loc='lower right')

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'fig6_explore_vs_full.png'),
                dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved fig6_explore_vs_full.png")


def plot_fig7_deviation_vs_pval(results, fig_dir):
    """Fig 7: Rate change magnitude vs stationarity p-values."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    valid = [r for r in results
             if r['p_mean_full'] is not None
             and not np.isnan(r['max_dev_pct'])]

    # Left: max deviation vs p_mean
    ax = axes[0]
    for ct in CELL_ORDER:
        sub = [r for r in valid if r['cell_type'] == ct]
        if not sub:
            continue
        devs = [r['max_dev_pct'] for r in sub]
        p_means = np.clip([r['p_mean_full'] for r in sub], 1e-20, 1)
        # Color by pre- vs post-reward deviation
        for r, dev, pm in zip(sub, devs, p_means):
            marker = 'v' if r['max_dev_is_pre_reward'] else 'o'
            ax.scatter(dev, pm, c=CELL_COLORS.get(ct, 'gray'),
                       marker=marker, alpha=0.5, s=15, edgecolors='none')
    # Dummy entries for legend
    ax.scatter([], [], c='gray', marker='v', label='Max dev in pre-reward')
    ax.scatter([], [], c='gray', marker='o', label='Max dev in post-reward')
    for ct in CELL_ORDER:
        if any(r['cell_type'] == ct for r in valid):
            ax.scatter([], [], c=CELL_COLORS.get(ct, 'gray'), marker='s',
                       label=ct, s=30)
    ax.axhline(0.05, color='red', ls='--', lw=0.8, alpha=0.5)
    ax.set_yscale('log')
    ax.set_xlabel('Max segment rate deviation from mean (%)')
    ax.set_ylabel('p_mean (Kruskal-Wallis)')
    ax.set_title('Rate Deviation vs Stationarity p-value')
    ax.legend(fontsize=6, loc='upper right')

    # Right: same for p_var
    ax = axes[1]
    for ct in CELL_ORDER:
        sub = [r for r in valid if r['cell_type'] == ct]
        if not sub:
            continue
        devs = [r['max_dev_pct'] for r in sub]
        p_vars = np.clip([r['p_var_full'] for r in sub], 1e-20, 1)
        ax.scatter(devs, p_vars, c=CELL_COLORS.get(ct, 'gray'),
                   alpha=0.5, s=15, edgecolors='none', label=ct)
    ax.axhline(0.05, color='red', ls='--', lw=0.8, alpha=0.5)
    ax.set_yscale('log')
    ax.set_xlabel('Max segment rate deviation from mean (%)')
    ax.set_ylabel('p_var (Levene\'s)')
    ax.set_title('Rate Deviation vs Variance Stationarity')
    ax.legend(fontsize=7, loc='upper right')

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'fig7_deviation_vs_pval.png'),
                dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved fig7_deviation_vs_pval.png")


# %% ============================================================
# CONSOLE SUMMARY
# ===============================================================

def print_summary(results):
    """Print comprehensive summary statistics to console."""
    valid = [r for r in results if r['stationary_full'] is not None]
    n_total = len(valid)
    n_stat = sum(1 for r in valid if r['stationary_full'] == True)

    print("\n" + "=" * 65)
    print("STATIONARITY DIAGNOSTICS SUMMARY")
    print("=" * 65)

    print(f"\n--- Overall stationarity (full trial, {N_STAT_SEGMENTS} segments) ---")
    print(f"  Stationary: {n_stat}/{n_total} ({100*n_stat/n_total:.1f}%)")
    n_fail_mean = sum(1 for r in valid if r['p_mean_full'] <= 0.05)
    n_fail_var = sum(1 for r in valid if r['p_var_full'] <= 0.05)
    n_fail_both = sum(1 for r in valid
                      if r['p_mean_full'] <= 0.05 and r['p_var_full'] <= 0.05)
    print(f"  Fail mean only:  {n_fail_mean - n_fail_both}")
    print(f"  Fail var only:   {n_fail_var - n_fail_both}")
    print(f"  Fail both:       {n_fail_both}")

    print(f"\n--- Per cell type ---")
    print(f"  {'Cell Type':32s} {'Stat':>6s} {'Total':>6s} {'%':>7s}")
    print(f"  {'-'*32} {'-'*6} {'-'*6} {'-'*7}")
    for ct in CELL_ORDER:
        sub = [r for r in valid if r['cell_type'] == ct]
        if not sub:
            continue
        n_s = sum(1 for r in sub if r['stationary_full'] == True)
        pct = 100 * n_s / len(sub) if sub else 0
        print(f"  {ct:32s} {n_s:6d} {len(sub):6d} {pct:6.1f}%")

    # Reward time statistics
    with_reward = [r for r in valid if r['t_reward'] is not None]
    if with_reward:
        t_rews = [r['t_reward'] for r in with_reward]
        frac_rews = [r['t_reward'] / r['duration_s'] for r in with_reward]
        print(f"\n--- Reward timing ---")
        print(f"  Neurons with valid t_reward: {len(with_reward)}/{n_total}")
        print(f"  Median t_reward: {np.median(t_rews):.0f} s "
              f"({np.median(frac_rews)*100:.1f}% of trial)")
        print(f"  Range: {np.min(t_rews):.0f} – {np.max(t_rews):.0f} s")

    # Leave-one-out recovery
    non_stat = [r for r in valid if r['stationary_full'] == False]
    if non_stat:
        n_any_loo = sum(1 for r in non_stat if any(r['loo_recovers']))
        print(f"\n--- Leave-one-out recovery (non-stationary neurons) ---")
        print(f"  Recoverable by dropping 1 of {N_STAT_SEGMENTS} segments: "
              f"{n_any_loo}/{len(non_stat)} "
              f"({100*n_any_loo/len(non_stat):.1f}%)")

        recovery_by_seg = np.zeros(N_STAT_SEGMENTS)
        for r in non_stat:
            for s in range(N_STAT_SEGMENTS):
                if r['loo_recovers'][s]:
                    recovery_by_seg[s] += 1
        for s in range(N_STAT_SEGMENTS):
            print(f"    Drop seg {s+1}: {int(recovery_by_seg[s])} neurons recover")

        # Does the recovered segment correlate with reward?
        n_rew_match = 0
        n_rew_checked = 0
        for r in non_stat:
            if r['reward_segment_4'] is not None:
                for s in range(N_STAT_SEGMENTS):
                    if r['loo_recovers'][s]:
                        n_rew_checked += 1
                        if s == r['reward_segment_4']:
                            n_rew_match += 1
        if n_rew_checked > 0:
            print(f"  Recovery segment matches reward segment: "
                  f"{n_rew_match}/{n_rew_checked}")

        # Per cell type
        print(f"\n  {'Cell Type':32s} {'Recoverable':>12s} {'Non-stat':>10s}")
        print(f"  {'-'*32} {'-'*12} {'-'*10}")
        for ct in CELL_ORDER:
            sub = [r for r in non_stat if r['cell_type'] == ct]
            if not sub:
                continue
            n_rec = sum(1 for r in sub if any(r['loo_recovers']))
            print(f"  {ct:32s} {n_rec:12d} {len(sub):10d}")

    # Free exploration recovery
    expl_tested = [r for r in valid
                   if r['stationary_explore'] is not None
                   and not r['explore_skipped']]
    expl_skipped = [r for r in valid if r['explore_skipped']]
    if expl_tested:
        n_stat_expl = sum(1 for r in expl_tested
                          if r['stationary_explore'] == True)
        n_stat_full_of_tested = sum(1 for r in expl_tested
                                     if r['stationary_full'] == True)
        # Count "recovered": was non-stationary (full), now stationary (explore)
        n_recovered = sum(1 for r in expl_tested
                          if r['stationary_full'] == False
                          and r['stationary_explore'] == True)
        n_was_nonstat = sum(1 for r in expl_tested
                            if r['stationary_full'] == False)

        print(f"\n--- Free-exploration stationarity "
              f"(post reward + {EATING_DURATION:.0f}s eating) ---")
        print(f"  Tested: {len(expl_tested)}  "
              f"(skipped {len(expl_skipped)} with <{MIN_ISI_COUNT} ISIs)")
        print(f"  Stationary (full trial):      "
              f"{n_stat_full_of_tested}/{len(expl_tested)} "
              f"({100*n_stat_full_of_tested/len(expl_tested):.1f}%)")
        print(f"  Stationary (explore only):     "
              f"{n_stat_expl}/{len(expl_tested)} "
              f"({100*n_stat_expl/len(expl_tested):.1f}%)")
        if n_was_nonstat > 0:
            print(f"  Recovered (non-stat -> stat):  "
                  f"{n_recovered}/{n_was_nonstat} "
                  f"({100*n_recovered/n_was_nonstat:.1f}%)")
        else:
            print(f"  Recovered: N/A (all were stationary on full trial)")

        print(f"\n  {'Cell Type':32s} {'Full':>10s} {'Explore':>10s} "
              f"{'Recovered':>12s}")
        print(f"  {'-'*32} {'-'*10} {'-'*10} {'-'*12}")
        for ct in CELL_ORDER:
            sub = [r for r in expl_tested if r['cell_type'] == ct]
            if not sub:
                continue
            nf = sum(1 for r in sub if r['stationary_full'] == True)
            ne = sum(1 for r in sub if r['stationary_explore'] == True)
            nr = sum(1 for r in sub
                     if r['stationary_full'] == False
                     and r['stationary_explore'] == True)
            n_ns = sum(1 for r in sub if r['stationary_full'] == False)
            print(f"  {ct:32s} {nf:>4d}/{len(sub):<4d} {ne:>4d}/{len(sub):<4d} "
                  f"{nr:>5d}/{n_ns:<5d}")


# %% ============================================================
# MAIN EXECUTION
# ===============================================================

if __name__ == '__main__':
    print("=" * 65)
    print("STATIONARITY DIAGNOSTICS")
    print("=" * 65)

    print("\n--- Discovering trials ---")
    recordings = discover_trials(EXPERIMENTS)

    print("\n--- Loading and filtering trials (>20 min) ---")
    trial_data = load_and_filter_trials(recordings)

    if not trial_data:
        print("No qualifying trials found. Exiting.")
        raise SystemExit

    print(f"\n--- Running stationarity analyses ---")
    results = run_all_analyses(trial_data)

    # Save results to CSV (excluding arrays)
    csv_rows = []
    for r in results:
        csv_row = {k: v for k, v in r.items()
                   if k not in ('seg_centers', 'seg_rates', 'seg_edges',
                                'seg_means_full', 'seg_vars_full',
                                'loo_recovers')}
        # Flatten loo_recovers into separate columns
        for s in range(N_STAT_SEGMENTS):
            csv_row[f'loo_drop_seg{s+1}_recovers'] = r['loo_recovers'][s]
        csv_rows.append(csv_row)
    df = pd.DataFrame(csv_rows)
    csv_path = os.path.join(FIG_DIR, 'stationarity_diagnostics.csv')
    df.to_csv(csv_path, index=False)
    print(f"  Results saved to {csv_path}")

    # Print summary
    print_summary(results)

    # Generate figures
    print("\n--- Generating figures ---")

    print("  Figure 1: p-value scatter...")
    plot_fig1_pvalue_scatter(results, FIG_DIR)

    print("  Figure 2: rate trajectories...")
    plot_fig2_rate_trajectories(results, FIG_DIR)

    print("  Figure 3: cumulative spike gallery...")
    plot_fig3_cumulative_spikes(results, trial_data, FIG_DIR)

    print("  Figure 4: rate heatmap...")
    plot_fig4_rate_heatmap(results, FIG_DIR)

    print("  Figure 5: leave-one-out recovery...")
    plot_fig5_leave_one_out(results, FIG_DIR)

    print("  Figure 6: full trial vs free exploration...")
    plot_fig6_explore_vs_full(results, FIG_DIR)

    print("  Figure 7: deviation vs p-values...")
    plot_fig7_deviation_vs_pval(results, FIG_DIR)

    print(f"\n--- Done! All outputs saved to {FIG_DIR} ---")