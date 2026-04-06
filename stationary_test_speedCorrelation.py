# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 17:02:35 2026

@author: Kelly
"""

# -*- coding: utf-8 -*-
"""
Stationarity Follow-Up: ISI–Speed and ISI–Position Correlations

For each neuron in long (~30 min) trials, this script:
  1. Plots ISI vs time with mouse speed overlay (example gallery)
  2. Correlates log10(ISI) with interpolated mouse speed at each spike
  3. Correlates log10(ISI) with radial position (distance from arena center)
  4. Computes speed-binned firing rate curves

Goal: determine whether the non-stationarity observed in the previous
diagnostics is explained by behavioral state (locomotion) and/or spatial
position (arena boundary effects).

Figures saved to: figs/stationarity_isi_speed/
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
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

EXCLUDED_PREFIXES = ()      # use all trial types

N_SPEED_BINS = 6            # bins for speed-binned firing rate curves
N_GALLERY    = 4            # example neurons per cell type in gallery

CELL_COLORS = {
    'Narrow Interneuron':         '#0050a0',
    'Wide Interneuron':           '#07CCC3',
    'Granule Cell':               '#c00021',
    'Mossy Cell':                 '#358940',
    'Bursty Narrow Interneuron':  '#87147A',
}
CELL_ORDER = ['Narrow Interneuron', 'Wide Interneuron',
              'Granule Cell', 'Mossy Cell', 'Bursty Narrow Interneuron']

FIG_DIR = os.path.join(cg.ROOT_DIR, 'figs', 'stationarity_isi_speed')
os.makedirs(FIG_DIR, exist_ok=True)


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
    """Extract reward time in seconds relative to trial start."""
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
    Load trials >= MIN_TRIAL_DURATION. For each qualifying neuron, store
    spike times, ISIs, velocity, r_center, time_ttl (all relative to
    trial start), and cell labels.
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

        # Extract behavioral data (relative to trial start)
        time_rel = edata.time_ttl - t_start
        velocity = np.array(edata.velocity, dtype=np.float64)
        r_center = np.array(edata.r_center, dtype=np.float64)  # (n, 2)

        for ni in range(n_neurons):
            if labels[ni] is None:
                continue

            spk = np.atleast_1d(edata.t_spikeTrains[ni])
            if spk.size == 0:
                continue

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
                'time_rel': time_rel,
                'velocity': velocity,
                'r_center': r_center,
            })

    print(f"\nQualifying neurons (trials>={MIN_TRIAL_DURATION}s, "
          f">={MIN_ISI_COUNT} ISIs, >={MIN_RATE_HZ}Hz): {len(trial_data)}")
    ct_counts = {}
    for td in trial_data:
        ct = td['cell_type']
        ct_counts[ct] = ct_counts.get(ct, 0) + 1
    for ct, n in sorted(ct_counts.items()):
        print(f"  {ct}: {n}")
    return trial_data


# %% ============================================================
# ANALYSIS FUNCTIONS
# ===============================================================

def interpolate_at_spikes(time_rel, signal, spike_times):
    """
    Interpolate a behavioral signal (velocity or position) at spike times.
    Returns array of same length as spike_times. NaN where interpolation
    is out of range or source is NaN.
    """
    # Remove NaN from source for interpolation
    valid = ~np.isnan(signal)
    if valid.sum() < 2:
        return np.full(len(spike_times), np.nan)
    return np.interp(spike_times, time_rel[valid], signal[valid],
                     left=np.nan, right=np.nan)


def compute_radial_distance(r_center, time_rel, spike_times):
    """
    Compute radial distance from arena center (0,0) at each spike time.
    """
    rx = interpolate_at_spikes(time_rel, r_center[:, 0], spike_times)
    ry = interpolate_at_spikes(time_rel, r_center[:, 1], spike_times)
    return np.sqrt(rx**2 + ry**2)


def run_correlations(trial_data):
    """
    For each neuron, compute:
      - Spearman correlation of log10(ISI) vs speed at spike time
      - Spearman correlation of log10(ISI) vs radial distance at spike time
    """
    results = []

    for i, td in enumerate(trial_data):
        if (i + 1) % 50 == 0:
            print(f"  Analyzing {i+1}/{len(trial_data)}...")

        spk = td['spike_times']
        isis = td['isis']
        log_isi = np.log10(isis)

        # Midpoint times for each ISI (between consecutive spikes)
        isi_times = 0.5 * (spk[:-1] + spk[1:])

        # Interpolate velocity at ISI midpoints
        speed_at_isi = interpolate_at_spikes(
            td['time_rel'], td['velocity'], isi_times)

        # Interpolate radial distance at ISI midpoints
        radial_at_isi = compute_radial_distance(
            td['r_center'], td['time_rel'], isi_times)

        # --- Speed correlation ---
        valid_speed = ~np.isnan(speed_at_isi) & ~np.isnan(log_isi)
        if valid_speed.sum() > 20:
            rho_speed, p_speed = stats.spearmanr(
                log_isi[valid_speed], speed_at_isi[valid_speed])
        else:
            rho_speed, p_speed = np.nan, np.nan

        # --- Radial distance correlation ---
        valid_rad = ~np.isnan(radial_at_isi) & ~np.isnan(log_isi)
        if valid_rad.sum() > 20:
            rho_radial, p_radial = stats.spearmanr(
                log_isi[valid_rad], radial_at_isi[valid_rad])
        else:
            rho_radial, p_radial = np.nan, np.nan

        results.append({
            'neuron_id': td['neuron_id'],
            'exp': td['exp'], 'mouse': td['mouse'],
            'day': td['day'], 'trial': td['trial'],
            'neuron_idx': td['neuron_idx'],
            'cell_type': td['cell_type'],
            'n_spikes': td['n_spikes'], 'n_isis': td['n_isis'],
            'mean_rate_hz': td['mean_rate'],
            'duration_s': td['duration'],
            't_reward': td['t_reward'],
            'rho_speed': float(rho_speed),
            'p_speed': float(p_speed),
            'rho_radial': float(rho_radial),
            'p_radial': float(p_radial),
            'n_valid_speed': int(valid_speed.sum()),
            'n_valid_radial': int(valid_rad.sum()),
            # Store arrays for plotting (not saved to CSV)
            '_isi_times': isi_times,
            '_log_isi': log_isi,
            '_speed_at_isi': speed_at_isi,
            '_radial_at_isi': radial_at_isi,
        })

    return results


# %% ============================================================
# PLOTTING FUNCTIONS
# ===============================================================

def plot_fig1_gallery(results, trial_data, fig_dir):
    """
    Fig 1: ISI vs time with speed overlay.
    Select N_GALLERY examples per cell type, chosen to span rho_speed range.
    """
    for ct in CELL_ORDER:
        sub = [r for r in results
               if r['cell_type'] == ct and not np.isnan(r['rho_speed'])]
        if len(sub) < 1:
            continue

        # Sort by rho_speed and pick evenly spaced examples
        sub_sorted = sorted(sub, key=lambda r: r['rho_speed'])
        n_pick = min(N_GALLERY, len(sub_sorted))
        if n_pick <= 1:
            picks = sub_sorted
        else:
            indices = np.linspace(0, len(sub_sorted) - 1, n_pick, dtype=int)
            picks = [sub_sorted[idx] for idx in indices]

        fig, axes = plt.subplots(n_pick, 1, figsize=(12, 3 * n_pick),
                                 squeeze=False)

        for row, r in enumerate(picks):
            ax = axes[row, 0]

            isi_times = r['_isi_times']
            log_isi = r['_log_isi']

            # Plot log10(ISI)
            ax.scatter(isi_times, log_isi, s=0.5, alpha=0.3,
                       color=CELL_COLORS.get(ct, 'gray'), rasterized=True)
            ax.set_ylabel('log₁₀(ISI) [s]', fontsize=9)

            # Overlay speed on twin axis
            ax2 = ax.twinx()
            # Find matching trial_data for the full velocity trace
            td_match = [td for td in trial_data
                        if td['neuron_id'] == r['neuron_id']
                        and td['trial'] == r['trial']]
            if td_match:
                td = td_match[0]
                ax2.plot(td['time_rel'], td['velocity'],
                         color='black', alpha=0.4, lw=0.5)
            ax2.set_ylabel('Speed (cm/s)', fontsize=9, color='black')

            # Reward line
            if r['t_reward'] is not None:
                ax.axvline(r['t_reward'], color='darkorange', ls='--',
                           lw=1.5, alpha=0.8, label='Reward')

            ax.set_title(f"{r['neuron_id']}  |  ρ_speed = {r['rho_speed']:.3f} "
                         f"(p = {r['p_speed']:.2e})  |  "
                         f"rate = {r['mean_rate_hz']:.1f} Hz",
                         fontsize=8)
            if row == n_pick - 1:
                ax.set_xlabel('Time (s)')

        fig.suptitle(f'{ct} — ISI vs Time with Speed Overlay\n'
                     f'(selected across ρ_speed range)',
                     fontsize=11, y=1.01)
        plt.tight_layout()
        safe_ct = ct.replace(' ', '_').replace('/', '_')
        plt.savefig(os.path.join(fig_dir, f'fig1_gallery_{safe_ct}.png'),
                    dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  Saved fig1_gallery_{safe_ct}.png")


def plot_fig2_speed_correlation_summary(results, fig_dir):
    """
    Fig 2: Violin/box plot of Spearman rho (ISI vs speed) by cell type.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: violin plot
    ax = axes[0]
    plot_data = []
    plot_labels = []
    plot_colors = []
    stats_text = []

    for ct in CELL_ORDER:
        rhos = [r['rho_speed'] for r in results
                if r['cell_type'] == ct and not np.isnan(r['rho_speed'])]
        if len(rhos) < 3:
            continue
        plot_data.append(rhos)
        plot_labels.append(ct)
        plot_colors.append(CELL_COLORS.get(ct, 'gray'))

        # One-sample Wilcoxon test vs zero
        stat_w, p_w = stats.wilcoxon(rhos)
        median_rho = np.median(rhos)
        stats_text.append(f"{ct}:\n  median ρ = {median_rho:.3f}, "
                          f"Wilcoxon p = {p_w:.2e}, n = {len(rhos)}")

    if plot_data:
        parts = ax.violinplot(plot_data, showmedians=True, showextrema=False)
        for pc, color in zip(parts['bodies'], plot_colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.6)
        parts['cmedians'].set_color('black')

        # Overlay individual points (jittered)
        for i, (data, color) in enumerate(zip(plot_data, plot_colors)):
            jitter = np.random.default_rng(42).normal(0, 0.05, len(data))
            ax.scatter(np.full(len(data), i + 1) + jitter, data,
                       s=5, alpha=0.3, color=color, edgecolors='none')

        ax.axhline(0, color='gray', ls='--', lw=0.8, alpha=0.5)
        ax.set_xticks(range(1, len(plot_labels) + 1))
        ax.set_xticklabels([l.replace(' ', '\n') for l in plot_labels],
                           fontsize=8)
        ax.set_ylabel('Spearman ρ (log₁₀ ISI vs Speed)')
        ax.set_title('ISI–Speed Correlation by Cell Type')

    # Right: stats summary text
    ax2 = axes[1]
    ax2.axis('off')
    summary = "One-sample Wilcoxon signed-rank test (H₀: ρ = 0)\n\n"
    summary += "\n\n".join(stats_text)

    # Kruskal-Wallis across cell types
    if len(plot_data) >= 2:
        h_stat, p_kw = stats.kruskal(*plot_data)
        summary += f"\n\nKruskal-Wallis across types: H = {h_stat:.2f}, p = {p_kw:.2e}"

    ax2.text(0.05, 0.95, summary, transform=ax2.transAxes,
             fontsize=9, va='top', family='monospace')

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'fig2_speed_correlation.png'),
                dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved fig2_speed_correlation.png")


def plot_fig3_radial_correlation_summary(results, fig_dir):
    """
    Fig 3: Violin/box plot of Spearman rho (ISI vs radial distance)
    by cell type.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    ax = axes[0]
    plot_data = []
    plot_labels = []
    plot_colors = []
    stats_text = []

    for ct in CELL_ORDER:
        rhos = [r['rho_radial'] for r in results
                if r['cell_type'] == ct and not np.isnan(r['rho_radial'])]
        if len(rhos) < 3:
            continue
        plot_data.append(rhos)
        plot_labels.append(ct)
        plot_colors.append(CELL_COLORS.get(ct, 'gray'))

        stat_w, p_w = stats.wilcoxon(rhos)
        median_rho = np.median(rhos)
        stats_text.append(f"{ct}:\n  median ρ = {median_rho:.3f}, "
                          f"Wilcoxon p = {p_w:.2e}, n = {len(rhos)}")

    if plot_data:
        parts = ax.violinplot(plot_data, showmedians=True, showextrema=False)
        for pc, color in zip(parts['bodies'], plot_colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.6)
        parts['cmedians'].set_color('black')

        for i, (data, color) in enumerate(zip(plot_data, plot_colors)):
            jitter = np.random.default_rng(42).normal(0, 0.05, len(data))
            ax.scatter(np.full(len(data), i + 1) + jitter, data,
                       s=5, alpha=0.3, color=color, edgecolors='none')

        ax.axhline(0, color='gray', ls='--', lw=0.8, alpha=0.5)
        ax.set_xticks(range(1, len(plot_labels) + 1))
        ax.set_xticklabels([l.replace(' ', '\n') for l in plot_labels],
                           fontsize=8)
        ax.set_ylabel('Spearman ρ (log₁₀ ISI vs Radial Distance)')
        ax.set_title('ISI–Radial Position Correlation by Cell Type')

    ax2 = axes[1]
    ax2.axis('off')
    summary = "One-sample Wilcoxon signed-rank test (H₀: ρ = 0)\n\n"
    summary += "\n\n".join(stats_text)

    if len(plot_data) >= 2:
        h_stat, p_kw = stats.kruskal(*plot_data)
        summary += f"\n\nKruskal-Wallis across types: H = {h_stat:.2f}, p = {p_kw:.2e}"

    ax2.text(0.05, 0.95, summary, transform=ax2.transAxes,
             fontsize=9, va='top', family='monospace')

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'fig3_radial_correlation.png'),
                dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved fig3_radial_correlation.png")


def plot_fig4_speed_binned_rate(results, trial_data, fig_dir):
    """
    Fig 4: Speed-binned firing rate curves by cell type.
    Bin velocity into N_SPEED_BINS quantile bins, compute firing rate in
    each bin for each neuron. Plot mean ± SEM by cell type.
    """
    # Collect all valid speeds to define common bin edges
    all_speeds = []
    for td in trial_data:
        v = td['velocity']
        valid = v[~np.isnan(v)]
        if len(valid) > 0:
            all_speeds.append(valid)
    if not all_speeds:
        print("  No valid velocity data for speed-binned rate plot.")
        return
    all_speeds = np.concatenate(all_speeds)
    bin_edges = np.quantile(all_speeds, np.linspace(0, 1, N_SPEED_BINS + 1))
    # Ensure unique edges
    bin_edges = np.unique(bin_edges)
    n_bins = len(bin_edges) - 1
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # For each neuron, compute time spent and spike count in each speed bin
    ct_rates = {ct: [] for ct in CELL_ORDER}

    for td in trial_data:
        ct = td['cell_type']
        if ct not in ct_rates:
            continue

        time_rel = td['time_rel']
        velocity = td['velocity']
        spk = td['spike_times']

        # Time in each speed bin (using time_ttl sampling intervals)
        dt = np.diff(time_rel)
        vel_mid = velocity[:-1]  # velocity at start of each interval
        valid_dt = ~np.isnan(vel_mid)

        # Spike speeds (interpolated)
        spk_speed = interpolate_at_spikes(time_rel, velocity, spk)
        valid_spk = ~np.isnan(spk_speed)

        rates = np.full(n_bins, np.nan)
        for b in range(n_bins):
            # Time in bin
            in_bin_dt = valid_dt & (vel_mid >= bin_edges[b]) & (vel_mid < bin_edges[b + 1])
            if b == n_bins - 1:  # include right edge for last bin
                in_bin_dt = valid_dt & (vel_mid >= bin_edges[b]) & (vel_mid <= bin_edges[b + 1])
            time_in_bin = np.sum(dt[in_bin_dt])

            # Spikes in bin
            in_bin_spk = valid_spk & (spk_speed >= bin_edges[b]) & (spk_speed < bin_edges[b + 1])
            if b == n_bins - 1:
                in_bin_spk = valid_spk & (spk_speed >= bin_edges[b]) & (spk_speed <= bin_edges[b + 1])
            n_spk_bin = np.sum(in_bin_spk)

            if time_in_bin > 0.5:  # at least 0.5 s occupancy
                rates[b] = n_spk_bin / time_in_bin

        ct_rates[ct].append(rates)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 5.5))

    for ct in CELL_ORDER:
        rate_arr = ct_rates.get(ct, [])
        if len(rate_arr) < 3:
            continue
        rate_matrix = np.array(rate_arr)  # (n_neurons, n_bins)

        # Normalize each neuron to its mean rate across bins for shape comparison
        row_means = np.nanmean(rate_matrix, axis=1, keepdims=True)
        row_means[row_means == 0] = np.nan
        norm_matrix = rate_matrix / row_means

        mean_curve = np.nanmean(norm_matrix, axis=0)
        n_valid = np.sum(~np.isnan(norm_matrix), axis=0)
        sem_curve = np.nanstd(norm_matrix, axis=0) / np.sqrt(np.maximum(n_valid, 1))

        color = CELL_COLORS.get(ct, 'gray')
        ax.plot(bin_centers, mean_curve, '-o', color=color, lw=2,
                markersize=5, label=f'{ct} (n={len(rate_arr)})')
        ax.fill_between(bin_centers, mean_curve - sem_curve,
                        mean_curve + sem_curve, color=color, alpha=0.15)

    ax.axhline(1.0, color='gray', ls='--', lw=0.8, alpha=0.5)
    ax.set_xlabel('Mouse Speed (cm/s)')
    ax.set_ylabel('Normalized Firing Rate (rate / mean)')
    ax.set_title('Speed-Binned Firing Rate by Cell Type\n'
                 '(each neuron normalized to its mean rate)')
    ax.legend(fontsize=8, loc='best')

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'fig4_speed_binned_rate.png'),
                dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved fig4_speed_binned_rate.png")


# %% ============================================================
# CONSOLE SUMMARY
# ===============================================================

def print_summary(results):
    """Print summary statistics."""
    print("\n" + "=" * 65)
    print("ISI–SPEED / ISI–POSITION CORRELATION SUMMARY")
    print("=" * 65)

    valid_speed = [r for r in results if not np.isnan(r['rho_speed'])]
    valid_rad = [r for r in results if not np.isnan(r['rho_radial'])]

    print(f"\nTotal neurons analyzed: {len(results)}")
    print(f"Valid speed correlations: {len(valid_speed)}")
    print(f"Valid radial correlations: {len(valid_rad)}")

    print(f"\n--- ISI–Speed Correlation (Spearman) ---")
    print(f"  {'Cell Type':32s} {'Median ρ':>10s} {'Mean ρ':>10s} "
          f"{'% sig':>8s} {'n':>5s}")
    print(f"  {'-'*32} {'-'*10} {'-'*10} {'-'*8} {'-'*5}")
    for ct in CELL_ORDER:
        sub = [r for r in valid_speed if r['cell_type'] == ct]
        if not sub:
            continue
        rhos = [r['rho_speed'] for r in sub]
        n_sig = sum(1 for r in sub if r['p_speed'] < 0.05)
        print(f"  {ct:32s} {np.median(rhos):10.3f} {np.mean(rhos):10.3f} "
              f"{100*n_sig/len(sub):7.1f}% {len(sub):5d}")

    print(f"\n--- ISI–Radial Distance Correlation (Spearman) ---")
    print(f"  {'Cell Type':32s} {'Median ρ':>10s} {'Mean ρ':>10s} "
          f"{'% sig':>8s} {'n':>5s}")
    print(f"  {'-'*32} {'-'*10} {'-'*10} {'-'*8} {'-'*5}")
    for ct in CELL_ORDER:
        sub = [r for r in valid_rad if r['cell_type'] == ct]
        if not sub:
            continue
        rhos = [r['rho_radial'] for r in sub]
        n_sig = sum(1 for r in sub if r['p_radial'] < 0.05)
        print(f"  {ct:32s} {np.median(rhos):10.3f} {np.mean(rhos):10.3f} "
              f"{100*n_sig/len(sub):7.1f}% {len(sub):5d}")


# %% ============================================================
# MAIN EXECUTION
# ===============================================================

if __name__ == '__main__':
    print("=" * 65)
    print("STATIONARITY FOLLOW-UP: ISI–SPEED & ISI–POSITION CORRELATIONS")
    print("=" * 65)

    print("\n--- Discovering trials ---")
    recordings = discover_trials(EXPERIMENTS)

    print("\n--- Loading and filtering trials ---")
    trial_data = load_and_filter_trials(recordings)

    if not trial_data:
        print("No qualifying trials found. Exiting.")
        raise SystemExit

    print(f"\n--- Computing correlations ---")
    results = run_correlations(trial_data)

    # Save CSV (exclude internal arrays)
    csv_rows = []
    for r in results:
        csv_rows.append({k: v for k, v in r.items() if not k.startswith('_')})
    df = pd.DataFrame(csv_rows)
    csv_path = os.path.join(FIG_DIR, 'isi_speed_radial_correlations.csv')
    df.to_csv(csv_path, index=False)
    print(f"  Results saved to {csv_path}")

    # Print summary
    print_summary(results)

    # Generate figures
    print("\n--- Generating figures ---")

    print("  Figure 1: ISI vs time gallery...")
    plot_fig1_gallery(results, trial_data, FIG_DIR)

    print("  Figure 2: ISI–speed correlation summary...")
    plot_fig2_speed_correlation_summary(results, FIG_DIR)

    print("  Figure 3: ISI–radial correlation summary...")
    plot_fig3_radial_correlation_summary(results, FIG_DIR)

    print("  Figure 4: Speed-binned firing rate curves...")
    plot_fig4_speed_binned_rate(results, trial_data, FIG_DIR)

    print(f"\n--- Done! All outputs saved to {FIG_DIR} ---")