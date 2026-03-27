# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 13:44:08 2026

@author: Kelly
"""

"""
Superburst (SB) analysis following Vandael et al. (2020, Neuron).

Fits 2- vs 3-component Gaussian mixture models on log10(ISI) distributions
to detect three regimes of activity:
  - Intra-burst ISIs (short, ~1-10 ms)
  - Inter-burst / intra-SB ISIs (medium, ~10-100 ms)
  - Inter-event ISIs (long, >100 ms)

If 3 components significantly outperform 2 (LLR bootstrap test),
this indicates superburst structure. Crossover points between
Gaussian components define thresholds for event classification.

Cell types are separated using cellLabels_BITP, with Excitatory
Principal Cells subdivided into Granule Cell / Mossy Cell via
cellLabels_pca.

Analyses:
  1. Per-neuron GMM fitting (2 vs 3 components) with bootstrap LLR
  2. Pooled log10(ISIH) by cell type with GMM overlay
  3. Event classification: single spikes, bursts, superbursts
  4. SB statistics: spikes/SB, duration, fraction of spikes in SBs
  5. Summary comparison across cell types
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy import stats
import glob, os, re, warnings, time
import pandas as pd

import lib_ephys_obj as elib
import config as cg

warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# %% ============================================================
# PARAMETERS
# ===============================================================
EXPERIMENTS = ['2024-02-15', '2024-11-09', '2025-01-21']

MIN_TRIAL_DURATION = 1200   # seconds (20 min) — long trials only
MIN_ISI_COUNT      = 200    # minimum ISIs for GMM fitting
MIN_RATE_HZ        = 0.5    # minimum mean firing rate
N_BOOTSTRAP        = 1000   # bootstrap replications for LLR test
ALPHA              = 0.05   # significance level for LLR test
LOG_ISI_BIN_WIDTH  = 0.05   # log10 units for histogram display

# Cell type colors (standard scheme)
CELL_COLORS = {
    'Narrow Interneuron':         '#0050a0',
    'Wide Interneuron':           '#07CCC3',
    'Granule Cell':               '#c00021',
    'Mossy Cell':                 '#358940',
    'Bursty Narrow Interneuron':  '#87147A',
}

CELL_ORDER = ['Narrow Interneuron', 'Wide Interneuron',
              'Granule Cell', 'Mossy Cell', 'Bursty Narrow Interneuron']

# Excluded trial prefixes — NONE for this analysis (use all trial types)
EXCLUDED_PREFIXES = ()

FIG_DIR = os.path.join(cg.ROOT_DIR, 'figs', 'superburst_analysis')
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
    """
    Get the final cell type label for a neuron.
    Uses cellLabels_BITP, then subdivides Excitatory Principal Cells
    into Granule Cell / Mossy Cell via cellLabels_pca.
    Returns label string, or None if the cell should be excluded.
    """
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
                    return None  # unclassified excitatory -> exclude
            except (IndexError, TypeError):
                return None
        else:
            return None

    return bitp


def load_spike_data(recordings):
    """
    Load all trials, pool spike trains per unique neuron
    (exp, mouse, day, neuron_idx). Only keep trials > MIN_TRIAL_DURATION.

    Returns dict: neuron_key -> {
        'isis': pooled ISI array,
        'cell_type': str,
        'total_duration': float,
        'mean_rate': float,
        ...
    }
    """
    neurons = {}   # key = (exp, mouse, day, neuron_idx)

    for rec_i, (exp, mouse, trial) in enumerate(recordings):
        print(f"  Loading [{rec_i+1}/{len(recordings)}] "
              f"{exp} M{mouse} T{trial}", end='\r')
        edata = elib.EphysTrial.Load(exp, mouse, trial)

        # Trial duration
        t = edata.time_ttl
        if t is None or len(t) < 2:
            continue
        duration = float(t[-1] - t[0])
        if duration < MIN_TRIAL_DURATION:
            continue

        day = str(edata.day)
        n_neurons = len(edata.t_spikeTrains)

        for i in range(n_neurons):
            label = _get_cell_label(edata, i)
            if label is None:
                continue

            spk = np.atleast_1d(edata.t_spikeTrains[i])
            if spk.size == 0:
                continue

            # Only keep spikes within the trial window
            t_start, t_end = float(t[0]), float(t[-1])
            spk = spk[(spk >= t_start) & (spk <= t_end)]

            nkey = (exp, mouse, day, i)
            if nkey not in neurons:
                neurons[nkey] = {
                    'spike_times_list': [],
                    'cell_type': label,
                    'durations': [],
                    'exp': exp,
                    'mouse': mouse,
                    'day': day,
                    'neuron_idx': i,
                }
            neurons[nkey]['spike_times_list'].append(spk)
            neurons[nkey]['durations'].append(duration)

    print()  # clear the \r line

    # Pool ISIs across trials (concatenate per-trial ISIs to avoid
    # inter-trial gaps being treated as ISIs)
    result = {}
    for nkey, info in neurons.items():
        all_isis = []
        for spk in info['spike_times_list']:
            if len(spk) > 1:
                isis = np.diff(spk)
                all_isis.append(isis[isis > 0])

        if len(all_isis) == 0:
            continue
        isis_pooled = np.concatenate(all_isis)
        total_dur = sum(info['durations'])
        total_spikes = sum(len(s) for s in info['spike_times_list'])
        mean_rate = total_spikes / total_dur

        if len(isis_pooled) < MIN_ISI_COUNT or mean_rate < MIN_RATE_HZ:
            continue

        result[nkey] = {
            'isis': isis_pooled,
            'cell_type': info['cell_type'],
            'total_duration': total_dur,
            'mean_rate': mean_rate,
            'n_spikes': total_spikes,
            'exp': info['exp'],
            'mouse': info['mouse'],
            'day': info['day'],
            'neuron_idx': info['neuron_idx'],
            'spike_times_list': info['spike_times_list'],
        }

    # Summary
    ct_counts = {}
    for v in result.values():
        ct = v['cell_type']
        ct_counts[ct] = ct_counts.get(ct, 0) + 1
    print(f"\nLoaded {len(result)} neurons passing filters:")
    for ct in CELL_ORDER:
        if ct in ct_counts:
            print(f"  {ct}: {ct_counts[ct]}")
    print()

    return result


# %% ============================================================
# GAUSSIAN MIXTURE MODEL FITTING
# ===============================================================

def fit_gmm(log_isis, n_components, n_init=10):
    """
    Fit a Gaussian mixture model to log10(ISI) data.
    Returns the fitted model and log-likelihood.
    """
    X = log_isis.reshape(-1, 1)
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type='full',
        n_init=n_init,
        max_iter=300,
        random_state=42
    )
    gmm.fit(X)
    ll = gmm.score(X) * len(X)  # total log-likelihood
    return gmm, ll


def bootstrap_llr_test(log_isis, n_bootstrap=N_BOOTSTRAP, verbose=True):
    """
    Bootstrap log-likelihood ratio test: 2 vs 3 Gaussian components.
    Following Vandael et al. (2020) Fig. 1C methodology.

    Returns:
        llr_observed: float, observed LLR
        p_value: float, proportion of bootstrap LLR >= observed
        gmm2, gmm3: fitted models
        bootstrap_llrs: array of bootstrap LLR values
    """
    # Fit models to original data
    gmm2, ll2 = fit_gmm(log_isis, 2)
    gmm3, ll3 = fit_gmm(log_isis, 3)
    llr_observed = ll3 - ll2

    # Bootstrap: generate data under H0 (2-component model),
    # fit both models, compute LLR distribution
    n = len(log_isis)
    bootstrap_llrs = np.zeros(n_bootstrap)
    t0 = time.time()

    for b in range(n_bootstrap):
        # Progress every 100 iterations
        if verbose and (b + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (b + 1) / elapsed
            remaining = (n_bootstrap - b - 1) / rate
            print(f"    bootstrap {b+1}/{n_bootstrap} "
                  f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s left)",
                  end='\r')

        # Sample from the 2-component model (null hypothesis)
        X_boot = gmm2.sample(n)[0].ravel()

        try:
            _, ll2_boot = fit_gmm(X_boot, 2, n_init=5)
            _, ll3_boot = fit_gmm(X_boot, 3, n_init=5)
            bootstrap_llrs[b] = ll3_boot - ll2_boot
        except Exception:
            bootstrap_llrs[b] = 0.0

    if verbose:
        print(f"    bootstrap {n_bootstrap}/{n_bootstrap} "
              f"done in {time.time()-t0:.0f}s               ")

    p_value = np.mean(bootstrap_llrs >= llr_observed)

    return llr_observed, p_value, gmm2, gmm3, bootstrap_llrs


def gmm_crossover_points(gmm):
    """
    Find crossover points between adjacent Gaussian components.
    These define the thresholds separating ISI regimes.

    Returns sorted list of crossover points in log10(ISI) space.
    """
    means = gmm.means_.ravel()
    order = np.argsort(means)
    means = means[order]
    stds = np.sqrt(gmm.covariances_.ravel()[order])
    weights = gmm.weights_[order]

    crossovers = []
    x_range = np.linspace(means[0] - 3*stds[0],
                          means[-1] + 3*stds[-1], 10000)

    for j in range(len(means) - 1):
        # Weighted PDFs of adjacent components
        pdf_j = weights[j] * stats.norm.pdf(x_range, means[j], stds[j])
        pdf_k = weights[j+1] * stats.norm.pdf(x_range, means[j+1], stds[j+1])

        # Find where they cross
        diff = pdf_j - pdf_k
        sign_changes = np.where(np.diff(np.sign(diff)))[0]

        if len(sign_changes) > 0:
            # Take the crossing closest to the midpoint between means
            midpoint = (means[j] + means[j+1]) / 2
            best = sign_changes[np.argmin(np.abs(x_range[sign_changes] - midpoint))]
            crossovers.append(x_range[best])

    return sorted(crossovers)


# %% ============================================================
# EVENT CLASSIFICATION
# ===============================================================

def classify_events(spike_times_list, burst_thresh_log, sb_thresh_log=None):
    """
    Walk through spike trains and classify events into:
      - Single spikes
      - Bursts (ISI < burst_thresh)
      - Superbursts (bursts separated by ISI < sb_thresh)

    burst_thresh_log and sb_thresh_log are in log10(seconds).
    If sb_thresh_log is None, only burst/single classification is done.

    Returns list of event dicts.
    """
    burst_thresh = 10**burst_thresh_log
    sb_thresh = 10**sb_thresh_log if sb_thresh_log is not None else None

    all_events = []

    for spk in spike_times_list:
        spk = np.sort(spk)
        if len(spk) < 2:
            if len(spk) == 1:
                all_events.append({'type': 'single', 'spikes': 1, 'duration': 0})
            continue

        isis = np.diff(spk)

        # Step 1: group consecutive spikes into bursts (ISI < burst_thresh)
        bursts = []
        current_burst = [0]  # indices into spk
        for k in range(len(isis)):
            if isis[k] < burst_thresh:
                current_burst.append(k + 1)
            else:
                bursts.append(current_burst)
                current_burst = [k + 1]
        bursts.append(current_burst)

        if sb_thresh is not None:
            # Step 2: group consecutive bursts into superbursts
            # Inter-burst intervals = time between last spike of one burst
            # and first spike of next burst
            super_groups = [[0]]  # indices into bursts list
            for b in range(1, len(bursts)):
                last_spike_prev = spk[bursts[b-1][-1]]
                first_spike_curr = spk[bursts[b][0]]
                ibi = first_spike_curr - last_spike_prev
                if ibi < sb_thresh:
                    super_groups[-1].append(b)
                else:
                    super_groups.append([b])

            for sg in super_groups:
                # Collect all spike indices in this super-group
                spike_indices = []
                for b_idx in sg:
                    spike_indices.extend(bursts[b_idx])
                n_spk = len(spike_indices)
                dur = spk[spike_indices[-1]] - spk[spike_indices[0]]
                n_bursts_in_sg = len(sg)

                if n_bursts_in_sg > 1 or (n_bursts_in_sg == 1
                                           and len(bursts[sg[0]]) > 1):
                    if n_bursts_in_sg > 1:
                        all_events.append({
                            'type': 'superburst',
                            'spikes': n_spk,
                            'duration': dur,
                            'n_sub_bursts': n_bursts_in_sg,
                        })
                    else:
                        all_events.append({
                            'type': 'burst',
                            'spikes': n_spk,
                            'duration': dur,
                        })
                else:
                    all_events.append({
                        'type': 'single',
                        'spikes': 1,
                        'duration': 0,
                    })
        else:
            # No superburst classification, just bursts and singles
            for b in bursts:
                n_spk = len(b)
                if n_spk > 1:
                    dur = spk[b[-1]] - spk[b[0]]
                    all_events.append({
                        'type': 'burst',
                        'spikes': n_spk,
                        'duration': dur,
                    })
                else:
                    all_events.append({
                        'type': 'single',
                        'spikes': 1,
                        'duration': 0,
                    })

    return all_events


def compute_event_stats(events, total_spikes):
    """Compute summary statistics from classified events."""
    if not events:
        return {}

    singles = [e for e in events if e['type'] == 'single']
    bursts = [e for e in events if e['type'] == 'burst']
    sbs = [e for e in events if e['type'] == 'superburst']

    n_events = len(events)
    n_spikes_single = sum(e['spikes'] for e in singles)
    n_spikes_burst = sum(e['spikes'] for e in bursts)
    n_spikes_sb = sum(e['spikes'] for e in sbs)

    result = {
        'n_events': n_events,
        'n_singles': len(singles),
        'n_bursts': len(bursts),
        'n_superbursts': len(sbs),
        'frac_events_single': len(singles) / n_events if n_events > 0 else 0,
        'frac_events_burst': len(bursts) / n_events if n_events > 0 else 0,
        'frac_events_sb': len(sbs) / n_events if n_events > 0 else 0,
        'frac_spikes_single': n_spikes_single / total_spikes if total_spikes > 0 else 0,
        'frac_spikes_burst': n_spikes_burst / total_spikes if total_spikes > 0 else 0,
        'frac_spikes_sb': n_spikes_sb / total_spikes if total_spikes > 0 else 0,
    }

    if sbs:
        sb_spikes = [e['spikes'] for e in sbs]
        sb_durs = [e['duration'] for e in sbs]
        result['mean_spikes_per_sb'] = np.mean(sb_spikes)
        result['std_spikes_per_sb'] = np.std(sb_spikes)
        result['mean_sb_duration_ms'] = np.mean(sb_durs) * 1000
        result['std_sb_duration_ms'] = np.std(sb_durs) * 1000

    if bursts:
        b_spikes = [e['spikes'] for e in bursts]
        b_durs = [e['duration'] for e in bursts]
        result['mean_spikes_per_burst'] = np.mean(b_spikes)
        result['mean_burst_duration_ms'] = np.mean(b_durs) * 1000

    return result


# %% ============================================================
# PLOTTING FUNCTIONS
# ===============================================================

def _gmm_pdf(gmm, x):
    """Evaluate weighted PDF of all GMM components at x."""
    X = x.reshape(-1, 1)
    log_prob = gmm.score_samples(X)
    return np.exp(log_prob)


def _gmm_component_pdfs(gmm, x):
    """Evaluate individual weighted component PDFs."""
    means = gmm.means_.ravel()
    stds = np.sqrt(gmm.covariances_.ravel())
    weights = gmm.weights_
    order = np.argsort(means)

    pdfs = []
    for idx in order:
        pdf = weights[idx] * stats.norm.pdf(x, means[idx], stds[idx])
        pdfs.append(pdf)
    return pdfs


def plot_neuron_gmm(neuron_info, gmm2, gmm3, llr, p_val, crossovers_3,
                    bootstrap_llrs, save_path):
    """
    Plot per-neuron GMM analysis:
      - Log10(ISI) histogram with 2-component fit
      - Log10(ISI) histogram with 3-component fit + crossovers
      - Bootstrap LLR histogram
    """
    log_isis = np.log10(neuron_info['isis'])

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    title = (f"{neuron_info['cell_type']} | {neuron_info['exp']} "
             f"M{neuron_info['mouse']} d{neuron_info['day']} "
             f"n{neuron_info['neuron_idx']}")
    fig.suptitle(title, fontsize=11, fontweight='bold')

    bins = np.arange(log_isis.min() - LOG_ISI_BIN_WIDTH,
                     log_isis.max() + 2 * LOG_ISI_BIN_WIDTH,
                     LOG_ISI_BIN_WIDTH)
    x_fit = np.linspace(log_isis.min() - 0.5, log_isis.max() + 0.5, 500)

    color = CELL_COLORS.get(neuron_info['cell_type'], '#555555')

    # --- Panel A: 2-component GMM ---
    ax = axes[0]
    ax.hist(log_isis, bins=bins, density=True, alpha=0.5, color=color,
            edgecolor='none')
    comp_colors_2 = ['#e07b39', '#3a7ca5']
    for k, pdf in enumerate(_gmm_component_pdfs(gmm2, x_fit)):
        ax.plot(x_fit, pdf, '--', color=comp_colors_2[k], lw=1.5)
    ax.plot(x_fit, _gmm_pdf(gmm2, x_fit), 'k-', lw=2, label='GMM-2 total')
    ax.set_xlabel('log₁₀(ISI) [s]')
    ax.set_ylabel('Density')
    ax.set_title('2-Component GMM')
    ax.legend(fontsize=8)

    # --- Panel B: 3-component GMM ---
    ax = axes[1]
    ax.hist(log_isis, bins=bins, density=True, alpha=0.5, color=color,
            edgecolor='none')
    comp_colors_3 = ['#d32f2f', '#388e3c', '#1976d2']
    comp_labels = ['Intra-burst', 'Intra-SB', 'Inter-event']
    for k, pdf in enumerate(_gmm_component_pdfs(gmm3, x_fit)):
        lbl = comp_labels[k] if k < len(comp_labels) else f'Comp {k+1}'
        ax.plot(x_fit, pdf, '--', color=comp_colors_3[k], lw=1.5, label=lbl)
    ax.plot(x_fit, _gmm_pdf(gmm3, x_fit), 'k-', lw=2)
    for cx in crossovers_3:
        ax.axvline(cx, color='gray', ls=':', lw=1)
        ax.text(cx, ax.get_ylim()[1] * 0.95, f'{10**cx*1000:.1f} ms',
                ha='center', va='top', fontsize=7, color='gray')
    sig_str = '*' if p_val < ALPHA else 'n.s.'
    ax.set_title(f'3-Component GMM (p={p_val:.3f} {sig_str})')
    ax.set_xlabel('log₁₀(ISI) [s]')
    ax.legend(fontsize=7, loc='upper right')

    # --- Panel C: Bootstrap LLR ---
    ax = axes[2]
    ax.hist(bootstrap_llrs, bins=40, density=True,
            alpha=0.6, color='gray', edgecolor='none')
    ax.axvline(llr, color='red', lw=2, label=f'Observed LLR = {llr:.1f}')
    ax.set_xlabel('Log-Likelihood Ratio')
    ax.set_ylabel('Density')
    ax.set_title('Bootstrap LLR Distribution')
    ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


def plot_pooled_by_celltype(all_results_df, neuron_data, save_path):
    """
    Plot pooled log10(ISI) histograms for each cell type,
    with GMM overlay from pooled ISIs.
    """
    cell_types_present = [ct for ct in CELL_ORDER
                          if ct in all_results_df['cell_type'].values]
    n_types = len(cell_types_present)
    if n_types == 0:
        return

    fig, axes = plt.subplots(1, n_types, figsize=(5 * n_types, 4.5),
                             squeeze=False)
    axes = axes.ravel()

    for idx, ct in enumerate(cell_types_present):
        ax = axes[idx]
        # Pool all ISIs for this cell type
        ct_isis = []
        for nkey, ninfo in neuron_data.items():
            if ninfo['cell_type'] == ct:
                ct_isis.append(ninfo['isis'])
        if not ct_isis:
            continue
        pooled = np.concatenate(ct_isis)
        log_pooled = np.log10(pooled)

        bins = np.arange(log_pooled.min() - LOG_ISI_BIN_WIDTH,
                         log_pooled.max() + 2 * LOG_ISI_BIN_WIDTH,
                         LOG_ISI_BIN_WIDTH)
        color = CELL_COLORS.get(ct, '#555555')
        ax.hist(log_pooled, bins=bins, density=True, alpha=0.5,
                color=color, edgecolor='none')

        x_fit = np.linspace(log_pooled.min() - 0.5,
                            log_pooled.max() + 0.5, 500)

        # Fit 2 and 3 component GMMs to pooled data
        try:
            gmm2, _ = fit_gmm(log_pooled, 2)
            gmm3, _ = fit_gmm(log_pooled, 3)

            ax.plot(x_fit, _gmm_pdf(gmm2, x_fit), 'k--', lw=1.5,
                    alpha=0.6, label='GMM-2')
            ax.plot(x_fit, _gmm_pdf(gmm3, x_fit), 'k-', lw=2,
                    label='GMM-3')

            comp_colors = ['#d32f2f', '#388e3c', '#1976d2']
            for k, pdf in enumerate(_gmm_component_pdfs(gmm3, x_fit)):
                ax.plot(x_fit, pdf, '--', color=comp_colors[k], lw=1)

            crossovers = gmm_crossover_points(gmm3)
            for cx in crossovers:
                ax.axvline(cx, color='gray', ls=':', lw=1)
                ax.text(cx, ax.get_ylim()[1] * 0.9,
                        f'{10**cx*1000:.1f} ms',
                        ha='center', va='top', fontsize=7, color='gray')
        except Exception as e:
            print(f"  GMM fit failed for pooled {ct}: {e}")

        n_neurons = sum(1 for v in neuron_data.values()
                        if v['cell_type'] == ct)
        ax.set_title(f'{ct} (n={n_neurons})', fontsize=10)
        ax.set_xlabel('log₁₀(ISI) [s]')
        if idx == 0:
            ax.set_ylabel('Density')
        ax.legend(fontsize=7)

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


def plot_event_summary(results_df, save_path):
    """
    Bar charts comparing event/spike proportions across cell types.
    """
    cell_types_present = [ct for ct in CELL_ORDER
                          if ct in results_df['cell_type'].values]
    if not cell_types_present:
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # --- A: Event proportions ---
    ax = axes[0, 0]
    x_pos = np.arange(len(cell_types_present))
    width = 0.25
    for i, etype in enumerate(['frac_events_single', 'frac_events_burst',
                                'frac_events_sb']):
        means = []
        sems = []
        for ct in cell_types_present:
            vals = results_df.loc[results_df['cell_type'] == ct, etype]
            means.append(vals.mean())
            sems.append(vals.sem() if len(vals) > 1 else 0)
        labels = ['Single AP', 'Burst', 'Superburst']
        colors = ['#5e81ac', '#a3be8c', '#bf616a']
        ax.bar(x_pos + i * width, means, width, yerr=sems,
               color=colors[i], alpha=0.8, label=labels[i],
               capsize=3, edgecolor='none')
    ax.set_xticks(x_pos + width)
    ax.set_xticklabels(cell_types_present, fontsize=8, rotation=20,
                       ha='right')
    ax.set_ylabel('Fraction of Events')
    ax.set_title('Event Type Proportions')
    ax.legend(fontsize=8)

    # --- B: Spike proportions ---
    ax = axes[0, 1]
    for i, etype in enumerate(['frac_spikes_single', 'frac_spikes_burst',
                                'frac_spikes_sb']):
        means = []
        sems = []
        for ct in cell_types_present:
            vals = results_df.loc[results_df['cell_type'] == ct, etype]
            means.append(vals.mean())
            sems.append(vals.sem() if len(vals) > 1 else 0)
        labels = ['In singles', 'In bursts', 'In SBs']
        colors = ['#5e81ac', '#a3be8c', '#bf616a']
        ax.bar(x_pos + i * width, means, width, yerr=sems,
               color=colors[i], alpha=0.8, label=labels[i],
               capsize=3, edgecolor='none')
    ax.set_xticks(x_pos + width)
    ax.set_xticklabels(cell_types_present, fontsize=8, rotation=20,
                       ha='right')
    ax.set_ylabel('Fraction of Spikes')
    ax.set_title('Spike Distribution by Event Type')
    ax.legend(fontsize=8)

    # --- C: Spikes per SB ---
    ax = axes[1, 0]
    sb_data = results_df[results_df['n_superbursts'] > 0]
    for i, ct in enumerate(cell_types_present):
        ct_data = sb_data[sb_data['cell_type'] == ct]
        if 'mean_spikes_per_sb' in ct_data.columns and len(ct_data) > 0:
            vals = ct_data['mean_spikes_per_sb'].dropna()
            color = CELL_COLORS.get(ct, '#555555')
            parts = ax.violinplot([vals.values], positions=[i],
                                  showmeans=True, showextrema=False)
            for pc in parts['bodies']:
                pc.set_facecolor(color)
                pc.set_alpha(0.5)
            parts['cmeans'].set_color(color)
            ax.scatter(np.full(len(vals), i) + np.random.uniform(-0.1, 0.1, len(vals)),
                       vals.values, color=color, s=15, alpha=0.6, zorder=3)
    ax.set_xticks(range(len(cell_types_present)))
    ax.set_xticklabels(cell_types_present, fontsize=8, rotation=20,
                       ha='right')
    ax.set_ylabel('Mean Spikes per SB')
    ax.set_title('Superburst Size')

    # --- D: SB duration ---
    ax = axes[1, 1]
    for i, ct in enumerate(cell_types_present):
        ct_data = sb_data[sb_data['cell_type'] == ct]
        if 'mean_sb_duration_ms' in ct_data.columns and len(ct_data) > 0:
            vals = ct_data['mean_sb_duration_ms'].dropna()
            color = CELL_COLORS.get(ct, '#555555')
            parts = ax.violinplot([vals.values], positions=[i],
                                  showmeans=True, showextrema=False)
            for pc in parts['bodies']:
                pc.set_facecolor(color)
                pc.set_alpha(0.5)
            parts['cmeans'].set_color(color)
            ax.scatter(np.full(len(vals), i) + np.random.uniform(-0.1, 0.1, len(vals)),
                       vals.values, color=color, s=15, alpha=0.6, zorder=3)
    ax.set_xticks(range(len(cell_types_present)))
    ax.set_xticklabels(cell_types_present, fontsize=8, rotation=20,
                       ha='right')
    ax.set_ylabel('Mean SB Duration (ms)')
    ax.set_title('Superburst Duration')

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


def plot_gmm_significance_summary(results_df, save_path):
    """
    Summary figure: fraction of neurons per cell type where
    3-component GMM significantly outperforms 2-component.
    """
    cell_types_present = [ct for ct in CELL_ORDER
                          if ct in results_df['cell_type'].values]
    if not cell_types_present:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- A: Fraction with significant 3-component fit ---
    ax = axes[0]
    fracs = []
    ns = []
    for ct in cell_types_present:
        ct_df = results_df[results_df['cell_type'] == ct]
        n = len(ct_df)
        n_sig = (ct_df['gmm3_pval'] < ALPHA).sum()
        fracs.append(n_sig / n if n > 0 else 0)
        ns.append(n)

    colors = [CELL_COLORS.get(ct, '#555555') for ct in cell_types_present]
    ax.bar(range(len(cell_types_present)), fracs, color=colors,
           alpha=0.8, edgecolor='none')
    for i, (f, n) in enumerate(zip(fracs, ns)):
        ax.text(i, f + 0.02, f'{int(f*n)}/{n}', ha='center', fontsize=9)
    ax.set_xticks(range(len(cell_types_present)))
    ax.set_xticklabels(cell_types_present, fontsize=8, rotation=20,
                       ha='right')
    ax.set_ylabel('Fraction with Significant 3-Component Fit')
    ax.set_title(f'GMM 3 vs 2 Component Test (α={ALPHA})')
    ax.set_ylim(0, 1.1)
    ax.axhline(ALPHA, color='gray', ls='--', lw=0.8, alpha=0.5)

    # --- B: LLR distribution by cell type ---
    ax = axes[1]
    for ct in cell_types_present:
        ct_df = results_df[results_df['cell_type'] == ct]
        llrs = ct_df['llr_observed'].dropna().values
        color = CELL_COLORS.get(ct, '#555555')
        ax.scatter(np.random.uniform(-0.2, 0.2, len(llrs))
                   + cell_types_present.index(ct),
                   llrs, color=color, s=20, alpha=0.6, label=ct)
    ax.set_xticks(range(len(cell_types_present)))
    ax.set_xticklabels(cell_types_present, fontsize=8, rotation=20,
                       ha='right')
    ax.set_ylabel('Log-Likelihood Ratio (3 vs 2)')
    ax.set_title('LLR Distribution')
    ax.axhline(0, color='gray', ls='-', lw=0.5)

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


# %% ============================================================
# MAIN
# ===============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("SUPERBURST ANALYSIS — Vandael et al. (2020) methodology")
    print("=" * 60)

    # --- 1. Load data ---
    recordings = discover_trials(EXPERIMENTS)
    neuron_data = load_spike_data(recordings)

    # --- 2. Per-neuron GMM fitting ---
    n_total = len(neuron_data)
    print(f"\n--- Per-neuron GMM analysis ({n_total} neurons) ---")
    results = []
    gallery_dir = os.path.join(FIG_DIR, 'gallery')
    os.makedirs(gallery_dir, exist_ok=True)

    t_start_all = time.time()

    for neuron_i, (nkey, ninfo) in enumerate(neuron_data.items()):
        log_isis = np.log10(ninfo['isis'])
        ct = ninfo['cell_type']
        uid = f"{ninfo['exp']}_M{ninfo['mouse']}_d{ninfo['day']}_n{ninfo['neuron_idx']}"

        print(f"\n[{neuron_i+1}/{n_total}] {ct:30s} | {uid} | "
              f"{len(ninfo['isis'])} ISIs")

        try:
            llr, p_val, gmm2, gmm3, boot_llrs = bootstrap_llr_test(log_isis)
            crossovers_3 = gmm_crossover_points(gmm3)

            # Sorted component means (log10 seconds)
            means_3 = np.sort(gmm3.means_.ravel())

            sig = p_val < ALPHA
            print(f"  Result: LLR={llr:.1f}, p={p_val:.3f} "
                  f"{'*** SIGNIFICANT' if sig else 'n.s.'}")

            # Event classification using crossovers
            if sig and len(crossovers_3) >= 2:
                burst_thresh = crossovers_3[0]
                sb_thresh = crossovers_3[1]
            elif sig and len(crossovers_3) == 1:
                burst_thresh = crossovers_3[0]
                sb_thresh = None
            else:
                # Use 2-component crossover for burst threshold only
                crossovers_2 = gmm_crossover_points(gmm2)
                burst_thresh = crossovers_2[0] if crossovers_2 else np.log10(0.006)
                sb_thresh = None

            events = classify_events(ninfo['spike_times_list'],
                                     burst_thresh, sb_thresh)
            event_stats = compute_event_stats(events, ninfo['n_spikes'])

            row = {
                'neuron_key': uid,
                'cell_type': ct,
                'exp': ninfo['exp'],
                'mouse': ninfo['mouse'],
                'day': ninfo['day'],
                'neuron_idx': ninfo['neuron_idx'],
                'mean_rate': ninfo['mean_rate'],
                'n_isis': len(ninfo['isis']),
                'llr_observed': llr,
                'gmm3_pval': p_val,
                'gmm3_significant': sig,
                'n_crossovers': len(crossovers_3),
                'burst_thresh_ms': 10**burst_thresh * 1000,
                'sb_thresh_ms': 10**sb_thresh * 1000 if sb_thresh is not None else np.nan,
                'gmm3_mean1_log': means_3[0],
                'gmm3_mean2_log': means_3[1],
                'gmm3_mean3_log': means_3[2] if len(means_3) > 2 else np.nan,
            }
            row.update(event_stats)
            results.append(row)

            # Save per-neuron figure
            fig_path = os.path.join(gallery_dir, f'{ct}_{uid}.png')
            plot_neuron_gmm(ninfo, gmm2, gmm3, llr, p_val, crossovers_3,
                            boot_llrs, fig_path)

        except Exception as e:
            print(f"  FAILED: {e}")
            results.append({
                'neuron_key': uid,
                'cell_type': ct,
                'llr_observed': np.nan,
                'gmm3_pval': np.nan,
                'gmm3_significant': False,
            })

    elapsed_total = time.time() - t_start_all
    print(f"\nGMM analysis complete in {elapsed_total/60:.1f} minutes")

    df = pd.DataFrame(results)

    # --- 3. Summary prints ---
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    for ct in CELL_ORDER:
        ct_df = df[df['cell_type'] == ct]
        if len(ct_df) == 0:
            continue
        n = len(ct_df)
        n_sig = ct_df['gmm3_significant'].sum()
        print(f"\n{ct} (n={n}):")
        print(f"  3-component significant: {n_sig}/{n} "
              f"({100*n_sig/n:.0f}%)")

        if n_sig > 0:
            sig_df = ct_df[ct_df['gmm3_significant']]
            print(f"  Burst threshold: "
                  f"{sig_df['burst_thresh_ms'].mean():.1f} ± "
                  f"{sig_df['burst_thresh_ms'].std():.1f} ms")
            if 'sb_thresh_ms' in sig_df.columns:
                sb_vals = sig_df['sb_thresh_ms'].dropna()
                if len(sb_vals) > 0:
                    print(f"  SB threshold: "
                          f"{sb_vals.mean():.1f} ± "
                          f"{sb_vals.std():.1f} ms")

        if 'n_superbursts' in ct_df.columns:
            sb_cells = ct_df[ct_df.get('n_superbursts', 0) > 0]
            if len(sb_cells) > 0:
                print(f"  Neurons with SBs: {len(sb_cells)}/{n}")
                if 'mean_spikes_per_sb' in sb_cells.columns:
                    vals = sb_cells['mean_spikes_per_sb'].dropna()
                    if len(vals) > 0:
                        print(f"  Mean spikes/SB: "
                              f"{vals.mean():.1f} ± {vals.std():.1f}")
                if 'mean_sb_duration_ms' in sb_cells.columns:
                    vals = sb_cells['mean_sb_duration_ms'].dropna()
                    if len(vals) > 0:
                        print(f"  Mean SB duration: "
                              f"{vals.mean():.0f} ± {vals.std():.0f} ms")
                if 'frac_spikes_sb' in sb_cells.columns:
                    vals = sb_cells['frac_spikes_sb'].dropna()
                    if len(vals) > 0:
                        print(f"  Frac spikes in SBs: "
                              f"{vals.mean():.2f} ± {vals.std():.2f}")

    # --- 4. Plots ---
    print("\n--- Generating figures ---")

    plot_pooled_by_celltype(
        df, neuron_data,
        os.path.join(FIG_DIR, 'pooled_log_isih_by_celltype.png'))
    print("  Saved pooled_log_isih_by_celltype.png")

    plot_gmm_significance_summary(
        df,
        os.path.join(FIG_DIR, 'gmm_significance_summary.png'))
    print("  Saved gmm_significance_summary.png")

    plot_event_summary(
        df,
        os.path.join(FIG_DIR, 'event_classification_summary.png'))
    print("  Saved event_classification_summary.png")

    # --- 5. Save CSV ---
    csv_path = os.path.join(FIG_DIR, 'superburst_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"  Saved results to {csv_path}")

    print("\nDone.")