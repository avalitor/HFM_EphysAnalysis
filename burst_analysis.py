"""
Comprehensive spike train analysis for burstiness characterization.
Implements analyses from Shinomoto et al. (2009), Turner et al. (1996),
Pasquale et al. (2010), and Cotterill et al. (2016).

Analyses:
  1. ISI histogram + distribution fitting (exponential, gamma, lognormal)
  2. Log10(ISIH) with LOESS smoothing
  3. Joint interval return map
  4. Stationarity test (divide into 4 segments)
  5. Serial correlation coefficient
  6. Autocorrelation and power spectral density
  7. Coefficient of variation (CV)
  8. Fixed-ISI burst detection (6 ms, Royer et al. 2012)
  9. Lv and LvR (Shinomoto et al. 2005, 2009)

Only analyzes trials with total duration > 20 minutes.
Uses raw spike trains (no velocity filter).
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats, signal
# from scipy.optimize import curve_fit
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
MIN_ISI_SHINOMOTO  = 2000   # Shinomoto's stricter threshold
MIN_RATE_HZ        = 0.5    # minimum mean rate (Hz)
ISI_BURST_THRESH   = 0.006  # 6 ms fixed ISI burst threshold (Royer 2012)
LVR_R              = 0.005  # refractoriness constant R=5ms (Shinomoto 2009)
N_EXAMPLE_NEURONS  = 3      # per cell type for gallery

# Cell type colors (standard scheme)
CELL_COLORS = {
    'Narrow Interneuron':         '#0050a0',
    'Wide Interneuron':           '#07CCC3',
    'Excitatory Principal Cell':  '#e36414',
    'Bursty Narrow Interneuron':  '#87147A',
}

CELL_ORDER = ['Narrow Interneuron', 'Wide Interneuron',
              'Excitatory Principal Cell', 'Bursty Narrow Interneuron']

# Excluded trial prefixes — NONE for this analysis (use all trial types)
EXCLUDED_PREFIXES = ()

FIG_DIR = os.path.join(cg.ROOT_DIR, 'figs', 'burst_analysis')
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
                # Check excluded prefixes
                skip = any(trial.startswith(p) for p in EXCLUDED_PREFIXES)
                if not skip:
                    recordings.append((exp, mouse, trial))
    print(f"Discovered {len(recordings)} recordings across {len(experiments)} experiments")
    return recordings


def load_and_filter_trials(recordings):
    """
    Load trials, keep only those > MIN_TRIAL_DURATION.
    For each qualifying trial, extract spike trains and cell labels.
    Deduplicate neurons by (exp, mouse, day) — only load cell_metrics once per day.
    Returns list of dicts with trial+neuron info.
    """
    trial_data = []
    loaded_days = {}  # (exp, mouse, day) -> cell_labels

    for i, (exp, mouse, trial) in enumerate(recordings):
        if (i + 1) % 20 == 0:
            print(f"  Loading {i+1}/{len(recordings)}...")
        try:
            edata = elib.EphysTrial.Load(exp, mouse, trial)
        except Exception as e:
            print(f"  Failed to load {exp} M{mouse} T{trial}: {e}")
            continue

        # Check trial duration
        if edata.time_ttl is None or len(edata.time_ttl) < 2:
            continue
        duration = edata.time_ttl[-1] - edata.time_ttl[0]
        if duration < MIN_TRIAL_DURATION:
            continue

        # Get day
        day = str(edata.day)

        # Get cell labels — load cell_metrics only once per (exp, mouse, day)
        day_key = (exp, mouse, day)
        if day_key not in loaded_days:
            try:
                edata_cm = elib.EphysTrial.Load(exp, mouse, trial,
                                                 load_cell_metrics=True)
                n_neurons = len(edata_cm.t_spikeTrains)
                labels = []
                for ni in range(n_neurons):
                    try:
                        lbl = edata_cm.cellLabels_BITP[ni][0]
                    except (IndexError, TypeError):
                        lbl = 'Unknown'
                    labels.append(str(lbl))
                loaded_days[day_key] = labels
            except Exception as e:
                print(f"  Failed to load cell_metrics for {day_key}: {e}")
                n_neurons = len(edata.t_spikeTrains)
                loaded_days[day_key] = ['Unknown'] * n_neurons

        labels = loaded_days[day_key]
        n_neurons = len(edata.t_spikeTrains)

        # Safety check for label count mismatch
        if len(labels) != n_neurons:
            print(f"  Label mismatch for {exp} M{mouse} T{trial}: "
                  f"{len(labels)} labels vs {n_neurons} neurons")
            labels = labels[:n_neurons] if len(labels) > n_neurons \
                     else labels + ['Unknown'] * (n_neurons - len(labels))

        for ni in range(n_neurons):
            spk = np.atleast_1d(edata.t_spikeTrains[ni])
            if spk.size == 0:
                continue

            # Compute ISIs
            isis = np.diff(spk)
            if isis.size < MIN_ISI_COUNT:
                continue

            mean_rate = len(spk) / duration
            if mean_rate < MIN_RATE_HZ:
                continue

            trial_data.append({
                'exp': exp, 'mouse': mouse, 'day': day, 'trial': trial,
                'neuron_idx': ni,
                'neuron_id': f"{exp}_M{mouse}_d{day}_n{ni}",
                'cell_type': labels[ni],
                'spike_times': spk,
                'isis': isis,
                'duration': duration,
                'mean_rate': mean_rate,
                'n_spikes': len(spk),
                'n_isis': len(isis),
                'meets_shinomoto': (len(isis) >= MIN_ISI_SHINOMOTO
                                   and mean_rate >= 5.0),
            })

    print(f"\nQualifying neurons (trials>20min, >=100 ISIs, >=0.5Hz): "
          f"{len(trial_data)}")
    # Summarize by cell type
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

# --- 1. ISI histogram + distribution fitting ---
def fit_isi_distributions(isis):
    """Fit exponential, gamma, and lognormal to ISI distribution."""
    isis_pos = isis[isis > 0]
    if len(isis_pos) < 50:
        return {}

    results = {}

    # Exponential
    try:
        loc_e, scale_e = stats.expon.fit(isis_pos, floc=0)
        ks_stat, ks_p = stats.kstest(isis_pos, 'expon', args=(loc_e, scale_e))
        results['exponential'] = {'scale': scale_e, 'ks_stat': ks_stat,
                                   'ks_p': ks_p}
    except Exception:
        pass

    # Gamma
    try:
        a_g, loc_g, scale_g = stats.gamma.fit(isis_pos, floc=0)
        ks_stat, ks_p = stats.kstest(isis_pos, 'gamma', args=(a_g, loc_g, scale_g))
        results['gamma'] = {'shape': a_g, 'scale': scale_g,
                            'ks_stat': ks_stat, 'ks_p': ks_p}
    except Exception:
        pass

    # Lognormal
    try:
        s_ln, loc_ln, scale_ln = stats.lognorm.fit(isis_pos, floc=0)
        ks_stat, ks_p = stats.kstest(isis_pos, 'lognorm',
                                      args=(s_ln, loc_ln, scale_ln))
        results['lognormal'] = {'s': s_ln, 'scale': scale_ln,
                                'ks_stat': ks_stat, 'ks_p': ks_p}
    except Exception:
        pass

    return results


# --- 2. Log10(ISIH) with LOESS smoothing ---
def log_isi_histogram(isis, bin_width=0.1):
    """
    Compute log10(ISI) histogram.
    bin_width in log10 units (0.1 as per Pasquale et al.).
    Returns bin_centers, counts, smoothed_counts.
    """
    isis_pos = isis[isis > 0]
    log_isis = np.log10(isis_pos)

    bins = np.arange(log_isis.min() - bin_width,
                     log_isis.max() + 2 * bin_width, bin_width)
    counts, edges = np.histogram(log_isis, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])

    # Normalize to probability
    total = counts.sum()
    if total > 0:
        prob = counts / total
    else:
        prob = counts.astype(float)

    # LOESS smoothing (local regression, 1st degree polynomial)
    # Using LOWESS from statsmodels-compatible approach via savgol or manual
    # Since we want local regression, use scipy's savgol as approximation
    # or implement simple weighted linear regression
    smoothed = _loess_smooth(centers, prob, frac=0.15)

    return centers, prob, smoothed


def _loess_smooth(x, y, frac=0.15):
    """Simple LOESS-like smoothing using weighted local linear regression."""
    n = len(x)
    span = max(int(np.ceil(frac * n)), 3)
    if span % 2 == 0:
        span += 1
    smoothed = np.copy(y).astype(float)

    for i in range(n):
        # Window around point i
        half = span // 2
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        xi = x[lo:hi]
        yi = y[lo:hi]

        # Tricube weights
        dists = np.abs(xi - x[i])
        max_dist = dists.max()
        if max_dist == 0:
            smoothed[i] = yi.mean()
            continue
        u = dists / (max_dist * 1.001)  # avoid exactly 1
        w = (1 - u**3)**3
        w[u >= 1] = 0

        # Weighted linear fit
        if w.sum() == 0:
            smoothed[i] = yi.mean()
            continue
        try:
            coeffs = np.polyfit(xi, yi, 1, w=w)
            smoothed[i] = np.polyval(coeffs, x[i])
        except Exception:
            smoothed[i] = np.average(yi, weights=w)

    return np.maximum(smoothed, 0)  # no negative probabilities


# --- 3. Return map ---
def return_map(isis):
    """Compute joint interval return map: ISI_i vs ISI_{i+1}."""
    return isis[:-1], isis[1:]


# --- 4. Stationarity test ---
def stationarity_test(spike_times, n_segments=4):
    """
    Test for wide-sense stationarity (Turner et al. 1996).
    Divide into n_segments, compare mean ISI and variance.
    Returns dict with per-segment stats and ANOVA p-values.
    """
    isis = np.diff(spike_times)
    n = len(isis)
    seg_size = n // n_segments
    if seg_size < 20:
        return {'stationary': None, 'p_mean': None, 'p_var': None}

    seg_means = []
    seg_vars = []
    segments = []
    for s in range(n_segments):
        start = s * seg_size
        end = start + seg_size if s < n_segments - 1 else n
        seg = isis[start:end]
        seg_means.append(np.mean(seg))
        seg_vars.append(np.var(seg))
        segments.append(seg)

    # ANOVA on means (Kruskal-Wallis as nonparametric alternative)
    try:
        _, p_mean = stats.kruskal(*segments)
    except Exception:
        p_mean = np.nan

    # Levene test for variance homogeneity
    try:
        _, p_var = stats.levene(*segments)
    except Exception:
        p_var = np.nan

    stationary = (p_mean > 0.05) and (p_var > 0.05)

    return {
        'stationary': stationary,
        'p_mean': p_mean,
        'p_var': p_var,
        'seg_means': seg_means,
        'seg_vars': seg_vars,
    }


# --- 5. Serial correlation ---
def serial_correlation(isis, max_lag=7):
    """
    Compute serial correlation coefficients for lags 1..max_lag.
    SC=0 implies renewal process.
    """
    n = len(isis)
    if n < max_lag + 10:
        return np.full(max_lag, np.nan)

    mean_isi = np.mean(isis)
    var_isi = np.var(isis)
    if var_isi == 0:
        return np.zeros(max_lag)

    sc = np.zeros(max_lag)
    for lag in range(1, max_lag + 1):
        cov = np.mean((isis[:-lag] - mean_isi) * (isis[lag:] - mean_isi))
        sc[lag - 1] = cov / var_isi

    return sc


# --- 6. Autocorrelation and PSD ---
def autocorrelation_psd(spike_times, duration, bin_width=0.001, max_lag_s=5.0):
    """
    Compute spike train autocorrelation and power spectral density.
    bin_width in seconds (1 ms default).
    """
    # Bin spike train
    bins = np.arange(spike_times[0], spike_times[-1], bin_width)
    spike_counts, _ = np.histogram(spike_times, bins=bins)

    # Subtract mean (to get autocorrelation of deviations)
    spike_counts = spike_counts.astype(float) - spike_counts.mean()

    # Autocorrelation via numpy correlate (normalized)
    max_lag_bins = int(max_lag_s / bin_width)
    n = len(spike_counts)

    # Use FFT-based approach for speed
    autocorr_full = signal.fftconvolve(spike_counts, spike_counts[::-1],
                                        mode='full')
    center = len(autocorr_full) // 2
    autocorr = autocorr_full[center:center + max_lag_bins]
    if autocorr[0] > 0:
        autocorr = autocorr / autocorr[0]

    lags = np.arange(len(autocorr)) * bin_width

    # PSD via Welch's method
    fs = 1.0 / bin_width
    nperseg = min(len(spike_counts), 2**14)
    freqs, psd = signal.welch(spike_counts + spike_counts.mean(),
                               fs=fs, nperseg=nperseg)

    # Keep only relevant frequencies (0-50 Hz typically)
    mask = freqs <= 50
    freqs = freqs[mask]
    psd = psd[mask]

    return lags, autocorr, freqs, psd


# --- 7. Coefficient of variation ---
def coefficient_of_variation(isis):
    """CV = std / mean. CV=1 for Poisson, <1 regular, >1 bursty."""
    if len(isis) < 2:
        return np.nan
    return np.std(isis) / np.mean(isis)


# --- 8. Fixed-ISI burst detection ---
def fixed_isi_burst_detection(spike_times, isis, threshold=ISI_BURST_THRESH,
                               min_spikes=3):
    """
    Detect bursts as sequences of >= min_spikes spikes with ISI < threshold.
    Returns burst statistics.
    """
    in_burst = False
    bursts = []
    current_burst = []

    for i in range(len(isis)):
        if isis[i] < threshold:
            if not in_burst:
                current_burst = [i]  # spike index
                in_burst = True
            current_burst.append(i + 1)
        else:
            if in_burst:
                if len(current_burst) >= min_spikes:
                    bursts.append(current_burst)
                current_burst = []
                in_burst = False

    # Close last burst
    if in_burst and len(current_burst) >= min_spikes:
        bursts.append(current_burst)

    if not bursts:
        return {
            'n_bursts': 0, 'burst_rate': 0,
            'mean_spikes_per_burst': 0, 'mean_burst_duration': 0,
            'fraction_spikes_in_bursts': 0,
            'burst_index_royer': 0,
        }

    n_bursts = len(bursts)
    spikes_per_burst = [len(b) for b in bursts]
    burst_durations = [spike_times[b[-1]] - spike_times[b[0]] for b in bursts]
    total_spikes_in_bursts = sum(spikes_per_burst)
    duration = spike_times[-1] - spike_times[0]

    # Burst index (Royer 2012): fraction of ISIs < threshold
    burst_index = np.sum(isis < threshold) / len(isis)

    return {
        'n_bursts': n_bursts,
        'burst_rate': n_bursts / duration if duration > 0 else 0,
        'mean_spikes_per_burst': np.mean(spikes_per_burst),
        'mean_burst_duration': np.mean(burst_durations),
        'fraction_spikes_in_bursts': total_spikes_in_bursts / len(spike_times),
        'burst_index_royer': burst_index,
    }


# --- 9. Lv and LvR ---
def compute_Lv(isis):
    """
    Local variation Lv (Shinomoto et al. 2005).
    Lv = (3/(n-1)) * sum((I_i - I_{i+1})^2 / (I_i + I_{i+1})^2)
    Lv = 0: perfectly regular. Lv = 1: Poisson. Lv > 1: bursty.
    """
    n = len(isis)
    if n < 2:
        return np.nan

    I_i = isis[:-1]
    I_ip1 = isis[1:]
    terms = ((I_i - I_ip1) / (I_i + I_ip1)) ** 2

    return (3.0 / (n - 1)) * np.sum(terms)


def compute_LvR(isis, R=LVR_R):
    """
    Revised local variation LvR (Shinomoto et al. 2009, Eq. 3).
    LvR = (3/(n-1)) * sum( (1 - 4*I_i*I_{i+1} / (I_i+I_{i+1})^2)
                            * (1 + 4*R / (I_i+I_{i+1})) )

    R = refractoriness constant (default 5 ms = 0.005 s).
    """
    n = len(isis)
    if n < 2:
        return np.nan

    I_i = isis[:-1]
    I_ip1 = isis[1:]
    sum_ii = I_i + I_ip1

    # Avoid division by zero
    mask = sum_ii > 0
    if not np.all(mask):
        I_i = I_i[mask]
        I_ip1 = I_ip1[mask]
        sum_ii = sum_ii[mask]
        n = len(I_i) + 1

    term1 = 1.0 - (4.0 * I_i * I_ip1) / (sum_ii ** 2)
    term2 = 1.0 + (4.0 * R) / sum_ii

    return (3.0 / (n - 1)) * np.sum(term1 * term2)


def compute_LvR_segments(isis, n_segments=20, R=LVR_R):
    """
    Compute LvR for fractional segments (for F-test analysis).
    Returns array of LvR values, one per segment.
    """
    n = len(isis)
    seg_size = n // n_segments
    if seg_size < 10:
        return np.array([compute_LvR(isis, R)])

    lvr_segs = []
    for s in range(n_segments):
        start = s * seg_size
        end = start + seg_size if s < n_segments - 1 else n
        seg = isis[start:end]
        if len(seg) >= 2:
            lvr_segs.append(compute_LvR(seg, R))
    return np.array(lvr_segs)


# %% ============================================================
# MAIN ANALYSIS LOOP
# ===============================================================

def analyze_all(trial_data):
    """Run all 9 analyses on each qualifying neuron."""
    results = []

    for i, td in enumerate(trial_data):
        if (i + 1) % 50 == 0:
            print(f"  Analyzing {i+1}/{len(trial_data)}...")

        isis = td['isis']
        spk = td['spike_times']

        # 1. Distribution fitting
        fits = fit_isi_distributions(isis)

        # 4. Stationarity
        stat = stationarity_test(spk)

        # 5. Serial correlation
        sc = serial_correlation(isis)

        # 7. CV
        cv = coefficient_of_variation(isis)

        # 8. Burst detection
        burst = fixed_isi_burst_detection(spk, isis)

        # 9. Lv and LvR
        lv = compute_Lv(isis)
        lvr = compute_LvR(isis)

        # Classify firing pattern based on LvR (Shinomoto categories)
        if lvr < 0.7:
            lvr_category = 'Regular'
        elif lvr < 1.3:
            lvr_category = 'Random'
        else:
            lvr_category = 'Bursty'

        row = {
            'neuron_id': td['neuron_id'],
            'exp': td['exp'],
            'mouse': td['mouse'],
            'day': td['day'],
            'trial': td['trial'],
            'neuron_idx': td['neuron_idx'],
            'cell_type': td['cell_type'],
            'n_spikes': td['n_spikes'],
            'n_isis': td['n_isis'],
            'duration_s': td['duration'],
            'mean_rate_hz': td['mean_rate'],
            'meets_shinomoto': td['meets_shinomoto'],
            # 1. Distribution fits
            'fit_exp_ks_p':     fits.get('exponential', {}).get('ks_p', np.nan),
            'fit_gamma_ks_p':   fits.get('gamma', {}).get('ks_p', np.nan),
            'fit_gamma_shape':  fits.get('gamma', {}).get('shape', np.nan),
            'fit_lognorm_ks_p': fits.get('lognormal', {}).get('ks_p', np.nan),
            # 4. Stationarity
            'stationary':    stat['stationary'],
            'stationarity_p_mean': stat['p_mean'],
            'stationarity_p_var':  stat['p_var'],
            # 5. Serial correlation (lag 1)
            'serial_corr_lag1': sc[0] if len(sc) > 0 else np.nan,
            'serial_corr_lag2': sc[1] if len(sc) > 1 else np.nan,
            # 7. CV
            'cv': cv,
            # 8. Burst detection
            'n_bursts':                burst['n_bursts'],
            'burst_rate':              burst['burst_rate'],
            'mean_spikes_per_burst':   burst['mean_spikes_per_burst'],
            'mean_burst_duration_ms':  burst['mean_burst_duration'] * 1000,
            'frac_spikes_in_bursts':   burst['fraction_spikes_in_bursts'],
            'burst_index_royer':       burst['burst_index_royer'],
            # 9. Lv and LvR
            'Lv':  lv,
            'LvR': lvr,
            'LvR_category': lvr_category,
        }
        results.append(row)

    df = pd.DataFrame(results)
    return df


# %% ============================================================
# PLOTTING FUNCTIONS
# ===============================================================

def plot_example_neuron(td, df_row, save_path):
    """
    Plot full example panel for a single neuron:
    ISI histogram + fits, log(ISIH), return map, autocorrelation.
    """
    isis = td['isis']
    spk = td['spike_times']

    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.35)

    title = (f"{td['cell_type']} | {td['exp']} M{td['mouse']} "
             f"d{td['day']} n{td['neuron_idx']} T{td['trial']}")
    fig.suptitle(title, fontsize=11, fontweight='bold')

    # --- A: ISI histogram with fits ---
    ax1 = fig.add_subplot(gs[0, 0])
    isis_ms = isis * 1000  # convert to ms
    max_isi = np.percentile(isis_ms, 99)
    bins_isi = np.linspace(0, max_isi, 80)
    ax1.hist(isis_ms, bins=bins_isi, density=True, alpha=0.6, color='gray',
             edgecolor='none')
    # Overlay gamma fit if available
    if not np.isnan(df_row.get('fit_gamma_shape', np.nan)):
        try:
            a, _, sc = stats.gamma.fit(isis[isis > 0], floc=0)
            x_fit = np.linspace(0, max_isi / 1000, 200)
            ax1.plot(x_fit * 1000, stats.gamma.pdf(x_fit, a, 0, sc) / 1000,
                     'r-', lw=1.5, label=f'Gamma (κ={a:.1f})')
            ax1.legend(fontsize=8)
        except Exception:
            pass
    ax1.set_xlabel('ISI (ms)')
    ax1.set_ylabel('Density')
    ax1.set_title('ISI Histogram')

    # --- B: Log10(ISIH) ---
    ax2 = fig.add_subplot(gs[0, 1])
    centers, prob, smoothed = log_isi_histogram(isis)
    ax2.bar(centers, prob, width=0.08, alpha=0.5, color='gray', edgecolor='none')
    ax2.plot(centers, smoothed, 'r-', lw=2, label='LOESS')
    ax2.set_xlabel('log₁₀(ISI) [s]')
    ax2.set_ylabel('Probability')
    ax2.set_title('Log₁₀(ISIH)')
    ax2.legend(fontsize=8)

    # --- C: Return map ---
    ax3 = fig.add_subplot(gs[0, 2])
    rm_x, rm_y = return_map(isis * 1000)
    ax3.scatter(rm_x, rm_y, s=1, alpha=0.3, color='k', rasterized=True)
    lim = np.percentile(np.concatenate([rm_x, rm_y]), 99)
    ax3.set_xlim(0, lim)
    ax3.set_ylim(0, lim)
    ax3.plot([0, lim], [0, lim], 'r--', alpha=0.3, lw=0.5)
    ax3.set_xlabel('ISI_n (ms)')
    ax3.set_ylabel('ISI_{n+1} (ms)')
    ax3.set_title('Return Map')
    ax3.set_aspect('equal')

    # --- D: Autocorrelation ---
    ax4 = fig.add_subplot(gs[1, 0])
    try:
        lags, autocorr, _, _ = autocorrelation_psd(spk, td['duration'],
                                                     bin_width=0.001,
                                                     max_lag_s=2.0)
        # Skip lag 0
        ax4.plot(lags[1:] * 1000, autocorr[1:], 'k-', lw=0.5)
        ax4.set_xlabel('Lag (ms)')
        ax4.set_ylabel('Autocorrelation')
        ax4.set_title('Autocorrelation')
        ax4.axhline(0, color='gray', lw=0.5)
    except Exception:
        ax4.text(0.5, 0.5, 'Error', transform=ax4.transAxes, ha='center')

    # --- E: PSD ---
    ax5 = fig.add_subplot(gs[1, 1])
    try:
        _, _, freqs, psd = autocorrelation_psd(spk, td['duration'],
                                                 bin_width=0.001)
        ax5.semilogy(freqs, psd, 'k-', lw=0.5)
        ax5.set_xlabel('Frequency (Hz)')
        ax5.set_ylabel('PSD')
        ax5.set_title('Power Spectral Density')
    except Exception:
        ax5.text(0.5, 0.5, 'Error', transform=ax5.transAxes, ha='center')

    # --- F: Serial correlation ---
    ax6 = fig.add_subplot(gs[1, 2])
    sc = serial_correlation(isis)
    lags_sc = np.arange(1, len(sc) + 1)
    ax6.bar(lags_sc, sc, color='steelblue', edgecolor='none')
    # 95% CI for null (renewal process)
    ci = 1.96 / np.sqrt(len(isis))
    ax6.axhline(ci, color='red', ls='--', lw=0.8, alpha=0.5)
    ax6.axhline(-ci, color='red', ls='--', lw=0.8, alpha=0.5)
    ax6.axhline(0, color='gray', lw=0.5)
    ax6.set_xlabel('Lag')
    ax6.set_ylabel('Serial Correlation')
    ax6.set_title('Serial Correlation')

    # --- G: Summary text ---
    ax7 = fig.add_subplot(gs[2, :])
    ax7.axis('off')
    summary = (
        f"CV = {df_row['cv']:.2f}  |  Lv = {df_row['Lv']:.3f}  |  "
        f"LvR = {df_row['LvR']:.3f} ({df_row['LvR_category']})  |  "
        f"Rate = {df_row['mean_rate_hz']:.1f} Hz  |  "
        f"Bursts (6ms): {df_row['n_bursts']}  |  "
        f"Frac in bursts: {df_row['frac_spikes_in_bursts']:.2f}  |  "
        f"BI_Royer: {df_row['burst_index_royer']:.3f}\n"
        f"Stationary: {df_row['stationary']} "
        f"(p_mean={df_row['stationarity_p_mean']:.3f}, "
        f"p_var={df_row['stationarity_p_var']:.3f})  |  "
        f"SC lag1 = {df_row['serial_corr_lag1']:.3f}  |  "
        f"Meets Shinomoto criteria: {df_row['meets_shinomoto']}\n"
        f"Fit KS p-values — Exp: {df_row['fit_exp_ks_p']:.2e}  "
        f"Gamma: {df_row['fit_gamma_ks_p']:.2e}  "
        f"Lognorm: {df_row['fit_lognorm_ks_p']:.2e}"
    )
    ax7.text(0.02, 0.7, summary, transform=ax7.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_summary_by_cell_type(df, fig_dir):
    """Generate summary figures comparing cell types."""

    # Filter to cell types with enough data
    valid_types = [ct for ct in CELL_ORDER if ct in df['cell_type'].values
                   and df[df['cell_type'] == ct].shape[0] >= 3]

    if not valid_types:
        print("Not enough cell types with data for summary plots.")
        return

    # --- A: LvR distribution by cell type ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # LvR
    ax = axes[0]
    data_lvr = [df[df['cell_type'] == ct]['LvR'].dropna().values
                for ct in valid_types]
    colors = [CELL_COLORS.get(ct, 'gray') for ct in valid_types]
    bp = ax.boxplot(data_lvr, labels=[ct.replace(' ', '\n') for ct in valid_types],
                    patch_artist=True, widths=0.5)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.axhline(1.0, color='gray', ls='--', lw=0.8, alpha=0.5, label='Poisson')
    ax.set_ylabel('LvR')
    ax.set_title('LvR by Cell Type')

    # CV
    ax = axes[1]
    data_cv = [df[df['cell_type'] == ct]['cv'].dropna().values
               for ct in valid_types]
    bp = ax.boxplot(data_cv, labels=[ct.replace(' ', '\n') for ct in valid_types],
                    patch_artist=True, widths=0.5)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.axhline(1.0, color='gray', ls='--', lw=0.8, alpha=0.5)
    ax.set_ylabel('CV')
    ax.set_title('CV by Cell Type')

    # Burst index
    ax = axes[2]
    data_bi = [df[df['cell_type'] == ct]['burst_index_royer'].dropna().values
               for ct in valid_types]
    bp = ax.boxplot(data_bi, labels=[ct.replace(' ', '\n') for ct in valid_types],
                    patch_artist=True, widths=0.5)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel('Burst Index (Royer)')
    ax.set_title('Burst Index by Cell Type')

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'summary_variability_by_celltype.png'),
                dpi=200, bbox_inches='tight')
    plt.close()

    # --- B: LvR vs CV scatter colored by cell type ---
    fig, ax = plt.subplots(figsize=(7, 6))
    for ct in valid_types:
        mask = df['cell_type'] == ct
        ax.scatter(df.loc[mask, 'cv'], df.loc[mask, 'LvR'],
                   c=CELL_COLORS.get(ct, 'gray'), label=ct, alpha=0.6,
                   s=20, edgecolors='none')
    ax.axhline(1.0, color='gray', ls='--', lw=0.5, alpha=0.5)
    ax.axvline(1.0, color='gray', ls='--', lw=0.5, alpha=0.5)
    ax.set_xlabel('CV')
    ax.set_ylabel('LvR')
    ax.set_title('LvR vs CV')
    ax.legend(fontsize=8, loc='upper right')
    plt.savefig(os.path.join(fig_dir, 'scatter_LvR_vs_CV.png'),
                dpi=200, bbox_inches='tight')
    plt.close()

    # --- C: LvR histogram (Shinomoto Fig 3A style) ---
    fig, ax = plt.subplots(figsize=(7, 4))
    for ct in valid_types:
        vals = df[df['cell_type'] == ct]['LvR'].dropna().values
        if len(vals) > 2:
            ax.hist(vals, bins=np.arange(0, 3.05, 0.15), alpha=0.5,
                    color=CELL_COLORS.get(ct, 'gray'), label=ct,
                    edgecolor='none')
    ax.set_xlabel('LvR')
    ax.set_ylabel('Count')
    ax.set_title('LvR Distribution')
    ax.legend(fontsize=8)
    plt.savefig(os.path.join(fig_dir, 'hist_LvR_by_celltype.png'),
                dpi=200, bbox_inches='tight')
    plt.close()

    # --- D: Stationarity summary ---
    fig, ax = plt.subplots(figsize=(6, 4))
    stat_counts = []
    for ct in valid_types:
        sub = df[df['cell_type'] == ct]
        n_stat = sub['stationary'].sum()
        n_total = len(sub)
        stat_counts.append((ct, n_stat, n_total))
    labels = [s[0].replace(' ', '\n') for s in stat_counts]
    fracs = [s[1] / s[2] if s[2] > 0 else 0 for s in stat_counts]
    bar_colors = [CELL_COLORS.get(s[0], 'gray') for s in stat_counts]
    bars = ax.bar(labels, fracs, color=bar_colors, alpha=0.7, edgecolor='none')
    for bar, sc in zip(bars, stat_counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f'{sc[1]}/{sc[2]}', ha='center', fontsize=8)
    ax.set_ylabel('Fraction Stationary')
    ax.set_title('Stationarity Test Results')
    ax.set_ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'stationarity_by_celltype.png'),
                dpi=200, bbox_inches='tight')
    plt.close()

    # --- E: Serial correlation lag 1 ---
    fig, ax = plt.subplots(figsize=(7, 4))
    data_sc = [df[df['cell_type'] == ct]['serial_corr_lag1'].dropna().values
               for ct in valid_types]
    bp = ax.boxplot(data_sc, labels=[ct.replace(' ', '\n') for ct in valid_types],
                    patch_artist=True, widths=0.5)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.axhline(0, color='gray', ls='--', lw=0.8)
    ax.set_ylabel('Serial Correlation (lag 1)')
    ax.set_title('Serial Correlation by Cell Type')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'serial_corr_by_celltype.png'),
                dpi=200, bbox_inches='tight')
    plt.close()

    # --- F: Statistical comparisons table ---
    print("\n=== Statistical Comparisons (LvR) ===")
    if len(valid_types) >= 2:
        for i, ct1 in enumerate(valid_types):
            for ct2 in valid_types[i+1:]:
                v1 = df[df['cell_type'] == ct1]['LvR'].dropna()
                v2 = df[df['cell_type'] == ct2]['LvR'].dropna()
                if len(v1) >= 3 and len(v2) >= 3:
                    u_stat, p_val = stats.mannwhitneyu(v1, v2,
                                                       alternative='two-sided')
                    print(f"  {ct1} vs {ct2}: U={u_stat:.0f}, "
                          f"p={p_val:.4f}, n=({len(v1)},{len(v2)})")

    print("\n=== Statistical Comparisons (CV) ===")
    if len(valid_types) >= 2:
        for i, ct1 in enumerate(valid_types):
            for ct2 in valid_types[i+1:]:
                v1 = df[df['cell_type'] == ct1]['cv'].dropna()
                v2 = df[df['cell_type'] == ct2]['cv'].dropna()
                if len(v1) >= 3 and len(v2) >= 3:
                    u_stat, p_val = stats.mannwhitneyu(v1, v2,
                                                       alternative='two-sided')
                    print(f"  {ct1} vs {ct2}: U={u_stat:.0f}, "
                          f"p={p_val:.4f}, n=({len(v1)},{len(v2)})")


# %% ============================================================
# MAIN EXECUTION
# ===============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("SPIKE TRAIN ANALYSIS — Burstiness Characterization")
    print("=" * 60)

    # Step 1: Discover trials
    print("\n--- Discovering trials ---")
    recordings = discover_trials(EXPERIMENTS)

    # Step 2: Load and filter
    print("\n--- Loading and filtering trials (>20 min) ---")
    trial_data = load_and_filter_trials(recordings)

    if not trial_data:
        print("No qualifying trials found. Exiting.")
        raise SystemExit

    # Step 3: Run all analyses
    print("\n--- Running analyses ---")
    df = analyze_all(trial_data)

    # Step 4: Save results CSV
    csv_path = os.path.join(FIG_DIR, 'spike_train_metrics.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    # Step 5: Summary statistics
    print("\n--- Summary Statistics ---")
    key_cols = ['cv', 'Lv', 'LvR', 'burst_index_royer',
                'frac_spikes_in_bursts', 'serial_corr_lag1', 'mean_rate_hz']
    for ct in CELL_ORDER:
        sub = df[df['cell_type'] == ct]
        if len(sub) == 0:
            continue
        print(f"\n  {ct} (n={len(sub)}):")
        for col in key_cols:
            vals = sub[col].dropna()
            if len(vals) > 0:
                print(f"    {col:30s}: {vals.mean():.3f} ± {vals.std():.3f} "
                      f"(median={vals.median():.3f})")

    # Step 6: Summary plots
    print("\n--- Generating summary plots ---")
    plot_summary_by_cell_type(df, FIG_DIR)

    # Step 7: Example gallery
    print("\n--- Generating example neuron gallery ---")
    for ct in CELL_ORDER:
        sub_df = df[df['cell_type'] == ct]
        if len(sub_df) == 0:
            continue

        # Pick N_EXAMPLE_NEURONS spanning the LvR range
        sub_sorted = sub_df.sort_values('LvR')
        n_pick = min(N_EXAMPLE_NEURONS, len(sub_sorted))
        indices = np.linspace(0, len(sub_sorted) - 1, n_pick, dtype=int)
        examples = sub_sorted.iloc[indices]

        for _, row in examples.iterrows():
            # Find corresponding trial_data entry
            td_match = [td for td in trial_data
                        if td['neuron_id'] == row['neuron_id']
                        and td['trial'] == row['trial']]
            if not td_match:
                continue
            td = td_match[0]
            safe_ct = ct.replace(' ', '_')
            fname = (f"gallery/{safe_ct}_{td['exp']}_M{td['mouse']}_"
                     f"d{td['day']}_n{td['neuron_idx']}_T{td['trial']}.png")
            plot_example_neuron(td, row, os.path.join(FIG_DIR, fname))

    print(f"\n--- Done! All outputs saved to {FIG_DIR} ---")
