# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 10:08:58 2026

@author: Kelly
"""

"""
Plot Trough-to-Peak vs LvR scatter, colored by original BITP classification.
Loads LvR from the spike_train_metrics.csv already generated,
pulls troughToPeak from cell_metrics, and uses cellLabels_BITP
(the original 4 categories) for coloring.
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

import lib_ephys_obj as elib
import config as cg

# %% === PARAMETERS ===
FIG_DIR = os.path.join(cg.ROOT_DIR, 'figs', 'burst_analysis2')
CSV_PATH = os.path.join(FIG_DIR, 'spike_train_metrics.csv')

# Original 4-category BITP colors and labels
BITP_COLORS = {
    # 'Excitatory Principal Cell':  '#e36414',
    # 'Narrow Interneuron':         '#0050a0',
    # 'Wide Interneuron':           '#07CCC3',
    # 'Bursty Narrow Interneuron':  '#87147A',
    'Excitatory Principal Cell':  'red',
    'Narrow Interneuron':         'blue',
    'Wide Interneuron':           'cyan',
    'Bursty Narrow Interneuron':  'magenta',
}
BITP_ORDER = ['Narrow Interneuron', 'Wide Interneuron',
              'Excitatory Principal Cell', 'Bursty Narrow Interneuron']

# Classification cutoffs (for reference lines)
TP_CUTOFF = 0.425
BI_CUTOFF = 1.8  # original Royer ACG-based cutoff (not used for axis)

# %% === LOAD LvR RESULTS ===
df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df)} rows from {CSV_PATH}")

# %% === LOAD TROUGH-TO-PEAK AND ORIGINAL BITP LABELS ===
# We need to pull TP and the original 4-category BITP label
# for each unique (exp, mouse, day, neuron_idx).

# Get unique days — only need to load cell_metrics once per day
unique_days = df[['exp', 'mouse', 'day', 'trial']].drop_duplicates(
    subset=['exp', 'mouse', 'day'])

tp_map = {}    # (exp, mouse, day, neuron_idx) -> troughToPeak
bitp_map = {}  # (exp, mouse, day, neuron_idx) -> original BITP label

for _, row in unique_days.iterrows():
    exp, mouse, day, trial = row['exp'], str(row['mouse']), str(row['day']), row['trial']
    day_key = (exp, mouse, day)

    try:
        edata = elib.EphysTrial.Load(exp, mouse, trial, load_cell_metrics=True)
    except Exception as e:
        print(f"  Failed to load cell_metrics for {day_key}: {e}")
        continue

    # Extract troughToPeak
    try:
        tp_arr = np.array(edata.cell_metrics['troughToPeak']).flatten()
    except (KeyError, TypeError):
        print(f"  No troughToPeak in cell_metrics for {day_key}")
        continue

    n_neurons = len(tp_arr)

    for ni in range(n_neurons):
        key = (exp, mouse, day, ni)
        tp_map[key] = tp_arr[ni]

        # Get original BITP label (4 categories, no PCA subdivision)
        try:
            bitp_map[key] = str(edata.cellLabels_BITP[ni][0])
        except (IndexError, TypeError):
            bitp_map[key] = 'Unknown'

print(f"Loaded TP for {len(tp_map)} neurons")

# %% === MERGE INTO DATAFRAME ===
df['troughToPeak'] = df.apply(
    lambda r: tp_map.get((r['exp'], str(r['mouse']), str(r['day']),
                          r['neuron_idx']), np.nan), axis=1)

df['bitp_label'] = df.apply(
    lambda r: bitp_map.get((r['exp'], str(r['mouse']), str(r['day']),
                            r['neuron_idx']), 'Unknown'), axis=1)

# Drop rows with missing TP
df_plot = df.dropna(subset=['troughToPeak', 'LvR']).copy()
print(f"Neurons with both TP and LvR: {len(df_plot)}")

for lbl in BITP_ORDER:
    n = (df_plot['bitp_label'] == lbl).sum()
    print(f"  {lbl}: {n}")

# %% === PLOT ===
fig, ax = plt.subplots(figsize=(8, 6))

for lbl in BITP_ORDER:
    mask = df_plot['bitp_label'] == lbl
    if mask.sum() == 0:
        continue
    ax.scatter(df_plot.loc[mask, 'troughToPeak'],
               df_plot.loc[mask, 'LvR'],
               c=BITP_COLORS[lbl], label=lbl,
               s=25, alpha=0.65, edgecolors='none', zorder=2)

# Reference lines
ax.axvline(TP_CUTOFF, color='gray', ls='--', lw=0.8, alpha=0.5,
           label=f'TP = {TP_CUTOFF} ms')
ax.axhline(1.0, color='gray', ls=':', lw=0.8, alpha=0.5,
           label='LvR = 1.0 (Poisson)')

ax.set_xlabel('Trough-to-Peak (ms)', fontsize=11)
ax.set_ylabel('LvR', fontsize=11)
ax.set_title('Trough-to-Peak vs LvR\n(colored by original BITP classification)',
             fontsize=12)
ax.legend(fontsize=8, loc='upper right', framealpha=0.8)

plt.tight_layout()
save_path = os.path.join(FIG_DIR, 'scatter_TP_vs_LvR_BITP.png')
plt.savefig(save_path, dpi=200, bbox_inches='tight')
plt.close()
print(f"\nSaved to {save_path}")