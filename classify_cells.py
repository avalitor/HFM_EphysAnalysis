# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 16:29:28 2026

Cell classification - uses buzaki's classification scheme to determine whether a cell is excitatory or inhibitory
then uses principle component analysis to determine if excitatory cells are mossy or pyramidal

@author: Kelly
"""

import lib_ephys_obj as elib
import config as cg
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import glob

def discover_recordings(data_root, experiments):
    recordings = []
    
    for exp in experiments:
        exp_folder = os.path.join(data_root, exp)
        for filepath in glob.glob(os.path.join(exp_folder, "hfmE_*.mat")):
            filename = os.path.basename(filepath)
            match = re.match(r"hfmE_\d{4}-\d{2}-\d{2}_M(\d+)_(.+)\.mat", filename)
            if match:
                mouse = match.group(1)
                trial = match.group(2)
                recordings.append((exp, mouse, trial))
        
    print(f"Found {len(recordings)} recordings across {len(experiments)} experiments")
    return recordings

recordings = discover_recordings(cg.PROCESSED_FILE_DIR, ["2024-11-09", "2025-01-21"])

def load_cell_data(recordings):
    seen_days = set()
    all_tp = []
    all_bi = []
    cell_source = []

    for exp, mouse, trial in recordings:
        edata = elib.EphysTrial.Load(exp, mouse, trial)
        day_key = (mouse, edata.day)

        if day_key in seen_days:
            continue

        metrics = edata.cell_metrics
        tp = np.array(metrics["troughToPeak"]).flatten()
        bi = np.array(metrics["burstIndex_Royer2012"]).flatten()

        all_tp.append(tp)
        all_bi.append(bi)
        cell_source.extend([(exp, mouse, edata.day)] * len(tp))

        seen_days.add(day_key)
        print(f"Loaded {len(tp)} cells from mouse {mouse}, day {edata.day}")

    all_tp = np.concatenate(all_tp)
    all_bi = np.concatenate(all_bi)

    print(f"\nTotal unique cells: {len(all_tp)}")
    return all_tp, all_bi, cell_source


tp_values, bi_values, sources = load_cell_data(recordings)

# --- Quick sanity check ---
print(f"TP range: {tp_values.min():.4f} - {tp_values.max():.4f}")
print(f"BI range: {bi_values.min():.4f} - {bi_values.max():.4f}")


#%%

# --- Classify cells ---
labels = np.empty(len(tp_values), dtype=object)

labels[(tp_values < 0.425) & (bi_values < 1.8)] = "Narrow Interneuron"
labels[(tp_values >= 0.425) & (bi_values < 1.8)] = "Wide Interneuron"
labels[(tp_values >= 0.425) & (bi_values >= 1.8)] = "Excitatory Principal Cells"
labels[(tp_values < 0.425) & (bi_values >= 1.8)] = "Bursty_Narrow"

# --- Scatter Plot ---
color_map = {
    "Narrow Interneuron": "blue",
    "Wide Interneuron": "cyan",
    "Excitatory Principal Cells": "red",
    "Bursty Narrow Interneuron": "magenta",
}

fig, ax = plt.subplots(figsize=(8, 6))

for cell_type, color in color_map.items():
    mask = labels == cell_type
    ax.scatter(tp_values[mask], bi_values[mask],
               c=color, label=cell_type, s=20, alpha=0.8)

# Quadrant dividers
ax.axvline(x=0.425, color="gray", linestyle="--", linewidth=0.8)
ax.axhline(y=1.8, color="gray", linestyle="--", linewidth=0.8)

ax.set_yscale("log")
ax.set_xlabel("Trough-to-Peak")
ax.set_ylabel("Burst Index (log scale)")
ax.legend(loc="upper left")
ax.set_title("Cell Classification by TP and BI")

plt.tight_layout()
plt.show()