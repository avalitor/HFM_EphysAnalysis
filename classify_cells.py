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

recordings = discover_recordings(cg.PROCESSED_FILE_DIR, ["2024-02-15", "2024-11-09", "2025-01-21"])

def load_cell_data(recordings):
    seen_days = set()
    all_tp = []
    all_bi = []
    all_tau_rise = []
    all_cell_labels = []
    cell_source = []

    for exp, mouse, trial in recordings:
        edata = elib.EphysTrial.Load(exp, mouse, trial, load_cell_metrics=True)
        day_key = (mouse, edata.day)

        if day_key in seen_days:
            continue

        metrics = edata.cell_metrics
        tp = np.array(metrics["troughToPeak"]).flatten()
        bi = np.array(metrics["burstIndex_Royer2012"]).flatten()
        tau = np.array(metrics["acg_tau_rise"]).flatten()

        cl = np.array(edata.cellLabels).flatten()
        cl = np.array([c[0] if isinstance(c, (list, np.ndarray)) else c for c in cl])

        n_cells = len(tp)
        all_tp.append(tp)
        all_bi.append(bi)
        all_tau_rise.append(tau)
        all_cell_labels.append(cl)
        cell_source.extend([(exp, mouse, edata.day, j) for j in range(n_cells)])

        seen_days.add(day_key)
        print(f"Loaded {len(tp)} cells from mouse {mouse}, day {edata.day}")

    all_tp = np.concatenate(all_tp)
    all_bi = np.concatenate(all_bi)
    all_tau_rise = np.concatenate(all_tau_rise)
    all_cell_labels = np.concatenate(all_cell_labels)

    print(f"\nTotal unique cells: {len(all_tp)}")
    return all_tp, all_bi, all_tau_rise, all_cell_labels, cell_source


tp_values, bi_values, tau_rise_values, cell_labels, sources = load_cell_data(recordings)

# --- Quick sanity check ---
print(f"TP range: {tp_values.min():.4f} - {tp_values.max():.4f}")
print(f"BI range: {bi_values.min():.4f} - {bi_values.max():.4f}")


#%%
# --- Classify cells ---
labels = np.empty(len(tp_values), dtype=object)

labels[(tp_values < 0.425) & (bi_values < 1.8)] = "Narrow Interneuron"
labels[(tp_values >= 0.425) & (bi_values < 1.8)] = "Wide Interneuron"
labels[(tp_values >= 0.425) & (bi_values >= 1.8)] = "Excitatory Principal Cell"
labels[(tp_values < 0.425) & (bi_values >= 1.8)] = "Bursty Narrow Interneuron"

#%%
# --- Scatter Plot ---
color_map = {
    "Narrow Interneuron": "blue",
    "Wide Interneuron": "cyan",
    "Excitatory Principal Cell": "red",
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
# plt.savefig(cg.ROOT_DIR+'/figs/ScatterClassification.png', dpi=600, bbox_inches='tight', pad_inches = 0)

plt.show()

#%%
# --- Plot Cell Explorer labels ---
ce_color_map = {
    "Narrow Interneuron": "blue",
    "Wide Interneuron": "cyan",
    "Pyramidal Cell": "red",
}

fig, ax = plt.subplots(figsize=(8, 6))

for label, color in ce_color_map.items():
    mask = cell_labels == label
    if mask.any():
        ax.scatter(tp_values[mask], bi_values[mask],
                   c=color, label=label, s=20, alpha=0.8)

# Catch any labels not in the map
known = set(ce_color_map.keys())
unknown = set(cell_labels) - known
if unknown:
    mask = np.isin(cell_labels, list(unknown))
    ax.scatter(tp_values[mask], bi_values[mask],
               c="gray", label="Other", s=20, alpha=0.8)

ax.axvline(x=0.425, color="gray", linestyle="--", linewidth=0.8)
ax.axhline(y=1.8, color="gray", linestyle="--", linewidth=0.8)

ax.set_yscale("log")
ax.set_xlabel("Trough-to-Peak")
ax.set_ylabel("Burst Index (log scale)")
ax.legend(loc="upper left")
ax.set_title("Cell Explorer Labels")

plt.tight_layout()
plt.savefig(cg.ROOT_DIR+'/figs/ScatterClassification_CellExplorerLabels.png', dpi=600, bbox_inches='tight', pad_inches = 0)
plt.show()

#%%

# Find disagreements: we say Wide Interneuron, Cell Explorer says Pyramidal Cell
mask = (labels == "Wide Interneuron") & (cell_labels == "Pyramidal Cell")
indices = np.where(mask)[0]

print(f"Found {len(indices)} disagreements (Wide IN vs Pyramidal Cell):\n")
for i in indices:
    exp, mouse, day, neuron_idx = sources[i]
    print(f"  Neuron {neuron_idx} | exp: {exp}, mouse: {mouse}, day: {day} | "
          f"TP: {tp_values[i]:.4f}, BI: {bi_values[i]:.4f}")
    
#%%
# --- Plot: acg_tau_rise vs TP, colored by Cell Explorer labels ---
ce_color_map = {
    "Narrow Interneuron": "blue",
    "Wide Interneuron": "cyan",
    "Pyramidal Cell": "red",
}

fig, ax = plt.subplots(figsize=(8, 6))

for label, color in ce_color_map.items():
    mask = cell_labels == label
    if mask.any():
        ax.scatter(tp_values[mask], tau_rise_values[mask],
                   c=color, label=label, s=20, alpha=0.8)

known = set(ce_color_map.keys())
unknown = set(cell_labels) - known
if unknown:
    mask = np.isin(cell_labels, list(unknown))
    ax.scatter(tp_values[mask], tau_rise_values[mask],
               c="gray", label="Other", s=20, alpha=0.8)

ax.set_xlabel("Trough-to-Peak")
ax.set_ylabel("ACG Tau Rise")
ax.legend(loc="upper left")
ax.set_title("Cell Explorer Labels: ACG Tau Rise vs TP")

plt.tight_layout()
plt.savefig(cg.ROOT_DIR+'/figs/ScatterClassification_ACGTauRise.png', dpi=600, bbox_inches='tight', pad_inches = 0)

plt.show()
#%% identify the excitatory cells
mask = labels == "Excitatory Principal Cell"
indices = np.where(mask)[0]

print(f"Found {len(indices)} Excitatory Principal Cells:\n")
for i in indices:
    exp, mouse, day, neuron_idx = sources[i]
    print(f"  Neuron {neuron_idx} | exp: {exp}, mouse: {mouse}, day: {day} | "
          f"TP: {tp_values[i]:.4f}, BI: {bi_values[i]:.4f}")
    
#%% save labels to all files
def save_labels_to_mat(recordings, our_labels, sources):
    day_labels = {}
    for i, (s_exp, s_mouse, s_day, s_idx) in enumerate(sources):
        key = (s_mouse, s_day)
        if key not in day_labels:
            day_labels[key] = []
        day_labels[key].append((s_idx, our_labels[i]))
    
    for key in day_labels:
        day_labels[key].sort(key=lambda x: x[0])
        day_labels[key] = np.array(
            [np.array([lbl]) for _, lbl in day_labels[key]], dtype=object
        ).reshape(1, -1)
    
    for exp, mouse, trial in recordings:
        edata = elib.EphysTrial.Load(exp, mouse, trial)
        day_key = (mouse, edata.day)
        
        edata.cellLabels_BITP = day_labels[day_key]
        edata.Update()
        
        print(f"Saved labels to {edata.filename}")

save_labels_to_mat(recordings, labels, sources)

#%% Load one file to verify

exp = '2023-10-16'
mouse = "101"
trial = '14'


edata = elib.EphysTrial.Load(exp, mouse, trial, load_cell_metrics=False)

#%% print how many cells of each type we have
unique, counts = np.unique(labels, return_counts=True)
print("Cell counts by BI/TP classification:")
for label, count in zip(unique, counts):
    print(f"  {label}: {count}")
print(f"  Total: {len(labels)}")
