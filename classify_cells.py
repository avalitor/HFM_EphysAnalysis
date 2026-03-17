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
        edata = elib.EphysTrial.Load(exp, mouse, trial, True)
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

#%%
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import config as cg

# --- Collect waveforms and firing rates for excitatory cells ---
def load_excitatory_waveforms(recordings, our_labels, sources):
    """Load waveforms for cells classified as Excitatory Principal Cells."""
    seen_days = set()
    all_waveforms = []
    all_firing_rates = []
    exc_sources = []

    for exp, mouse, trial in recordings:
        edata = elib.EphysTrial.Load(exp, mouse, trial, load_cell_metrics=True)
        day_key = (mouse, edata.day)

        if day_key in seen_days:
            continue

        wf_filt = edata.cell_metrics["waveforms"]["filt"]
        fr = np.array(edata.cell_metrics["firingRate"]).flatten()

        # Find excitatory cells from this day
        for i, (s_exp, s_mouse, s_day, s_idx) in enumerate(sources):
            if s_mouse == mouse and s_day == edata.day and our_labels[i] == "Excitatory Principal Cell":
                W_cell = np.array(wf_filt[s_idx, 0])
                all_waveforms.append(W_cell)
                all_firing_rates.append(fr[s_idx])
                exc_sources.append(sources[i])

        seen_days.add(day_key)

    W = np.vstack(all_waveforms)
    firing_rate = np.array(all_firing_rates)
    print(f"Loaded {W.shape[0]} excitatory cell waveforms, shape: {W.shape}")
    return W, firing_rate, exc_sources


W, firing_rate, exc_sources = load_excitatory_waveforms(recordings, labels, sources)
W = W.astype(np.float64)

# --- 2nd derivative ---
W2 = np.diff(np.diff(W, axis=1), axis=1)

# Drop non-finite rows
ok = np.isfinite(W2).all(axis=1)
if not np.all(ok):
    print(f"Dropping {np.sum(~ok)} cells with non-finite values.")
    W, W2, firing_rate = W[ok], W2[ok], firing_rate[ok]
    exc_sources = [exc_sources[i] for i in range(len(ok)) if ok[i]]

# --- PCA ---
pca = PCA(n_components=2, svd_solver='full', random_state=42)
X_pca = pca.fit_transform(W2)
print("Explained variance ratio:", np.round(pca.explained_variance_ratio_, 4))
print("Cumulative:", np.round(pca.explained_variance_ratio_.cumsum(), 4))

# --- Silhouette scores for k = 2..5 ---
max_clusters = 5
silhouette_scores = []
ks = []
for k in range(2, min(max_clusters + 1, X_pca.shape[0])):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(X_pca)
    if len(np.unique(labels)) > 1:
        score = silhouette_score(X_pca, labels)
        silhouette_scores.append(score)
        ks.append(k)

optimal_clusters = ks[int(np.argmax(silhouette_scores))]
print(f"Optimal k: {optimal_clusters} (silhouette={max(silhouette_scores):.3f})")
optimal_clusters = 2
# --- Final KMeans ---
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init='auto')
cluster_labels = kmeans.fit_predict(X_pca)

# --- Color mapping ---
unique_labels = np.sort(np.unique(cluster_labels))
K = unique_labels.size
palette = plt.cm.viridis(np.linspace(0, 1, K))
label_to_color = {lab: palette[i] for i, lab in enumerate(unique_labels)}

def display_name(lab):
    return f"Cluster {lab + 1}"

# --- Figure ---
fig = plt.figure(figsize=(20, 15))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.25], hspace=0.35, wspace=0.3)

ax_sil = fig.add_subplot(gs[0, 0])
ax_pca = fig.add_subplot(gs[0, 1])
ax_fr = fig.add_subplot(gs[0, 2])
bottom = gs[1, :].subgridspec(1, 2, wspace=0.25)
ax_d2 = fig.add_subplot(bottom[0, 0])
ax_raw = fig.add_subplot(bottom[0, 1])

# Silhouette
ax_sil.plot(ks, silhouette_scores, marker='o')
ax_sil.set_xlabel('Number of clusters', fontsize=15)
ax_sil.set_ylabel('Silhouette Score', fontsize=15)
ax_sil.set_title('Silhouette Score vs k', fontsize=15)
ax_sil.tick_params(labelsize=13)

# PCA scatter
for lab in unique_labels:
    mask = cluster_labels == lab
    ax_pca.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   s=35, alpha=0.8, color=label_to_color[lab],
                   label=display_name(lab), edgecolor='none')
ax_pca.set_xlabel('PC1', fontsize=15)
ax_pca.set_ylabel('PC2', fontsize=15)
ax_pca.set_title('PCA of 2nd Derivative', fontsize=15)
ax_pca.legend(fontsize=12)
ax_pca.tick_params(labelsize=13)

# Firing rate by cluster
rng = np.random.default_rng(123)
for lab in unique_labels:
    fr_vals = firing_rate[cluster_labels == lab]
    x = rng.normal(lab, 0.08, size=fr_vals.size)
    ax_fr.scatter(x, fr_vals, alpha=0.65, color=label_to_color[lab],
                  label=display_name(lab))
ax_fr.set_xlabel('Cluster', fontsize=15)
ax_fr.set_ylabel('Firing Rate (Hz)', fontsize=15)
ax_fr.set_title('Firing Rates by Cluster', fontsize=15)
ax_fr.set_xticks(range(K))
ax_fr.set_xticklabels([display_name(i) for i in range(K)], fontsize=13)
ax_fr.tick_params(labelsize=13)

# 2nd derivative waveforms
t2 = np.arange(W2.shape[1])
for lab in unique_labels:
    mask = cluster_labels == lab
    avg = W2[mask].mean(axis=0)
    std = W2[mask].std(axis=0)
    ax_d2.plot(t2, avg, color=label_to_color[lab], linewidth=2, label=display_name(lab))
    ax_d2.fill_between(t2, avg - std, avg + std, color=label_to_color[lab], alpha=0.2)
ax_d2.set_xlabel('Time', fontsize=15)
ax_d2.set_ylabel('Amplitude', fontsize=15)
ax_d2.set_title('Average 2nd Derivative by Cluster', fontsize=15)
ax_d2.legend(fontsize=14)
ax_d2.tick_params(labelsize=13)

# Raw waveforms
t_raw = np.arange(W.shape[1])
for lab in unique_labels:
    mask = cluster_labels == lab
    avg = W[mask].mean(axis=0)
    std = W[mask].std(axis=0)
    ax_raw.plot(t_raw, avg, color=label_to_color[lab], linewidth=2, label=display_name(lab))
    ax_raw.fill_between(t_raw, avg - std, avg + std, color=label_to_color[lab], alpha=0.2)
ax_raw.set_xlabel('Time', fontsize=15)
ax_raw.set_ylabel('Amplitude', fontsize=15)
ax_raw.set_title('Average Waveform by Cluster', fontsize=15)
ax_raw.legend(fontsize=14)
ax_raw.tick_params(labelsize=13)

plt.tight_layout()
fig.savefig(os.path.join(cg.ROOT_DIR, 'figs', 'excitatory_pca_classification.png'), dpi=200)
plt.close()

# --- Summary ---
print("\nCluster sizes:")
for lab in unique_labels:
    n = np.sum(cluster_labels == lab)
    fr_vals = firing_rate[cluster_labels == lab]
    print(f"  {display_name(lab)}: {n} cells, FR = {np.mean(fr_vals):.2f} ± {np.std(fr_vals):.2f} Hz")