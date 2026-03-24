# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 16:29:28 2026

Cell classification - uses buzaki's classification scheme to determine whether a cell is excitatory or inhibitory
then uses principle component analysis to determine if excitatory cells are mossy or pyramidal

@author: Kelly
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"

import lib_ephys_obj as elib
import config as cg
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import re
import glob
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.ndimage import gaussian_filter1d

# =============================================================================
# Global color palette
# =============================================================================
CELL_COLORS = {
    "Granule Cell": "#c00021",
    "Mossy Cell": "#358940",
    "Narrow Interneuron": "#0050a0",
    "Wide Interneuron": "#07CCC3",
    "Excitatory Principal Cell": "#e36414",
    "Bursty Narrow Interneuron": "#87147A",
}

# =============================================================================
# Discover and load recordings
# =============================================================================
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

print(f"TP range: {tp_values.min():.4f} - {tp_values.max():.4f}")
print(f"BI range: {bi_values.min():.4f} - {bi_values.max():.4f}")

# =============================================================================
# BI/TP Classification
# =============================================================================
our_labels = np.empty(len(tp_values), dtype=object)
our_labels[(tp_values < 0.425) & (bi_values < 1.8)] = "Narrow Interneuron"
our_labels[(tp_values >= 0.425) & (bi_values < 1.8)] = "Wide Interneuron"
our_labels[(tp_values >= 0.425) & (bi_values >= 1.8)] = "Excitatory Principal Cell"
our_labels[(tp_values < 0.425) & (bi_values >= 1.8)] = "Bursty Narrow Interneuron"

# =============================================================================
# Scatter Plot - BI/TP classification
# =============================================================================
fig, ax = plt.subplots(figsize=(8, 6))
for cell_type in ["Narrow Interneuron", "Wide Interneuron", "Excitatory Principal Cell", "Bursty Narrow Interneuron"]:
    mask = our_labels == cell_type
    ax.scatter(tp_values[mask], bi_values[mask],
               c=CELL_COLORS[cell_type], label=cell_type, s=20, alpha=0.8)
ax.axvline(x=0.425, color="gray", linestyle="--", linewidth=0.8)
ax.axhline(y=1.8, color="gray", linestyle="--", linewidth=0.8)
ax.set_yscale("log")
ax.set_xlabel("Trough-to-Peak")
ax.set_ylabel("Burst Index (log scale)")
ax.legend(loc="upper left")
ax.set_title("Cell Classification by TP and BI")
plt.tight_layout()
fig.savefig(os.path.join(cg.ROOT_DIR, 'figs', 'ScatterClassification.png'), dpi=600, bbox_inches='tight', pad_inches=0)
plt.close()

# =============================================================================
# Scatter Plot - Cell Explorer labels
# =============================================================================
ce_color_map = {
    "Narrow Interneuron": CELL_COLORS["Narrow Interneuron"],
    "Wide Interneuron": CELL_COLORS["Wide Interneuron"],
    "Pyramidal Cell": CELL_COLORS["Excitatory Principal Cell"],
}

fig, ax = plt.subplots(figsize=(8, 6))
for label, color in ce_color_map.items():
    mask = cell_labels == label
    if mask.any():
        ax.scatter(tp_values[mask], bi_values[mask], c=color, label=label, s=20, alpha=0.8)
known = set(ce_color_map.keys())
unknown = set(cell_labels) - known
if unknown:
    mask = np.isin(cell_labels, list(unknown))
    ax.scatter(tp_values[mask], bi_values[mask], c="gray", label="Other", s=20, alpha=0.8)
ax.axvline(x=0.425, color="gray", linestyle="--", linewidth=0.8)
ax.axhline(y=1.8, color="gray", linestyle="--", linewidth=0.8)
ax.set_yscale("log")
ax.set_xlabel("Trough-to-Peak")
ax.set_ylabel("Burst Index (log scale)")
ax.legend(loc="upper left")
ax.set_title("Cell Explorer Labels")
plt.tight_layout()
fig.savefig(os.path.join(cg.ROOT_DIR, 'figs', 'ScatterClassification_CellExplorerLabels.png'), dpi=600, bbox_inches='tight', pad_inches=0)
plt.close()

# =============================================================================
# Disagreements: Wide IN (ours) vs Pyramidal (Cell Explorer)
# =============================================================================
mask = (our_labels == "Wide Interneuron") & (cell_labels == "Pyramidal Cell")
indices = np.where(mask)[0]
print(f"\nFound {len(indices)} disagreements (Wide IN vs Pyramidal Cell):\n")
for i in indices:
    exp, mouse, day, neuron_idx = sources[i]
    print(f"  Neuron {neuron_idx} | exp: {exp}, mouse: {mouse}, day: {day} | "
          f"TP: {tp_values[i]:.4f}, BI: {bi_values[i]:.4f}")

# =============================================================================
# ACG Tau Rise vs TP - Cell Explorer labels
# =============================================================================
fig, ax = plt.subplots(figsize=(8, 6))
for label, color in ce_color_map.items():
    mask = cell_labels == label
    if mask.any():
        ax.scatter(tp_values[mask], tau_rise_values[mask], c=color, label=label, s=20, alpha=0.8)
if unknown:
    mask = np.isin(cell_labels, list(unknown))
    ax.scatter(tp_values[mask], tau_rise_values[mask], c="gray", label="Other", s=20, alpha=0.8)
ax.set_xlabel("Trough-to-Peak")
ax.set_ylabel("ACG Tau Rise")
ax.legend(loc="upper left")
ax.set_title("Cell Explorer Labels: ACG Tau Rise vs TP")
plt.tight_layout()
fig.savefig(os.path.join(cg.ROOT_DIR, 'figs', 'ScatterClassification_ACGTauRise.png'), dpi=600, bbox_inches='tight', pad_inches=0)
plt.close()

# =============================================================================
# Identify excitatory cells
# =============================================================================
mask = our_labels == "Excitatory Principal Cell"
indices = np.where(mask)[0]
print(f"\nFound {len(indices)} Excitatory Principal Cells:\n")
for i in indices:
    exp, mouse, day, neuron_idx = sources[i]
    print(f"  Neuron {neuron_idx} | exp: {exp}, mouse: {mouse}, day: {day} | "
          f"TP: {tp_values[i]:.4f}, BI: {bi_values[i]:.4f}")

# =============================================================================
# Save BI/TP labels to mat files
# =============================================================================
def save_labels_to_mat(recordings, label_array, sources):
    day_labels = {}
    for i, (s_exp, s_mouse, s_day, s_idx) in enumerate(sources):
        key = (s_mouse, s_day)
        if key not in day_labels:
            day_labels[key] = []
        day_labels[key].append((s_idx, label_array[i]))
    for key in day_labels:
        day_labels[key].sort(key=lambda x: x[0])
        day_labels[key] = np.array(
            [np.array([lbl]) for _, lbl in day_labels[key]], dtype=object
        ).reshape(1, -1)
    for exp, mouse, trial in recordings:
        edata = elib.EphysTrial.Load(exp, mouse, trial, load_cell_metrics=True)
        day_key = (mouse, edata.day)
        edata.cellLabels_BITP = day_labels[day_key]
        edata.Update()
        print(f"Saved labels to {edata.filename}")

# save_labels_to_mat(recordings, our_labels, sources)  # uncomment to run

# =============================================================================
# Cell count summary
# =============================================================================
unique, counts = np.unique(our_labels, return_counts=True)
print("\nCell counts by BI/TP classification:")
for label, count in zip(unique, counts):
    print(f"  {label}: {count}")
print(f"  Total: {len(our_labels)}")

# =============================================================================
# Load excitatory waveforms for PCA
# =============================================================================
def load_excitatory_waveforms(recordings, label_array, sources):
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
        for i, (s_exp, s_mouse, s_day, s_idx) in enumerate(sources):
            if s_mouse == mouse and s_day == edata.day and label_array[i] == "Excitatory Principal Cell":
                W_cell = np.array(wf_filt[s_idx, 0])
                all_waveforms.append(W_cell)
                all_firing_rates.append(fr[s_idx])
                exc_sources.append(sources[i])
        seen_days.add(day_key)
    W = np.vstack(all_waveforms)
    firing_rate = np.array(all_firing_rates)
    print(f"\nLoaded {W.shape[0]} excitatory cell waveforms, shape: {W.shape}")
    return W, firing_rate, exc_sources

W_all, fr_all, exc_sources_all = load_excitatory_waveforms(recordings, our_labels, sources)
W_all = W_all.astype(np.float64)

# =============================================================================
# PCA helper function
# =============================================================================
def run_pca_analysis(W, firing_rate, exc_sources, max_clusters=5):
    W2 = np.diff(np.diff(W, axis=1), axis=1)
    ok = np.isfinite(W2).all(axis=1)
    if not np.all(ok):
        print(f"Dropping {np.sum(~ok)} cells with non-finite values.")
        W = W[ok]
        W2 = W2[ok]
        firing_rate = firing_rate[ok]
        exc_sources = [exc_sources[i] for i in range(len(ok)) if ok[i]]

    pca = PCA(n_components=2, svd_solver='full', random_state=42)
    X_pca = pca.fit_transform(W2)
    print("Explained variance ratio:", np.round(pca.explained_variance_ratio_, 4))
    print("Cumulative:", np.round(pca.explained_variance_ratio_.cumsum(), 4))

    sil_scores = []
    ks = []
    for k in range(2, min(max_clusters + 1, X_pca.shape[0])):
        km = KMeans(n_clusters=k, random_state=42, n_init='auto')
        kl = km.fit_predict(X_pca)
        if len(np.unique(kl)) > 1:
            score = silhouette_score(X_pca, kl)
            sil_scores.append(score)
            ks.append(k)

    optimal_k = ks[int(np.argmax(sil_scores))]
    print(f"Optimal k: {optimal_k} (silhouette={max(sil_scores):.3f})")

    km_final = KMeans(n_clusters=optimal_k, random_state=42, n_init='auto')
    cluster_labels = km_final.fit_predict(X_pca)

    return W, W2, firing_rate, exc_sources, X_pca, cluster_labels, ks, sil_scores, optimal_k

# =============================================================================
# PCA Figure helper
# =============================================================================
def plot_pca_figure(W, W2, firing_rate, X_pca, cluster_labels, ks, sil_scores,
                    label_map, color_map, save_path):
    unique_labels = list(label_map.values())
    mapped_labels = np.array([label_map[c] for c in cluster_labels])

    fig = plt.figure(figsize=(20, 15))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.25], hspace=0.35, wspace=0.3)
    ax_sil = fig.add_subplot(gs[0, 0])
    ax_pca = fig.add_subplot(gs[0, 1])
    ax_fr = fig.add_subplot(gs[0, 2])
    bottom = gs[1, :].subgridspec(1, 2, wspace=0.25)
    ax_d2 = fig.add_subplot(bottom[0, 0])
    ax_raw = fig.add_subplot(bottom[0, 1])

    # Silhouette
    ax_sil.plot(ks, sil_scores, marker='o')
    ax_sil.set_xlabel('Number of clusters', fontsize=15)
    ax_sil.set_ylabel('Silhouette Score', fontsize=15)
    ax_sil.set_title('Silhouette Score vs k', fontsize=15)
    ax_sil.tick_params(labelsize=13)

    # PCA scatter
    for label in unique_labels:
        mask = mapped_labels == label
        ax_pca.scatter(X_pca[mask, 0], X_pca[mask, 1],
                       s=35, alpha=0.8, color=color_map[label],
                       label=label, edgecolor='none')
    ax_pca.set_xlabel('PC1', fontsize=15)
    ax_pca.set_ylabel('PC2', fontsize=15)
    ax_pca.set_title('PCA of 2nd Derivative', fontsize=15)
    ax_pca.legend(fontsize=12)
    ax_pca.tick_params(labelsize=13)

    # Firing rate
    rng = np.random.default_rng(123)
    for i, label in enumerate(unique_labels):
        mask = mapped_labels == label
        fr_vals = firing_rate[mask]
        x = rng.normal(i, 0.08, size=fr_vals.size)
        ax_fr.scatter(x, fr_vals, alpha=0.65, color=color_map[label], label=label)
    ax_fr.set_xlabel('Cell Type', fontsize=15)
    ax_fr.set_ylabel('Firing Rate (Hz)', fontsize=15)
    ax_fr.set_title('Firing Rates by Cell Type', fontsize=15)
    ax_fr.set_xticks(range(len(unique_labels)))
    ax_fr.set_xticklabels(unique_labels, fontsize=13)
    ax_fr.tick_params(labelsize=13)

    # 2nd derivative
    t2 = np.arange(W2.shape[1])
    for label in unique_labels:
        mask = mapped_labels == label
        avg = W2[mask].mean(axis=0)
        std = W2[mask].std(axis=0)
        ax_d2.plot(t2, avg, color=color_map[label], linewidth=2, label=label)
        ax_d2.fill_between(t2, avg - std, avg + std, color=color_map[label], alpha=0.2)
    ax_d2.set_xlabel('Time', fontsize=15)
    ax_d2.set_ylabel('Amplitude', fontsize=15)
    ax_d2.set_title('Average 2nd Derivative by Cell Type', fontsize=15)
    ax_d2.legend(fontsize=14)
    ax_d2.tick_params(labelsize=13)

    # Raw waveforms
    t_raw = np.arange(W.shape[1])
    for label in unique_labels:
        mask = mapped_labels == label
        avg = W[mask].mean(axis=0)
        std = W[mask].std(axis=0)
        ax_raw.plot(t_raw, avg, color=color_map[label], linewidth=2, label=label)
        ax_raw.fill_between(t_raw, avg - std, avg + std, color=color_map[label], alpha=0.2)
    ax_raw.set_xlabel('Time', fontsize=15)
    ax_raw.set_ylabel('Amplitude', fontsize=15)
    ax_raw.set_title('Average Waveform by Cell Type', fontsize=15)
    ax_raw.legend(fontsize=14)
    ax_raw.tick_params(labelsize=13)

    plt.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close()

    # Summary
    print(f"\nSaved: {save_path}")
    for label in unique_labels:
        mask = mapped_labels == label
        fr_vals = firing_rate[mask]
        print(f"  {label}: {mask.sum()} cells, FR = {np.mean(fr_vals):.2f} ± {np.std(fr_vals):.2f} Hz")

# =============================================================================
# PCA Pass 1: All excitatory cells (no outlier removal)
# =============================================================================
print("\n" + "="*60)
print("PCA Pass 1: All excitatory cells")
print("="*60)

W1, W2_1, fr1, src1, Xpca1, cl1, ks1, sil1, optk1 = run_pca_analysis(
    W_all.copy(), fr_all.copy(), list(exc_sources_all))

# Default viridis-style colors for the initial pass
palette1 = plt.cm.viridis(np.linspace(0, 1, optk1))
label_map1 = {i: f"Cluster {i+1}" for i in range(optk1)}
color_map1 = {f"Cluster {i+1}": palette1[i] for i in range(optk1)}

plot_pca_figure(W1, W2_1, fr1, Xpca1, cl1, ks1, sil1,
                label_map1, color_map1,
                os.path.join(cg.ROOT_DIR, 'figs', 'excitatory_pca_all_cells.png'))

# Print cluster membership for outlier identification
for c in range(optk1):
    mask = cl1 == c
    print(f"\nCluster {c+1} ({mask.sum()} cells):")
    for i in np.where(mask)[0]:
        exp, mouse, day, neuron_idx = src1[i]
        print(f"  Neuron {neuron_idx} | exp: {exp}, mouse: {mouse}, day: {day}")

# =============================================================================
# PCA Pass 2: Outliers removed
# =============================================================================
print("\n" + "="*60)
print("PCA Pass 2: Outliers removed")
print("="*60)

exclude = {
    ("2024-02-15", "105", np.str_('4'), 14),
    ("2024-02-15", "105", np.str_('5'), 14),
    ("2024-02-15", "105", np.str_('7'), 6),
    ("2024-02-15", "105", np.str_('3'), 11),
}

keep = [i for i, s in enumerate(exc_sources_all) if tuple(s) not in exclude]
W_clean = W_all[keep].copy()
fr_clean = fr_all[keep].copy()
exc_sources_clean = [exc_sources_all[i] for i in keep]
print(f"Kept {W_clean.shape[0]} cells after excluding {len(exclude)} outliers")

W2c, W2_2c, fr2c, src2c, Xpca2, cl2, ks2, sil2, optk2 = run_pca_analysis(
    W_clean, fr_clean, exc_sources_clean)

# Force k=2 for granule vs mossy
km_final = KMeans(n_clusters=2, random_state=42, n_init='auto')
cl2 = km_final.fit_predict(Xpca2)

label_map2 = {0: "Granule Cell", 1: "Mossy Cell"}
color_map2 = {"Granule Cell": CELL_COLORS["Granule Cell"], "Mossy Cell": CELL_COLORS["Mossy Cell"]}

plot_pca_figure(W2c, W2_2c, fr2c, Xpca2, cl2, ks2, sil2,
                label_map2, color_map2,
                os.path.join(cg.ROOT_DIR, 'figs', 'excitatory_pca_classification.png'))

pca_labels = np.array([label_map2[c] for c in cl2])
exc_sources = src2c

# =============================================================================
# Save PCA labels to mat files
# =============================================================================
def save_pca_labels_to_mat(recordings, pca_labels, exc_sources):
    pca_lookup = {}
    for i, (exp, mouse, day, neuron_idx) in enumerate(exc_sources):
        pca_lookup[(mouse, day, neuron_idx)] = pca_labels[i]
    seen_days = set()
    day_info = {}
    for exp, mouse, trial in recordings:
        edata = elib.EphysTrial.Load(exp, mouse, trial, load_cell_metrics=True)
        day_key = (mouse, edata.day)
        if day_key in seen_days:
            continue
        n_cells = len(np.array(edata.cell_metrics["troughToPeak"]).flatten())
        label_array = np.array(
            [np.array([pca_lookup.get((mouse, edata.day, j), "")])
             for j in range(n_cells)],
            dtype=object
        ).reshape(1, -1)
        day_info[day_key] = label_array
        seen_days.add(day_key)
    for exp, mouse, trial in recordings:
        edata = elib.EphysTrial.Load(exp, mouse, trial, load_cell_metrics=True)
        day_key = (mouse, edata.day)
        edata.cellLabels_pca = day_info[day_key]
        edata.Update()
        print(f"Saved PCA labels to {edata.filename}")

# save_pca_labels_to_mat(recordings, pca_labels, exc_sources)  # uncomment to run

# =============================================================================
# Firing rate distribution by cell type
# =============================================================================
fig, ax = plt.subplots(figsize=(16, 6))

fr_by_type = {}
seen_days = set()
for exp, mouse, trial in recordings:
    edata = elib.EphysTrial.Load(exp, mouse, trial)
    day_key = (mouse, edata.day)
    if day_key in seen_days:
        continue

    fr = np.array(edata.firingRate).flatten()
    bitp = np.array(edata.cellLabels_BITP).flatten()
    bitp = np.array([b[0] if isinstance(b, (list, np.ndarray)) else b for b in bitp])

    has_pca = hasattr(edata, 'cellLabels_pca')
    if has_pca:
        pca_lab = np.array(edata.cellLabels_pca).flatten()
        pca_lab = np.array([p[0] if isinstance(p, (list, np.ndarray)) and len(p) > 0 else "" for p in pca_lab])

    for j in range(len(fr)):
        if bitp[j] == "Excitatory Principal Cell" and has_pca and pca_lab[j] != "":
            label = pca_lab[j]
        elif bitp[j] == "Excitatory Principal Cell":
            continue
        else:
            label = bitp[j]
        if label not in fr_by_type:
            fr_by_type[label] = []
        fr_by_type[label].append(fr[j])

    seen_days.add(day_key)

for key in fr_by_type:
    fr_by_type[key] = np.array(fr_by_type[key])

plot_order = ["Granule Cell", "Mossy Cell", "Wide Interneuron", "Narrow Interneuron"]
x_max = 60
bins = np.arange(0, x_max + 1, 1)

for label in plot_order:
    if label in fr_by_type:
        vals = fr_by_type[label]
        vals = vals[vals <= x_max]
        ax.hist(vals, bins=bins, density=True, alpha=0.3, color=CELL_COLORS[label])
        counts, _ = np.histogram(vals, bins=bins, density=True)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        smoothed = gaussian_filter1d(counts, sigma=1.5)
        ax.plot(bin_centers, smoothed, color=CELL_COLORS[label], linewidth=2,
                label=f"{label} (n={len(fr_by_type[label])})")

ax.set_xlabel('Firing Rate (Hz)', fontsize=15)
ax.set_ylabel('Density', fontsize=15)
ax.set_title('Distribution of Firing Rates by Cell Type', fontsize=15)
ax.legend(fontsize=12)
ax.tick_params(labelsize=13)
ax.set_xlim(0, x_max)

plt.tight_layout()
fig.savefig(os.path.join(cg.ROOT_DIR, 'figs', 'firing_rate_distributions.png'), dpi=200)
plt.close()