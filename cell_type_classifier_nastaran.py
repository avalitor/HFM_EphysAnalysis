#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import scipy.io as sio
import numpy as np
import mat73
import matplotlib.pyplot as plt
from pylab import *
from scipy.stats import gamma
from scipy import io
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt')
import pandas as pd
import seaborn as sns
import os 
import glob 
from tqdm.notebook import tqdm
import scipy.io as sio
from elephant import kernels
from elephant.statistics import instantaneous_rate
from neo.core import SpikeTrain
from quantities import ms, s, Hz
#import xlwt
from xlwt import Workbook
from collections import defaultdict
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import scipy.stats as stats
from scipy.stats import wilcoxon
from scipy.stats import mannwhitneyu


# # **New - from meta data**

# ## Importing data from Meta_data - Loading all files

# In[2]:


class TTL_class:

    def __init__(self, light_in_tray_ttl=0, poke_in_tray_ttl=0, image_ttl=0, second_poke_ttl=0, action_ttl=0, feedback_ttl=0, reward_consumption_ttl=0):
        self.light_in_tray_ttl = light_in_tray_ttl
        self.poke_in_tray_ttl = poke_in_tray_ttl
        self.image_ttl = image_ttl
        self.second_poke_ttl = second_poke_ttl
        self.action_ttl = action_ttl
        self.feedback_ttl = feedback_ttl
        self.reward_consumption_ttl = reward_consumption_ttl


class labels_class:

    def __init__(self, response=0, image='V', correction_trial=0, tray_light_exc=0, poke_in_tray_exc=0, image_exc=0, second_poke_exc=0, second_poke_est_exc=0, action_exc=0, reward_consumption_exc=0, no_reward_exc=0):
        self.response = response
        self.image = image
        self.correction_trial = correction_trial
        self.tray_light_exc = tray_light_exc
        self.poke_in_tray_exc = poke_in_tray_exc
        self.image_exc = image_exc
        self.second_poke_exc = second_poke_exc
        self.second_poke_est_exc = second_poke_est_exc
        self.action_exc = action_exc
        self.reward_consumption_exc = reward_consumption_exc
        self.no_reward_exc = no_reward_exc


# In[3]:


pickle_files = []

# Get list of all pickle files in the directory / one folder
# directory_path = r'J:\Spike_train_analysis\pickle\multi-image\good'
# directory_path = r'J:\Spike_train_analysis\pickle\multi-image - NR'
# directory_path = r'J:\Spike_train_analysis\pickle\multi-image - RD\good'
# directory_path = r'J:\Spike_train_analysis\pickle\multi-image - OI\good'
#pickle_files = glob.glob(os.path.join(directory_path, '*'))

# Get list of all pickle files in the directory / two or more folders
directory_paths = [r'J:\Spike_train_analysis\pickle\multi-image\good_final' , r'J:\Spike_train_analysis\pickle\multi-image - OI\good', r'J:\Spike_train_analysis\pickle\multi-image - NR',  r'J:\Spike_train_analysis\pickle\multi-image - RD\good']
for directory in directory_paths:
    pickle_files.extend(glob.glob(os.path.join(directory, '*')))

# Load data from each pickle file and append to mice_meta_data
mice_meta_data = []
for file_path in pickle_files:
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
        mice_meta_data.append(data)


# In[4]:


pickle_files


# In[5]:


a = 0
b = 11
ti_m = ['multi','multi','multi','multi','multi','multi','multi','multi','multi','multi', 'multi', 'OI','OI','OI','OI','OI','OI','NR','NR','NR','RD','RD','RD','RD'] 

for mice in range(a,b):
#for mice in range(len(mice_meta_data)):
    meta_data = mice_meta_data[mice]
    dict_key = os.path.basename(pickle_files[mice])[10:]
    print(dict_key)


# In[7]:


## extracting the values and make dictionaries

# added_list = ["A1, Dec 10, c3", "A1, Dec 10, c4", "A1, Dec 10, c14", "A1, Dec 13, c12", "A1, Dec 13, c33", "A2, Jan 24, c12", "A2, Jan 24, c23", "A2, Jan 24, c40", "A5, Dec 13, c7", "N11, Aug 2, c9", "N12, Oct 11, c15", "N12, Oct 11, c24", "N12, Oct 11, c33", "N12, Oct 6, c4", "N12, Oct 6, c35", "N12, Oct 6, c36", "N12, Oct 6, c68", "N12, Oct 6, c72", "N12, Oct 6, c77", "N13, Oct 29, c30", "N13, Oct 29, c65", "N13, Oct 29, c111", "N13, Oct 29, c124"]

#####################
burst_idx = 1.8
through_peak = 0.425

cell_met = {'cell_id':[], 'trough_t_peak':[], 'burst_Royer': [], 'cell_type':[], 'cell_type_F':[], 'FR':[], 'waveform':[]}

cell_type_Fr = {'Pyramidal Cell': [], 'Wide Interneuron':[], 'Narrow Interneuron':[], 'Bursty_Narrow':[], 'Trash':[]}
cell_type_ID = {'Pyramidal Cell': [], 'Wide Interneuron':[], 'Narrow Interneuron':[], 'Bursty_Narrow':[], 'Trash': []}

for mice in range(a,b):
#for mice in range(len(mice_meta_data)):
    meta_data = mice_meta_data[mice]
    dict_key = os.path.basename(pickle_files[mice])[10:]
    print(dict_key)
    for st in range(len(meta_data['cell_metrics']['cellID'][0][0])):
        cell_id = meta_data['cell_metrics']['cellID'][0][0][st]
        trough_t_peak = meta_data['cell_metrics']['troughToPeak'][0][0][st]
        burst_Royer = meta_data['cell_metrics']['burstIndex_Royer2012'][0][0][st]
        cell_type = meta_data['cell_metrics']['putativeCellType'][0][0][st][0]
        Firing_rate = meta_data['cell_metrics']['firingRate'][0][0][st]
        waveform = meta_data['cell_metrics']['waveforms'][0][0]['filt'][0][0][st]

        if cell_type != 'Trash': #or f"{dict_key}, c{st+1}" in added_list:
            print(f"{dict_key}, c{st+1}")
            if burst_Royer > burst_idx and trough_t_peak > through_peak:
                cell_type_F = 'Pyramidal Cell'
                cell_type_Fr['Pyramidal Cell'].append(Firing_rate)
                cell_type_ID['Pyramidal Cell'].append(f"{dict_key}, c{cell_id}")

            elif burst_Royer <= burst_idx and trough_t_peak > through_peak:
                cell_type_F = 'Wide Interneuron'
                cell_type_Fr['Wide Interneuron'].append(Firing_rate)
                cell_type_ID['Wide Interneuron'].append(f"{dict_key}, c{cell_id}")

            elif burst_Royer < burst_idx and trough_t_peak <= through_peak:
                cell_type_F = 'Narrow Interneuron'
                cell_type_Fr['Narrow Interneuron'].append(Firing_rate)
                cell_type_ID['Narrow Interneuron'].append(f"{dict_key}, c{cell_id}")

            elif burst_Royer > burst_idx and trough_t_peak <= through_peak:
                cell_type_F = 'Bursty_Narrow'
                cell_type_Fr['Bursty_Narrow'].append(Firing_rate)
                cell_type_ID['Bursty_Narrow'].append(f"{dict_key}, c{cell_id}")
        else:
            cell_type_F = 'Trash'
            cell_type_Fr['Trash'].append(Firing_rate)
            cell_type_ID['Trash'].append(f"{dict_key}, c{cell_id}")

        cell_met['cell_id'].append(f"{dict_key}, c{cell_id}")
        cell_met['trough_t_peak'].append(trough_t_peak)
        cell_met['burst_Royer'].append(burst_Royer)
        cell_met['cell_type'].append(cell_type)
        cell_met['cell_type_F'].append(cell_type_F)
        cell_met['FR'].append(round(Firing_rate, 2))
        cell_met['waveform'].append(waveform)

        # print(f"{dict_key}, c{cell_id}")
        # # print(trough_t_peak)
        # # print(burst_Royer)
        # print(cell_type)


# ## Main code

# In[8]:


# Import required libraries
import os
import scipy.io as sio
import numpy as np

# # Define function to process individual folders
# def process_folder(folder_path):
#     # wake_nrem = sio.loadmat(os.path.join(folder_path, "Wake_NREM_ratio.mat"))['Wake_NREM_ratio']
#     # ds2 = sio.loadmat(os.path.join(folder_path, "DS2_amp.mat"))['DS2_amp']
#     cell_metrics = sio.loadmat(os.path.join(folder_path, "recording.cell_metrics.cellinfo.mat"))['cell_metrics']

#     waveform = cell_metrics['waveforms'][0][0]['filt'][0][0]
#     firing_rate = cell_metrics['firingRate'][0][0][0]
#     labels = cell_metrics['putativeCellType'][0][0][0]
#     cell_id = cell_metrics[0][0][0][0]

#     # Ensure all data is 1D
#     # wake_nrem = wake_nrem.flatten()
#     # ds2 = ds2.flatten()
#     waveform = waveform[0].flatten()
#     labels = np.array([label[0] for label in labels])  # Convert to 1D array of strings

#     return firing_rate, waveform, labels, cell_id

# # Set the root folder path
# root_folder = r"J:\Spike_train_analysis\cell_classifier_input"

# # Initialize empty lists for each metric and labels
# # total_wake_nrem = []
# # total_ds2 = []
# total_firing_rate = []
# total_waveform = []
# total_labels = []
# total_cell_id = {'mice_id': [], 'cell_id':[], 'labels': []}

# # Process each subfolder
# for subfolder in os.listdir(root_folder):
#     folder_path = os.path.join(root_folder, subfolder)
#     print(folder_path)
#     if os.path.isdir(folder_path):
#         try:
#             firing_rate, waveform, labels, cell_id = process_folder(folder_path)

#             total_firing_rate.extend(firing_rate)
#             total_waveform.extend(waveform)
#             total_cell_id['labels'].extend(labels)
#             total_cell_id['cell_id'].extend(cell_id)

#             total_cell_id['mice_id'].extend(len(cell_id) * [subfolder])

#             print(f"Processed folder: {subfolder}")
#         except Exception as e:
#             print(f"Error processing folder {subfolder}: {str(e)}")

# Convert lists to numpy arrays
# total_wake_nrem = np.array(total_wake_nrem)
# total_ds2 = np.array(total_ds2)
total_firing_rate =  np.array(cell_met['FR'])
total_waveform = np.squeeze(np.array(cell_met['waveform']))
total_labels = np.array(cell_met['cell_type_F'])
# total_cell_id = np.array(cell_id)

# Print the final shapes and label information
print("\nTotal shapes:")
# print(f"Wake_NREM: {total_wake_nrem.shape}")
# print(f"DS2: {total_ds2.shape}")
print(f"firing_rate: {total_firing_rate.shape}")
print(f"waveform: {total_waveform.shape}")
print(f"Labels: {total_labels.shape}")
# print(f"mice_id: {len(total_cell_id['mice_id'])}")
print(f"cell_id: {len(cell_met['cell_id'])}")
print(f"labels: {len(cell_met['cell_type_F'])}")

# Print unique cell types and their counts
unique_labels, counts = np.unique(total_labels, return_counts=True)
print("\nCell type distribution:")
for label, count in zip(unique_labels, counts):
    print(f"{label}: {count}")


# In[9]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# --- Select pyramidal cells ---
pyramidal_mask = np.array(cell_met['cell_type_F']) == 'Pyramidal Cell'
W = np.asarray(total_waveform[pyramidal_mask])        # shape: (n_cells, n_timepoints)
print(f"Number of pyramidal cells: {W.shape[0]}")
print(f"Waveform shape: {W.shape}")

# --- Second derivative along time (no preprocessing) ---
# np.diff defaults to axis=-1, but we make it explicit:
W2 = np.diff(np.diff(W, axis=1), axis=1)              # shape: (n_cells, n_timepoints-2)

# Optional guardrail: drop any rows with NaN/Inf to avoid PCA errors
ok = np.isfinite(W2).all(axis=1)
if not np.all(ok):
    print(f"Dropping {np.sum(~ok)} cells with non-finite values in second derivative.")
    W2 = W2[ok]

# --- PCA (no scaling; sklearn PCA will mean-center features by default) ---
# Keep all possible components (limited by min(n_cells, n_features))
pca = PCA(n_components=None, svd_solver='full', random_state=None)
X_pca = pca.fit_transform(W2)

expl_var = pca.explained_variance_                  # absolute variance per PC
expl_var_ratio = pca.explained_variance_ratio_      # fraction per PC
cum_var_ratio = np.cumsum(expl_var_ratio)

print(f"n_components returned: {pca.n_components_}")
print("First 10 explained variance ratios:", np.round(expl_var_ratio[:10], 4))

# --- Plots: Scree (variance per PC) + cumulative variance ---
fig, ax = plt.subplots(1, 1, figsize=(8, 5))
idx = np.arange(1, len(expl_var_ratio) + 1)

# Scree: variance ratio per PC
ax.bar(idx, expl_var_ratio)
ax.set_xlabel("Principal Component", fontsize = 20)
ax.set_ylabel("Explained variance ratio", fontsize = 20)
ax.set_title("PCA on waveform second derivative (scree)", fontsize = 20)
ax.tick_params(labelsize=20)

# Overlay cumulative variance on a twin axis
ax2 = ax.twinx()
ax2.plot(idx, cum_var_ratio, marker='o')
ax2.set_ylabel("Cumulative explained variance", fontsize = 20)
ax2.tick_params(labelsize=20)

# Nice x-limits and grid
ax.set_xlim(0.5, len(expl_var_ratio) + 0.5)
ax.grid(True, axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()


# In[10]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# --- Select pyramidal cells ---
pyramidal_mask = np.array(cell_met['cell_type_F']) == 'Pyramidal Cell'
firing_rate = total_firing_rate[pyramidal_mask]
W = np.asarray(total_waveform[pyramidal_mask])           # (n_cells, n_timepoints)
print(f"Number of pyramidal cells: {W.shape[0]}")

# --- 2nd derivative along time (no preprocessing) ---
W2 = np.diff(np.diff(W, axis=1), axis=1)                 # (n_cells, n_timepoints-2)

# Optional: drop any rows with bad values
ok = np.isfinite(W2).all(axis=1)
if not np.all(ok):
    print(f"Dropping {np.sum(~ok)} cells with non-finite values.")
    W, W2 = W[ok], W2[ok]

# --- PCA to 2 components (for clustering & viz) ---
pca = PCA(n_components=2, svd_solver='full', random_state=42)
X_pca = pca.fit_transform(W2)
print("Explained variance ratio:", np.round(pca.explained_variance_ratio_, 4))
print("Cumulative:", np.round(pca.explained_variance_ratio_.cumsum(), 4))

# --- Choose k via silhouette on PCA space ---
max_clusters = 5
silhouette_scores = []
ks = []

n_samples = X_pca.shape[0]
for k in range(2, max_clusters + 1):
    if n_samples <= k:  # need at least one sample per cluster
        break
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(X_pca)
    # silhouette requires >=2 labels and no singleton-only clusters
    if len(np.unique(labels)) > 1:
        score = silhouette_score(X_pca, labels)
        silhouette_scores.append(score)
        ks.append(k)

optimal_clusters = ks[int(np.argmax(silhouette_scores))]
print(f"Optimal k based on silhouette: {optimal_clusters} (score={max(silhouette_scores):.3f})")

# --- Final KMeans ---
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init='auto')
cluster_labels = kmeans.fit_predict(X_pca)

# # --- (Optional) Quick plots ---
# # 1) Silhouette vs k
# plt.figure(figsize=(12,4))
# plt.subplot(1,3,1)
# plt.plot(ks, silhouette_scores, marker='o')
# plt.xlabel('k'); plt.ylabel('Silhouette'); plt.title('Silhouette vs k')
# plt.xticks(fontsize=12); plt.yticks(fontsize=12)

# # 2) PCA scatter
# plt.subplot(1,3,2)
# sc = plt.scatter(X_pca[:,0], X_pca[:,1], c=cluster_labels, cmap='viridis')
# plt.xlabel('PC1'); plt.ylabel('PC2'); plt.title('PCA(2nd-deriv)'); plt.colorbar(sc)
# plt.xticks(fontsize=12); plt.yticks(fontsize=12)

# # 3) Mean ORIGINAL waveform per cluster (more interpretable)
# plt.subplot(1,3,3)
# t = np.arange(W.shape[1])
# colors = plt.cm.viridis(np.linspace(0,1,optimal_clusters))
# for i, col in enumerate(colors):
#     m = (cluster_labels == i)
#     if m.any():
#         mean_w = W[m].mean(axis=0)
#         std_w  = W[m].std(axis=0)
#         plt.plot(t, mean_w, color=col, label=f'Cluster {i+1}')
#         plt.fill_between(t, mean_w-std_w, mean_w+std_w, color=col, alpha=0.2)
# plt.title('Mean original waveform by cluster'); plt.xlabel('Time'); plt.ylabel('Amplitude')
# plt.legend(fontsize=10)
# plt.tight_layout(); plt.show()

# --- Cluster sizes ---
sizes = np.bincount(cluster_labels, minlength=optimal_clusters)
print("\nCluster sizes:")
for i, s in enumerate(sizes, start=1):
    print(f"  Cluster {i}: {s} cells")

# If you really want centers, these are in PCA space (PC1, PC2):
print("\nCluster centers in PCA space (PC1, PC2):")
print(kmeans.cluster_centers_)


# In[12]:


import numpy as np
import matplotlib.pyplot as plt

waveform_pca = X_pca 
waveform = W2
waveform_abs = W
# -----------------------------
# 1) Build a stable color mapping once and reuse everywhere
# -----------------------------
unique_labels = np.sort(np.unique(cluster_labels))
K = unique_labels.size

# Choose a palette (viridis), reversed so Cluster 1 is brighter (like your example)
# palette = plt.cm.viridis(np.linspace(0, 1, K))[::-1]
palette = plt.cm.viridis(np.linspace(0, 1, K))

# Map actual labels -> indices 0..K-1 (and colors)
label_to_idx = {lab: i for i, lab in enumerate(unique_labels)}
idx_to_color = {i: palette[i] for i in range(K)}
label_to_color = {lab: idx_to_color[label_to_idx[lab]] for lab in unique_labels}

def display_name_for_label(lab):
    # Display clusters as "Cluster 1, 2, ..." in the order of unique_labels
    return f"Cluster {label_to_idx[lab] + 1}"

# Convenience arrays with 0..K-1 indices (if you ever need them)
cluster_idx = np.array([label_to_idx[lab] for lab in cluster_labels])

# -----------------------------
# 2) Figure layout
# -----------------------------
fig = plt.figure(figsize=(20, 15))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.25], hspace=0.35, wspace=0.3)

# top row (same as before)
ax_sil = fig.add_subplot(gs[0, 0])
ax_pca = fig.add_subplot(gs[0, 1])
ax_fr  = fig.add_subplot(gs[0, 2])

# bottom row → split the [1, :] slot into two columns
bottom = gs[1, :].subgridspec(1, 2, wspace=0.25)
ax_wav    = fig.add_subplot(bottom[0, 0])  # left bottom
ax_wavabs = fig.add_subplot(bottom[0, 1])  # right bottom

# -----------------------------
# 3) Silhouette curve
# -----------------------------
ax_sil.plot(range(2, max_clusters + 1), silhouette_scores, marker='o')
ax_sil.set_xlabel('Number of clusters', fontsize=15)
ax_sil.set_ylabel('Silhouette Score', fontsize=15)
ax_sil.set_title('Silhouette Score vs Number of Clusters', fontsize=15)
ax_sil.tick_params(labelsize=13)

# -----------------------------
# 4) PCA scatter (consistent colors)
# -----------------------------
for lab in unique_labels:
    mask = (cluster_labels == lab)
    ax_pca.scatter(
        waveform_pca[mask, 0],
        waveform_pca[mask, 1],
        s=35, alpha=0.8,
        color=label_to_color[lab],
        label=display_name_for_label(lab),
        edgecolor='none'
    )
ax_pca.set_xlabel('PC1', fontsize=15)
ax_pca.set_ylabel('PC2', fontsize=15)
ax_pca.set_title('PC1 vs PC2 of Waveform', fontsize=15)
ax_pca.legend(fontsize=12)
ax_pca.tick_params(labelsize=13)

# -----------------------------
# 5) Firing rates by cluster (jittered)
# -----------------------------
rng = np.random.default_rng(123)
for lab in unique_labels:
    i = label_to_idx[lab]
    fr_vals = firing_rate[cluster_labels == lab]
    x = rng.normal(i, 0.08, size=fr_vals.size)
    ax_fr.scatter(x, fr_vals, alpha=0.65, color=label_to_color[lab], label=display_name_for_label(lab))
ax_fr.set_xlabel('Cluster', fontsize=15)
ax_fr.set_ylabel('Firing Rate', fontsize=15)
ax_fr.set_title('Firing Rates by Cluster', fontsize=15)
ax_fr.set_xticks(range(K))
ax_fr.set_xticklabels([f'Cluster {i+1}' for i in range(K)], fontsize=13)
ax_fr.tick_params(labelsize=13)

# Only one legend (top-right plot) is enough; remove duplicates if you prefer
handles, labels = ax_fr.get_legend_handles_labels()
ax_fr.legend(handles[:K], labels[:K], fontsize=12, loc='upper right')

# -----------------------------
# 6) Average waveform (with ±1 SD) by cluster
# -----------------------------
T = waveform.shape[1]
time_points = np.arange(T)

for lab in unique_labels:
    mask = (cluster_labels == lab)
    avg_w = waveform[mask].mean(axis=0)
    std_w = waveform[mask].std(axis=0)

    color = label_to_color[lab]
    ax_wav.plot(time_points, avg_w, color=color, linewidth=2, label=display_name_for_label(lab))
    ax_wav.fill_between(time_points, avg_w - std_w, avg_w + std_w, color=color, alpha=0.20)

ax_wav.set_xlabel('Time', fontsize=15)
ax_wav.set_ylabel('Amplitude', fontsize=15)
ax_wav.set_title('Average second derivative of Waveforms by Cluster', fontsize=15)
ax_wav.legend(fontsize=14)
ax_wav.tick_params(labelsize=13)

plt.tight_layout()
plt.show()

# -----------------------------
# 7) Average waveform (with ±1 SD) by cluster
# -----------------------------
waveform_abs = total_waveform[pyramidal_mask]
T = waveform_abs.shape[1]
time_points = np.arange(T)

for lab in unique_labels:
    mask = (cluster_labels == lab)
    avg_w_abs = waveform_abs[mask].mean(axis=0)
    std_w_abs = waveform_abs[mask].std(axis=0)

    color = label_to_color[lab]
    ax_wavabs.plot(time_points, avg_w_abs, color=color, linewidth=2, label=display_name_for_label(lab))
    ax_wavabs.fill_between(time_points, avg_w_abs - std_w_abs, avg_w_abs + std_w_abs, color=color, alpha=0.20)

ax_wavabs.set_xlabel('Time', fontsize=15)
ax_wavabs.set_ylabel('Amplitude', fontsize=15)
ax_wavabs.set_title('Average of Waveforms by Cluster', fontsize=15)
ax_wavabs.legend(fontsize=14)
ax_wavabs.tick_params(labelsize=13)

plt.tight_layout()
plt.show()

# -----------------------------
# 8) Console summary (sizes + firing rates)
# -----------------------------
print("\nCluster mapping (actual_label -> display_name):")
for lab in unique_labels:
    print(f"  {lab} -> {display_name_for_label(lab)}")

print("\nCluster Sizes:")
for lab in unique_labels:
    size = np.sum(cluster_labels == lab)
    print(f"{display_name_for_label(lab)}: {size} cells")

print("\nAverage Firing Rates:")
for lab in unique_labels:
    fr_vals = firing_rate[cluster_labels == lab]
    print(f"{display_name_for_label(lab)}: {np.mean(fr_vals):.2f} ± {np.std(fr_vals):.2f}")


# In[ ]:


np.array(cell_met['cell_id'])[pyramidal_mask]


# In[ ]:


cluster_labels


# In[14]:


M_C_ids = {'mossy':[], 'granule':[]}

c_i = np.array(cell_met['cell_id'])[pyramidal_mask]

for n in range(len(cluster_labels)):
    if cluster_labels[n] == 0:
        M_C_ids['mossy'].append(f'{c_i[n]}')
    else:
        M_C_ids['granule'].append(f'{c_i[n]}')



# In[15]:


len(M_C_ids['granule'])


# In[ ]:


M_C_ids['granule']


# In[16]:


len(M_C_ids['mossy'])


# In[ ]:


M_C_ids['mossy']


# In[ ]:


M_C_ids


# In[18]:


# Save the dictionary to a pickle file 
path_p = r'J:'
file_name = f'mossy-granule-multi-final-added'

with open('{}/{}'.format(path_p, file_name), 'wb') as f:
    pickle.dump(M_C_ids, f)


# In[ ]:


with open("J:\mossy-granule-all", 'rb') as f:
    mossy_granule = pickle.load(f)


# ## Plotting separate waveforms and second derivatives

# In[ ]:


# cell_metrics = sio.loadmat(os.path.join('J:\Spike_train_analysis\mice multi\\A6, Feb 7', "recording.cell_metrics.cellinfo.mat"))['cell_metrics']

# waveform2 = cell_metrics['waveforms'][0][0]['filt'][0][0]
# labels2 = cell_metrics['putativeCellType'][0][0][0]

# plt.plot(np.diff(waveform2[0][0][0]))

# Plot 4: Average waveforms
waveform = np.diff(total_waveform[pyramidal_mask])
time_points = np.arange(waveform.shape[1])
colors = plt.cm.viridis(np.linspace(0, 1, optimal_clusters))
labels = ['Mossy cell', 'Granule cell']

for i in range(optimal_clusters):
    cluster_waveforms = waveform[cluster_labels == i]
    avg_waveform = np.mean(cluster_waveforms, axis=0)
    std_waveform = np.std(cluster_waveforms, axis=0)

    plt.plot(time_points, avg_waveform, label=labels[i], color=colors[i])
    plt.fill_between(time_points, 
                     avg_waveform - std_waveform, 
                     avg_waveform + std_waveform, 
                     alpha=0.2, color=colors[i])

plt.xlabel('Time', fontsize=15)
plt.ylabel('Amplitude', fontsize=15)
plt.title('Average Waveforms by Cluster', fontsize=15)
plt.legend(fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

plt.tight_layout()
plt.show()


# In[ ]:


waveform2 = (total_waveform[pyramidal_mask])


# Plot 4: Average waveforms

time_points = np.arange(waveform2.shape[1])
colors = plt.cm.viridis(np.linspace(0, 1, optimal_clusters))
labels = ['Mossy cell', 'Granule cell']

for i in range(optimal_clusters):
    cluster_waveforms = waveform2[cluster_labels == i]
    avg_waveform = np.mean(cluster_waveforms, axis=0)
    std_waveform = np.std(cluster_waveforms, axis=0)

    plt.plot(time_points, avg_waveform, label=labels[i], color=colors[i])
    plt.fill_between(time_points, 
                     avg_waveform - std_waveform, 
                     avg_waveform + std_waveform, 
                     alpha=0.2, color=colors[i])

plt.xlabel('Time', fontsize=15)
plt.ylabel('Amplitude', fontsize=15)
plt.title('Average Waveforms by Cluster', fontsize=15)
plt.legend(fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

plt.tight_layout()
plt.show()


# In[ ]:


len(labels2)


# In[ ]:


for i in labels2:
    print(i)


# In[ ]:


A6 = ["G",	"G",	"M",	"N",	"M",	"N",	"M",	"M",	"M",	"M",	"M",	"G",	"N",	"M",	"N",	"M",	"G",	"G",	"G",	"N",	"G",	"M",	"G",	"M",	"M",	"G",	"N",	"M",	"N"]


# In[ ]:


N12 = ["N",	"N",	"N",	"N",	"M",	"M",	"N",	"N",	"N",	"N",	"N",	"N",	"M",	"G",	"G",	"G",	"N",	"M",	"N",	"N",	"N",	"N",	"N",	"N",	"N",	"N",	"N",	"N",	"G",	"N",	"N",	"N",	"N",	"G",	"N",	"N",	"N",	"N",	"N",	"N",	"M",	"N",	"N",	"G",	"N",	"N",	"N",	"N",	"N",	"N",	"N",	"N",	"N",	"N",	"N",	"N",	"N",	"N",	"N",	"N",	"N",	"N",	"N",	"N",	"N",	"N",	"N",	"N",	"N",	"N",	"N",	"N",	"N",	"M",	"M",	"N",	"N",	"N",	"G",	"G",	"M",	"G",	"G",	"G",	"N"]


# In[ ]:


for a in range(len(A6)):
    if A6[a] == 'G':
        M_C_ids['granule'].append(f'A6, Feb 7, c{a+1}')
        print((f'A6, Feb 7, c{a+1}'))
    elif A6[a] == 'M':
        M_C_ids['mossy'].append(f'A6, Feb 7, c{a+1}')
        print((f'A6, Feb 7, c{a+1}'))


# In[ ]:


for N in range(len(N12)):
    if N12[N] == 'G':
        M_C_ids['granule'].append(f'N12, Oct 6, c{N+1}')
        print((f'N12, Oct 6, c{N+1}'))
    if N12[N] == 'M':
        M_C_ids['mossy'].append(f'N12, Oct 6, c{N+1}')
        print((f'N12, Oct 6, c{N+1}'))   


# In[ ]:


len(M_C_ids['mossy'])


# In[ ]:


len(M_C_ids['granule'])


# In[ ]:





# # **Cell-type analysis and plotting**

# ## Importing data from Meta_data - Loading all files

# In[19]:


class TTL_class:

    def __init__(self, light_in_tray_ttl=0, poke_in_tray_ttl=0, image_ttl=0, second_poke_ttl=0, action_ttl=0, feedback_ttl=0, reward_consumption_ttl=0):
        self.light_in_tray_ttl = light_in_tray_ttl
        self.poke_in_tray_ttl = poke_in_tray_ttl
        self.image_ttl = image_ttl
        self.second_poke_ttl = second_poke_ttl
        self.action_ttl = action_ttl
        self.feedback_ttl = feedback_ttl
        self.reward_consumption_ttl = reward_consumption_ttl


class labels_class:

    def __init__(self, response=0, image='V', correction_trial=0, tray_light_exc=0, poke_in_tray_exc=0, image_exc=0, second_poke_exc=0, second_poke_est_exc=0, action_exc=0, reward_consumption_exc=0, no_reward_exc=0):
        self.response = response
        self.image = image
        self.correction_trial = correction_trial
        self.tray_light_exc = tray_light_exc
        self.poke_in_tray_exc = poke_in_tray_exc
        self.image_exc = image_exc
        self.second_poke_exc = second_poke_exc
        self.second_poke_est_exc = second_poke_est_exc
        self.action_exc = action_exc
        self.reward_consumption_exc = reward_consumption_exc
        self.no_reward_exc = no_reward_exc


# In[20]:


pickle_files = []

# Get list of all pickle files in the directory / one folder
# directory_path = r'J:\Spike_train_analysis\pickle\multi-image\good'
# directory_path = r'J:\Spike_train_analysis\pickle\multi-image - NR'
# directory_path = r'J:\Spike_train_analysis\pickle\multi-image - RD\good'
# directory_path = r'J:\Spike_train_analysis\pickle\multi-image - OI\good'
#pickle_files = glob.glob(os.path.join(directory_path, '*'))

# Get list of all pickle files in the directory / two or more folders
directory_paths = [r'J:\Spike_train_analysis\pickle\multi-image\good_final' , r'J:\Spike_train_analysis\pickle\multi-image - OI\good', r'J:\Spike_train_analysis\pickle\multi-image - NR',  r'J:\Spike_train_analysis\pickle\multi-image - RD\good']
for directory in directory_paths:
    pickle_files.extend(glob.glob(os.path.join(directory, '*')))

# Load data from each pickle file and append to mice_meta_data
mice_meta_data = []
for file_path in pickle_files:
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
        mice_meta_data.append(data)


# In[21]:


pickle_files


# ## Analysis

# In[22]:


a = 0
b = 11
ti_m = ['multi','multi','multi','multi','multi','multi','multi','multi','multi','multi', 'multi', 'OI','OI','OI','OI','OI','OI','NR','NR','NR','RD','RD','RD','RD'] 

for mice in range(a,b):
#for mice in range(len(mice_meta_data)):
    meta_data = mice_meta_data[mice]
    dict_key = os.path.basename(pickle_files[mice])[10:]
    print(dict_key)


# In[8]:


len(mice_meta_data[0]['cell_metrics']['troughToPeak'][0][0])


# In[9]:


len(mice_meta_data[0]['cell_metrics']['burstIndex_Royer2012'][0][0])


# In[ ]:


len(mice_meta_data[0]['cell_metrics']['putativeCellType'][0][0])


# In[ ]:


len(meta_data['cell_metrics']['cellID'][0][0])


# In[ ]:


(mice_meta_data[0]['cell_metrics']['putativeCellType'][0][0])


# In[ ]:


## extracting the values and make dictionaries

# added_list = ["A1, Dec 10, c3", "A1, Dec 10, c4", "A1, Dec 10, c14", "A1, Dec 13, c12", "A1, Dec 13, c33", "A2, Jan 24, c12", "A2, Jan 24, c23", "A2, Jan 24, c40", "A5, Dec 13, c7", "N11, Aug 2, c9", "N12, Oct 11, c15", "N12, Oct 11, c24", "N12, Oct 11, c33", "N12, Oct 6, c4", "N12, Oct 6, c35", "N12, Oct 6, c36", "N12, Oct 6, c68", "N12, Oct 6, c72", "N12, Oct 6, c77", "N13, Oct 29, c30", "N13, Oct 29, c65", "N13, Oct 29, c111", "N13, Oct 29, c124"]
###########
burst_idx = 1.8
through_peak = 0.425

cell_met = {'cell_id':[], 'trough_t_peak':[], 'burst_Royer': [], 'cell_type':[], 'cell_type_F':[], 'FR':[]}

cell_type_Fr = {'Pyramidal Cell': [], 'Wide Interneuron':[], 'Narrow Interneuron':[], 'Bursty_Narrow':[], 'Trash':[]}
cell_type_ID = {'Pyramidal Cell': [], 'Wide Interneuron':[], 'Narrow Interneuron':[], 'Bursty_Narrow':[], 'Trash': []}

for mice in range(a,b):
#for mice in range(len(mice_meta_data)):
    meta_data = mice_meta_data[mice]
    dict_key = os.path.basename(pickle_files[mice])[10:]
    print(dict_key)
    for st in range(len(meta_data['cell_metrics']['cellID'][0][0])):
        cell_id = meta_data['cell_metrics']['cellID'][0][0][st]
        trough_t_peak = meta_data['cell_metrics']['troughToPeak'][0][0][st]
        burst_Royer = meta_data['cell_metrics']['burstIndex_Royer2012'][0][0][st]
        cell_type = meta_data['cell_metrics']['putativeCellType'][0][0][st][0]
        Firing_rate = meta_data['cell_metrics']['firingRate'][0][0][st]

        if cell_type != 'Trash': #or f"{dict_key}, c{st+1}" in added_list:
            if burst_Royer > burst_idx and trough_t_peak > through_peak:
                cell_type_F = 'Pyramidal Cell'
                cell_type_Fr['Pyramidal Cell'].append(Firing_rate)
                cell_type_ID['Pyramidal Cell'].append(f"{dict_key}, c{cell_id}")

            elif burst_Royer <= burst_idx and trough_t_peak > through_peak:
                cell_type_F = 'Wide Interneuron'
                cell_type_Fr['Wide Interneuron'].append(Firing_rate)
                cell_type_ID['Wide Interneuron'].append(f"{dict_key}, c{cell_id}")

            elif burst_Royer < burst_idx and trough_t_peak <= through_peak:
                cell_type_F = 'Narrow Interneuron'
                cell_type_Fr['Narrow Interneuron'].append(Firing_rate)
                cell_type_ID['Narrow Interneuron'].append(f"{dict_key}, c{cell_id}")

            elif burst_Royer > burst_idx and trough_t_peak <= through_peak:
                cell_type_F = 'Bursty_Narrow'
                cell_type_Fr['Bursty_Narrow'].append(Firing_rate)
                cell_type_ID['Bursty_Narrow'].append(f"{dict_key}, c{cell_id}")
        else:
            cell_type_F = 'Trash'
            cell_type_Fr['Trash'].append(Firing_rate)
            cell_type_ID['Trash'].append(f"{dict_key}, c{cell_id}")

        cell_met['cell_id'].append(f"{dict_key}, c{cell_id}")
        cell_met['trough_t_peak'].append(trough_t_peak)
        cell_met['burst_Royer'].append(burst_Royer)
        cell_met['cell_type'].append(cell_type)
        cell_met['cell_type_F'].append(cell_type_F)
        cell_met['FR'].append(round(Firing_rate, 2))

        print(f"{dict_key}, c{cell_id}")
        # print(trough_t_peak)
        # print(burst_Royer)
        print(cell_type)


# In[ ]:


for indx, cell_id in enumerate(cell_met['cell_id']):
    print(cell_id)


# In[ ]:


for indx, cell_id in enumerate(cell_met['cell_id']):
    print(cell_met['cell_type_F'][indx])


# ### Plots

# In[24]:


## dot plot based in the cell explorer result
import matplotlib.pyplot as plt

# Example data
x = cell_met['trough_t_peak']
y = cell_met['burst_Royer']
cell_type = cell_met['cell_type']

# Make a color map for cell types
colors = {"Pyramidal Cell": "red", "Wide Interneuron": "cyan", "Narrow Interneuron": "blue"}

# Plot each point with the color based on cell_type
for xi, yi, ci in zip(x, y, cell_type):
    if ci == 'Trash':
        continue
    plt.scatter(xi, yi, color=colors[ci], s=60, label=ci)

plt.yscale("log")
plt.xlabel("X values")
plt.ylabel("Y values (log scale)")
plt.title("Dot Plot by Cell Type")

# Add vertical and horizontal lines
plt.axvline(x=0.425, color="black", linestyle="--", linewidth=1)  # vertical at x=0.5
plt.axhline(y=1.8, color="black", linestyle="--", linewidth=1)   # horizontal at y=10

# Add legend without duplicates
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())

plt.show()


# In[14]:


cell_met['cell_type_F']


# In[27]:


## dot plots based on the manual criterea 
import matplotlib.pyplot as plt

burst_idx = 1.8
through_peak = 0.425

# Example data
x = cell_met['trough_t_peak']
y = cell_met['burst_Royer']

# Plot each point with conditional color
for idx, cell in enumerate(cell_met['cell_type_F']):
    if cell == "Pyramidal Cell":
        color = "red" 
        xi = x[idx]
        yi = y[idx]
        label = "Excitatory Principal Cells"

    elif cell == "Wide Interneuron":
        color = "cyan"  
        xi = x[idx]
        yi = y[idx]
        label = "Wide Interneuron"

    elif cell == "Narrow Interneuron":
        color = "blue"
        xi = x[idx]
        yi = y[idx]  
        label = "Narrow Interneuron"

    elif cell == "Bursty_Narrow":
        color = "violet"  
        xi = x[idx]
        yi = y[idx]
        label = "Bursty_Narrow"
    else:
        continue

    plt.scatter(xi, yi, color=color, s=60, label=label)

plt.yscale("log")
plt.xlabel("Through-to-Peak", fontsize=25)
plt.ylabel("Burst index (log scale)", fontsize=25)
plt.title("Cell Type classification", fontsize=25)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

# Add vertical and horizontal lines
plt.axvline(x=through_peak, color="gray", linestyle="--", linewidth=1)
plt.axhline(y=burst_idx, color="gray", linestyle="--", linewidth=1)

# Add legend without duplicates
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), fontsize=20)

plt.show()


# In[23]:


## dot plots based on the manual criterea 
import matplotlib.pyplot as plt

burst_idx = 1.8
through_peak = 0.425

# Example data
x = cell_met['trough_t_peak']
y = cell_met['burst_Royer']

# Plot each point with conditional color
for xi, yi in zip(x, y):
    if xi > through_peak and yi > burst_idx:
        color = "red"   # Pyramidal Cell
        label = "Excitatory Principal Cells"
    elif xi > through_peak and yi <= burst_idx:
        color = "cyan"  # Wide Interneuron
        label = "Wide Interneuron"
    else: 
        color = "blue"  # Narrow Interneuron
        label = "Narrow Interneuron"

    plt.scatter(xi, yi, color=color, s=60, label=label)

plt.yscale("log")
plt.xlabel("Through-to-Peak", fontsize=25)
plt.ylabel("Burst index (log scale)", fontsize=25)
plt.title("Cell Type classification", fontsize=25)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

# Add vertical and horizontal lines
plt.axvline(x=through_peak, color="gray", linestyle="--", linewidth=1)
plt.axhline(y=burst_idx, color="gray", linestyle="--", linewidth=1)

# Add legend without duplicates
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), fontsize=20)

plt.show()


# In[28]:


## histogram of the firing rates - in subplots

import matplotlib.pyplot as plt

# Example: dictionary where keys are cell types, values are lists of firing rates
firing_rates = cell_type_Fr.copy()
firing_rates.pop("Trash", None)

# Colors for consistency
colors = {
    "Pyramidal Cell": "red",
    "Wide Interneuron": "cyan",
    "Narrow Interneuron": "blue",
    "Bursty_Narrow": "violet"
}

# Create 2x2 subplot grid
fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
axes = axes.flatten()  # flatten so we can index in a loop

# Loop through dictionary and plot histogram for each cell type
for ax, (cell_type, rates) in zip(axes, firing_rates.items()):
    ax.hist(
        rates, 
        bins=15, 
        alpha=0.7,
        color=colors[cell_type], 
        edgecolor="black"
    )
    ax.set_title(cell_type, fontsize=20)
    ax.set_xlabel("Firing Rate (Hz)", fontsize=20)
    ax.set_ylabel("Count", fontsize=20)
    ax.tick_params(axis="x", labelsize=20)
    ax.tick_params(axis="y", labelsize=20)

fig.suptitle("Histograms of Firing Rates by Cell Type", fontsize=20)
plt.tight_layout(rect=[0, 0, 1, 0.96])  # leave space for suptitle
plt.show()


# In[29]:


## histogram of firing rates
import matplotlib.pyplot as plt

# Example: dictionary where keys are cell types, values are lists of firing rates
firing_rates = cell_type_Fr
firing_rates.pop("Trash", None)

# Colors for consistency
colors = {
    "Pyramidal Cell": "red",
    "Wide Interneuron": "cyan",
    "Narrow Interneuron": "blue",
    "Bursty_Narrow": "violet"
}

plt.figure(figsize=(8,6))

# Loop through dictionary and plot histogram for each cell type
for cell_type, rates in firing_rates.items():
    plt.hist(
        rates, 
        bins=15, 
        alpha=0.5,         # transparency so they overlay
        color=colors[cell_type], 
        label=cell_type,
        edgecolor="black"
    )

plt.xlabel("Firing Rate (Hz)", fontsize=25)
plt.ylabel("Cell Count", fontsize=25)
plt.title("Histogram of Firing Rates by Cell Type", fontsize=25)
plt.legend(fontsize=25)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.show()


# In[30]:


## histogram of the firing rates - with guassian filter curve

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

# Example: dictionary where keys are cell types, values are lists of firing rates
firing_rates = cell_type_Fr.copy()
firing_rates.pop("Trash", None)

# Colors for consistency
colors = {
    "Pyramidal Cell": "red",
    "Wide Interneuron": "cyan",
    "Narrow Interneuron": "blue",
    "Bursty_Narrow": "violet"
}

plt.figure(figsize=(8,6))

# Loop through dictionary and plot histogram + smoothed curve
for cell_type, rates in firing_rates.items():
    # histogram with counts (not density)
    counts, bins, _ = plt.hist(
        rates, 
        bins=15, 
        alpha=0.3,
        color=colors[cell_type], 
        label=cell_type,
        edgecolor="black"
    )

    # Bin centers
    bin_centers = 0.5 * (bins[1:] + bins[:-1])

    # Smooth the counts with a Gaussian filter
    smoothed_counts = gaussian_filter1d(counts, sigma=2)

    # Plot smoothed line
    plt.plot(bin_centers, smoothed_counts, color=colors[cell_type], linewidth=2)

plt.xlabel("Firing Rate (Hz)", fontsize=25)
plt.ylabel("Cell Count", fontsize=25)
plt.title("Histogram of Firing Rates by Cell Type (Smoothed with Gaussian)", fontsize=25)
plt.legend(fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.show()


# In[31]:


## guassian filter curve of firing rates

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

# Example: dictionary where keys are cell types, values are lists of firing rates
firing_rates = cell_type_Fr.copy()
firing_rates.pop("Trash", None)

# Colors for consistency
colors = {
    "Pyramidal Cell": "red",
    "Wide Interneuron": "cyan",
    "Narrow Interneuron": "blue",
    "Bursty_Narrow": "violet"
}

plt.figure(figsize=(8,6))

# Loop through dictionary and plot only smoothed histogram curves
for cell_type, rates in firing_rates.items():
    # Get histogram counts (absolute counts, not density)
    counts, bins = np.histogram(rates, bins=15)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])

    # Smooth the counts with a Gaussian filter
    smoothed_counts = gaussian_filter1d(counts, sigma=1.5)

    # Plot smoothed line
    plt.plot(bin_centers, smoothed_counts, color=colors[cell_type], 
             label=cell_type, linewidth=2)

plt.xlabel("Firing Rate (Hz)", fontsize=25)
plt.ylabel("Cell Count", fontsize=25)
plt.title("Smoothed Firing Rate Distributions by Cell Type", fontsize=25)
plt.legend(fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.show()


# In[32]:


## firing rate kde gaussian - min max

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

# Example: dictionary where keys are cell types, values are lists of firing rates
firing_rates = cell_type_Fr.copy()
firing_rates.pop("Trash", None)

# Colors for consistency
colors = {
    "Pyramidal Cell": "red",
    "Wide Interneuron": "cyan",
    "Narrow Interneuron": "blue",
    "Bursty_Narrow": "violet"
}

plt.figure(figsize=(8,6))

# Loop through dictionary and plot KDE curve for each cell type
for cell_type, rates in firing_rates.items():
    rates = np.array(rates)
    kde = gaussian_kde(rates)   # fit KDE
    x_vals = np.linspace(rates.min(), rates.max(), 100)  # range for smooth curve
    y_vals = kde(x_vals)

    plt.plot(x_vals, y_vals, color=colors[cell_type], label=cell_type, linewidth=2)

plt.xlabel("Firing Rate (Hz)", fontsize=20)
plt.ylabel("Density", fontsize=20)
plt.title("Firing Rate Distributions by Cell Type (KDE curves)", fontsize=20)
plt.legend(fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.show()


# In[33]:


## firing rate kde gaussian - 0 max

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

plt.figure(figsize=(8,6))

for cell_type, rates in firing_rates.items():
    rates = np.array(rates)
    kde = gaussian_kde(rates)

    # extend x-range to start from 0
    x_min = 0
    x_max = rates.max()
    x_vals = np.linspace(x_min, x_max, 500)
    y_vals = kde(x_vals)

    # prepend (0,0) so curve starts at baseline
    x_vals = np.insert(x_vals, 0, 0)
    y_vals = np.insert(y_vals, 0, 0)

    plt.plot(x_vals, y_vals, color=colors[cell_type], label=cell_type, linewidth=2)

plt.xlabel("Firing Rate (Hz)", fontsize=20)
plt.ylabel("Density", fontsize=20)
plt.title("Firing Rate Distributions by Cell Type (KDE curves)", fontsize=20)
plt.legend(fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.tight_layout()
plt.show()



# In[34]:


## firing rate - kde seaborn

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8,6))

# KDE plots only (no histograms)
sns.kdeplot(firing_rates['Pyramidal Cell'], label="Pyramidal Cell", bw_adjust=0.6, color="red", linewidth=2)
sns.kdeplot(firing_rates['Wide Interneuron'], label="Wide Interneuron", bw_adjust=0.6, color="cyan", linewidth=2)
sns.kdeplot(firing_rates['Narrow Interneuron'], label="Narrow Interneuron", bw_adjust=0.6, color="blue", linewidth=2)
sns.kdeplot(firing_rates['Bursty_Narrow'], label="Bursty Narrow", bw_adjust=0.6, color="violet", linewidth=2)

plt.xlabel("Firing Rate (Hz)", fontsize=20)
plt.ylabel("Density", fontsize=20)
plt.title("Firing Rate Distributions by Cell Type (KDE curves)", fontsize=20)
plt.legend(fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.tight_layout()
plt.show()


# In[33]:


## firing rate - kde seaborn

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))

for label, arr in firing_rates.items():
    arr = np.asarray(arr)
    # match manual range (no extension beyond data)
    xmin, xmax = arr.min(), arr.max()

    sns.kdeplot(
        arr,
        label=label,
        bw_adjust=0.6,     # your chosen smoothness
        cut=0,             # don’t extend beyond data range
        gridsize=100,      # match your 100-point linspace
        clip=(xmin, xmax), # hard-clip to data range
        linewidth=2,
        color=colors[label]
    )

plt.xlabel("Firing Rate (Hz)", fontsize=20)
plt.ylabel("Density", fontsize=20)
plt.title("Firing Rate Distributions by Cell Type (KDE curves)", fontsize=20)
plt.legend(fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.tight_layout()
plt.show()


# ### extra

# In[ ]:


cell_type_Fr


# In[ ]:


cell_type_ID


# In[ ]:


id_list = []
for idx, fr in enumerate(cell_type_Fr['Narrow Interneuron']):
    if fr < 5:
        id_list.append(cell_type_ID['Narrow Interneuron'][idx])


# In[ ]:


id_list


# In[ ]:





# # **Final code for cell_features dictionary**

# In[ ]:


## extracting the values and make dictionaries
# added_list = ["A1, Dec 10, c3", "A1, Dec 10, c4", "A1, Dec 10, c14", "A1, Dec 13, c12", "A1, Dec 13, c33", "A2, Jan 24, c12", "A2, Jan 24, c23", "A2, Jan 24, c40", "A5, Dec 13, c7", "N11, Aug 2, c9", "N12, Oct 11, c15", "N12, Oct 11, c24", "N12, Oct 11, c33", "N12, Oct 6, c4", "N12, Oct 6, c35", "N12, Oct 6, c36", "N12, Oct 6, c68", "N12, Oct 6, c72", "N12, Oct 6, c77", "N13, Oct 29, c30", "N13, Oct 29, c65", "N13, Oct 29, c111", "N13, Oct 29, c124"]

#####
with open("J:\mossy-granule-multi-final", 'rb') as f:
    mossy_granule = pickle.load(f)

# with open("J:\mossy-granule-all-final", 'rb') as f:
#     mossy_granule = pickle.load(f)
#####

burst_idx = 1.8
through_peak = 0.425
sr = 30000
epsilon = 1e-10  # A small constant 

############
# generating a dictionary to store the results
cell_features = {}
############

#for mice in range(20,21):
for mice in range(len(mice_meta_data)):
    meta_data = mice_meta_data[mice]
    dict_key = os.path.basename(pickle_files[mice])[10:]
    print(dict_key)
    #####################################################################
    #exctracting ttl and lables from the meta_data
    light_in_tray_ttl = []
    poke_in_tray_ttl = []
    image_ttl = []
    second_poke_ttl = []
    action_ttl = []
    feedback_ttl = []
    reward_consumption_ttl = []
    no_reward_ttl = []
    response = []
    images = []
    cr_trial = []
    no_reward_exc = []
    ITI_ttl = []

    for tri in range(len(meta_data['session_recording']['TTL'])):
        light_in_tray_ttl.append(meta_data['session_recording']['TTL'][tri].light_in_tray_ttl)
        poke_in_tray_ttl.append(meta_data['session_recording']['TTL'][tri].poke_in_tray_ttl)
        image_ttl.append(meta_data['session_recording']['TTL'][tri].image_ttl)
        second_poke_ttl.append(meta_data['session_recording']['TTL'][tri].second_poke_ttl)
        action_ttl.append(meta_data['session_recording']['TTL'][tri].action_ttl)
        feedback_ttl.append(meta_data['session_recording']['TTL'][tri].feedback_ttl)
        reward_consumption_ttl.append(meta_data['session_recording']['TTL'][tri].reward_consumption_ttl)

        if (meta_data['session_recording']['TTL'][tri].action_ttl) is not None:
            no_reward_ttl.append((meta_data['session_recording']['TTL'][tri].action_ttl) + (1*sr))
        else:
            no_reward_ttl.append(meta_data['session_recording']['TTL'][tri].action_ttl)

        if (meta_data['session_recording']['TTL'][tri].reward_consumption_ttl) is not None:
            if (meta_data['session_recording']['TTL'][tri].reward_consumption_ttl) > 0:
                ITI_ttl.append((meta_data['session_recording']['TTL'][tri].reward_consumption_ttl) + (5*sr))  
            else:
                ITI_ttl.append((meta_data['session_recording']['TTL'][tri].action_ttl) + (6*sr))  
        else:
            ITI_ttl.append(meta_data['session_recording']['TTL'][tri].reward_consumption_ttl)  

    for tri in range(len(meta_data['session_recording']['labels'])):
        response.append(meta_data['session_recording']['labels'][tri].response)
        images.append(meta_data['session_recording']['labels'][tri].image)
        cr_trial.append(meta_data['session_recording']['labels'][tri].correction_trial)
        no_reward_exc.append(meta_data['session_recording']['labels'][tri].no_reward_exc) 

    spike_train = meta_data['session_recording']['spike_time_beh']
    spike_train_base = meta_data['session_recording']['spike_time_base']
    cells = meta_data['session_recording']['spike_time_beh']
    cell_metrics = meta_data['cell_metrics']
    #####################################################################
    for st in range((len(cells))):
        cell_id = meta_data['cell_metrics']['cellID'][0][0][st]
        trough_t_peak = meta_data['cell_metrics']['troughToPeak'][0][0][st]
        burst_Royer = meta_data['cell_metrics']['burstIndex_Royer2012'][0][0][st]
        cell_type = meta_data['cell_metrics']['putativeCellType'][0][0][st][0]
        Firing_rate = meta_data['cell_metrics']['firingRate'][0][0][st]

        ######################################
        if cell_type != 'Trash': #or f"{dict_key}, c{st+1}" in added_list:
            if burst_Royer > burst_idx and trough_t_peak > through_peak:
                cell_type_F = 'Pyramidal Cell'

            elif burst_Royer <= burst_idx and trough_t_peak > through_peak:
                cell_type_F = 'Wide Interneuron'

            elif burst_Royer < burst_idx and trough_t_peak <= through_peak:
                cell_type_F = 'Narrow Interneuron'

            elif burst_Royer > burst_idx and trough_t_peak <= through_peak:
                cell_type_F = 'Bursty_Narrow'
        else:
            cell_type_F = 'Trash'
        #######################################
        if cell_type_F == 'Pyramidal Cell':
            #print('yes', st+1)
            #print(f"{dict_key}, c{st+1}")
            if f"{dict_key}, c{st+1}" in mossy_granule['mossy']:
                #print('yes mossy')
                cell_type_FF = 'Mossy' 
            elif f"{dict_key}, c{st+1}" in mossy_granule['granule']:
                #print('yes granule')
                cell_type_FF = 'Granule'  
            else:
                cell_type_FF = 'Pyramidal Cell' 
        else:
            cell_type_FF = cell_type_F
        #########################################
        # Calculate mean baseline firing rate beh
        f_baseline_total = []
        for tri in range((len(response))-1):
            spike_sample = spike_train[st]
            baseline_t = np.where((spike_sample > light_in_tray_ttl[tri]) & (spike_sample < light_in_tray_ttl[tri + 1])) 
            f_baseline_t = len(baseline_t[0]) / (abs(light_in_tray_ttl[tri+1] - light_in_tray_ttl[tri]) / sr)
            f_baseline_total.append(f_baseline_t)
        mean_f_baseline_total = np.mean(f_baseline_total)
        #########################################
        # Calculate mean baseline firing rate ITI
        f_ITI_total = []
        for tri in range((len(response))-1):
            spike_sample = spike_train[st]
            ITI_t = np.where((spike_sample > ITI_ttl[tri]) & (spike_sample < light_in_tray_ttl[tri + 1])) 
            f_ITI_t = len(ITI_t[0]) / (abs(light_in_tray_ttl[tri+1] - ITI_ttl[tri]) / sr)
            f_ITI_total.append(f_ITI_t)
        mean_f_ITI_total = np.mean(f_ITI_total)
        ######################################### 
        #initializing the rest of the dictionary
        cell_features[f"{dict_key}, c{st+1}"] = {
        'cell_type':[],
        'firing_rate':[],
        'firing_rate_beh':[],
        'firing_rate_ITI':[], 
        'trough_to_peak':[],
        'burst_Royer':[]
        }

        ##############################################################################################    
        cell_features[f"{dict_key}, c{st+1}"]['cell_type'] = cell_type_FF
        cell_features[f"{dict_key}, c{st+1}"]['firing_rate'] = round(Firing_rate, 2)  
        cell_features[f"{dict_key}, c{st+1}"]['firing_rate_beh'] = round(mean_f_baseline_total, 2)    
        cell_features[f"{dict_key}, c{st+1}"]['firing_rate_ITI'] = round(mean_f_ITI_total, 2) 
        cell_features[f"{dict_key}, c{st+1}"]['trough_to_peak'] = round(trough_t_peak, 3)  
        cell_features[f"{dict_key}, c{st+1}"]['burst_Royer'] = round(burst_Royer, 3)  


# In[ ]:


for cell in cell_features:
    print(cell)


# In[ ]:


for cell in cell_features:
    print(cell_features[cell]['firing_rate'])


# In[ ]:


# old
# # extracting cell features ##

# ##########################################
# # Load the dictionary from the pickle file
# with open("J:\Spike_train_analysis\pickle\mossy-granule-all", 'rb') as f:
#     mossy_granule = pickle.load(f)

# # with open("C:\NASTARAN\analysis\feb 2025\pickle\mossy-granule-all", 'rb') as f:
# #     mossy_granule = pickle.load(f)

# ##########################################
# sr = 30000
# epsilon = 1e-10  # A small constant 

#    # generating a dictionary to store the results
# cell_features = {}
# # for file in pickle_files:
# #     key = os.path.basename(file)[10:]
# #     cell_features[key] = {}
# ###########

# #for mice in range(20,21):
# for mice in range(len(mice_meta_data)):
#     meta_data = mice_meta_data[mice]
#     dict_key = os.path.basename(pickle_files[mice])[10:]
#     print(dict_key)
#     #####################################################################
#     #exctracting ttl and lables from the meta_data
#     light_in_tray_ttl = []
#     poke_in_tray_ttl = []
#     image_ttl = []
#     second_poke_ttl = []
#     action_ttl = []
#     feedback_ttl = []
#     reward_consumption_ttl = []
#     no_reward_ttl = []
#     response = []
#     images = []
#     cr_trial = []
#     no_reward_exc = []
#     ITI_ttl = []

#     for tri in range(len(meta_data['session_recording']['TTL'])):
#         light_in_tray_ttl.append(meta_data['session_recording']['TTL'][tri].light_in_tray_ttl)
#         poke_in_tray_ttl.append(meta_data['session_recording']['TTL'][tri].poke_in_tray_ttl)
#         image_ttl.append(meta_data['session_recording']['TTL'][tri].image_ttl)
#         second_poke_ttl.append(meta_data['session_recording']['TTL'][tri].second_poke_ttl)
#         action_ttl.append(meta_data['session_recording']['TTL'][tri].action_ttl)
#         feedback_ttl.append(meta_data['session_recording']['TTL'][tri].feedback_ttl)
#         reward_consumption_ttl.append(meta_data['session_recording']['TTL'][tri].reward_consumption_ttl)

#         if (meta_data['session_recording']['TTL'][tri].action_ttl) is not None:
#             no_reward_ttl.append((meta_data['session_recording']['TTL'][tri].action_ttl) + (1*sr))
#         else:
#             no_reward_ttl.append(meta_data['session_recording']['TTL'][tri].action_ttl)

#         if (meta_data['session_recording']['TTL'][tri].reward_consumption_ttl) is not None:
#             if (meta_data['session_recording']['TTL'][tri].reward_consumption_ttl) > 0:
#                 ITI_ttl.append((meta_data['session_recording']['TTL'][tri].reward_consumption_ttl) + (5*sr))  
#             else:
#                 ITI_ttl.append((meta_data['session_recording']['TTL'][tri].action_ttl) + (6*sr))  
#         else:
#             ITI_ttl.append(meta_data['session_recording']['TTL'][tri].reward_consumption_ttl)  

#     for tri in range(len(meta_data['session_recording']['labels'])):
#         response.append(meta_data['session_recording']['labels'][tri].response)
#         images.append(meta_data['session_recording']['labels'][tri].image)
#         cr_trial.append(meta_data['session_recording']['labels'][tri].correction_trial)
#         no_reward_exc.append(meta_data['session_recording']['labels'][tri].no_reward_exc) 

#     spike_train = meta_data['session_recording']['spike_time_beh']
#     cells = meta_data['session_recording']['spike_time_beh']
#     cell_metrics = meta_data['cell_metrics']
#     #####################################################################
#     for st in range((len(cells))):
#     #for st in range(0,1):  # Modify range as needed
#     #####################################################################
#         # Calculate mean baseline firing rate
#         f_baseline_total = []
#         for tri in range((len(response))-1):
#             spike_sample = spike_train[st]
#             baseline_t = np.where((spike_sample > light_in_tray_ttl[tri]) & (spike_sample < light_in_tray_ttl[tri + 1])) 
#             f_baseline_t = len(baseline_t[0]) / (abs(light_in_tray_ttl[tri+1] - light_in_tray_ttl[tri]) / sr)
#             f_baseline_total.append(f_baseline_t)
#         mean_f_baseline_total = np.mean(f_baseline_total)
#     ####################################################################### 
#         # Calculate mean baseline firing rate
#         f_ITI_total = []
#         for tri in range((len(response))-1):
#             spike_sample = spike_train[st]
#             ITI_t = np.where((spike_sample > ITI_ttl[tri]) & (spike_sample < light_in_tray_ttl[tri + 1])) 
#             f_ITI_t = len(ITI_t[0]) / (abs(light_in_tray_ttl[tri+1] - ITI_ttl[tri]) / sr)
#             f_ITI_total.append(f_ITI_t)
#         mean_f_ITI_total = np.mean(f_ITI_total)
#     ####################################################################### 
#         #initializing the rest of the dictionary
#         cell_features[f"{dict_key}, c{st+1}"] = {
#         'cell_type':[],
#         'firing_rate':[],
#         'firing_rate_beh':[],
#         'firing_rate_ITI':[]
#         }
#     #######################################################################
#         if cell_metrics['putativeCellType'][0][0][st][0] == 'Pyramidal Cell':
#             #print('yes', st+1)
#             #print(f"{dict_key}, c{st+1}")
#             if f"{dict_key}, c{st+1}" in mossy_granule['mossy']:
#                 #print('yes mossy')
#                 cell_features[f"{dict_key}, c{st+1}"]['cell_type'] = 'Mossy' 
#             elif f"{dict_key}, c{st+1}" in mossy_granule['granule']:
#                 #print('yes granule')
#                 cell_features[f"{dict_key}, c{st+1}"]['cell_type'] = 'Granule'   
#         else:
#             cell_features[f"{dict_key}, c{st+1}"]['cell_type'] = cell_metrics['putativeCellType'][0][0][st][0]          

#         cell_features[f"{dict_key}, c{st+1}"]['firing_rate'] = cell_metrics['firingRate'][0][0][st]     
#         cell_features[f"{dict_key}, c{st+1}"]['firing_rate_beh'] = mean_f_baseline_total     
#         cell_features[f"{dict_key}, c{st+1}"]['firing_rate_ITI'] = mean_f_ITI_total   


# In[ ]:




