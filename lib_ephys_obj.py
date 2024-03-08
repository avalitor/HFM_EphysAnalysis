# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 17:17:09 2023

library of classes and functions for ephys

@author: student
"""
import matplotlib.pyplot as plt
import pickle
from modules.lib_process_data_to_mat import TrialData
import glob #for file search
import config as cfg
import scipy.io
import os
import numpy as np


class EphysEpoch(object):
    def __init__(self, trial='', t_TTL=[], t_spike_train=[], spike_labels=[], sample_rate=[], firingRate=[] ):
        self.trial = trial
        self.t_TTL = t_TTL
        self.t_spike_train = t_spike_train
        self.spike_labels = spike_labels
        self.sample_rate = sample_rate
        self.firingRate = firingRate
      
    def plot_raster(self, neuron = None):
        '''
        Make a simple raster plot of all neurons or a single neuron
        '''
        if neuron == None:    
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 8))
            lineoffsets1 = range(len(self.t_spike_train))
            ax.eventplot(self.t_spike_train, lineoffsets=lineoffsets1)
            plt.vlines(self.t_TTL, min(lineoffsets1)-1, max(lineoffsets1)+1, color='r')
            plt.show()
        else:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 4))
            ax.eventplot(self.t_spike_train[neuron], lineoffsets=0, linelengths=1)
            plt.vlines(self.t_TTL, -1, 1, colors='r')
            plt.title(f'Neuron {neuron}')
            plt.show()
    
    def Load(self, exp, mouse, epoch):
        file = open(rf'F:\Spike Sorting\Data\3_Raster\{exp}_M{mouse}\{epoch}.EphysEpoch', 'rb')
        m = pickle.load(file)
        # self.trial = m.trial
        # self.t_TTL = m.t_TTL
        # self.t_spike_train = m.t_spike_train
        # self.spike_labels = m.spike_labels
        # self.sample_rate = m.sample_rate
        file.close()
        return m
    def Update(self, exp, mouse, epoch):
        file=open(rf'F:\Spike Sorting\Data\3_Raster\{exp}_M{mouse}\{epoch}.EphysEpoch', 'wb')
        pickle.dump(self, file)
        file.close()
        

        
class EphysTrial(TrialData):
    def __init__(self, exp='', protocol_name='', protocol_description='',
                    eth_file=[], bkgd_img='', img_extent=[], experimenter='',
                    mouse_number='', mouse_sex = '', day='', trial='', entrance='', target=[],
                    time=[], r_nose=[], r_center=[], r_tail=[], filename='', 
                    spikeTrains=[], cellLabels=[], firingRates=[], sampleRate=[]):
      super().__init__(exp, protocol_name, protocol_description, eth_file, 
                       bkgd_img, img_extent, experimenter, mouse_number, mouse_sex, 
                       day, trial, entrance, target, time, r_nose, r_center, r_tail, filename)
      self.t_spikeTrains = spikeTrains
      self.cellLabels = cellLabels
      self.firingRates = firingRates
      self.sampleRate = sampleRate
    
    def Store(self): #stores all experimental data and metadata as a mat file
        save_path = os.path.join(cfg.PROCESSED_FILE_DIR, self.exp, self.filename)
        
        #creates experiment directory if it does not already exist
        if not os.path.exists(os.path.dirname(save_path)):
            os.mkdir(os.path.dirname(save_path))
        #checks if save file already exists
        if os.path.exists(save_path):
            raise IOError(f'File {self.filename} already exists.')
        
        #create mat file here
        scipy.io.savemat(save_path,self.__dict__,long_field_names=True)
        
    def Load(self, exp, mouse, trial):
        super().Load(exp, mouse, trial)
        
        try: path = glob.glob(glob.glob(cfg.PROCESSED_FILE_DIR+'/'+exp+'/')[0]+'*M%s_%s.mat'%(mouse, trial), 
                         recursive = True)[0] #finds file path based on ethovision trial number
        except: raise ValueError('The specified file does not exist ::: %s Mouse %s Trial %s'%(exp,mouse,trial))
        m = scipy.io.loadmat(path)
        if 'time_ttl' in m:
            self.time_ttl = m['time_ttl'][0]
        self.t_spikeTrains = [np.squeeze(sp) for sp in m['t_spikeTrains'][0].tolist()]
        self.cellLabels = m['cellLabels'][0]
        self.firingRates = m['firingRate'][0]
        self.sampleRate = m['sampleRate'][0][0]
        
    # def Load(self, exp, mouse, trial): #loads mat file
    #     try: path = glob.glob(glob.glob(cfg.PROCESSED_FILE_DIR+'/'+exp+'/')[0]+'*M%s_%s.mat'%(mouse, trial), 
    #                      recursive = True)[0] #finds file path based on ethovision trial number
    #     except: raise ValueError('The specified file does not exist ::: %s Mouse %s Trial %s'%(exp,mouse,trial))
    #     m = scipy.io.loadmat(path)
    #     self.exp = exp
    #     self.protocol_name=m['protocol_name'][0]
    #     self.protocol_description=m['protocol_description'][0]
    #     self.eth_file = m['eth_file'][0][0]
    #     self.bkgd_img=m['bkgd_img'][0]
    #     if isinstance(m['img_extent'], str): #if is string, convert to floats
    #         self.img_extent=[float(x) for x in m['img_extent'][0].split(',')]
    #     else: self.img_extent=m['img_extent'][0]
    #     self.experimenter=m['experimenter'][0]
    #     self.mouse_number=m['mouse_number'][0]
    #     self.mouse_sex=m['mouse_sex'][0]
    #     self.day=m['day'][0]
    #     self.trial=m['trial'][0]
    #     self.entrance=m['entrance'][0]
    #     self.target=m['target'][0]
    #     if 'target_reverse' in m:
    #         self.target_reverse=m['target_reverse'][0]
    #     self.time=m['time'][0]
    #     self.r_nose=m['r_nose']
    #     self.r_center=m['r_center']
    #     self.r_tail=m['r_tail']
    #     self.filename=m['filename'][0]
    #     if "velocity" in m: #checks if file has velocity info
    #         self.velocity = m['velocity'][0]
    #     if "head_direction" in m: #checks if file has velocity info
    #         self.head_direction = m['head_direction'][0]
    #     if 'r_arena_holes' in m:
    #         self.r_arena_holes = m['r_arena_holes']
    #     if 'arena_circle' in m:
    #         self.arena_circle = m['arena_circle'][0]
    #     if 'k_reward' in m:
    #         self.k_reward = m['k_reward'][0][0]
    #     if 'k_hole_checks' in m:
    #         self.k_hole_checks = m['k_hole_checks']
    #     if 'time_ttl' in m:
    #         self.time_ttl = m['time_ttl'][0]
    #     # self.t_spikeTrain =  m['t_spikeTrain'][0].tolist()
    #     self.t_spikeTrains = [np.squeeze(sp) for sp in m['t_spikeTrain'][0].tolist()]
    #     self.cellLabels = m['cellLabels'][0]
    #     self.firingRates = m['firingRate'][0]
    #     self.sampleRate = m['sampleRate'][0][0]
        
    def Update(self):
        save_path = os.path.join(cfg.PROCESSED_FILE_DIR, self.exp, self.filename)
        
        #checks if save file already exists
        if not os.path.exists(save_path):
            raise IOError(f'File {self.filename} does not exist.')
            
        #update mat file here
        scipy.io.savemat(save_path,self.__dict__,long_field_names=True)
            