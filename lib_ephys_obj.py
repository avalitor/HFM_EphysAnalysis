# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 17:17:09 2023

library of classes and functions for ephys

@author: student
"""
import matplotlib.pyplot as plt
import pickle

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
        
        
class EphysTrial(object):
    def __init__(self, exp, mouse_number, trial, filename, spikeTrains, spikeLabels, firingRate, sampleRate):
        self.exp = exp
        self.mouse_number = mouse_number
        self.trial = trial
        self.filename = filename
        
        self.t_spikeTrain = spikeTrains
        self.spikeLabels = spikeLabels
        self.firingRate = firingRate
        self.sampleRate = sampleRate
        
        
class EphysTrial(TrialData):
  def __init__(self, fname, lname, year):
    super().__init__(fname, lname)
    self.graduationyear = year
        