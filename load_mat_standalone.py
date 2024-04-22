# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 23:31:40 2024

change MATLAB_FILE_DIR to where you stored all the experiment files

@author: Kelly
"""
import scipy.io
import glob
import numpy as np
import os

class EphysTrial:
    def __init__(self):
        pass
    
    @classmethod
    def Load(cls, exp, mouse, trial):
        #replace this with the path to the directory with experiment folders containing the mat files
        MATLAB_FILE_DIR = r'C:\Users\Kelly\PythonGraphingProjects\GitProject\HFM_EphysAnalysis\data\processedData'
        
        try: path = glob.glob(glob.glob(MATLAB_FILE_DIR+'/'+exp+'/')[0]+'*M%s_%s.mat'%(mouse, trial), 
                         recursive = True)[0] #finds file path based on ethovision trial number
        except: raise ValueError('The specified file does not exist ::: %s Mouse %s Trial %s'%(exp,mouse,trial))
        # Load MATLAB file
        mat_data = scipy.io.loadmat(path)
        # Create an instance of the class
        obj = cls()
        # Assign attributes dynamically based on the keys of mat_data
        for key in mat_data:
            if key.startswith('__'): pass
            elif key == 't_spikeTrains':
                setattr(obj, key, [np.squeeze(sp) for sp in mat_data['t_spikeTrains'][0].tolist()])
            else:
                if mat_data[key].shape == (1,1):
                    setattr(obj, key, mat_data[key][0][0])
                elif key.startswith('r_') or key == 'k_hole_checks':
                    setattr(obj, key, mat_data[key])
                else:
                    setattr(obj, key, mat_data[key][0])
        return obj
    
if __name__ == "__main__":
    
    test = EphysTrial.Load(exp='2024-02-15', mouse='105', trial=14)