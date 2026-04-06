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
import json

# File paths
PROCESSED_FILE_DIR = r'C:\Users\Kelly\PythonCode\GitProject\HFM_EphysAnalysis\data\processedData'

# helper function
def unwrap(val):
    """Recursively unwrap numpy arrays/void to clean Python types."""
    # Unwrap single-element arrays
    while isinstance(val, np.ndarray) and val.shape == (1,) and val.dtype.names is None:
        val = val[0]
    
    # Handle structured arrays (numpy void or named dtype)
    if isinstance(val, np.ndarray) and val.dtype.names:
        return {name: unwrap(val[name]) for name in val.dtype.names}
    if isinstance(val, np.void):
        return {name: unwrap(val[name]) for name in val.dtype.names}
    
    # Handle object arrays (often contain strings or nested arrays)
    if isinstance(val, np.ndarray) and val.dtype == object:
        if val.size == 1:
            return unwrap(val.flat[0])
        unwrapped = [unwrap(v) for v in val.flat]
        try:
            return np.array(unwrapped).reshape(val.shape)
        except ValueError:
            try:
                return np.array(unwrapped, dtype=object).reshape(val.shape)
            except ValueError:
                return np.array(unwrapped, dtype=object)
    
    # Handle regular arrays - return as-is or convert to list
    if isinstance(val, np.ndarray):
        if val.size == 1:
            return val.flat[0].item() if hasattr(val.flat[0], 'item') else val.flat[0]
        return val  # keep as numpy array for numeric data (waveforms, acg, etc.)
    
    # Convert numpy scalars to Python native types
    if isinstance(val, (np.integer, np.floating)):
        return val.item()
    
    return val

class EphysTrial: #this is the actual important class for combining behav data and ephys
    def __init__(self):
        pass
    
    # 3 JSON helper methods for cell_metrics
    @staticmethod
    def _json_default(obj):
        """Handle numpy types for JSON serialization."""
        if isinstance(obj, np.ndarray):
            return {'__ndarray__': True, 'data': obj.tolist(), 'dtype': str(obj.dtype), 'shape': list(obj.shape)}
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, bytes):
            return obj.decode('utf-8', errors='replace')
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    @staticmethod
    def _json_object_hook(obj):
        """Reconstruct numpy arrays on load."""
        if obj.get('__ndarray__'):
            return np.array(obj['data'], dtype=obj['dtype']).reshape(obj['shape'])
        return obj

    def _prepare_dict_for_mat(self):
        """Prepare __dict__ for savemat, serializing cell_metrics as JSON."""
        out = dict(self.__dict__)
        if 'cell_metrics' in out:
            out['cell_metrics'] = json.dumps(out['cell_metrics'], default=self._json_default)
        return out
    
    def Store(self): #stores all experimental data and metadata as a mat file
        save_path = os.path.join(PROCESSED_FILE_DIR, self.exp, self.filename)
        
        #creates experiment directory if it does not already exist
        if not os.path.exists(os.path.dirname(save_path)):
            os.mkdir(os.path.dirname(save_path))
        #checks if save file already exists
        if os.path.exists(save_path):
            raise IOError(f'File {self.filename} already exists.')
        
        #create mat file here
        scipy.io.savemat(save_path,self._prepare_dict_for_mat(),long_field_names=True)
        
    @classmethod
    def Load(cls, exp, mouse, trial, load_cell_metrics=False):
        try: path = glob.glob(glob.glob(PROCESSED_FILE_DIR+'/'+exp+'/')[0]+'*M%s_%s.mat'%(mouse, trial), 
                         recursive = True)[0] #finds file path based on ethovision trial number
        except: raise ValueError('The specified file does not exist ::: %s Mouse %s Trial %s'%(exp,mouse,trial))
        
        # Load MATLAB file
        mat_data = scipy.io.loadmat(path)
        #Create an instance of the class
        obj = cls()
        
        # Assign attributes dynamically based on the keys of mat_data
        for key in mat_data:
            if key.startswith('__'): pass
            elif key == 't_spikeTrains':
                setattr(obj, key, [np.squeeze(sp) for sp in mat_data['t_spikeTrains'][0].tolist()])
            elif key == 'cell_metrics':
                if not load_cell_metrics:
                    pass
                else:
                    val = mat_data[key]
                    # New format: JSON string
                    if val.dtype.kind in ('U', 'S'):  # unicode or byte string
                        raw = str(np.squeeze(val))
                        setattr(obj, key, json.loads(raw, object_hook=cls._json_object_hook))
                    # Old format: MATLAB struct — use unwrap
                    else:
                        squeezed = np.squeeze(val)
                        setattr(obj, key, {name: unwrap(squeezed[name]) for name in squeezed.dtype.names})
            else:
                if mat_data[key].shape == (1,1):
                    setattr(obj, key, mat_data[key][0][0])
                elif key.startswith('r_') or key == 'k_hole_checks':
                    setattr(obj, key, mat_data[key])
                else:
                    setattr(obj, key, mat_data[key][0])
        
        return obj
    
    def Update(self):
        save_path = os.path.join(PROCESSED_FILE_DIR, self.exp, self.filename)
        
        #checks if save file already exists
        if not os.path.exists(save_path):
            raise IOError(f'File {self.filename} does not exist.')
        
        # Safeguard: don't overwrite if cell_metrics was not loaded
        if not hasattr(self, 'cell_metrics'):
            raise RuntimeError('Cannot Update: cell_metrics was not loaded. Reload with load_cell_metrics=True to save.')
            
        #update mat file here
        scipy.io.savemat(save_path,self._prepare_dict_for_mat(),long_field_names=True)


    
if __name__ == "__main__":
    
    edata = EphysTrial.Load(exp='2025-01-21', mouse='112', trial=14)