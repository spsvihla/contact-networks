import os
import numpy as np
from scipy.io import loadmat


def load_data():
    data_dir = "../data"
    data = dict()

    for fname in os.listdir(data_dir):
        fpath = os.path.join(data_dir, fname)
        if not os.path.isfile(fpath) or not fname.endswith('.mat'):
            continue
        
        # Strip the extension and get base name
        name = os.path.splitext(fname)[0]
        
        # Check if the name has '_pres' suffix
        if name.startswith("A_pres_"):
            base_name = name[7:]  # Remove 'A_pres_' to get base key
            is_copresent = True
        elif name.startswith("A_"):
            base_name = name[2:]  # Remove 'A_' to get base key
            is_copresent = False
        else:
            # Skip unexpected files that don't match the expected pattern
            continue
        
        # Load the data
        X = np.ascontiguousarray(loadmat(fpath)['U']).astype(np.int64)
        
        # Initialize nested dictionary for each base name if not already present
        if base_name not in data:
            data[base_name] = {'face-to-face': None, 'co-present': None}
        
        # Assign data to 'pres' or 'regular' entry
        if is_copresent:
            data[base_name]['co-present'] = X
        else:
            data[base_name]['face-to-face'] = X
    return data