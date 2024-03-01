import numpy as np
from .misc import mag2db


# x: Nrois x Nfreq
# orginal name is std_psd
def stdz_psd(x):
    assert x.shape[-1] == 40
    res = (x- x.mean(axis=1).reshape(-1, 1))/x.std(axis=1).reshape(-1, 1)
    return res

# original name is std_vec
stdz_vec = lambda x: (x-x.mean())/x.std()
minmax_vec = lambda x: (x-x.min())/(x.max()-x.min())

def minmax_fn(x, byrow=False):
    if x.ndim == 1:
        minmax_x = (x-x.min())/(x.max()-x.min())
    elif x.ndim == 2:
        if not byrow:
            x = x.T
        minmax_x = ((x - x.min(axis=1, keepdims=1))/(x.max(axis=1, keepdims=1) - x.min(axis=1, keepdims=1)))
        
        if not byrow:
            minmax_x = minmax_x.T
    return minmax_x



# convert psd to training format
# abs -> mag2db -> stdize for each ROIs
def psd_2tr(psd):
    psd = np.abs(psd)
    psd_db = mag2db(psd)
    std_psd_db = stdz_psd(psd_db)
    return std_psd_db

# convert psd to training format
# abs -> mag2db -> stdize across all ROIs
def psd_2tr_vec(psd):
    psd = np.abs(psd)
    psd_db = mag2db(psd)
    std_psd_db_vec = stdz_vec(psd_db.flatten())
    return std_psd_db_vec
