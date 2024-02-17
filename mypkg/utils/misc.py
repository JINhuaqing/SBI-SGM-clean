import numpy as np
import scipy
import pickle
from easydict import EasyDict as edict
import logging


def _set_verbose_level(verbose, logger):
    if verbose == 0:
        verbose_lv = logging.ERROR
    elif verbose == 1:
        verbose_lv = logging.WARNING
    elif verbose == 2:
        verbose_lv = logging.INFO
    elif verbose == 3:
        verbose_lv = logging.DEBUG
    if len(logger.handlers)>0:
        logger.handlers[0].setLevel(verbose_lv)
    else:
        logger.setLevel(verbose_lv)

def _update_params(input_params, def_params, logger):
    for ky, v in input_params.items():
        if ky not in def_params.keys():
            logger.warning(f"Check your input, {ky} is not used.")
        else:
            if v is not None:
                def_params[ky] = v
    return edict(def_params)

def mat_power(mat, pv=-1):
    """mat is a symmetric positive-semi-definite matrix
    """
    eps = 1e-10
    
    # p-inv
    S, U = np.linalg.eig(mat)
    assert np.sum(S[np.abs(S)>eps] <0) == 0, "mat should be PSD mat"
    
    S = np.abs(S) # S >=0 for PSD
    S1 = S.copy()
    S1[S<=eps] = 0
    S1[S>eps] = (S[S>eps])**(pv)
    
    mat_r = U @ np.diag(S1) @ U.T
    return mat_r

def geodesic_dist(Q1, Q2):
    """Calculate the geodesic distance between two sys-PSD matrices. 
       Strictly, Q1 and Q2 should be invertible, but I let it can be PSD
       Follows https://github.com/makto-toruk/FC_geodesic/blob/master/utils/distance_FC/distance_FC.py
    """
    eps = 1e-10
    Q1_neg_half = mat_power(Q1, -1/2)
    Q = Q1_neg_half @ Q2 @ Q1_neg_half
    eigvs, _ = np.linalg.eig(Q)
    assert np.sum(eigvs[np.abs(eigvs)>eps] <0) == 0, "mat should be PSD mat"
    
    eigvs = np.abs(eigvs) # Q is PSD
    eigvs_part = eigvs[eigvs>eps]
    dist = np.sqrt(np.sum(np.log(eigvs_part)**2))
    return dist


# load file from pkl
def load_pkl(fil, verbose=True):
    if verbose:
        print(f"Load file {fil}")
    with open(fil, "rb") as f:
        result = pickle.load(f)
    return result

# load file from folder and save it to dict
def load_pkl_folder2dict(folder, excludes=[], verbose=True):
    all_exc = [list(folder.glob(exclude+".pkl")) for exclude in excludes]
    all_exc_stem = []
    for exc in all_exc:
        cur_stem = [fil.stem for fil in exc]
        all_exc_stem.extend(cur_stem)
    all_exc_stem = list(set(all_exc_stem))
    res = edict()
    for fil in folder.glob('*.pkl'):
        if fil.stem not in all_exc_stem:
            res[fil.stem] = load_pkl(fil, verbose)
    return res

# save a dict into a folder
def save_pkl_dict2folder(folder, res, excludes=[], is_force=False):
    assert isinstance(res, dict)
    for ky, v in res.items():
        if ky not in excludes:
            save_pkl(folder/f"{ky}.pkl", v, is_force=is_force)

# save file to pkl
def save_pkl(fil, result, is_force=False):
    if not fil.parent.exists():
        fil.parent.mkdir()
        print(fil.parent)
        print(f"Create a folder {fil.parent}")
    if is_force or (not fil.exists()):
        print(f"Save to {fil}")
        with open(fil, "wb") as f:
            pickle.dump(result, f)
    else:
        print(f"{fil} exists! Use is_force=True to save it anyway")

# get the mode of a vec
def get_mode(x):
    kde_est = scipy.stats.gaussian_kde(x)
    xs = np.linspace(x.min(), x.max(), 500)
    ys = kde_est(xs)
    return xs[np.argmax(ys)]

def mag2db(y):
    """Convert magnitude response to decibels for a simple array.

    Args:
        y (numpy array): Power spectrum, raw magnitude response.

    Returns:
        dby (numpy array): Power spectrum in dB

    """
    dby = 20 * np.log10(y)
    return dby



# calculate the spatial feature for MEG under the given band
def meg_psd2spatialfeature(C, PSD, freqs, band="alpha"):
    """
        args:
            C: Connectome
            PSD: spectrum
            freqs: Frequency of the PSD
            band: the freq band selected
    """
    if band == "alpha":
        freqband = np.where((freqs>=8) & (freqs<=12))[0]
    elif band == "beta":
        freqband = np.where((freqs>=13) & (freqs<=25))[0]
    else:
        freqband = np.arange(len(freqs))
    
    summed_psd = np.sum(PSD[:,freqband], axis = 1)
    summed_psd = summed_psd/np.linalg.norm(summed_psd)

    w_spat = 10.0

    rowdegree = np.transpose(np.sum(C, axis=1))
    coldegree = np.sum(C, axis=0)
    qind = rowdegree + coldegree < 0.2 * np.mean(rowdegree + coldegree)
    rowdegree[qind] = np.inf
    coldegree[qind] = np.inf
    L2 = np.divide(1, np.sqrt(np.multiply(rowdegree, coldegree)) + np.spacing(1))
    Cc = np.matmul(np.diag(L2), C)
    
    C2 = Cc + w_spat*np.eye(86)
    rowdegree = np.transpose(np.sum(C2, axis=1))
    coldegree = np.sum(C2, axis=0)
    qind = rowdegree + coldegree < 0.2 * np.mean(rowdegree + coldegree)
    rowdegree[qind] = np.inf
    coldegree[qind] = np.inf
    L22 = np.divide(1, np.sqrt(np.multiply(rowdegree, coldegree)) + np.spacing(1))
    Cc2 = np.matmul(np.diag(L22), C2)    
    
    func1 = np.matmul(Cc2[0:68,0:68], summed_psd)
    return func1, PSD[:, freqband] # first for calculating the spatial correlation, second for plotting on brain Nrois x Nfreqs
