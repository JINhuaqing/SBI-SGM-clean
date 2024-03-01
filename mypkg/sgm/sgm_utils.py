import numpy as np
import mne
from mne_connectivity import spectral_connectivity_epochs
from scipy import signal
from easydict import EasyDict as edict
from scipy.interpolate import interp1d
from utils.misc import _set_verbose_level, _update_params
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if not logger.hasHandlers():
    ch = logging.StreamHandler() # for console. 
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch) 
    
_set_verbose_level(1, logger)
mne.set_log_level(logging.WARNING)
    
def _2matf(mat):
    mat_f = mat + mat.T
    mat_f = mat_f - np.diag(np.diag(mat))
    return mat_f

def get_fc(input_signal, fs, bds, fc_params={}):
    """
    Calculate functional connectivity from input_signal using the parameters in paras.

    Parameters:
    input_signal (numpy.ndarray): Input signal with shape (n_channels, n_timepont)
    fs (int): Sampling freq of the data, in Hz
    bds (list of two): freq band limits, [low, high], in Hz
    fc_params (dict): Dictionary containing the following keys and values:
        - nepoch (int): Number of epochs
        - fc_type (str): Functional connectivity method
        - f_skip (int): Frequency skip
        - num_taps (int): Length of the filter 

    Returns:
    numpy.ndarray: Functional connectivity matrix
    """
    fc_params_def = edict({
        "nepoch": 50, 
        "fc_type": "coh", 
        "f_skip": 10, 
        "num_taps": 351
    })
    fc_params = _update_params(fc_params, fc_params_def, logger)
    
    input_signal = signal.detrend(input_signal, axis=1, type="linear", bp=0, overwrite_data=False);
    bii = signal.firwin(fc_params.num_taps, 
                        bds,
                        pass_zero=False, 
                        fs=fs, 
                        window="hamming");


    input_signal = signal.filtfilt(bii, 1, input_signal, axis=1);
    
    if fc_params.nepoch == 1:
        input_signal = input_signal[np.newaxis]
    else:
        input_signal = input_signal.reshape(68, fc_params.nepoch, -1).transpose(1, 0, 2)
    
    ts_con = spectral_connectivity_epochs(input_signal,
                                          names=None, 
                                          method=fc_params.fc_type, 
                                          indices=None, 
                                          sfreq=fs, 
                                          mode='multitaper',
                                          fmin=None, 
                                          fmax=np.inf,
                                          fskip=fc_params.f_skip, 
                                          faverage=True, 
                                          tmin=None, 
                                          tmax=None,  
                                          mt_bandwidth=None, 
                                          mt_adaptive=False, 
                                          mt_low_bias=True, 
                                          cwt_freqs=None, 
                                          cwt_n_cycles=7, 
                                          block_size=1000, 
                                          n_jobs=1, 
                                          verbose=False)
    mat = ts_con.get_data(output='dense').squeeze();
    return np.abs(_2matf(mat))


def obt_psd_at_freqs(psd_raw, f, fvec):
    """
    Calculate the power spectral density (PSD) at given frequency points.

    Parameters:
    psd_raw (array-like): The estimated PSD from Welch's method, not in dB not squared
    f (array-like): The frequency vector corresponding to the PSD.
    fvec (array-like): The frequency points at which to calculate the PSD.

    Returns:
    array-like: The PSD values at the given frequency points (not in dB)

    Notes:
    - The input PSD is expected to be in linear scale, i.e., not in dB
    - The PSD values are converted to dB scale using a small epsilon value to avoid taking the logarithm of zero.
    - The PSD is smoothed using a 5-point symmetric linear-phase FIR filter.
    - The PSD values at the given frequency points are obtained using linear interpolation.

    """
    eps = 1e-10
    psd_dB = 20*np.log10(psd_raw+eps)
    
    # Smooth the PSD
    lpf = np.array([1, 2, 5, 2, 1]) 
    lpf = lpf/np.sum(lpf)
    psd_dB = np.convolve(psd_dB, lpf, 'same')
    
    fit_psd = interp1d(f, psd_dB)
    return 10**(fit_psd(fvec)/20)


def get_psd(input_signal, freqs, fs, psd_params={}):
    """
    Calculate functional connectivity from input_signal using the parameters in paras.

    Parameters:
    input_signal (numpy.ndarray): Input signal with shape (n_channels, n_timepont)
    freqs (array): The freq pts at which you want to get PSD, in Hz
    fs (int): Sampling freq of the data, in Hz
    psd_params (dict): Dictionary containing the following keys and values:
        - nperseg (int): Length of each segment.
        - num_taps (int): Length of the filter 

    Returns:
    numpy.ndarray: the PSD at freqs, in abs mag, 20log10 to dB
    """
    psd_params_def = edict({
        "num_taps": 351,
        "nperseg": 128
    })
    bds = [freqs[0], freqs[-1]]
    psd_params = _update_params(psd_params, psd_params_def, logger)
    
    input_signal = signal.detrend(input_signal, axis=1, type="linear", bp=0, overwrite_data=False);
    bii = signal.firwin(psd_params.num_taps, 
                        bds,
                        pass_zero=False, 
                        fs=fs, 
                        window="hamming");


    input_signal = signal.filtfilt(bii, 1, input_signal, axis=1);
    
    # not in dB, not squared
    f, Pxx = signal.welch(input_signal, fs=fs, axis=1, detrend=False)
    low_idx = np.argmin(np.abs(f - bds[0]))
    if f[low_idx] > bds[0]:
        low_idx = low_idx -1
    high_idx = np.argmin(np.abs(f - bds[1]))
    if f[high_idx] < bds[1]:
        high_idx = high_idx +1
    f_sub = f[low_idx:(high_idx+1)]
    Pxx_sub = Pxx[:, low_idx:(high_idx+1)]
    
    Pxx_target = np.array([obt_psd_at_freqs(Pxx_sub[roi_ix], f_sub, freqs) 
                           for roi_ix in range(input_signal.shape[0])]);
    return Pxx_target