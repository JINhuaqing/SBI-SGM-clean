import numpy as np
from scipy.interpolate import interp1d
def obt_psd_at_freqs(psd_raw, f, fvec):
    """
    Calculate the power spectral density (PSD) at given frequency points.

    Parameters:
    psd_raw (array-like): The estimated PSD from Welch's method.
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
    psd_dB = 10*np.log10(psd_raw+eps)
    
    # Smooth the PSD
    lpf = np.array([1, 2, 5, 2, 1]) 
    lpf = lpf/np.sum(lpf)
    psd_dB = np.convolve(psd_dB, lpf, 'same')
    
    fit_psd = interp1d(f, psd_dB)
    return 10**(fit_psd(fvec)/10)

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
