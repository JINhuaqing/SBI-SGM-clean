from fooof import FOOOF
import numpy as np


def obt_psm_fs(raw_psd, freqs):
    """
        extract features from Power spectrum models
        two components:
            1. Peaks, [center freq, power, bandwidth], I choose two peaks.
            2. Aperiodic part: offset and Exponent
        args:
            raw_psd: the power of spectrum to extract features
            freqs: corresponding freqs
    """
    freq_range = [np.amin(freqs), np.amax(freqs)]
    
    # to smooth the raw psd
    lpf = np.array([1, 2, 3, 2, 1])
    lpf = lpf/np.sum(lpf)
    sm_psd = np.convolve(raw_psd,lpf,'same')
    
    # the fooof obj
    fm = FOOOF(peak_width_limits=[2*np.diff(freqs)[0], 12.0])
    fm.fit(freqs, sm_psd, freq_range)
    if fm.n_peaks_ == 1:
        peaks_fs = np.concatenate([fm.peak_params_[0, :], fm.peak_params_[0, :]])
    elif fm.n_peaks_ == 0:
        peaks_fs = np.zeros(6) # if no enough peak, pool it with 0
    else:
        peaks_fs = fm.peak_params_[:2, :].flatten()
    
    # Power Spectrum Models
    PSM_fs  =  peaks_fs # 6-dim vector, we do not need the exponential term
    #PSM_fs  = np.concatenate([fm.aperiodic_params_[1:], peaks_fs]) # 7-dim vector, we do not need offset
    return PSM_fs

def obt_psm_raw(raw_psd, freqs, is_knee=False):
    """
        extract features from Power spectrum models
        two components:
            1. Peaks, [center freq, power, bandwidth], I choose two peaks.
            2. Aperiodic part: offset and Exponent
        args:
            raw_psd: the power of spectrum to extract features
            freqs: corresponding freqs
    """
    freq_range = [np.amin(freqs), np.amax(freqs)]
    
    # to smooth the raw psd
    lpf = np.array([1, 2, 3, 2, 1])
    lpf = lpf/np.sum(lpf)
    sm_psd = np.convolve(raw_psd,lpf,'same')
    if is_knee:
        aperiodic_mode='knee'
    else:
        aperiodic_mode='fixed'
    # the fooof obj
    fm = FOOOF(peak_width_limits=[2*np.diff(freqs)[0], 12.0], aperiodic_mode=aperiodic_mode)
    fm.fit(freqs, sm_psd, freq_range)
    return fm