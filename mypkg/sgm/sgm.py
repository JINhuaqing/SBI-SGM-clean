import numpy as np
import logging
from utils.misc import _set_verbose_level
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if not logger.hasHandlers():
    ch = logging.StreamHandler() # for console. 
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch) 
    


class SGM:
    """
    A SGM model for both FC and PSD forward
    """
    def __init__(self, C, D, freqs, verbose=2):
        """args:
            C (array): sc matrix, nroi x nroi
            D (array): dist matrix, nroi x nroi
            freqs: the freqs to get the FC, in Hz. 
            verbose (int, optional): Verbosity level. Defaults to 2.
        """
        _set_verbose_level(verbose, logger)
        if isinstance(freqs, str):
            msg = "only support delta, theta and alpha bands, for other band, plz input a list."
            assert freqs.lower() in ["delta", "theta", "alpha"], msg 
            bands_rgs = {"delta": [2, 3.5], "theta": [4, 7], "alpha": [8, 12]}
            bdlmt = bands_rgs[freqs]
            logger.info(f"The range of {freqs} band is {bdlmt}.")
            freqs = np.linspace(bdlmt[0], bdlmt[1], 10)
        self.C = C
        self.D = D
        self.freqs = freqs
        
        logger.info(f"Num of ROI is {C.shape[0]}.")
        logger.debug(f"Be careful about your input, freq should be in Hz!")
        logger.debug(f"All tau's params should be in second!")
        logger.debug(f"The output  PSD is be in abs magnitude (20log10(psd) toi dB)!")
        
    def _get_lap_result(self, alpha, speed, freq):
        """Get the eig results from Laplaician matrix at give freq
        args:
            alpha (scale): alpha param
            speed (scale): speed param
            freq (float): The freq, in Hz
        """
        C = self.C
        D = self.D
        w = 2 * np.pi * freq # from Hz to angular
        
        # define sum of degrees for rows and columns for laplacian normalization
        C = C/np.linalg.norm(C)
        # define sum of degrees for rows and columns for laplacian normalization
        rowdegree = np.transpose(np.sum(C, axis=1))
        coldegree = np.sum(C, axis=0)
    
        degree = (rowdegree + coldegree)/2
        eps = np.percentile(degree,5)
        nroi = C.shape[0]
        Tau = 0.001 * D / speed
        Cc = C * np.exp(-1j * Tau * w)
    
        # Eigen Decomposition of Complex Laplacian Here
        L1 = np.identity(nroi)
        L2 = np.divide(1, np.sqrt(np.multiply(rowdegree, coldegree)) + eps)
        L = L1 - alpha * np.matmul(np.diag(L2), Cc)
    
        d, v = np.linalg.eig(L)  
        eig_ind = np.argsort(np.abs(d))  # sorting in ascending order and absolute value
        eig_vec = v[:, eig_ind]  # re-indexing eigen vectors according to sorted index
        eig_val = d[eig_ind]  # re-indexing eigen values with same sorted index
        
        return eig_val, eig_vec
    
    def _get_psd_at_freq(self, params, freq):
        """Network Transfer Function for spectral graph model for give freq w (in angular, not in Hz)
           i.e. SGM forward function for a specific w
    
        Args:
            params (dict): parameters for ntf, including alpha, gei, gii, taue, tauG, taui, speed
                              The tau's should in second.
            freq (float): frequency at which to calculate NTF, in Hz
    
        Returns:
            model_out (numpy asarray):  the psd at given freq
    
        """
        params1 = {}
        for ky, v in params.items():
            params1[ky.lower()] = v
        alpha = params1["alpha"]
        gei = params1["gei"]
        gii = params1["gii"]
        taue = params1["taue"]
        tauG = params1["taug"]
        taui = params1["taui"]
        speed = params1["speed"]
        gee = 1
        nroi = self.C.shape[0]
        w = 2 * np.pi * freq # from Hz to angular
        
        # Defining some other parameters used:
        zero_thr = 0.05
    
        eig_val, eig_vec = self._get_lap_result(alpha, speed, freq)
    
        # Cortical model
        Fe = np.divide(1 / taue ** 2, (1j * w + 1 / taue) ** 2)
        Fi = np.divide(1 / taui ** 2, (1j * w + 1 / taui) ** 2)
        FG = np.divide(1 / tauG ** 2, (1j * w + 1 / tauG) ** 2)
    
        Hed = (1 + (Fe * Fi * gei)/(taue * (1j * w + Fi * gii/taui)))/(1j * w + Fe * gee/taue + (Fe * Fi * gei)**2/(taue * taui * (1j * w + Fi * gii / taui)))
        Hid = (1 - (Fe * Fi * gei)/(taui * (1j * w + Fe * gee/taue)))/(1j * w + Fi * gii/taui + (Fe * Fi * gei)**2/(taue * taui * (1j * w + Fe * gee / taue)))
        Htotal = Hed + Hid
    
    
        q1 = (1j * w + 1 / tauG * FG * eig_val)
        qthr = zero_thr * np.abs(q1[:]).max()
        magq1 = np.maximum(np.abs(q1), qthr)
        angq1 = np.angle(q1)
        q1 = np.multiply(magq1, np.exp(1j * angq1))
        frequency_response = np.divide(Htotal, q1)
        
        model_out = 0
        for k in range(nroi):
            model_out += (frequency_response[k]) * np.outer(eig_vec[:, k], np.conjugate(eig_vec[:, k])) 
        model_out = np.linalg.norm(model_out,axis=1)

        return model_out
    
    def _get_fc_at_freq(self, params, freq):
        """Network Transfer Function for spectral graph model.
    
        Args:
            params (dict): params for SGM, including alpha, tauG and speed
            freq (float): frequency at which to calculate FC, in Hz 
    
        Returns:
            fc(numpy asarray):  The FC for the given frequency (freq)
        """
        
        params1 = {}
        for ky, v in params.items():
            params1[ky.lower()] = v
        alpha = params1["alpha"]
        tauG = params1["taug"]
        speed = params1["speed"]
        w = 2*np.pi*freq # change from Hz to angular freq
        nroi = self.C.shape[0]
        
        # Defining some other parameters used:
        zero_thr = 0.05 # in my paper, it is 0.01, but to make it consistent with PSD-SGM, I change it to 0.05
    
        eig_val, eig_vec = self._get_lap_result(alpha, speed, freq)
    
        # Cortical model
        FG = np.divide(1 / tauG ** 2, (1j * w + 1 / tauG) ** 2)
    
    
        q1 = (1j * w + 1 / tauG * FG * eig_val)
        qthr = zero_thr * np.abs(q1[:]).max()
        magq1 = np.maximum(np.abs(q1), qthr)
        angq1 = np.angle(q1)
        q1 = np.multiply(magq1, np.exp(1j * angq1))
        frequency_response = np.divide(1, np.abs(q1)**2)
        
        fc = eig_vec @ np.diag(frequency_response) @ np.conjugate(eig_vec.T)
        fc = np.abs(fc)
    
        return fc
    
    def forward_fc(self, params):

        """
        Output:
        estFC, the mean normalized estimated FC at the given frequency computed 
                over the range given in freqrange.
        """
        estFC = 0
        for cur_freq in self.freqs:
            cur_estFC = self._get_fc_at_freq(params, cur_freq)
            estFC = cur_estFC/len(self.freqs) + estFC
    
        # Now normalize estFC
        diagFC = np.diag(np.abs(estFC))
        diagFC = 1./np.sqrt(diagFC)
        D = np.diag(diagFC)
        estFC = np.matmul(D, estFC)
        estFC = np.matmul(estFC , np.matrix.getH(D)) # f_ij/\sqrt(f_ii)\sqrt(f_jj)
        estFC = estFC - np.diag(np.diag( estFC ))
    
        return estFC
            
    def forward_psd(self, params):
        """run_forward. Function for running the forward model over the passed in range of frequencies,
        for the handed set of parameters (which must be passed in as a dictionary)
    
        Args:
            params (dict): Dictionary of a setting of parameters for the NTF model.
    
        Returns:
            model_out(array): the modelled PSD from SGM model
    
        """
        model_out = []
        for freq in self.freqs:
            freq_model = self._get_psd_at_freq(params, freq)
            model_out.append(freq_model)
        model_out = np.asarray(model_out).T
        return model_out
        
