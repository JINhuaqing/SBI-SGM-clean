import scipy.signal as signal
from scipy.optimize import dual_annealing, minimize
import numpy as np
from easydict import EasyDict as edict

from utils.measures import reg_R_fn
from .utils import minmax_fn, obt_psd_at_freqs
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler() # for console. 
ch.setLevel(logging.WARNING)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# add formatter to ch
ch.setFormatter(formatter)
logger.addHandler(ch)

class sgm_fMRI():
    """This class is to implement SGM for fMRI, a python version of Ben's matlab code
       Refer to https://github.com/Raj-Lab-UCSF/SGMforFMRI
    """
    def __init__(self, raw_sc):
        """args:
            raw_sc (numpy.ndarray): The raw structural connectivity matrix, num_rois x num_rois
        """
        self.raw_sc = raw_sc
        self.num_rois = raw_sc.shape[0]
        self.eps = 1e-10
        self.U, self.ev = self._prepare_sc(raw_sc.copy())
        self.data = None
        self.emp_fc = None
        self.emp_psd = None
        self.fitted_theta = None

        
        # default params
        self.params = edict()
        self.params.TR = 2 # 
        self.params.fband = [0.008, 0.08]
        self.params.costtype = "corr"
        self.params.perc_thresh = False
        self.params.eig_weights = True
        self.params.deconvHRF = False
        self.params.is_ann = False
        self.params.model_focus = "both"
        self.params.fitmean = False
    
    def _prepare_sc(self, sc):
        # Get Laplacian and eigmode
        sc = sc/np.sum(sc)
        cd, rd = sc.sum(axis=0), sc.sum(axis=1);
        L = np.eye(self.num_rois) - np.diag(1/(np.sqrt(rd)+self.eps))@sc@np.diag(1/(np.sqrt(cd)+self.eps));
        ev, U = np.linalg.eig(L);
        sorted_idx = np.argsort(np.abs(ev)) # ascending
        ev = ev[sorted_idx]
        U = U[:, sorted_idx];
        return U, ev
        
    def add_data(self, data, **in_params):
        """Add observed fMRI data
           args:
               data: num_pts x num_rois, the fMRI data
               in_params: params for fitting 
                   TR (float): The repetition time in seconds, i.e., 1/fs
                   fband (list): The frequency band of interest, [low, high]
                   costtype (str): The cost type used for optimization, corr or mse
                   perc_thresh (bool): Whether to use percentile thresholding.
                   eig_weights (bool): Whether to use eigenvalue weights.
                   deconvHRF (bool): Whether to perform HRF deconvolution.
                   is_ann (bool): Whether to use artificial neural networks.
                   model_focus (str): The focus of the model (e.g., "both", "FX(psd)", "FC").
                   fitmean (bool): Whether to fit to the mean signal or not
        """
        self.data = data
        self.num_pts, _ = data.shape
        for key in in_params.keys():
            if key not in self.params.keys():
                logger.warning(f"Arg {key} is not used, "
                               f"please check your input.")
            else:
                self.params[key] = in_params[key]
        if self.num_pts < 64:
            logger.warning(f"Not enough timepoints ({self.num_pts}) for a good FFT; "
                           f"therefore SGM is only fitting to FC.")
            self.params.model_focus = "FC"
        elif self.num_pts < 128:
            nfft = 64
        else: 
            nfft = 128
        
        self.fvec = np.linspace(self.params.fband[0], 
                           self.params.fband[1], 
                           nfft);
        self.omegavec = 2 * np.pi * self.fvec
        self.fs = 1/self.params.TR;
                
        assert not self.params.perc_thresh, f"perc_thresh is not implemented yet."
        assert not self.params.deconvHRF, f"deconvHDF is not implemented yet."
        if self.params.model_focus.lower().startswith("fc"):
            logger.warning("When focusing on FC, params.fitmean is not used.")
        
        
        logger.info(self.params)
        self._prepare_data()
        self.fitted_theta = None
    
    def _prepare_data(self):
        # Preprocessing time series
        # demean
        tmp_data = self.data - self.data.mean(axis=0, keepdims=True);
        # detrend along the time axis
        tmp_data = signal.detrend(tmp_data, axis=0);
        # lowpass filter
        sos = signal.butter(N=5, Wn=self.params.fband[1], 
                            btype="low", 
                            fs=self.fs, 
                            output="sos")
        tmp_data = signal.sosfilt(sos, tmp_data, axis=0);
        
        if self.params.deconvHRF:
            # current not defined this function. 
            # difficult to know the definition. 
            tmp_data = self.deconv_HRF(tmp_data)
        self.data = tmp_data
        self._get_emp_fc()
        self._get_emp_psd()
        
    def _get_emp_fc(self):
        # get empirical FC, diagonal term is 0    
        emp_fc = np.corrcoef(self.data.T)
        np.fill_diagonal(emp_fc, 0);
        
        if self.params.perc_thresh:
            # not definied, refer to Ben's github
            emp_fc = self.perc_thresh(emp_fc)
            
        # make it symmetric
        emp_fc =  np.triu(emp_fc, 1) + np.triu(emp_fc).T;
        self.emp_fc = emp_fc
        
        if self.params.eig_weights:
            ev_weight = np.abs(np.diag(self.U.T @ self.emp_fc @ self.U))
        else:
            ev_weight = np.ones(self.num_rois)
        ev_weight[0] = 0
        self.ev_weight = ev_weight
            
    
    def _get_emp_psd(self):
        nblock = 128
        win = signal.windows.hanning(nblock, True)
        f, Pxx = signal.welch(self.data, window=win, fs=self.fs, 
                               nperseg=nblock, 
                               noverlap=int(nblock/2), axis=0);

        kpidx = np.bitwise_and(self.params.fband[0]<f, self.params.fband[1]>f)
        self.fvec = f[kpidx]
        self.omegavec = 2 * np.pi * self.fvec
        # not in dB, not squared
        PSD = np.sqrt(Pxx[kpidx])
        
        f_at_max = self.fvec[np.argmax(PSD, axis=0)];
        self.omega = 2*np.pi*np.median(f_at_max);
        self.emp_psd = PSD
    
        
        
    def deconv_HRF(self):
        pass
    
    def perc_thresh(self):
        # not definied, refer to Ben's github
        pass
        
    
    
    def forward_FC(self, theta=None):
        """Checked with matlab code.
        """
        if theta is None:
            theta = self.fitted_theta
        alpha = np.tanh(theta[0])
        tau = theta[1]
        He = 1/tau**2/(1/tau+self.omega*1j)**2
        newev = 1/(1j*self.omega + 1/tau*He*(1-alpha*(1-self.ev)));
        newev = (np.abs(newev))**2 * self.ev_weight;
        out_fc = self.U @ (newev.reshape(-1, 1) * np.conjugate(self.U).T);
        dg = 1/(1e-4+np.sqrt(np.diag(out_fc)));
        out_fc = out_fc * dg.reshape(-1, 1) * dg.reshape(1, -1)
        return out_fc
    
    def forward_FX(self, theta=None):
        """Not that the output is in abs maginitude (20log10(psd) to dB)
        """
        if theta is None:
            theta = self.fitted_theta
        alpha = np.tanh(theta[0])
        tau = theta[1]
        He = 1/tau**2/(1/tau+1j*self.omegavec)**2;
        tmp_vec = 1j * self.omegavec;
        tmp_mat = (1/tau*(1-alpha*(1-self.ev))).reshape(-1, 1) * He.reshape(1, -1)
        frequency_response = self.ev_weight.reshape(-1, 1)/(tmp_mat + tmp_vec.reshape(1, -1));
        
        UtP = self.U.conj().T @ np.ones(self.ev.shape[0]);
        out_psd = (self.U@(frequency_response * UtP[:, np.newaxis])).T;
        return np.abs(out_psd)
    
    def _obj_fn(self, theta):
        err_fc = 0
        err_psd = 0
        if (self.params.model_focus.lower().startswith("both") or  
        self.params.model_focus.lower().startswith("fc")):
            out_fc  = self.forward_FC(theta)
            kp_idxs = np.where(np.triu(out_fc, 1) != 0);
            r_fc = np.corrcoef(out_fc[kp_idxs], self.emp_fc[kp_idxs])[0, 1]
            err_fc = np.abs(1-r_fc)
        
        if (self.params.model_focus.lower().startswith("both") or  
        self.params.model_focus.lower().startswith("fx")):
            out_psd = self.forward_FX(theta)
            if self.params.fitmean:
                qdata = np.abs(self.emp_psd.mean(axis=1))[np.newaxis];
                qmodel = np.abs(out_psd.mean(axis=1))[np.newaxis];
            else:
                qdata = np.abs(self.emp_psd).T
                qmodel = np.abs(out_psd).T
        
        
            if self.params.costtype.lower().startswith("corr"):
                rvec_psd = reg_R_fn(qdata, qmodel)
                errvec_psd = np.abs(1-rvec_psd)
                    
            elif self.params.costtype.lower().startswith("mse"):
                qdata = minmax_fn(qdata, byrow=True)
                qmodel = minmax_fn(qmodel, byrow=True)
                errvec_psd = np.mean((qdata-qmodel)**2, axis=1)
                
            errvec_psd[np.isnan(errvec_psd)] = 0
            err_psd = np.nanmean(errvec_psd)
        return err_fc + err_psd
    
        
    def fit(self, **in_fit_params):
        fit_params = edict({
            "theta0": [0.5, 1],
            "maxiter": 1000,
            "bds": [[0.1, 10], [0.1, 5]]
        })
        
        for key in in_fit_params.keys():
            if key not in fit_params.keys():
                logger.warning(f"Arg {key} is not used, "
                               f"please check your input.")
            else:
                fit_params[key] = in_fit_params[key]
        logger.info(fit_params)
        
        if not self.params.is_ann:
            logger.warning(f"Compared with minimize function in scipy, Annealing provides higher optimization accuracy.")
            fit_res = minimize(self._obj_fn, 
                               x0=fit_params.theta0, 
                               bounds=fit_params.bds, 
                               options={"maxiter":fit_params.maxiter, 
                                        "disp":False})
        else:
            fit_res = dual_annealing(self._obj_fn, 
                                     x0=fit_params.theta0, 
                                     bounds=fit_params.bds, 
                                     maxiter=fit_params.maxiter)
        if not fit_res.success:
            logger.warning("The fitting is not successfuly, consider to use other way to fit.")
        self.fit_res = fit_res
        self.fitted_theta = fit_res.x
