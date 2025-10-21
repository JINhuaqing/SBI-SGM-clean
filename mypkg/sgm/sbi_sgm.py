import torch
import scipy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi
from sbi import analysis as analysis
from easydict import EasyDict as edict
from functools import partial
from torch.distributions.multivariate_normal import MultivariateNormal
from numbers import Number


from utils.standardize import psd_2tr, stdz_vec, minmax_vec
from utils.reparam import theta_raw_2out, logistic_torch
from constants import RES_ROOT
from utils.misc import save_pkl, load_pkl
from utils.misc import _set_verbose_level, _update_params
from .sgm_utils import get_fc, get_psd
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if not logger.hasHandlers():
    ch = logging.StreamHandler() # for console. 
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch) 

class SBI_SGM():
    def __init__(self, sgmmodel, save_folder=None, verbose=2, params={}):
        _set_verbose_level(verbose, logger)
            
        # default params
        params_def = edict({
            # noise level added to output
            "noise_sd": 0.50,
            # num of sbi sps to train the NN
            "num_prior_sps": int(1e3),
            # the density estimator
            "den_est": "nsf",
            # the sd of Gaussian prior for SBI
            "prior_sd": 1, 
            # the mean of Gaussian prior for SBI
            "prior_mean": 0, 
            # the parameter to control how to reparam the SGM parameters
            "k": 1, 
            # the num of rounds to train SBI
            "num_round": 1,
            # prior bds of parameters
            "prior_bds": {
                "alpha": [0.1, 1], 
                "gei": [0.001, 0.7],
                "gii": [0.001, 2], 
                "speed": [5, 20],
                "taue": [0.005, 0.03], 
                "taug": [0.005, 0.03], 
                "taui": [0.005, 0.20], 
            }, 
            # fit on PSD or FC or both, 
            "fit_target": "both", 
        })
        
        if "prior_sd" in params.keys() or "k" in params.keys():
            logger.warning(f"k control the reparam of SGM-FC. "
                           f"The default value is compatible with prior_sd."
                           f"Be careful if you change k or prior_sd."
                          )
        
        params = _update_params(params, params_def, logger)
        params.fit_target = params.fit_target.lower()
        
        self.params = params
        logger.info(self.params)
        
        
        if isinstance(save_folder, str):
            self.save_path = RES_ROOT/save_folder
        elif save_folder:
            self.save_path = RES_ROOT/(f"./sbi-sgm_numsps{params.num_prior_sps:.0f}" +
                                      f"_sd{params.noise_sd*100:.0f}" +
                                      f"_denest{params.den_est}")
        else:
            self.save_path = None
        if self.save_path is not None:
            if not self.save_path.exists():
                self.save_path.mkdir()
        if self.save_path is None:
            logger.info(f"Results will not be saved.")
        else:
            logger.info(f"Results will be saved at {self.save_path}.")
            
            
        # prepare for the prior
        if params.fit_target.startswith("fc"):
            n_sgm_param = 3
            self.sgm_param_order_list = ["alpha", "speed", "taug"]
            self.sgm_param_prior_bds = np.array([params.prior_bds[ky] for ky 
                                           in self.sgm_param_order_list])
            logger.warning("Fit target is FC, so we only fit for three global parameters, alpha, speed, taug.")
        else:
            n_sgm_param = 7
            self.sgm_param_order_list = ["alpha", "gei", "gii", "speed", "taue", "taug", "taui"]
            self.sgm_param_prior_bds = np.array([params.prior_bds[ky] for ky 
                                           in self.sgm_param_order_list])
        if isinstance(params.prior_mean, Number):
            prior_mean = torch.zeros(n_sgm_param) + params.prior_mean
        else:
            assert len(params.prior_mean) == n_sgm_param, "prior_mean is not compbatile with our fitting."
            if not isinstance(params.prior_mean):
                prior_mean = torch.tensor(params.prior_mean)
            else:
                prior_mean = params.prior_mean
        prior_cov_mat = torch.eye(n_sgm_param) * (params.prior_sd**2)
            
            
        self.init_prior = MultivariateNormal(loc=prior_mean,
                                             covariance_matrix=prior_cov_mat)
            
        self._reparam_fn = partial(theta_raw_2out, map_fn=partial(logistic_torch, k=params.k))
        
        
        self.sgmmodel = sgmmodel
        self.posterior = None
        self.curX = None
        self.post_sps = None
        self.pt_est_name = None
        self.pt_est = None
        # the model/emp_fc/psd are 
        # fc is minmaxed 
        # psd is in dB then stdized.
        self.model_fc = self.model_psd = None
        self.emp_fc = self.emp_psd = None
        # the raw version is 
        # fc in org scale
        # psd in org scale (in abs mag)
        self.model_rawfc = self.model_rawpsd = None
        self.emp_rawfc = self.emp_rawpsd = None
    
    def simulator(self, sgm_reparams_vec, is_train=True):
        """ Generate data with SGM model
            args:
                sgm_reparams: the transformed sgm params
                is_train: (bool) if True, generating SGM data with noise (for training)
                                 if False, generating clean SGM data (no noise, for inference)
                    
        """
        if self.params.fit_target.startswith("fc"):
            assert len(sgm_reparams_vec) == 3, "Fit target is FC, so len of input should be 3."
        else:
            assert len(sgm_reparams_vec) == 7, "Fit target is PSD/Both, so len of input should be 7."

        if not isinstance(sgm_reparams_vec, torch.Tensor):
            sgm_reparams_vec = torch.tensor(sgm_reparams_vec)
            
        sgm_params_vec = self._reparam_fn(sgm_reparams_vec, 
                                self.sgm_param_prior_bds)
        sgm_params_dict = {
                 ky:sgm_params_vec[0, idx].item() for 
                 idx, ky in enumerate(self.sgm_param_order_list)
             }
        
        
        psd_vec = fc_vec = []
        mpsd = mfc = []
        if not self.params.fit_target.startswith("fc"):
            mpsd = self.sgmmodel.forward_psd(sgm_params_dict)
            psd_vec = psd_2tr(mpsd).flatten();
        if not self.params.fit_target.startswith("psd"):
            mfc = np.abs(self.sgmmodel.forward_fc(sgm_params_dict));
            fc_vec = stdz_vec(mfc[np.triu_indices(mfc.shape[0], k=1)]);
            
        all_vec = np.concatenate([psd_vec, fc_vec])
            
        
        if is_train:
            noise = np.random.randn(*all_vec.shape)*self.params.noise_sd 
            return all_vec + noise
        else:
            return mpsd, mfc

    
    def SBI_fit(self, load=False, n_jobs=20):
        """ Fit the SBI_SGM
        args:
            load: if you have saved results, you can set load = true to load it
        """
        if not load:
            if self.params.num_round > 1: 
                assert self.curX is not None, "If you want to train with multiple rounds, plz add_data first."
            simulator_wrapper, prior = prepare_for_sbi(self.simulator, self.init_prior)
            inference = SNPE(prior=prior, density_estimator=self.params.den_est)
            proposal = prior
            for t_ix in range(self.params.num_round):
                sgm_reparamss, x = simulate_for_sbi(simulator_wrapper, proposal, 
                                                num_simulations=int(self.params.num_prior_sps), 
                                                num_workers=n_jobs)
                density_estimator = inference.append_simulations(sgm_reparamss, x, proposal=proposal).train()
                posterior = inference.build_posterior(density_estimator)
                proposal = posterior.set_default_x(self.curX)
            self.posterior = posterior
            if self.save_path is not None:
                save_pkl(self.save_path/"posterior.pkl", self.posterior, is_force=True)
        else:
            assert (self.save_path/"posterior.pkl").exists(), "No file to load!!"
            self.posterior = load_pkl(self.save_path/"posterior.pkl")
        
        
    def add_data(self, psd=None, fc=None, ts=None, fs=None, 
                 fc_params={}, psd_params={}):
        """Add fs data to do inference
        fc: nroi x nroi array, note that we do not consider the diag terms
        psd: nroi x nfreq, should in abs magnitude
        ts: the time series, nroi x ntimes
        fs: Sampling freqs, in Hz
        """
        fc_params_def = edict({
        "nepoch": 50, 
        "fc_type": "coh", 
        "f_skip": 10, 
        "num_taps": 351, 
        })
        
        psd_params_def = edict({
        "num_taps": 351,
        "nperseg": 1024, 
        })
        
        
        is_fc_nn = fc is not None
        is_psd_nn = psd is not None
        is_ts_nn = ts is not None
        curX_psd = curX_fc = []
        if not self.params.fit_target.startswith("fc"):
            if not is_psd_nn:
                # estimate psd from ts
                assert ts is not None
                assert fs is not None, "Plz provide sampling freqs in Hz"
                psd_params = _update_params(psd_params, psd_params_def, logger)
                psd = get_psd(ts, freqs=self.sgmmodel.freqs,
                                 fs=fs, psd_params=psd_params) 
                logger.warning("Avoid placing too much trust in the estimated PSD, as it is derived through interpolation at various frequencies.")
            curX_psd = psd_2tr(psd).flatten()
        else:
            if is_psd_nn:
                logger.warning(f"fit-target is fc, psd is not used")
            if is_fc_nn and is_ts_nn:
                logger.warning("ts is not used")
        if not self.params.fit_target.startswith("psd"):
            if not is_fc_nn:
                # estimate fc from ts
                assert ts is not None
                fc_params = _update_params(fc_params, fc_params_def, logger)
                assert fs is not None, "Plz provide sampling freqs in Hz"
                fc = get_fc(ts, fs=fs, 
                            bds=[self.sgmmodel.freqs[0], self.sgmmodel.freqs[-1]], 
                            fc_params=fc_params)
            fc = np.abs(fc)
            curX_fc = stdz_vec(fc[np.triu_indices(fc.shape[0], k=1)])
        else:
            if is_fc_nn:
                logger.warning(f"fit-target is psd, fc is ignored")
            if is_psd_nn and is_ts_nn:
                logger.warning("ts is not used")
        if (is_psd_nn and is_fc_nn) and is_ts_nn:
            logger.warning("ts is not used")
        
        curX = np.concatenate([curX_psd, curX_fc])
        self.curX = torch.tensor(curX)
        
        if not self.params.fit_target.startswith("psd"):
            self.emp_rawfc = fc
            stdfc, _ = self.get_stdfc(fc)
            self.emp_fc = stdfc
        
        if not self.params.fit_target.startswith("fc"):
            self.emp_rawpsd = psd
            self.emp_psd = psd_2tr(psd)

        self.post_sps = None
        
        
        
    def get_post_sps(self, n=10000):
        """Get post sps of sgm parameters
        args:
            n: num of sps to draw
        """
        assert self.posterior is not None, "Train SBI first with SBI_fit()"
        assert self.curX is not None, "You should add data first with add_data(psd)"
        
        post_sps_reparam = self.posterior.sample((n, ), x=self.curX, max_sampling_batch_size=100)
        
        self.post_sps = self._reparam_fn(post_sps_reparam, self.sgm_param_prior_bds).numpy()
        return self.post_sps
    
    def get_point_est(self, mode="mean"):
        """Get the point estimation of the SGM parameters
        """
        assert self.post_sps is not None, "You should get posterior sps with get_post_sps(n)"
        mode = mode.lower()
        if mode.startswith("mean"):
            pt_est = np.mean(self.post_sps, axis=0)
         
        elif mode.startswith("medi"):
            pt_est = np.median(self.post_sps, axis=0)
        self.pt_est = pt_est
        self.pt_est_name = edict({})
        for idx, ky in enumerate(self.sgm_param_order_list):
            self.pt_est_name[ky] = self.pt_est[idx]
        
        
    def get_model_psd(self):
        """Get modelled psd based on self.post_sps
        """
        assert not self.params.fit_target.startswith("fc"), "Not support to get modeled psd as you only fit on FC."
        if self.pt_est_name is None:
            self.get_point_est()
        psd = self.sgmmodel.forward_psd(self.pt_est_name)
        self.model_rawpsd = psd
        psd = psd_2tr(psd)
        self.model_psd = psd
        return psd
    
    def get_model_fc(self):
        """Get modelled fc based on self.post_sps
        """
        if self.params.fit_target.startswith("psd"):
            logger.warning(f"You fit on psd not on fc.")
        if self.pt_est_name is None:
            self.get_point_est()
        fc = np.abs(self.sgmmodel.forward_fc(self.pt_est_name))
        self.model_rawfc = fc
        stdfc, _ = self.get_stdfc(fc)
        self.model_fc = stdfc
        return stdfc
    
    def get_stdfc(self, fc):
        fc = np.abs(fc)
        nroi = fc.shape[0]
        half_idxs = np.triu_indices(nroi, k=1)
        
        res = minmax_vec(fc[half_idxs])
        stdfc = np.zeros_like(fc)
        stdfc[half_idxs] = res
        stdfc = stdfc + stdfc.T
        return stdfc, res
    
    def plot_pairdensity(self, figsize=(10, 10)):
        """Plot the density of estimated post samples from sgm params
        """
        assert self.post_sps is not None, "You should get posterior sps with get_post_sps(n)"
        analysis.pairplot(self.post_sps,
                      limits=self.sgm_param_prior_bds, 
                      labels=self.sgm_param_order_list, 
                      figsize=figsize);
        
    def plot_compare(self, type_=None):
        """
        Plot psd or fc via modeled one vs empirical one
        """
        if type_ is None:
            type_ = self.params.fit_target
        else:
            type_ = type_.lower()
            assert type_ in ['fc', 'psd', 'both'], "type_ should be among fc, psd, both."
        if type_.startswith("both"):
            plt.figure(figsize=[14, 5])
        else:
            plt.figure(figsize=[7, 5])
            
        if not type_.startswith("fc"):
            assert self.model_psd is not None
            assert self.emp_psd is not None
            if type_.startswith("both"):
                plt.subplot(121)
            plt.plot(self.sgmmodel.freqs, self.model_psd.mean(axis=0), label="modelled PSD")
            plt.plot(self.sgmmodel.freqs, self.emp_psd.mean(axis=0), label="empirical PSD")
            plt.xlabel("Freq (Hz)", fontsize=15)
            plt.ylabel("Standardized PSD (dB)", fontsize=15)
            plt.legend()
            
        if not type_.startswith("psd"):
            assert self.model_fc is not None
            assert self.emp_fc is not None
            if type_.startswith("both"):
                plt.subplot(122)
            uidxs = np.triu_indices_from(self.model_fc, k=1)
            lidxs = np.tril_indices_from(self.model_fc, k=1)
            model_fc_vec = self.model_fc[uidxs]
            emp_fc_vec = self.emp_fc[uidxs]
            corr = scipy.stats.pearsonr(model_fc_vec, emp_fc_vec)[0]
            
            mat = np.zeros_like(self.model_fc)
            mat[lidxs] = self.emp_fc[lidxs]
            mat[uidxs] = self.model_fc[uidxs]
            plt.title(f"Modelled FC vs Empirical FC (corr: {corr:.3f})",  fontsize=15)
            heatmap = sns.heatmap(mat,  
                                  cmap="viridis", 
                                  square=True)
            plt.text(-max(plt.xlim())*.01, 
                     max(plt.ylim())/2, 
                     "Empirical FC", 
                     rotation=90, 
                     fontsize=15,
                     verticalalignment='center', 
                     horizontalalignment='right')
            plt.text(max(plt.xlim())*1.01, 
                     max(plt.ylim())/2, 
                     "Modelled FC ", 
                     rotation=90, 
                     fontsize=15,
                     verticalalignment='center', 
                     horizontalalignment='left')
            plt.xticks([])
            plt.yticks([]);
            
            
            
