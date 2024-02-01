import torch
import numpy as np
import xarray as xr

from sbi import utils as sutils
from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi
from torch.distributions.multivariate_normal import MultivariateNormal
from easydict import EasyDict as edict
from pathlib import Path
from functools import partial


from utils.standardize import stdz_vec, psd_2tr
from utils.reparam import theta_raw_2out, logistic_torch
from utils.stable import paras_table_check
from constants import RES_ROOT
from models.embedding_nets import SummaryNet
from utils.misc import save_pkl, load_pkl


import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler() # for console. 
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


class SBI_SGM():
    """This class is to fit SGM with SBI.
    It can be only used for DK altas (68). 
    You should use it with SGM class
    """
    def __init__(self, sgmmodel, save_folder=None, verbose=False, **input_params):
        
        
        if verbose:
            logger.handlers[0].setLevel(logging.INFO)
        else:
            logger.handlers[0].setLevel(logging.WARNING)
            
        # default params
        params = edict()
        
        # noise level added to SGM output
        params.noise_sd = 0.20
        # num of sbi sps to train the NN
        params.num_prior_sps = int(1e4)
        # the density estimator
        params.den_est = "nsf"
        # whether using a embed net to reduce the output dim or not
        params.is_embed = False
        # the sd of Gaussian prior for SBI
        params.prior_sd = 10
        
        # the parameter to control how to reparam the SGM parameters
        params.k = 0.1
        
        # the default SGM parameters bds with order in params.names
        params.names = ["Taue", "Taui", "TauC", "Speed", "alpha", "gii", "gei"]
        par_low = np.asarray([0.005,0.005,0.005,5, 0.1,0.001,0.001])
        par_high = np.asarray([0.03, 0.20, 0.03,20,  1,    2,  0.7])
        params.prior_bds = np.array([par_low, par_high]).T
        
        for key in input_params.keys():
            if key not in params.keys():
                logger.warning(f"Plz check your input {key}, it is not used.")
            else:
                params[key] = input_params[key]
        if "prior_sd" in input_params.keys() or "k" in input_params.keys():
            logger.warning(f"k control the reparam of SGM. "
                           f"The default value is compatible with prior_sd."
                           f"Be careful if you change k or prior_sd."
                          )
        
        self.params = params
        logger.info(f"In this class, the sgm parameter order is {self.params.names}.")
        logger.info(self.params)
        
        
        if isinstance(save_folder, str):
            self.save_path = RES_ROOT/save_folder
        elif save_folder:
            self.save_path = RES_ROOT/(f"./sbi-sgm_results_numsps{paras.num_prior_sps:.0f}" +
                                      f"_sd{paras.noise_sd*100:.0f}" +
                                      f"_denest{paras.den_est}" +
                                      f"_embed{paras.is_embed}")
        else:
            self.save_path = None
        if self.save_path is not None:
            if not self.save_path.exists():
                self.save_path.mkdir()
        if self.save_path is None:
            logger.info(f"Results will not be saved.")
        else:
            logger.info(f"Results will be saved at {self.save_path}.")
            
        
        self.init_prior = MultivariateNormal(loc=torch.zeros(7), 
                                             covariance_matrix=torch.eye(7)*(self.params.prior_sd**2))
            
        self._trans_fn=partial(logistic_torch, k=self.params.k)
        self._reparam_fn = partial(theta_raw_2out, map_fn=partial(logistic_torch, k=self.params.k))
        
        self.sgmmodel = sgmmodel
        self.posterior = None
        self.embedding_net = None
        self.curX = None
        self.post_sps = None
        
        assert not self.params.is_embed, f"The embed network is not fully tested!! If using, comment out this sentence."
        if self.params.is_embed:
            self.embedding_net = SummaryNet(num_in_fs=68*41)
            self.params.den_est = sutils.posterior_nn(model=self.params.den_est, 
                                                      embedding_net=self.embedding_net)
            
        
    def simulator(self, sgm_reparams, is_train=True):
        """ Generate data with SGM model
            args:
                is_train: (bool) if True, generating SGM data with noise (for training)
                                 if False, generating clean SGM data (no noise, for inference)
                    
        """
        sgm_params = (self._trans_fn(sgm_reparams)*(self.params.prior_bds[:, 1]-self.params.prior_bds[:, 0]) +
                       self.params.prior_bds[:, 0])
        #sgm_params = self._reparam_fn(sgm_reparams, self.params.prior_bds)
        psd, spatialFs = self.sgmmodel.run_local_coupling_forward(sgm_params)
        psd = psd[:68, :]
        
        sp_fs= spatialFs.sum(axis=1)
        if is_train: 
            sp_fs = stdz_vec(sp_fs) # std it
        
        std_psd_DB = psd_2tr(psd)
        psd_fs = std_psd_DB.flatten()
        
        res = np.concatenate([psd_fs, sp_fs]) 
        if is_train:
            noise =  np.random.randn(*res.shape)*self.params.noise_sd 
        else:
            noise = np.zeros_like(res)
        
        return res+noise
    
    def _filter_unstable(self, theta_raw, x=None):
        """This fn is to remove unstable SGM parameters
            args: theta_raw: parameters: num of sps x dim
                    order: ['Taue', 'Taui', 'TauC', 'Speed', 'alpha', 'gii', 'gei']
        """
        theta = self._reparam_fn(theta_raw, self.params.prior_bds)
        stable_idxs = paras_table_check(theta.numpy())
        
        # keep stable sps only
        theta_raw_stable = theta_raw[stable_idxs==0]
        if x is not None:
            x_stable = x[stable_idxs==0]
            return theta_raw_stable, x_stable
        else:
            return theta_raw_stable
    
    def SBI_fit(self, load=False):
        """ Fit the SBI_SGM
        args:
            load: if you have saved results, you can set load = true to load it
        """
        if not load:
            simulator_wrapper, prior = prepare_for_sbi(self.simulator, self.init_prior)
            sgm_reparamss, x = simulate_for_sbi(simulator_wrapper, prior, 
                                            num_simulations=int(self.params.num_prior_sps), 
                                            num_workers=20)
            sgm_reparamss_stable, x_stable = self._filter_unstable(sgm_reparamss, x)
            inference = SNPE(prior=prior, density_estimator=self.params.den_est)
            density_estimator = inference.append_simulations(sgm_reparamss_stable, x_stable).train()
            posterior = inference.build_posterior(density_estimator)
            self.posterior = posterior
            if self.save_path is not None:
                save_pkl(self.save_path/"posterior.pkl", self.posterior, is_force=True)
        else:
            assert (self.save_path/"posterior.pkl").exists(), "No file to load!!"
            self.posterior = load_pkl(self.save_path/"posterior.pkl")
        
    def add_data(self, psd):
        """Add psd data to do inference
        psd: nroi x nfreq array, should in abs magnitude
        """
        assert psd.shape[0] == 68, "Make sure the input psd is nroi x nfreq"
        # only get spatial feature from alpha band
        freqband = np.where((self.sgmmodel.freqs>=8) & (self.sgmmodel.freqs<=12))[0]
        raw_sps = psd[:, freqband]
    
        std_spv = stdz_vec(raw_sps.sum(axis=1))
        std_psd_DB = psd_2tr(psd)
        curX_raw = np.concatenate([std_psd_DB.flatten(), std_spv])
        self.curX = torch.Tensor(curX_raw)
        
        # if add a data, clean the previous sps
        self.post_sps = None
    
    def get_post_sps(self, n=10000):
        """Get post sps of sgm parameters
        args:
            n: num of sps to draw
        """
        assert self.posterior is not None, "Train SBI first with SBI_fit()"
        assert self.curX is not None, "You should add data first with add_data(psd)"
        
        post_sps_reparam = self.posterior.sample((n, ), x=self.curX, max_sampling_batch_size=100)
        
        # only keep stable params
        post_sps_reparam = self._filter_unstable(post_sps_reparam)
        self.post_sps = self._reparam_fn(post_sps_reparam, self.params.prior_bds) 
        return self.post_sps
        
    def get_model_psd_sp(self):
        """Get modelled PSD and spatial features based on self.post_sps
        """
        assert self.post_sps is not None, "You should get posterior sps with get_post_sps(n)"
        pt_est = np.median(self.post_sps, axis=0)
        cur_psd, cur_sp = self.sgmmodel.run_local_coupling_forward(pt_est)
        cur_psd = cur_psd[:68, :]
        cur_psd_DB = psd_2tr(cur_psd)
        return cur_psd_DB, cur_sp.sum(axis=1)
        

    def get_post_psd_sps(self, n=100):
        """You only need it when you want a Full-Bayesian inference.
           Get post PSD 
        args:
            n: num of sps to draw
        """
        assert self.posterior is not None, "Train SBI first with SBI_fit()"
        assert self.curX is not None, "You should add data first with add_data(psd, sc)"
        simulator_data_sp = partial(self.simulator, is_train=False)
        simulator_data_wrapper, _ = prepare_for_sbi(simulator_data_sp, self.init_prior)
        cur_post = self.posterior.set_default_x(self.curX)
        
        tmp_sps, post_psd = simulate_for_sbi(simulator_data_wrapper, cur_post, 
                            num_simulations=n,
                            num_workers=20)
        
        _, post_psd_stable = self._filter_unstable(tmp_sps, post_psd)
        self.post_psd = post_psd_stable[:, :-68].reshape(-1, 68, len(self.sgmmodel.freqs)).numpy()
        return self.post_psd