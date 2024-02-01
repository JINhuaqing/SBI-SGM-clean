import torch
import numpy as np
from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi
from easydict import EasyDict as edict
from functools import partial
from torch.distributions.multivariate_normal import MultivariateNormal


from utils.reparam import theta_raw_2out, logistic_torch
from constants import RES_ROOT
from utils.misc import save_pkl, load_pkl
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler() # for console. 
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

class SBI_SGMFC():
    def __init__(self, sgmfcmodel, save_folder=None, verbose=False, input_params={}):
        if verbose:
            logger.handlers[0].setLevel(logging.INFO)
        else:
            logger.handlers[0].setLevel(logging.WARNING)
            
        # default params
        params = edict()
        
        # noise level added to SGM-FC output
        params.noise_sd = 1.20
        # num of sbi sps to train the NN
        params.num_prior_sps = int(1e4)
        # the density estimator
        params.den_est = "nsf"
        # the sd of Gaussian prior for SBI
        params.prior_sd = 1
        # the parameter to control how to reparam the SGM parameters
        params.k = 1
        # the num of rounds to train SBI
        params.num_round = 1
        
        # the default SGM parameters bds with order in params.names
        params.names = ["TauC", "Speed", "alpha"]
        par_low = np.asarray([0.005,5, 0.1])
        par_high = np.asarray([0.03,20,  17])
        params.prior_bds = np.array([par_low, par_high]).T
        
        
        for key in input_params.keys():
            if key not in params.keys():
                logger.warning(f"Plz check your input {key}, it is not used.")
            else:
                params[key] = input_params[key]
        if "prior_sd" in input_params.keys() or "k" in input_params.keys():
            logger.warning(f"k control the reparam of SGM-FC. "
                           f"The default value is compatible with prior_sd."
                           f"Be careful if you change k or prior_sd."
                          )
        
        self.params = params
        logger.info(f"In this class, the sgm parameter order is {self.params.names}.")
        logger.info(self.params)
        
        
        if isinstance(save_folder, str):
            self.save_path = RES_ROOT/save_folder
        elif save_folder:
            self.save_path = RES_ROOT/(f"./sbi-sgmfc_results_numsps{paras.num_prior_sps:.0f}" +
                                      f"_sd{paras.noise_sd*100:.0f}" +
                                      f"_denest{paras.den_est}")
        else:
            self.save_path = None
        if self.save_path is not None:
            if not self.save_path.exists():
                self.save_path.mkdir()
        if self.save_path is None:
            logger.info(f"Results will not be saved.")
        else:
            logger.info(f"Results will be saved at {self.save_path}.")
            
        
        self.init_prior = MultivariateNormal(loc=torch.zeros(3), 
                                             covariance_matrix=torch.eye(3)*(self.params.prior_sd**2))
            
        self._trans_fn=partial(logistic_torch, k=self.params.k)
        self._reparam_fn = partial(theta_raw_2out, map_fn=partial(logistic_torch, k=self.params.k))
        
        self.sgmfcmodel = sgmfcmodel
        self.posterior = None
        self.curX = None
        self.post_sps = None
    
    def _minmax_vec(self, x):    
        return (x-np.min(x))/(np.max(x)-np.min(x))
    
    def simulator(self, sgm_reparams, is_train=True):
        """ Generate data with SGM model
            args:
                sgm_reparams: the transformed sgm params
                is_train: (bool) if True, generating SGM data with noise (for training)
                                 if False, generating clean SGM data (no noise, for inference)
                    
        """
        sgm_params = (self._trans_fn(sgm_reparams)*(self.params.prior_bds[:, 1]-self.params.prior_bds[:, 0]) +
                       self.params.prior_bds[:, 0])
        
        
        fc = self.sgmfcmodel.get_fc(sgm_params)
        fc_abs = np.abs(fc[:68, :68])
        
        if is_train:
            res = self._minmax_vec(fc_abs[np.triu_indices(fc_abs.shape[0], k=1)])
            noise = np.random.randn(*res.shape)*self.params.noise_sd 
            return res + noise
        else:
            return fc_abs

    
    def SBI_fit(self, load=False):
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
                                                num_workers=20)
                density_estimator = inference.append_simulations(sgm_reparamss, x, proposal=proposal).train()
                posterior = inference.build_posterior(density_estimator)
                proposal = posterior.set_default_x(self.curX)
            self.posterior = posterior
            if self.save_path is not None:
                save_pkl(self.save_path/"posterior.pkl", self.posterior, is_force=True)
        else:
            assert (self.save_path/"posterior.pkl").exists(), "No file to load!!"
            self.posterior = load_pkl(self.save_path/"posterior.pkl")
        
    def add_data(self, fc=None, ts=None):
        """Add fs data to do inference
        fc: nroi x nroi array, note that we do not consider the diag terms
        ts: the time series, nroi x ntimes
        """
        assert not ((fc is None) and (ts is None)), "You must provide your data, either FC or original ts."
        if fc is not None:
            if ts is not None:
                logger.warnning(f"You fc is given, ts is ignored.")
            fc_abs = np.abs(fc)
            curX = torch.Tensor(self._minmax_vec(fc_abs[np.triu_indices(fc_abs.shape[0], k=1)]))
        else:
            pass
            
        self.curX = curX
        
        self.post_sps = None
        
        
    def get_stdfc(self, fc):
        fc_abs = np.abs(fc)
        nroi = fc_abs.shape[0]
        half_idxs = np.triu_indices(nroi, k=1)
        
        res = self._minmax_vec(fc_abs[half_idxs])
        stdfc = np.zeros_like(fc_abs)
        stdfc[half_idxs] = res
        stdfc = stdfc + stdfc.T
        return stdfc, res
        
    def get_post_sps(self, n=10000):
        """Get post sps of sgm parameters
        args:
            n: num of sps to draw
        """
        assert self.posterior is not None, "Train SBI first with SBI_fit()"
        assert self.curX is not None, "You should add data first with add_data(psd)"
        
        post_sps_reparam = self.posterior.sample((n, ), x=self.curX, max_sampling_batch_size=100)
        
        self.post_sps = self._reparam_fn(post_sps_reparam, self.params.prior_bds).numpy()
        return self.post_sps
    
    def get_model_fc(self):
        """Get modelled fc based on self.post_sps
        """
        assert self.post_sps is not None, "You should get posterior sps with get_post_sps(n)"
        pt_est = np.median(self.post_sps, axis=0)
        fc = self.sgmfcmodel.get_fc(pt_est)
        stdfc, _ = self.get_stdfc(fc)
        return stdfc
    
