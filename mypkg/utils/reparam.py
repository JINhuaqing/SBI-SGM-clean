# this file contains fns for reparameterizing SGM
import numpy as np
import torch
from scipy.stats import norm as sci_normal
from torch.distributions.normal import Normal as tor_normal

def normcdf_np(x, sd=10):
    return sci_normal(loc=0, scale=sd).cdf(x)

def normcdf_torch(x, sd=10):
    return tor_normal(loc=0, scale=sd).cdf(x)

def logistic_torch(x, k=0.10):
    """k=0.1 fits prior N(0, 100)
    """
    num = torch.exp(k*x)
    den = torch.exp(k*x) + 1
    
    # fix inf issue
    res = num/den
    res[torch.isinf(num)] = 1
    return res


def logistic_np(x, k=0.10):
    """k=0.1 fits prior N(0, 100)
    """
    num = np.exp(k*x)
    den = np.exp(k*x) + 1
    # fix inf issue
    res = num/den
    res[np.isinf(num)] = 1
    return res

def theta_raw_2out(thetas_raw, prior_bds, map_fn=logistic_np):
    """transform reparameterized theta to orignal theta
        args: thetas_raw: an array with num_sps x 7
              prior_bds: an array with 7 x 2
    """
    if thetas_raw.ndim == 1:
        thetas_raw = thetas_raw.reshape(1, -1)
    assert prior_bds.shape[0] == thetas_raw.shape[1]
    thetas_out = map_fn(thetas_raw)*(prior_bds[:, 1] -  prior_bds[:, 0]) + prior_bds[:, 0]
    return thetas_out