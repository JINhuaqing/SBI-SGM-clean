# This file contains functions for checking whether 
# the SGM parameter is stable or not
# Refer to Verma_et_al_NNS_2022.pdf, formula (28) in appendix
import numpy as np
import sympy as sp


def obt_poly_coefs(theta):
    """The fn to obtain the polynomial coefs based on SGM parameters
            args: theta: parameters: num of sps x dim
                order: ['Taue', 'Taui', 'TauC', 'Speed', 'alpha', 'gii', 'gei']
    """
    if len(theta.shape) == 1:
        theta = theta.reshape(1, -1)
    tes = 1/theta[:, 0]
    tis = 1/theta[:, 1]
    geis = theta[:, -1]
    giis = theta[:, -2]
    gee = 1.0
    # part 1
    p1 = np.array([
        gee*tes**3*tis**2,
        tis**2*tes**2+2*gee*tes**3*tis, 
        gee*tes**3+2*tes*tis**2+2*tis*tes**2, 
        tes**2+tis**2+4*tes*tis, 
        2*(tes+tis), 
        np.ones(len(tes)),
    ])
    
    p2 = np.array([
        giis*tis**3*tes**2,
        tis**2*tes**2+2*giis*tis**3*tes, 
        giis*tis**3+2*tes*tis**2+2*tis*tes**2, 
        tes**2+tis**2+4*tes*tis, 
        2*(tes+tis), 
        np.ones(len(tes)),
    ])
    
    p1p2 = np.array([
        np.ones(len(tes)),
        2*p1[4, :], 
        2*p1[3, :] + p1[4, :]**2, 
        p1[2, :] + p2[2, :] + 2*p1[3, :]*p1[4, :], 
        p1[1, :] + p2[1, :] + p1[4, :]*(p1[2, :]+p2[2, :]) + p1[3, :]**2, 
        p1[0, :] + p2[0, :] + p1[4, :]*(p1[1, :]+p2[1, :]) + p1[3, :]*(p1[2, :]+p2[2, :]), 
        p1[4, :]*(p1[0, :]+p2[0, :]) + p1[3, :]*(p1[1, :]+p2[1, :]) + p1[2, :]*p2[2, :],
        p1[3, :]*(p1[0, :]+p2[0, :]) + p1[2, :]*p2[1, :] + p1[1, :]*p2[2, :], 
        p1[2, :]*p2[0, :] + p2[2, :]*p1[0, :] + p1[1, :]*p2[1, :],
        p1[1, :]*p2[0, :]+p2[1, :]*p1[0, :],
        p1[0, :]*p2[0, :]
    ])
    
    coefs = p1p2
    coefs[-1, :] = coefs[-1, :] + geis**2*tes**5*tis**5
    return coefs.T


def paras_table_check(theta):
    """The fn to obtain the polynomial coefs based on SGM parameters
            args: theta: parameters: num of sps x dim
                order: ['Taue', 'Taui', 'TauC', 'Speed', 'alpha', 'gii', 'gei']
        return: a vec of 0 or 1.  0 if stable, 1 if not stable
    """
    if len(theta.shape) == 1:
        theta = theta.reshape(1, -1)
    coefs = obt_poly_coefs(theta)
    stb_idxs = []
    for ix in range(coefs.shape[0]):
        res = np.roots(coefs[ix])
        stb_idxs.append(int(np.sum(res.real >0) >0))
    stb_idxs = np.array(stb_idxs)
    return stb_idxs

# the function by Parul, it is not very fast
def paras_table_check1(paras):
    """ the function to determine whether the set of SGM parameters is stable or not
        based on (tau_e, tau_i, gei, gii). tau_e and tau_i are in second
        args:
            paras: the SGM parameters
                    order: ['Taue', 'Taui', 'TauC', 'Speed', 'alpha', 'gii', 'gei']
        return: 0 if stable, 1 if not stable
    """

    tau_e = paras[0]
    tau_i = paras[1]
    gei = paras[-1]  # excitatory-inhibitory synaptic conductance as ratio of E-E syn
    gii = paras[-2]  # inhibitory-inhibitory synaptic conductance as ratio of E-E syn

    gee = 1

    fe = float(1/tau_e)
    fi = float(1/tau_i)
    s = sp.Symbol('s')

    a = sp.poly(
        ( (s * (s+fe)**2 * (s+fi)**2 + gee * fe**3 * (s+fi)**2) *  
        (s * (s+fe)**2 * (s+fi)**2 + gii * fi**3 * (s+fe)**2) + 
        gei**2 * fe**5 * fi**5 ) / (fe**5 * fi**5), 
        s
    )


    b = a.all_coeffs()
    roots = np.roots(b)

    st = 0

    for result in roots:
        if result.real>0:
            st = 1

    return st