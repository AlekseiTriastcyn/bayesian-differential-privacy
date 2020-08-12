#!/usr/bin/env python3
# Author:   Aleksei Triastcyn
# Date:     12 August 2020

import itertools
import numpy as np
import scipy as sp
import torch

from scipy.stats import t, binom
from scipy.special import logsumexp


def scaled_renyi_gaussian(alphas, ldistr, rdistr):
    """
        Computes scaled Renyi divergence D(p_left|p_right) between pairs of Gaussian distributions (with the same variance).
        
        
        Parameters
        ----------
        alpha : number, required
            Order of the Renyi divergence.
        ldistr : tuple or array, required
            Parameters of the left Gaussians (mu_left [n_samples * n_features], sigma [scalar]).
        rdistr : tuple or array, required
            Parameters of the right Gaussians (mu_right [n_samples * n_features], sigma [scalar]).
        
        Returns
        -------
        out : array
            Scaled Renyi divergences
    """
    lmu, lsigma = ldistr
    rmu, rsigma = rdistr
    
    if not (np.isscalar(lsigma) and np.isscalar(rsigma)):
        raise NotImplementedError("Not implemented for Gaussians with diagonal or full covariances.")
    if lsigma != rsigma:
        raise NotImplementedError("Not implemented for Gaussians with different variances.")
    
    distances = lmu - rmu
    # ensure it is a tensor
    if np.isscalar(distances):
        distances = torch.tensor(distances)
    distances = torch.norm(distances, p=2, dim=-1).view(-1).to(alphas)
    return torch.ger(distances**2, alphas * (alphas - 1) / (2 * lsigma**2))