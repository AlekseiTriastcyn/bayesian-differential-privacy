#!/usr/bin/env python3
# Author:   Aleksei Triastcyn
# Date:     12 August 2020

import itertools
import numpy as np
import scipy as sp
import torch
import warnings

from scipy.stats import t, binom
from scipy.special import logsumexp

from scaled_renyi import scaled_renyi_gaussian


class BayesianPrivacyAccountant:
    """ 
        Bayesian privacy accountant for Bayesian DP.
        See details in A. Triastcyn, B. Faltings (2019). Bayesian Differential Privacy for Machine Learning. 
        https://arxiv.org/pdf/1901.09697.pdf
    """
    
    def __init__(self, powers=32, total_steps=1, scaled_renyi_fn=None, conf=1-1e-16, bayesianDP=True):
        """
            Creates BayesianPrivacyAccountant object.
            
            Parameters
            ----------
            power : number, required
                Order of the Renyi divergence used in accounting.
            total_steps : number, optional
                Total number of privacy mechanism invocations envisioned.
                Only needed if bayesianDP==True. Can be an upper bound.
            scaled_renyi_fn : function, required
                Function pointer to compute scaled Renyi divergence (defines privacy mechanism). 
                If None, sampled Gaussian mechanism is assumed.
            conf : number, required
                Required confidence level in Bayesian accounting (only used if bayesianDP==True).
            bayesianDP : boolean, optional
                Return Bayesian DP bound. If False, returns the classical DP bound.
        """
        self.powers = powers
        self.holder_correction = total_steps
        self.scaled_renyi_fn = scaled_renyi_gaussian if scaled_renyi_fn is None else scaled_renyi_fn
        self.conf = conf
        self.bayesianDP = bayesianDP
        # total accumulated privacy costs (log of moment generating function) per Renyi order
        self.privacy_cost = np.zeros_like(self.powers, dtype=np.float_)
        # total accumulated confidence of Bayesian estimator (eventually, incorporated in delta)
        self.logconf = 0
        # history of minimum privacy cost per iteration
        self.history = []
        self.steps_accounted = 0
    
    
    def get_privacy(self, target_eps=None, target_delta=None):
        """
            Computes privacy parameters (eps, delta).
            
            Parameters
            ----------
            target_eps : number, required
                Target epsilon.
            target_delta : number, required
                Target delta.
            
            Returns
            -------
            out : tuple
                A pair (eps, delta)
        """
        if (target_eps is None) and (target_delta is None):
            raise ValueError("Both parameters cannot be None")
        if (target_eps is not None) and (target_delta is not None):
            raise ValueError("One of the parameters has to be None")
        if self.steps_accounted > self.holder_correction:
            warnings.warn(
                f"Accountant invoked for {self.steps_accounted} steps, "
                f"but corrected only for {self.holder_correction}. "
                "Privacy may be underestimated due to incorrect parameters in Hölder's inequality. "
                "To fix the issue, specify the total number of times a privacy mechanism will be "
                "invoked in 'total_steps' when creating BayesianPrivacyAccountant."
            )
        if target_eps is None:
            return np.min((self.privacy_cost - np.log(target_delta - (1 - np.exp(self.logconf)))) / self.powers), target_delta
        else:
            return target_eps, np.min(np.exp(self.privacy_cost - self.powers * target_eps))
    
    
    def accumulate(self, ldistr, rdistr, q=1, steps=1):
        """
            Accumulates privacy cost for a given number of steps.
            
            Parameters
            ----------
            ldistr : tuple or array, required
                Parameters of the left distribution (i.e., imposed by D).
            rdistr : tuple or array, required
                Parameters of the right distribution (i.e., imposed by D').
            q : number, required
                Subsampling probability (for subsampled mechanisms).
            steps : number, required
                Number of steps/invocations of the privacy mechanism.

            Returns
            -------
            out : tuple
                Total privacy cost and log confidence.
        """
        if np.isscalar(self.powers):
            bdp = self._compute_bdp(self.powers, self.scaled_renyi_fn, ldistr, rdistr, q, steps)
        else:
            bdp = np.array([self._compute_bdp(power, self.scaled_renyi_fn, ldistr, rdistr, q, steps) 
                                     for power in self.powers])
        self.privacy_cost += bdp
        self.history += [np.min(self.privacy_cost)]
        self.logconf += np.log(self.conf) if self.bayesianDP else 0
        self.steps_accounted += steps
        return self.history[-1], self.logconf

    
    def _compute_bdp(self, power, scaled_renyi_fn, ldistr, rdistr, q, steps):
        """
            Compute privacy cost for a given number of steps.
            
            Parameters
            ----------
            power : number, required
                Order of the Renyi divergence used in privacy cost computation.
            scaled_renyi_div_fn : function, required
                Function pointer to compute scaled Renyi divergence.
            ldistr : tuple or array, required
                Parameters of the left distribution (i.e., imposed by D).
            rdistr : tuple or array, required
                Parameters of the right distribution (i.e., imposed by D').
            q : number, required
                Subsampling probability (for subsampled mechanisms).
            steps : number, required
                Number of steps/invocations of the privacy mechanism.

            Returns
            -------
            out : tuple
                Total privacy cost and log confidence.
        """
        c_L = self._log_binom_expect(power + 1, q, scaled_renyi_fn, ldistr, rdistr)
        c_R = self._log_binom_expect(power + 1, q, scaled_renyi_fn, rdistr, ldistr)
        logmgf_samples = self.holder_correction * steps * torch.max(c_L, c_R).cpu().numpy()
        n_samples = np.size(logmgf_samples)

        if not self.bayesianDP:
            return np.asscalar(logmgf_samples) / self.holder_correction

        if n_samples < 3:
            raise ValueError("Number of samples is too low for estimating privacy cost.")

        # What we want to compute (numerically unstable):
        #    mgf_mean = np.mean(np.exp(logmgf))
        #    mgf_std = np.std(np.exp(logmgf))
        #    self.privacy_cost += np.log(mgf_mean + t.ppf(q=self.conf, df=n_samples-1) * mgf_std / np.sqrt(n_samples - 1))

        # Numerically stable implementation
        max_logmgf = np.max(logmgf_samples)
        log_mgf_mean = -np.log(n_samples) + logsumexp(logmgf_samples)
        if np.std(logmgf_samples) < np.finfo(np.float).eps:
            warnings.warn("Variance of privacy cost samples is 0. Privacy estimate may not be reliable!")
            bdp = log_mgf_mean / self.holder_correction
        else:
            log_mgf_std = 0.5 * (2 * max_logmgf - np.log(n_samples) +\
                                 np.log(np.sum(np.exp(2 * logmgf_samples - 2 * max_logmgf) -\
                                               np.exp(2 * log_mgf_mean - 2 * max_logmgf))))
            log_conf_pentalty = np.log(t.ppf(q=self.conf, df=n_samples-1)) + log_mgf_std - 0.5 * np.log(n_samples - 1)
            bdp = logsumexp([log_mgf_mean, log_conf_pentalty]) / self.holder_correction
        
        return bdp
    
    
    def _log_binom_expect(self, n, p, scaled_renyi_fn, ldistr, rdistr):
        """
            Computes logarithm of expectation over binomial distribution with parameters (n, p).
            
            Parameters
            ----------
            n : number, required
                Number of Bernoulli trials.
            p : number, required
                Probability of success.
            scaled_renyi_fn : function, required
                Function pointer to compute scaled Renyi divergence (inside Bernoulli expectation).
            ldistr : tuple or array, required
                Parameters of the left distribution (i.e., imposed by D).
            rdistr : tuple or array, required
                Parameters of the right distribution (i.e., imposed by D').

            Returns
            -------
            out : tuple
                Logarithm of expectation of scaled_renyi_fn over binomial distribution.
        """
        k = torch.arange(n + 1, dtype=torch.float)
        log_binom_coefs = torch.tensor(binom.logpmf(k, n=n, p=p))
        return torch.logsumexp(log_binom_coefs + scaled_renyi_fn(k, ldistr, rdistr), dim=1)
   