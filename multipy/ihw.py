# -*- encoding: utf-8 -*-
"""Methods for FDR control with independent hypothesis weighting (IHW).

Author: Tuomas Puoliväli
Email: tuomas.puolivali@helsinki.fi
Last modified: 6th March 2019
License: Revised 3-clause BSD

References:

[1] Ignatiadis N, Klaus B, Zaugg JB, Huber W (2016): Data-driven hypothesis
    weighting increases detection power in genome-scale multiple testing.
    Nature Methods 13:577-580.

[2] Genovese CR, Roeder K, Wasserman L (2006): False discovery control with
    p-value weighting. Biometrika 93(3):509-524.
"""

import numpy as np

from scipy.optimize import minimize

def ihw_grw(pvals, weights, method, alpha=0.05):
    """The IHW method by Genovese, Roeder, and Wasserman (GRW) [1, 2].

    Input arguments:
    ================
    pvals : ndarray [n_tests, ]
        The uncorrected p-values.

    weights : ndarray [n_test, ]
        The hypothesis weights. These must be independent of the p-values
        (verify mathematically or empirically).

    method : function
        The applied FDR procedure.

    alpha : float
        The desired critical level.

    Output arguments:
    =================
    significant : [n_tests, ]
        Significant p-values after the correction has been applied.
    """

    """Check that the weights average to one."""
    tolerance = 1e-7
    if (np.abs(np.mean(weights)-1) > tolerance):
        raise Exception('The weights must average to one!')

    """Check that the method is supported."""
    supported_methods = ['lsu']
    if (method.__name__ not in supported_methods):
        raise Exception('The method %s is not supported! % method.__name__')

    """Apply the correction."""
    weighted_pvals = pvals / weights
    significant = method(weighted_pvals, alpha)
    return significant


def _f_naive_ihw(weights, pvals, groups, method, alpha):
    """The minimized objective function for the naive IHW method."""
    weights = weights / np.mean(weights)
    pval_weights = np.zeros(np.shape(pvals))
    for i, g in enumerate(np.unique(groups)):
        pval_weights[groups == g] = weights[i]
    weighted_pvals = pvals / pval_weights

    significant = method(weighted_pvals, alpha)
    return -np.sum(significant)

def ihw_naive(pvals, groups, method, alpha=0.05):
    """The "naive IHW" method by Ignatiadis et al. [1].

    Input arguments:
    ================
    """

    """Initialize the weights."""
    n_groups = len(np.unique(groups))
    initial_weights = np.ones(n_groups, dtype='float')

    """Optimize the weights."""
    weights = minimize(fun=_f_naive_ihw, x0=initial_weights,
                       args=(pvals, groups, method, alpha),
                       method='nelder-mead').x

    """Apply the correction."""
    pval_weights = np.zeros(np.shape(pvals))
    for i, g in enumerate(np.unique(groups)):
        pval_weights[groups == g] = weights[i]
    weighted_pvals = pvals / pval_weights
    significant = method(weighted_pvals, alpha)

    return significant, weighted_pvals, weights

