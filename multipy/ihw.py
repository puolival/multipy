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
"""

def ihw_naive(pvals, weights, method, alpha=0.05):
    """The "naive IHW" method.

    Input arguments:
    ================
    pvals : ndarray [n_tests, 1]
        The uncorrected p-values.

    weights : ndarray [n_test, 1]
        The hypothesis weights. These must be independent of the p-values
        (verify mathematically or empirically).

    method : function
        The applied FDR procedure.

    alpha : float
        The desired critical level.

    Output arguments:
    =================
    significant : [n_tests, 1]
        Significant p-values after the correction has been applied.
    """

    """Check that the weights average to one."""
    tolerance = 1e-7
    if (np.abs(np.mean(weights)-1) > tolerance):
        raise Exception('The weights must average to one!')

    """Check that the method is supported."""
    supported_methods = ['lsu']
    if (method.__name__ not in supported_methods):
        raise Exception('The method %s is not supported! % method.__name__)

    """Apply the correction."""
    weighted_pvals = pvals / weights
    significant = method(weighted_pvals, alpha)
    return significant
