# -*- encoding: utf-8 -*-
"""Functions for testing reproducibility.

Author: Tuomas Puoliväli
Email: tuomas.puolivali@helsinki.fi
Last modified: 18th April 2019
License: Revised 3-clause BSD

References:

[1] Heller R, Bogomolov M, Benjamini Y (2014): Deciding whether follow-up
    studies have replicated findings in a preliminary large-scale omics
    study. Proceedings of the National Academy of Sciences of the United
    states of America 111(46):16262-16267.

[2] The Heller et al. [1] R implementation of their proposed method
    (followed closely in the function fdr_rvalue).

[3] Bogomolov M, Heller R (2013): Discovering findings that replicate from
    a primary study of high dimension to a follow-up study. Journal of the
    American Statistical Association 108(504):1480-1492.

TODO: consider somehow unifying the FWER replicability methods into one
function and several helper functions. The issue here is that the method
is needed to be applied to different inputs: t-statistics, p-values, or
the original data.

"""

import numpy as np

from scipy.optimize import root
from scipy.stats import rankdata

def fwer_replicability(pvals_primary, pvals_followup, emph_primary, method,
                       alpha=0.05):
    """The Bogomolov & Heller family-wise error rate (FWER) replicability
    method.

    Input arguments:
    ================
    pvals_primary : ndarray [n_tests, ]
        P-values from the primary study.

    pvals_followup : ndarray [n_tests, ]
        P-values from the follow-up study. Variables that were not selected
        for further investigation should be marked with not-a-number (NaN)
        values.

    emph_primary : float in the open interval (0, 1)
        Emphasis given to the primary study.

    method : function
        The method used to control for false positives.

    alpha : float
        The critical level. Defaults to 0.05 when unspecified.

    Output arguments:
    =================
    replicable : ndarray [n_tests, ]
        Array of booleans indicating which tests can be considered to be
        reproducible.
    """

    """Compute emphasis given to the follow-up study."""
    if ((emph_primary < 0) or (emph_primary > 1)):
        raise Exception('Emphasis given to the primary study must be' +
                        ' in (0, 1)!')
    emph_followup = 1-emph_primary

    """Apply the correction to both sets of p-values."""
    supported_methods = ['bonferroni', 'hochberg', 'holm_bonferroni',
                         'sidak', 'lsu', 'tst']
    if (method.__name__ in supported_methods):
        significant_primary = method(pvals_primary, emph_primary*alpha)
        """Exclude not-a-number values. These have two possible sources:
        (1) tests not significant in the primary study, set here, and
        (2) tests marked as nan by the user in advance, for other
            possible reasons.
        """
        pvals_followup[~significant_primary] = np.nan
        ind = np.isnan(pvals_followup) == False

        # First check whether anything was significant in primary study.
        if (np.sum(ind) > 0):
            significant_followup = method(pvals_followup[ind],
                                          emph_followup*alpha)
    elif (method.__name__ == 'qvalue'):
        """The q-value method also returns the q-values corresponding to
        the p-values, so we have to handle it separately."""
        significant_primary = method(pvals_primary, emph_primary*alpha)[0]

        # Exclude not-a-number values.
        pvals_followup[~significant_primary] = np.nan
        ind = np.isnan(pvals_followup) == False
        if (np.sum(ind) > 0):
            significant_followup = method(pvals_followup[ind],
                                          emph_followup*alpha)[0]
    else:
        raise Exception('Unsupported correction method!')

    """Decide which tests are replicable."""
    n_tests = len(pvals_primary)
    replicable = np.zeros(n_tests, dtype='bool')
    if (np.sum(ind) > 0):
        replicable[ind] = significant_primary[ind] & significant_followup
    return replicable

def fwer_replicability_permutation(rvs_a_primary, rvs_b_primary,
                                   rvs_a_followup, rvs_b_followup,
                                   method, emph_primary, alpha=0.05):
    """The Bogomolov & Heller family-wise error rate (FWER) replicability
    method for permutation testing approaches.

    Input arguments:
    ================
    rvs_a_primary, rvs_b_primary, rvs_a_followup, rvs_b_followup : ndarray
        Primary and follow-up data.

    method : function
        The applied correction technique. For now, the only supported
        method is tfr_permutation_test.

    emph_primary : float in the open interval (0, 1)
        Emphasis given to the primary study.

    alpha : float
        The critical level. Defaults to 0.05 when unspecified.

    Output arguments:
    =================
    replicable : ndarray [n_tests, ]
        Array of booleans indicating which tests can be considered to be
        reproducible.
    """

    """Settings."""
    # TODO: consider making these adjustable parameters.
    n_permutations, t_threshold = 100, 1.0

    """Compute emphasis given to the follow-up study."""
    if ((emph_primary < 0) or (emph_primary > 1)):
        raise Exception('Emphasis given to the primary study must be' +
                        ' in (0, 1)!')
    emph_followup = 1-emph_primary

    """Compute the new critical levels."""
    alpha_primary, alpha_followup = (emph_primary * alpha, emph_followup
                                                           * alpha)

    """Perform the reproducibility testing."""
    if (method.__name__ == 'tfr_permutation_test'):
        significant_primary = method(rvs_a_primary, rvs_b_primary,
                                     n_permutations, alpha_primary,
                                     t_threshold)
        significant_followup = method(rvs_a_followup, rvs_b_followup,
                                      n_permutations, alpha_followup,
                                      t_threshold)
    else:
        raise Exception('Unsupported correction method!')

    """Decide which tests are replicable."""
    replicable = significant_primary & significant_followup
    return replicable

def fwer_replicability_rft(tstat, method, emph_primary, alpha=0.05):
    """The FWER replicability method for random field theory (RFT) based
    approaches.

    Input arguments:
    ================
    tstat : ndarray
        T-statistic for each variable.

    method : str
        The applied correction method. For now, only 'rft_2d' is supported.

    emph_primary : float
        Amount of emphasis placed on the primary study.

    alpha : float
        The desired critical level. Set as 0.05 by default.
    """

    """Compute emphasis given to the follow-up study."""
    if ((emph_primary < 0) or (emph_primary > 1)):
        raise Exception('Emphasis given to the primary study must be' +
                        ' in (0, 1)!')
    emph_followup = 1-emph_primary

    # TODO: find which are replicable.
    return replicable

def partial_conjuction(pvals_primary, pvals_followup, method, alpha=0.05):
    """The partial conjuction method for deciding reproducible effects.

    Input arguments:
    ================
    pvals_primary : ndarray [n_tests, ]
        P-values from the primary study.

    pvals_followup : ndarray [n_tests, ]
        P-values from the follow-up study. Variables that were not selected
        for further investigation should be marked with not-a-number (NaN)
        values.

    method : function
        The method used to control FWER.

    alpha : float
        The critical level. Defaults to 0.05 when unspecified.

    Output arguments:
    =================
    reproducible : ndarray [n_tests, ]
        Array of booleans indicating which tests can be considered to be
        reproducible.
    """

    """Set the p-values of those tests that were not followed up as 1."""
    pvals_followup[np.isnan(pvals_followup)] = 1.0

    """Get element-wise maximum p-values."""
    pvals_primary[np.isnan(pvals_primary)] = 1.0
    pvals = np.maximum(pvals_primary, pvals_followup)

    """Apply the correction."""
    reproducible = method(pvals.flatten(), alpha)
    return reproducible

def _fdr_rvalue_f(x, m, p1, p2, c2=0.5, l00=0.8):
    """Function for computing the FDR r-value of a given feature.

    Input arguments:
    ================
    c2 : float in range (0, 1)
        The emphasis given to the follow-up study.

    l00 : float in range [0, 1)
        The lower bound on f00, which is "the fraction of features, of the
        m features examined in the primary study, that are null in both
        studies".
    """

    # TODO: initially named to match the reference R code
    R1 = len(p1)
    c1 = (1.-c2) / (1.-l00*(1.-c2*x))

    e = np.max([m*p1 / c1, R1*p2 / c2], axis=0)
    e = e / rankdata(e, method='max')

    oe = np.argsort(e)[::-1]
    oer = np.argsort(oe)

    r = np.minimum.accumulate(e[oe])[oer]
    r = np.min([r, np.ones(np.shape(r))], axis=0)
    return r

def _fdr_rvalue_f_aux(x, i, m, p1, p2, c2, l00):
    """Auxiliary function used to enable solving f(x) = x."""
    return _fdr_rvalue_f(x, m, p1, p2, c2, l00)[i] - x

def fdr_rvalue(p1, p2, m, c2=0.5, l00=0.8):
    """Function for computing FDR r-values using the method suggested by
    Heller et al.

    Input arguments:
    ================
    p1, p2 : ndarray [n_tests, ]
        The p-values that were selected for follow-up from the primary
        study (p1) and the corresponding p-values observed in the
        follow-up study (p2).

    m : int
        The number of tests considered in the primary study.

    c2 : float in range (0, 1)
        The emphasis given to the follow-up study.

    l00 : float in range [0, 1)
        The lower bound on f00, which is "the fraction of features, of the
        m features examined in the primary study, that are null in both
        studies".
    """

    # TODO: Consider implementing the other variations of computing c1.

    tol = np.min([np.min([p1, p2]), 0.0001])

    """Compute the r-values."""
    rvalue = np.zeros(np.shape(p2), dtype='float')
    for i, _ in enumerate(p2):
        if (_fdr_rvalue_f_aux(1, i, m, p1, p2, c2, l00) >= 0):
            rvalue[i] = 1
        else:
            sol = root(_fdr_rvalue_f_aux, x0=0.5,
                       args=(i, m, p1, p2, c2, l00), tol=tol)['x']
            rvalue[i] = sol

    return rvalue
