# -*- encoding: utf-8 -*-
"""Functions for testing reproducibility.

Author: Tuomas Puoliväli
Email: tuomas.puolivali@helsinki.fi
Last modified: 30th January 2019
License: Revised 3-clause BSD

References:

[1] Heller R, Bogomolov M, Benjamini Y (2014): Deciding whether follow-up
    studies have replicated findings in a preliminary large-scale omics
    study. Proceedings of the National Academy of Sciences of the United
    states of America 111(46):16262-16267.

[2] Bogomolov M, Heller R (2013): Discovering findings that replicate from
    a primary study of high dimension to a follow-up study. Journal of the
    American Statistical Association 108(504):1480-1492.

NOTE: Work in progress. Untested code.

"""

import numpy as np

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
        # Exclude not-a-number values.
        ind = np.isnan(pvals_followup) == False
        significant_followup = method(pvals_followup[ind],
                                      emph_followup*alpha)
    elif (method.__name__ == 'qvalue'):
        """The q-value method also returns the q-values corresponding to
        the p-values, so we have to handle it separately."""
        significant_primary = method(pvals_primary, emph_primary*alpha)[0]
        # Exclude not-a-number values.
        ind = np.isnan(pvals_followup) == False
        significant_followup = method(pvals_followup[ind],
                                      emph_followup*alpha)[0]
    else:
        raise Exception('Unsupported correction method!')

    """Decide which tests are replicable."""
    n_tests = len(pvals_primary)
    replicable = np.zeros(n_tests, dtype='bool')
    replicable[ind] = significant_primary[ind] & significant_followup
    return replicable

def fwer_replicability_permutation(rvs_a_primary, rvs_b_primary,
                                   rvs_a_followup, rvs_b_followup,
                                   emph_primary, alpha):
    """The Bogomolov & Heller family-wise error rate (FWER) replicability
    method for permutation testing approaches.

    Input arguments:
    ================
    rvs_a_primary, rvs_b_primary, rvs_a_followup, rvs_b_followup : ndarray
        Primary and follow-up data.

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
    raise NotImplementedError('Unfinished implementation!')

    """Compute emphasis given to the follow-up study."""
    if ((emph_primary < 0) or (emph_primary > 1)):
        raise Exception('Emphasis given to the primary study must be' +
                        ' in (0, 1)!')
    emph_followup = 1-emph_primary

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
    raise NotImplementedError('Not implemented yet!')
