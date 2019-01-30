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

"""

def fwer_replicability(pvals_primary, pvals_followup, emph_primary, method):
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
        The method used to control FWER.
    """
    pass

def partial_conjuction(pvals_primary, pvals_followup, method):
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
    """
    pass
