# -*- coding: utf-8 -*-
"""General-purpose utility functions.

This program code is part of the MultiPy (Multiple Hypothesis Testing in
Python) package.

Author: Tuomas Puoliväli (tuomas.puolivali@helsinki.fi)
Last modified: 17th September 2018.
License: Revised 3-clause BSD
Source: https://github.com/puolival/multipy/blob/master/util.py

WARNING: These functions have not been entirely validated yet.

"""

import numpy as np

def print_result(pvals, significant_pvals, sort_pvals=True, pval_digits=4):
    """Print a list of (p-value, is_significant) tuples showing which
    p-values are significant.

    Input arguments:
    pvals             - P-values corresponding to a family of hypotheses.
    significant_pvals - An array of flags indicating which p-values are
                        significant.
    sort_pvals        - Whether to sort the p-values before printing.
    pval_digits       - Number of printed digits after the decimal place.
    """
    if (sort_pvals):
        sort_ind = np.argsort(pvals)
        pvals, significant = pvals[sort_ind], significant_pvals[sort_ind]

    # Print output directly to console.
    format_str = '{:.' + str(pval_digits) + 'f}'
    print(zip([format_str.format(p) for p in pvals], significant_pvals))

def grid_model_counts(Y, nl, sl):
    """Function for counting the number of true and false positives and
    true and false negatives in Bennett et al. like simulations.

    Input arguments:
    ================
    Y : ndarray
        Truth values indicating which point were declared significant.
    nl : int
        Side length of the noise region.
    sl : int
        Side length of the signal region.

    Output arguments:
    =================
    tp : int
        Number of true positives.
    fp : int
        Number of false positives.
    tn : int
        Number of true negatives.
    fn : int
        Number of false negatives.
    """
    Y = np.asarray(Y, dtype='bool')
    d = (nl-sl) // 2
    """Process data within the signal region."""
    tp, fn = (np.sum(Y[d:(nl-d), d:(nl-d)] == True),
              np.sum(Y[d:(nl-d), d:(nl-d)] == False))
    """Process data within the noise region."""
    mask = np.ones(np.shape(Y), dtype='bool')
    mask[d:(nl-d), d:(nl-d)] = False
    fp, tn = np.sum(Y[mask] == True), np.sum(Y[mask] == False)
    return tp, fp, tn, fn

def roc(counts):
    """Function computing TPR and FPR.

    Output arguments:
    =================
    tpr : float
        True positive rate
    fpr : float
        False positive rate
    """
    tp, fp, tn, fn = counts
    return float(tp) / (tp+fn), float(fp) / (fp+tn)

