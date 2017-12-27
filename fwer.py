# -*- coding: utf-8 -*-
"""Functions for controlling the family-wise error rate (FWER).

This program code is part of the MultiPy (Multiple Hypothesis Testing in
Python) package.

Author: Tuomas Puoliväli (tuomas.puolivali@helsinki.fi)
Last modified: 27th December 2017.
License: Revised 3-clause BSD
Source: https://github.com/puolival/multipy/blob/master/fwer.py

References:

[1] Hochberg Y (1988): A sharper Bonferroni procedure for multiple tests of
    significance. Biometrika 75(4):800-802.
[2] Holm S (1979): A simple sequentially rejective multiple test procedure.
    Scandinavian Journal of Statistics 6(2):65-70.
[3] Sidak Z (1967): Confidence regions for the means of multivariate normal
    distributions. Journal of the American Statistical Association 62(318):
    626-633.

WARNING: These functions have not been entirely validated yet.

"""

import numpy as np

def bonferroni(pvals, alpha=0.05):
    """A function for controlling the FWER at some level alpha using the
    classical Bonferroni procedure.

    Input arguments:
    pvals - P-values corresponding to a family of hypotheses.
    alpha - The desired family-wise error rate.
    """
    n, pvals = len(pvals), np.asarray(pvals)
    return pvals < alpha/float(n)

def hochberg(pvals, alpha=0.05):
    """A function for controlling the FWER using Hochberg's procedure [1].

    Input arguments:
    pvals - P-values corresponding to a family of hypotheses.
    alpha - The desired family-wise error rate.
    """
    m, pvals = len(pvals), np.asarray(pvals)
    ind = np.argsort(pvals)
    test = [p <= alpha/(m+1-(k+1)) for k, p in enumerate(pvals[ind])]
    if (np.all(test) is True):
        return np.ones(m, dtype='bool')
    elif (np.all(test) is False):
        return np.zeros(m, dtype='bool')
    else:
        h = np.zeros(m, dtype='bool')
        h[ind[0:np.sum(test)]] = True
        return h

def holm_bonferroni(pvals, alpha=0.05):
    """A function for controlling the FWER using the Holm-Bonferroni
    procedure [2].

    Input arguments:
    pvals - P-values corresponding to a family of hypotheses.
    alpha - The desired family-wise error rate.
    """
    m, pvals = len(pvals), np.asarray(pvals)
    ind = np.argsort(pvals)
    test = [p > alpha/(m+1-k) for k, p in enumerate(pvals[ind])]
    if (np.all(test) is True):
        """If the comparison is true for every test (all p-values are too
        large) then return a vector of falses."""
        return np.zeros(m, dtype='bool')
    elif (np.all(test) is False):
        """If the comparison is false for every test (all p-values survive)
        return a vector of trues."""
        return np.ones(m, dtype='bool')
    else:
        h = np.zeros(m, dtype='bool')
        h[0:m-np.sum(test)] = True
        return h

def sidak(pvals, alpha=0.05):
    """A function for controlling the FWER at some level alpha using the
    procedure by Sidak [3].

    Input arguments:
    pvals - P-values corresponding to a family of hypotheses.
    alpha - The desired family-wise error rate.
    """
    n, pvals = len(pvals), np.asarray(pvals)
    return pvals < 1. - (1.-alpha) ** (1./n)

