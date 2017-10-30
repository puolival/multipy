# -*- coding: utf-8 -*-
"""Adaptive false discovery rate (FDR) procedures.

The procedures implemented here are:

abh - The adaptive linear step-up procedure, which is defined on page 494 
      in [1] and originally in [2]
lsu - The non-adaptive one-stage linear step-up procedure
orc - The 'Oracle' procedure, which is defined on pages 493 and 500 in [1]
tst - The two-stage linear step-up procedure, which is defined on page 495 
      in reference [1]

References:

[1] Benjamini Y, Krieger AM, Yekutieli D (2006): Adaptive linear step-up 
procedures that control the false discovery rate. Biometrika 93(3):491-507.

[2] Benjamini Y, Hochberg Y (2000): On the adaptive control of the false 
discovery rate in multiple testing with independent statistics. Journal of 
Educational and Behavioral Statistics 25:60-83.

[3] Benjamini Y, Hochberg Y (1995): Controlling the false discovery rate: 
A practical and powerful approach to multiple testing. Journal of Royal 
Statistical Society. Series B (Methodological): 57(1):289-300.

WARNING: This program code has not been thoroughly tested yet.

Last modified 10th October 2017
"""

import numpy as np

def lsu(pvals, q=0.05):
    """The (non-adaptive) one-stage linear step-up procedure (LSU) for 
    controlling the false discovery rate, i.e. the classic FDR method 
    proposed by Benjamini & Hochberg (1995).
    
    Input arguments:
    pvals - P-values corresponding to a family of hypotheses.
    q     - The desired false discovery rate.

    Output arguments:
    List of booleans indicating which p-values are significant (encoded 
    as boolean True values) and which are not (False values).
    """
    m = len(pvals)
    sort_ind = np.argsort(pvals)
    k = [i for i, p in enumerate(pvals[sort_ind]) if p < (i+1.)*q/m]
    significant = np.zeros(m, dtype='bool')
    if k:
        significant[sort_ind[0:k[-1]+1]] = True
    return significant

def abh(pvals, q=0.05):
    """The adaptive linear step-up procedure for controlling the false 
    discovery rate.

    Input arguments:
    pvals - P-values corresponding to a family of hypotheses.
    q     - The desired false discovery rate.
    """
    # P-values equal to 1. will cause a division by zero.
    pvals[pvals>0.99] = 0.99

    # Step 1.
    # If lsu does not reject any hypotheses, stop
    significant = lsu(pvals, q)
    if significant.all() is False:
        return significant

    # Steps 2 & 3
    m = len(pvals)
    sort_ind = np.argsort(pvals)
    m0k = [(m+1-(k+1))/(1-p) for k, p in enumerate(pvals[sort_ind])]
    j = [i for i, k in enumerate(m0k[1:]) if k > m0k[i-1]]

    # Step 4
    mhat0 = int(np.ceil(min(m0k[j[0]+1], m)))

    # Step 5
    qstar = q*m/mhat0
    return lsu(pvals, qstar)

def orc(pvals, m0, q=0.05):
    """The 'Oracle' procedure.

    Input arguments:
    pvals - P-values corresponding to a family of hypotheses.
    m0    - The number of null hypotheses that are true.
    q     - The desired false discovery rate.
    """
    m = len(pvals)
    if (m0 == 0):
        return np.ones(m, dtype='bool')
    else:
        return lsu(pvals, q*m/m0)

def tst(pvals, q=0.05):
    """The two-stage linear step-up procedure (TST) for controlling the 
    false discovery rate.

    Input arguments:
    pvals - P-values corresponding to a family of hypotheses.
    q     - The desired false discovery rate.
    """
    m, pvals = len(pvals), np.asarray(pvals)
    """Step 1. Estimate the number of null hypotheses that are true."""
    qprime = q / (1 + q)
    significant = lsu(pvals, qprime)
    r1 = np.sum(significant)
    if (r1 == 0):
        # Do not reject any hypothesis
        return np.zeros(m, dtype='bool')
    elif (r1 == m):
        return np.ones(m, dtype='bool')
    else:
        # Step 2.
        mhat0 = m - r1
        # Step 3.
        qstar = qprime*m / mhat0
        return lsu(pvals, qstar)
