# -*- coding: utf-8 -*-
"""Functions for generating test data.

This program code is part of the MultiPy (Multiple Hypothesis Testing in
Python) package.

Author: Tuomas Puoliväli
Email: tuomas.puolivali@helsinki.fi
License: Revised 3-clause BSD
Source: https://github.com/puolival/multipy/blob/master/data.py
Last modified: 23th January 2018.

References:

[1] Neuhaus KL, von Essen R, Tebbe U, Vogt A, Roth M, Riess M, Niederer W,
    Forycki F, Wirtzfeld A, Maeurer W, Limbourg P, Merx W, Haerten K (1992):
    Improved thrombolysis in acute myocardial infarction with front-loaded
    administration of alteplase: Results of the rt-PA-APSAC patency study
    (TAPS). Journal of the American College of Cardiology 19(5):385-391.

[2] Benjamini Y, Hochberg Y (1995): Controlling the false discovery rate:
    A practical and powerful approach to multiple testing. Journal of Royal
    Statistical Society. Series B (Methodological): 57(1):289-300.

[3] Reiss PT, Schwartzman A, Lu F, Huang L, Proal E (2012): Paradoxical
    results of adaptive false discovery rate procedures in neuroimaging
    studies. NeuroImage 63(4):1833-1840.
"""

import numpy as np
from numpy.random import normal, permutation

from scipy.stats import ttest_ind

def neuhaus(permute=False):
    """Function that returns the Neuhaus et al. data that was re-analyzed in
    the classic Benjamini & Hochberg (1995) FDR paper.

    Input arguments:
    permute - If true, the p-values are returned in random order. If false,
              the p-values are returned in ascending order.
    """
    pvals = np.array([0.0001, 0.0004, 0.0019, 0.0095, 0.0201, 0.0278, 0.0298,
                      0.0344, 0.0459, 0.3240, 0.4262, 0.5719, 0.6528, 0.7590,
                      1.000], dtype='float')
    if (permute):
        m = len(pvals)
        pvals = pvals[permutation(m)]
    return pvals

def two_group_model(N=25, m=1000, pi0=0.1, delta=0.7):
    """A two-group model for generating test data (described in [3] and
    elsewhere). The default input arguments can be used to reproduce the
    result reported by Reiss and colleagues in Figure 2.A.

    Input arguments:
    N     - Number of samples in each group or condition.
    m     - Number of variables
    pi0   - Proportion of null effects among the m variables.
    delta - Location parameter of the non-null part of the distribution of Y,
            which controls the effect size.

    Output arguments:
    tstat - Test statistics (Student's t's)
    pvals - P-values corresponding to the test statistics
    """
    X = normal(loc=0, scale=1, size=(N, m))
    Y = np.hstack([normal(loc=0, scale=1, size=(N, int(pi0*m))),
                   normal(loc=delta, scale=1, size=(N, int(round((1-pi0)*m, 1))))])
    # Two-sample t-test
    tstat, pvals = ttest_ind(X, Y, axis=0)
    return tstat, pvals

