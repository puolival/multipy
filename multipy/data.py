# -*- coding: utf-8 -*-
"""Functions for generating test data.

This program code is part of the MultiPy (Multiple Hypothesis Testing in
Python) package.

Author: Tuomas Puoliväli
Email: tuomas.puolivali@helsinki.fi
License: Revised 3-clause BSD
Source: https://github.com/puolival/multipy/blob/master/data.py
Last modified: 25th July 2018.

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

[4] Efron B (2008): Simultaneous inference: When should hypothesis testing
    problems be combined? The Annals of Applied Statistics 2(1):197-223.

[5] Bennett CM, Wolford GL, Miller MB (2009): The principled control of
    false positives in neuroimaging. Social Cognitive and Affective
    Neuroscience 4(4):417-422.
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

def separate_class_model(a_N=25, b_N=25, a_m=500, b_m=500, a_pi0=0.25,
                         b_pi0=0.75, a_delta=0.5, b_delta=0.6):
    """The separate classes model described by Efron [4]. The prefixes
    a_ and b_ refer to the two separate classes. See two_group_model() for
    the definition of the variables."""
    A_tstat, A_pvals = two_group_model(a_N, a_m, a_pi0, a_delta)
    B_tstat, B_pvals = two_group_model(b_N, b_m, b_pi0, b_delta)
    # Combine
    tstats, pvals = (np.hstack([A_tstat, B_tstat]),
                     np.hstack([A_pvals, B_pvals]))
    return tstats, pvals

def square_grid_model(nl=100, sl=60, N=25, delta=0.7, equal_var=True):
    """Bennett et al [5] model.

    Input arguments:
    nl : int
        The side length of the noise region.
    sl : int
        The side length of the signal region.
    N : int
        Sample size in each of the two groups.
    delta : float
        Effect size. Difference in means between the two normal
        distributions.
    equal_var : bool
        Whether to assume equal population variances while performing
        the Student's t-tests. See also scipy.stats.ttest_ind.

    Output arguments:
    P : ndarray
        T-test p-values.
    tstats : ndarray
        The corresponding test statistics.
    X, Y : ndarray
        Samples in the two groups.
    """

    """Generate the noise statistics."""
    X_noise = normal(loc=0, scale=1, size=(nl, nl, N))
    Y_noise = normal(loc=0, scale=1, size=(nl, nl, N))
    noise_tstats, noise_pvals = ttest_ind(
        X_noise, Y_noise, axis=2, equal_var=equal_var)

    """Generate the signal statistics."""
    X_signal = normal(loc=delta, scale=1, size=(sl, sl, N))
    Y_signal = normal(loc=0, scale=1, size=(sl, sl, N))
    signal_tstats, signal_pvals = ttest_ind(
        X_signal, Y_signal, axis=2, equal_var=equal_var)

    """Combine the data so that the desired spatial structure is
    obtained."""
    P = noise_pvals
    d = (nl-sl) // 2
    P[d:(nl-d), d:(nl-d)] = signal_pvals
    tstats = noise_tstats
    tstats[d:(nl-d), d:(nl-d)] = signal_tstats
    X, Y = X_noise, Y_noise
    X[d:(nl-d), d:(nl-d)] = X_signal
    Y[d:(nl-d), d:(nl-d)] = Y_signal
    return P, tstats, X, Y


def spatial_separate_classes_model(delta1, delta2, N=25):
    """Function for simulating data under the spatial separate-classes
    model.

    Input arguments:
    ================
    delta1, delta : float
        The effect sizes at the first and second signal regions
        respectively.
    N : int
        The sample size in each group.

    Output arguments:
    =================
    pvals, tstats : ndarray
        P-values and t-statistics of each test.
    X, Y : ndarray
        The simulated data.
    """
    # TODO: make nl, sl, & N parameters of the function.
    P_a, tstats_a, X_a, Y_a = square_grid_model(nl=45, sl=15, N=N,
                                                delta=delta1, equal_var=True)
    P_b, tstats_b, X_b, Y_b = square_grid_model(nl=45, sl=15, N=N,
                                                delta=delta2, equal_var=True)

    return (np.hstack([P_a, P_b]), np.hstack([tstats_a, tstats_b]),
            np.hstack([X_a, X_b]), np.hstack([Y_a, Y_b]))

def two_class_grid_model():
    """The Bennett et al. [5] model extended for the two separate classes
    case.

    Input arguments:
    nl : int
        The side length of the noise region.
    sl<i> : int
        The side length of the ith signal region.
    N : int
        Sample size in each group.
    delta<i> : float
        Effect size in ith class. Difference in means between the two
        normal distributions.
    """

    P_a, T_a, X_a, Y_a = square_grid_model(nl=45, sl=25, N=25, delta=0.25)
    P_b, T_b, X_b, Y_b = square_grid_model(nl=45, sl=25, N=25, delta=0.50)
    P_c, T_c, X_c, Y_c = square_grid_model(nl=45, sl=25, N=25, delta=0.75)
    P_d, T_d, X_d, Y_d = square_grid_model(nl=45, sl=25, N=25, delta=1.00)

    P = np.vstack([np.hstack([P_a, P_b]), np.hstack([P_c, P_d])])
    T = np.vstack([np.hstack([T_a, T_b]), np.hstack([T_c, T_d])])
    X = np.vstack([np.hstack([X_a, X_b]), np.hstack([X_c, X_d])])
    Y = np.vstack([np.hstack([Y_a, Y_b]), np.hstack([Y_c, Y_d])])

    return P, T, X, Y
