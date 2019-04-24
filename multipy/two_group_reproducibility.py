# -*- encoding: utf-8 -*-
"""Programs for visualizing reproducibility of true effects as a function
of effect size in the spatial two-group model.

Author: Tuomas Puoliväli
Email: tuomas.puolivali@helsinki.fi
Last modified: 24th April 2019
License: Revised 3-clause BSD

WARNING: There is unfinished code and only partial testing has been
         performed.

"""
from data import square_grid_model

from fdr import lsu, tst

import matplotlib.pyplot as plt

import numpy as np

from reproducibility import fdr_rvalue, fwer_replicability

from scipy.optimize import curve_fit

import seaborn as sns

from util import empirical_power, grid_model_counts, logistic_function

def two_group_reproducibility_null_density():
    """Function for estimating reproducibility as a function of non-null
    density at a fixed effect size."""
    emphasis = np.asarray([0.02, 0.5, 0.98])
    sls = np.arange(14, 90, 16)
    nl = 90
    n_iter = 10

    """Compute the tested null density under the selected parameters."""
    null_density = (sls**2) / float(nl**2)
    n_null_density, n_emphasis = len(sls), len(emphasis)

    reproducibility = np.zeros([n_iter, n_null_density, n_emphasis])
    for j in np.arange(0, n_iter):
        for i, sl in enumerate(sls):
            r = two_group_reproducibility(sl=sl, effect_sizes=[1.0],
                                          emphasis_primary=emphasis)
            reproducibility[j, i, :] = r[0]
    reproducibility = np.mean(reproducibility, axis=0)

    """Visualize the obtained results."""
    sns.set_style('darkgrid')
    fig = plt.figure(figsize=(8, 5))

    ax = fig.add_subplot(111)
    ax.plot(null_density, reproducibility, '.-')
    ax.set_xlabel('Non-null proportion $1-\pi_{0}$')
    ax.set_ylabel('Reproducibility')
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlim([null_density[0]-0.05, null_density[-1]+0.05])
    ax.legend(emphasis, loc='upper left')
    fig.tight_layout()
    plt.show()

def simulate_two_group_reproducibility():
    """Perform the simulation."""
    effect_sizes = np.linspace(0.2, 2.4, 12)
    emphasis_primary = np.asarray([0.02, 0.5, 0.98])
    reproducibility = two_group_reproducibility(effect_sizes,
                                                emphasis_primary)
    plot_two_group_reproducibility(effect_sizes, emphasis_primary,
                                   reproducibility)

def two_group_reproducibility(effect_sizes, emphasis_primary, nl=90, sl=30,
                              alpha=0.05, N=25, n_iter=10, method=tst):
    """Function for computing reproducibility in the two-group model under
    various effect sizes and amounts of emphasis on the primary study.

    Input arguments:
    ================
    effect_sizes : ndarray [n_effect_sizes, ]
        The tested effect sizes.

    emphasis_primary : ndarray [n_emphasis_values, ]
        The tested amounts of emphasis on the primary study.

    TODO: document rest of the parameters.

    Output arguments
    ================
    reproducibility : ndarray [n_effect_sizes, n_emphasis_values]
        The observed reproducibility at the tested effect sizes and amounts
        of emphasis on the primary study.
    """
    n_effect_sizes, n_emphasis = len(effect_sizes), len(emphasis_primary)

    """Compute the reproducibility rate for each effect size and
    primary study emphasis, for several iterations."""
    reproducible = np.zeros([n_effect_sizes, n_emphasis, n_iter])

    for ind in np.ndindex(n_effect_sizes, n_emphasis, n_iter):
        # Simulate new data.
        delta, emphasis = effect_sizes[ind[0]], emphasis_primary[ind[1]]
        X_pri = square_grid_model(nl, sl, N, delta, equal_var=True)[0]
        X_fol = square_grid_model(nl, sl, N, delta, equal_var=True)[0]
        X_pri, X_fol = X_pri.flatten(), X_fol.flatten()

        # Apply the correction and compute reproducibility.
        R = fwer_replicability(X_pri, X_fol, emphasis, method, alpha)
        R = np.reshape(R, [nl, nl])
        tp, _, _, fn = grid_model_counts(R, nl, sl)
        reproducible[ind] = tp / float(tp+fn)

    reproducible = np.mean(reproducible, axis=2)
    return reproducible

def plot_two_group_reproducibility(effect_sizes, emphasis_primary,
                                   reproducibility):
    """Function for visualizing reproducibility in the two-group model.

    Input arguments:
    ================
    effect_sizes : ndarray [n_effect_sizes, ]
        The tested effect sizes.

    emphasis_primary : ndarray [n_emphasis_values, ]
        The tested primary study emphases.

    reproducibility ndarray [n_effect_sizes, n_emphasis_values]
        The observed reproducibility at each combination of effect size and
        emphasis of primary study.

    Output arguments:
    =================
    fig : Figure
        Instance of matplotlib Figure class.
    """
    n_emphs = len(emphasis_primary)

    """Fit logistic functions to the data."""
    logistic_k, logistic_x0 = np.zeros(n_emphs), np.zeros(n_emphs)
    for i in np.arange(0, n_emphs):
        params = curve_fit(logistic_function, effect_sizes,
                           reproducibility[:, i])[0]
        logistic_k[i], logistic_x0[i] = params

    """Visualize the results."""
    sns.set_style('darkgrid')
    fig = plt.figure(figsize=(8, 5))

    ax = fig.add_subplot(111)
    ax.plot(effect_sizes, reproducibility, '.')

    for i in np.arange(0, n_emphs):
        logistic_x = np.linspace(effect_sizes[0], effect_sizes[-1], 100)
        logistic_y = logistic_function(logistic_x, logistic_k[i],
                                       logistic_x0[i])
        plt.plot(logistic_x, logistic_y, '-')

    ax.set_xlim([effect_sizes[0]-0.05, effect_sizes[-1]+0.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel('Effect size $\Delta$')
    ax.set_ylabel('Reproducibility rate')
    ax.set_title('Two-stage FDR')
    ax.legend(emphasis_primary, loc='lower right')

    fig.tight_layout()
    return fig

def rvalue_test(effect_sizes=np.linspace(0.2, 2.4, 12),
                emphasis=np.asarray([0.02, 0.5, 0.98]), method=tst,
                nl=90, sl=30, N=25, alpha=0.05, n_iter=10):
    """Function for testing the FDR r-value method.

    Input arguments:
    ================
    effect_sizes : ndarray [n_effect_sizes, ]
        The tested effect sizes.

    emphasis : ndarray [n_emphasis, ]
        The tested amounts of emphasis placed on the primary study.

    n_iter : int
        The number of repetitions of each simulation.

    method : function
        The applied correction procedure.

    nl, sl : int
        The sizes of the noise and signal regions respectively.

    N : int
        The sample size in both groups.

    alpha : float
        The critical level. Default value is 0.05.

    n_iter : int
        The number of repetitions each simulation.
    """
    n_emphasis = len(emphasis)
    n_effect_sizes = len(effect_sizes)
    reproducibility = np.zeros([n_iter, n_effect_sizes, n_emphasis])

    for ind in np.ndindex(n_iter, n_effect_sizes, n_emphasis):
        """Simulate primary and follow-up experiments."""
        delta, emph = effect_sizes[ind[1]], emphasis[ind[2]]
        p1 = square_grid_model(delta=delta, nl=nl, sl=sl, N=N)[0]
        p2 = square_grid_model(delta=delta, nl=nl, sl=sl, N=N)[0]

        """Test which hypotheses are significant in the primary study.
        This is done for selecting hypotheses for the follow-up study."""
        significant_primary = method(p1.flatten(), alpha)
        significant_primary = np.reshape(significant_primary, [nl, nl])

        """If there were significant hypotheses in the primary study,
        apply the r-value method to test which ones can be replicated in
        the follow-up study."""
        if (np.sum(s1) > 0):
            rvals = fdr_rvalue(p1=p1[significant_primary],
                               p2=p2[significant_primary], m=nl**2, c2=emph)
            R = np.ones(np.shape(p1))
            R[s1] = rvals
            tp, _, _, fn = grid_model_counts(R < alpha, nl, sl)
            reproducibility[ind] = tp / float(tp+fn)

    reproducibility = np.mean(reproducibility, axis=0)
    return reproducibility

def plot_rvalue_test():
    """Visualize the result."""
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(effect_sizes, reproducibility)
    ax.legend(emphasis, loc='lower right')

    fig.tight_layout()
    plt.show()


def two_group_reproducibility_sample_size():
    """Function for computing reproducibility in the two-group model at
    various sample sizes but fixed effect size.

    Input arguments:
    ================
    effect_size : float
        The tested effect size.

    emphasis_primary : ndarray
        The amount of emphasis placed on the primary study.

    sample_sizes : ndarray [n_sample_sizes, ]
        The tested sample sizes.

    n_iter : int
        The number of repetitions of each simulation.
    """

    # TODO: make this a proper function
    """Perform the simulation."""
    effect_sizes = [1.0]
    emphasis_primary = np.asarray([0.02, 0.5, 0.98])
    sample_sizes = np.arange(8, 80, 8)
    n_emphasis, n_sample_sizes = len(emphasis_primary), len(sample_sizes)
    n_iter = 10

    reproducibility = np.zeros([n_iter, n_sample_sizes, n_emphasis])
    for ind in np.ndindex(n_iter, n_sample_sizes):
        sample_size = sample_sizes[ind[1]]
        output = two_group_reproducibility(effect_sizes, emphasis_primary,
                                           N=sample_size)
        reproducibility[ind] = output

    reproducibility = np.mean(reproducibility, axis=0)

    # TODO: separate visualization
    fig = plot_two_group_reproducibility(sample_sizes, emphasis_primary,
                                         reproducibility)
    fig.axes[0].set_xlabel('Sample size $N$')
    plt.show()

