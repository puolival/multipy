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

from fdr import abh, lsu, tst, qvalue

from fwer import bonferroni, hochberg

import matplotlib.pyplot as plt

import numpy as np

from permutation import tfr_permutation_test

from reproducibility import (fdr_rvalue, fwer_replicability,
                             fwer_replicability_permutation as fwer_prep,
                             partial_conjuction)

from scipy.optimize import curve_fit

import seaborn as sns

from util import empirical_power, grid_model_counts, logistic_function

from viz import plot_logistic

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
    method = bonferroni # tst
    n_iter = 20
    reproducibility = two_group_reproducibility(effect_sizes,
                                                emphasis_primary, n_iter=n_iter,
                                                method=method)
    fig = plot_two_group_reproducibility(effect_sizes, emphasis_primary,
                                         reproducibility)
    fig.axes[0].set_title('Correction method: %s' % method.__name__)
    plt.show()

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
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel('Effect size')
    ax.set_ylabel('Reproducibility rate')
    ax.set_title('Two-stage FDR')
    ax.legend(emphasis_primary, loc='lower right')

    fig.tight_layout()
    return fig

def rvalue_test(effect_sizes=np.linspace(0.2, 2.4, 12),
                emphasis=np.asarray([0.02, 0.5, 0.98]), method=tst,
                nl=90, sl=30, N=25, alpha=0.05, n_iter=10):
    """Function for simulating primary and follow-up experiments using the
    two-group model and testing which effects are reproducible using the FDR
    r-value method.

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
        print ind
        """Simulate primary and follow-up experiments."""
        delta, emph = effect_sizes[ind[1]], emphasis[ind[2]]
        p1 = square_grid_model(delta=delta, nl=nl, sl=sl, N=N)[0]
        p2 = square_grid_model(delta=delta, nl=nl, sl=sl, N=N)[0]

        """Test which hypotheses are significant in the primary study.
        This is done for selecting hypotheses for the follow-up study."""
        if (method.__name__ == 'qvalue'):
            significant_primary = method(p1.flatten(), alpha)[0]
        else:
            significant_primary = method(p1.flatten(), alpha)
        significant_primary = np.reshape(significant_primary, [nl, nl])

        """If there were significant hypotheses in the primary study,
        apply the r-value method to test which ones can be replicated in
        the follow-up study."""
        if (np.sum(significant_primary) > 0):
            rvals = fdr_rvalue(p1=p1[significant_primary],
                               p2=p2[significant_primary], m=nl**2, c2=emph)
            R = np.ones(np.shape(p1))
            R[significant_primary] = rvals
            tp, _, _, fn = grid_model_counts(R < alpha, nl, sl)
            reproducibility[ind] = tp / float(tp+fn)

    reproducibility = np.mean(reproducibility, axis=0)
    return reproducibility

def simulate_rvalue():
    """Function for simulating primary and follow-up experiments using the
    two-group model and testing which hypotheses are reproducible. The
    FDR-based r-value method is used to decide which findings are considered
    reproducible. We compare here the BH FDR, two-stage FDR, and q-value
    methods."""

    """Define settings for the r-value method simulations."""
    methods = [abh]
    n_methods = len(methods)
    effect_sizes = np.linspace(0.2, 2.4, 12)
    n_effect_sizes = len(effect_sizes)
    emphasis = np.asarray([0.02, 0.5, 0.98])
    n_emphasis = len(emphasis)
    n_iter = 20

    """Compute reproducibility of true effects for each of the
    three different methods."""
    reproducibility = np.zeros([n_methods, n_effect_sizes, n_emphasis])
    for i, method in enumerate(methods):
        reproducibility[i, :] = rvalue_test(effect_sizes=effect_sizes,
                                            emphasis=emphasis,
                                            n_iter=n_iter, method=method)

    """Visualize the results."""
    plot_rvalue_test(effect_sizes, reproducibility, emphasis)

def plot_rvalue_test(effect_sizes, reproducibility, emphasis):
    """Visualize the result."""
    sns.set_style('darkgrid')
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)

    n_methods = np.shape(reproducibility)[0]
    method_colors = ['r', 'g', 'b']

    for i in np.arange(0, n_methods):
        ax = plot_logistic(effect_sizes, reproducibility[i, :], ax,
                           color=method_colors[i])

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
    fig.tight_layout()
    plt.show()

def direct_replication_fwer_partial_conjunction():
    """Perform a comparison of the partial conjuction and FWER
    replicability methods using the two-group model."""

    N, nl, sl = 25, 90, 30
    effect_sizes = np.linspace(0.6, 2.4, 12)
    n_effect_sizes = len(effect_sizes)
    method = lsu # hochberg #bonferroni
    emphasis = np.asarray([0.02, 0.05, 0.10, 0.30, 0.50,
                           0.70, 0.90, 0.95, 0.98])
    n_emphasis = len(emphasis)

    """Generate the test data."""
    print('Simulating primary and follow-up experiments ..')

    # Allocate memory.
    pvals_pri = np.zeros([n_effect_sizes, nl, nl])
    pvals_sec = np.zeros(np.shape(pvals_pri))

    # Obtain the uncorrected p-values.
    for i, delta in enumerate(effect_sizes):
        pvals_pri[i] = square_grid_model(nl, sl, N, delta)[0]
        pvals_sec[i] = square_grid_model(nl, sl, N, delta)[0]

    """Find reproducible effects using the FWER replicability
    method."""
    print('Estimating reproducibility: FWER replicability ..')

    repr_fwer = np.zeros([n_effect_sizes, n_emphasis])

    for i in np.ndindex(n_effect_sizes, n_emphasis):
        # Find reproducible effects and rearrange the data.
        result = fwer_replicability(pvals_pri[i[0]].flatten(),
                                    pvals_sec[i[0]].flatten(),
                                    emphasis[i[1]], method)
        result = np.reshape(result, [nl, nl])

        # Compute the number reproducible true effects.
        repr_fwer[i] = (grid_model_counts(result, nl, sl)[0] /
                        float(sl ** 2))

    """Find reproducible effects using the partial conjuction
    method."""
    print('Estimating reproducibility: Partial conjuction ..')

    repr_part = np.zeros([n_effect_sizes])

    for i in np.ndindex(n_effect_sizes):
        result = partial_conjuction(pvals_pri[i].flatten(),
                                    pvals_sec[i].flatten(), method)
        result = np.reshape(result, [nl, nl])
        repr_part[i] = (grid_model_counts(result, nl, sl)[0] /
                        float(sl ** 2))

    """Visualize the data."""
    sns.set_style('white')
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)

    plot_logistic(effect_sizes, repr_fwer[:, emphasis<=0.5],
                  ax=ax, color='k')
    plot_logistic(effect_sizes, repr_fwer[:, emphasis>0.5],
                  ax=ax, color='g')
    plot_logistic(effect_sizes, repr_part, ax=ax, color='b')

    ax.set_xlabel('Effect size')
    ax.set_ylabel('Reproducibility rate')

    fig.tight_layout()
    plt.show()

def permutation_test_fwer_replicability(effect_sizes, emphasis_primary,
                                        nl=90, sl=30, alpha=0.05, N=25,
                                        n_iter=20, t_threshold=1.0):
    """Estimate reproducibility in the two-group model using the
    Maris-Oostenveld permutation test with the Phipson-Smyth p-value
    correction.

    Input arguments:
    ================
    effect_sizes : ndarray
        Tested effect sizes (Cohen's d's).

    emphasis_primary : ndarray
        Amount of emphasis placed on the primary study.

    nl, sl : int
        The sizes of the noise and signal regions respectively.

    alpha : float
        The desired critical level.

    N : int
        Sample size in each of the two groups.

    n_iter : int
        Number of repetitions of the simulation at each distinct
        effect size.

    t_threshold : float
        The t-threshold used in the permutation test.
    """

    n_effect_sizes = len(effect_sizes)
    n_emphasis = len(emphasis_primary)
    reproducibility = np.zeros([n_effect_sizes, n_emphasis, n_iter])

    """Estimate reproducibility at each effect size."""
    for ind in np.ndindex(n_effect_sizes, n_emphasis, n_iter):
        print ind

        # Generate new raw data.
        delta, emphasis = effect_sizes[ind[0]], emphasis_primary[ind[1]]
        X_raw_p, Y_raw_p = square_grid_model(nl, sl, N, delta)[2:4]
        X_raw_f, Y_raw_f = square_grid_model(nl, sl, N, delta)[2:4]

        # Here *_p = primary study, *_f = follow-up study.
        R = fwer_prep(X_raw_p, Y_raw_p, X_raw_f, Y_raw_f,
                      tfr_permutation_test, emphasis, alpha)
        tp, _, _, fn = grid_model_counts(R, nl, sl)
        reproducibility[ind] = tp / float(tp+fn)

    reproducibility = np.mean(reproducibility, axis=2)

    """Visualize the results."""
    sns.set_style('white')
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    colors = ['r', 'g', 'b']
    ax.plot(effect_sizes, reproducibility, '.')
    ax.plot(effect_sizes, reproducibility, '-')
    fig.tight_layout()
    plt.show()

def simulate_permutation_fwer_replicability():
    effect_sizes = np.linspace(0.2, 2.4, 12)
    emphasis = np.asarray([0.02, 0.5, 0.98], dtype='float')
    permutation_test_fwer_replicability(effect_sizes, emphasis)
