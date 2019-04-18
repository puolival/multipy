# -*- encoding: utf-8 -*-
"""Function for visualizing empirical power as a function of effect size
in the spatial two-group or separate-classes model.

Author: Tuomas Puoliväli
Email: tuomas.puolivali@helsinki.fi
Last modified: 22th March 2019
License: Revised 3-clause BSD

WARNING: There is unfinished code and only partial testing has been
         performed.

"""
from data import square_grid_model, spatial_separate_classes_model

from fdr import lsu, qvalue, tst
from fwer import bonferroni, hochberg
from permutation import tfr_permutation_test

import matplotlib.pyplot as plt

import numpy as np

from reproducibility import fdr_rvalue, fwer_replicability

from scipy.optimize import curve_fit

import seaborn as sns

from util import (empirical_power, empirical_fpr,
                  grid_model_counts, separate_classes_model_counts)

from viz import plot_separate_classes_model

def logistic_function(x, k, x0):
    """Logistic function with a maximum value of one.

    Input arguments:
    ================
    x : float
        Value at which to evaluate the function.

    k : float
        Steepness of the curve.

    x0 : float
        The x-value of the sigmoid's midpoint.
    """
    return 1. / (1. + np.exp(-k*(x-x0)))

def plot_power(effect_sizes, empirical_power, ax=None):
    """Function for plotting empirical power as a function of
    effect size.

    Input arguments:
    ================
    effect_sizes : ndarray [n_effect_sizes, ]
        The evaluated effect sizes.

    empirical_power : ndarray[n_effect_sizes, ]
        The empirical power at each effect size.

    Output arguments:
    =================
    fig : Figure
        The figure instance for further plotting and/or style
        adjustments.
    """

    """Fit a logistic function to the data."""
    logistic_k, logistic_x0 = curve_fit(logistic_function, effect_sizes,
                                        empirical_power)[0]
    logistic_x = np.linspace(effect_sizes[0], effect_sizes[-1], 100)
    logistic_y = logistic_function(logistic_x, logistic_k, logistic_x0)

    """Plot the data and fitted line."""
    if (ax is None):
        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(121)
    ax.plot(effect_sizes, empirical_power, '.', markersize=9)
    ax.plot(logistic_x, logistic_y, '-', linewidth=1.5)

    """Label the axes etc."""
    ax.set_xlim([effect_sizes[0]-0.05, effect_sizes[-1]+0.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel('Effect size $\Delta$', fontsize=14)
    ax.set_ylabel('Empirical power', fontsize=14)

def two_group_model_power(nl=90, sl=30, deltas=np.linspace(0.2, 2.4, 12),
                          alpha=0.05, N=25, n_iter=10, verbose=True,
                          method=tst):
    """Function for generating data under two-group model at various effect
    sizes and computing the corresponding empirical power.

    Input arguments:
    ================
    nl : int
        The length of a side of the simulated grid. There will be a total
        of nl squared tests.
    sl : int
        The length of a side of the signal region. In the simulation, there
        will be a total of sl squared tests where the alternative
        hypothesis is true.
    deltas : ndarray
        The tested effect sizes.
    alpha : float
        The desired critical level.
    N : int
        Sample size is both of the two groups.
    n_iter : int
        Number of iterations used for estimating the power.
    verbose : bool
        Flag for deciding whether to print progress reports to the console.
    """

    """Allocate memory for the results."""
    n_deltas = len(deltas)
    pwr = np.zeros([n_deltas, n_iter])

    """Simulate data at each effect size and compute empirical power."""
    for i, delta in enumerate(deltas):
        if (verbose):
            print('Effect size: %1.3f' % delta)
        for j in np.arange(0, n_iter):
            X = square_grid_model(nl, sl, N, delta, equal_var=True)[0]
            Y = method(X.flatten(), alpha)
            Y = Y.reshape(nl, nl)
            tp, _, _, fn = grid_model_counts(Y, nl, sl)
            pwr[i, j] = empirical_power(tp, tp+fn)

    return np.mean(pwr, axis=1)

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

def simulate_two_group_model():
    """Function for performing the two-group simulations and visualizing
    the results."""
    # TODO: document
    deltas = np.linspace(0.2, 2.4, 12)
    pwr = two_group_model_power(deltas=deltas)

    sns.set_style('darkgrid')
    fig = plt.figure(figsize=(8, 5))
    ax1 = fig.add_subplot(111)
    ax1.set_title('Two-stage FDR')
    plot_power(deltas, pwr, ax=ax1)
    fig.tight_layout()
    plt.show()

def separate_classes_model_power(deltas, n_iter=10, alpha=0.05, nl=45,
                                 sl=15):
    """Function for simulating data under the spatial separate-classes
    model at various effect sizes.

    Input arguments:
    ================
    deltas : ndarray
        The tested effect sizes are all possible effect size pairs
        (delta1, delta2) among the given array.
    n_iter : int
        Number of repetitions of each simulation.
    alpha : float
        The desired critical level.
    nl, sl : int
        The side lengths of the signal and noise regions within a single
        class.
    """
    # TODO: make a proper function
    n_deltas = len(deltas)
    pwr = np.zeros([n_deltas, n_deltas, n_iter])

    for ind in np.ndindex(n_deltas, n_deltas, n_iter):
        delta1, delta2 = deltas[ind[0]], deltas[ind[1]]
        X = spatial_separate_classes_model(delta1, delta2)[0]
        Y = tst(X.flatten(), alpha)
        Y = Y.reshape([nl, 2*nl])
        tp, _, _, fn = separate_classes_model_counts(Y, nl, sl)
        pwr[ind] = empirical_power(tp, tp+fn)

    return np.mean(pwr, axis=2)

def plot_separate_classes_model_power(effect_sizes, pwr):
    """Function for visualizing empirical power in the separate-classes
    model as a function of the effect size at the first and second signal
    regions.

    Input arguments:
    ================
    effect_sizes : ndarray
        The tested effect sizes.
    pwr : ndarray
        The power at each combination of effect sizes.

    Output arguments:
    =================
    fig : Figure
        An instance of the matplotlib Figure class for further editing.
    """
    sns.set_style('darkgrid')
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    ax.imshow(pwr, origin='lower', cmap='viridis', interpolation='none')
    ax.grid(False)

    # Only display every other <x/y>tick.
    n_effect_sizes = len(effect_sizes)
    ax.set_xticks(np.arange(0, n_effect_sizes, 2))
    ax.set_yticks(np.arange(0, n_effect_sizes, 2))
    ax.set_xticklabels(effect_sizes[0::2])
    ax.set_yticklabels(effect_sizes[0::2])

    ax.set_xlabel('Effect size $\Delta_1$')
    ax.set_ylabel('Effect size $\Delta_2$')
    fig.tight_layout()
    return fig

def simulate_separate_classes_model():
    effect_sizes = np.linspace(0.2, 2.4, 12)
    pwr = separate_classes_model_power(effect_sizes)
    fig = plot_separate_classes_model_power(effect_sizes, pwr)
    plt.show()

#sl = np.arange(2, 80, 4)
#pwr = np.zeros(np.shape(sl))
#
#for i, s in enumerate(sl):
#    pwr[i] = two_group_model_power(deltas=[0.7], method=bonferroni, sl=s)


def rvalue_test(nl=90, sl=30, N=25, alpha=0.05, n_iter=10):
    """Function for testing the FDR r-value method.

    Input arguments:
    ================
    effect_sizes : ndarray [n_effect_sizes, ]
        The tested effect sizes.

    emphasis : ndarray [n_emphasis, ]
        The tested amounts of emphasis placed on the primary study.

    n_iter : int
        The number of repetitions of each simulation.

    """

    # TODO: document all variables.
    # TODO: refactor and clean the code

    emphasis = np.asarray([0.02, 0.5, 0.98])
    n_emphasis = len(emphasis)
    effect_sizes = np.linspace(0.2, 2.4, 12)
    n_effect_sizes = len(effect_sizes)

    reproducibility = np.zeros([n_iter, n_effect_sizes, n_emphasis])

    for ind in np.ndindex(n_iter, n_effect_sizes, n_emphasis):
        # simulate data
        delta, emph = effect_sizes[ind[1]], emphasis[ind[2]]
        p1 = square_grid_model(delta=delta, nl=nl, sl=sl, N=N)[0]
        p2 = square_grid_model(delta=delta, nl=nl, sl=sl, N=N)[0]

        # apply fdr for the primary-study data
        s1 = tst(p1.flatten(), alpha)
        s1 = np.reshape(s1, [nl, nl])

        # apply the r-value method
        if (np.sum(s1) > 0):
            r_p1 = p1[s1]
            r_p2 = p2[s1]
            rvals = fdr_rvalue(p1=r_p1, p2=r_p2, m=nl**2, c2=emph)
            R = np.ones(np.shape(p1))
            R[s1] = rvals
            tp, _, _, fn = grid_model_counts(R < alpha, nl, sl)
            reproducibility[ind] = tp / float(tp+fn)
        else:
            reproducibility[ind] = 0

    reproducibility = np.mean(reproducibility, axis=0)

    """Visualize the result."""
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(effect_sizes, reproducibility)
    ax.legend(emphasis, loc='lower right')

    fig.tight_layout()
    plt.show()
