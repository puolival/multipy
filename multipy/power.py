# -*- encoding: utf-8 -*-
"""Function for visualizing empirical power as a function of effect size
in the spatial two-group or separate-classes model.

Author: Tuomas Puoliväli
Email: tuomas.puolivali@helsinki.fi
Last modified: 22th March 2019
License: Revised 3-clause BSD
"""
from data import square_grid_model, spatial_separate_classes_model

from fdr import lsu, qvalue, tst
from fwer import bonferroni, hochberg
from permutation import tfr_permutation_test

import matplotlib.pyplot as plt

import numpy as np

from repeat import fwer_replicability

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
                          alpha=0.05, N=25, n_iter=10, verbose=True):
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
            Y = tst(X.flatten(), alpha)
            Y = Y.reshape(nl, nl)
            tp, _, _, fn = grid_model_counts(Y, nl, sl)
            pwr[i, j] = empirical_power(tp, tp+fn)

    return np.mean(pwr, axis=1)

def two_group_reproducibility():
    # TODO: make parameters
    method = tst
    alpha = 0.05
    emph_primary = np.asarray([0.02, 0.5, 0.98])
    n_emphs = len(emph_primary)
    nl, sl = 90, 30
    deltas = np.linspace(0.2, 2.4, 12)
    n_deltas = len(deltas)
    N = 25
    n_iter = 10

    """Compute the reproducibility rate for each effect size and
    primary study emphasis, for several iterations."""
    reproducible = np.zeros([n_deltas, n_emphs, n_iter])

    for ind in np.ndindex(n_deltas, n_emphs, n_iter):
        delta, emphasis = deltas[ind[0]], emph_primary[ind[1]]
        X_pri = square_grid_model(nl, sl, N, delta, equal_var=True)[0]
        X_fol = square_grid_model(nl, sl, N, delta, equal_var=True)[0]

        X_pri, X_fol = X_pri.flatten(), X_fol.flatten()

        R = fwer_replicability(X_pri, X_fol, emphasis,
                               method, alpha)
        R = np.reshape(R, [nl, nl])
        tp, _, _, fn = grid_model_counts(R, nl, sl)
        reproducible[ind] = tp / float(tp+fn)

    reproducible = np.mean(reproducible, axis=2)

    """Fit logistic functions."""
    logistic_k, logistic_x0 = np.zeros(n_emphs), np.zeros(n_emphs)
    for i in np.arange(0, n_emphs):
        params = curve_fit(logistic_function, deltas, reproducible[:, i])[0]
        logistic_k[i], logistic_x0[i] = params

    # TODO: separate visualization from computations
    """Visualize the results."""
    sns.set_style('darkgrid')
    fig = plt.figure(figsize=(8, 5))

    ax = fig.add_subplot(111)
    ax.plot(deltas, reproducible, '.')

    for i in np.arange(0, n_emphs):
        logistic_x = np.linspace(deltas[0], deltas[-1], 100)
        logistic_y = logistic_function(logistic_x, logistic_k[i],
                                       logistic_x0[i])
        plt.plot(logistic_x, logistic_y, '-')

    ax.set_xlim([deltas[0]-0.05, deltas[-1]+0.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel('Effect size $\Delta$')
    ax.set_ylabel('Reproducibility rate')
    ax.set_title('Two-stage FDR')
    ax.legend(emph_primary, loc='lower right')

    fig.tight_layout()
    plt.show()

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

def simulate_separate_classes_model():
    # TODO: document
    effect_sizes = np.linspace(0.2, 2.4, 12)
    pwr = separate_classes_model_power(effect_sizes)

    sns.set_style('darkgrid')
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    ax.imshow(pwr, origin='lower', cmap='viridis', interpolation='none')
    ax.grid(False)

    ax.set_xticks(np.arange(0, 12, 2))
    ax.set_yticks(np.arange(0, 12, 2))
    ax.set_xticklabels(effect_sizes[0::2])
    ax.set_yticklabels(effect_sizes[0::2])

    ax.set_xlabel('Effect size $\Delta_1$')
    ax.set_ylabel('Effect size $\Delta_2$')

    fig.tight_layout()
    plt.show()
