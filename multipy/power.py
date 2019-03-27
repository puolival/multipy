# -*- encoding: utf-8 -*-
"""Function for visualizing empirical power as a function of effect size.

Author: Tuomas Puoliväli
Email: tuomas.puolivali@helsinki.fi
Last modified: 22th March 2019
License: Revised 3-clause BSD
"""
from data import square_grid_model

from fdr import lsu, tst
from fwer import bonferroni

import matplotlib.pyplot as plt

import numpy as np

from scipy.optimize import curve_fit

import seaborn as sns

from util import empirical_power, grid_model_counts

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
        ax = fig.add_subplot(111)
    ax.plot(effect_sizes, empirical_power, '.', markersize=9)
    ax.plot(logistic_x, logistic_y, '-', linewidth=1.5)

    """Label the axes etc."""
    ax.set_xlim([effect_sizes[0]-0.05, effect_sizes[-1]+0.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel('Effect size $\Delta$', fontsize=14)
    ax.set_ylabel('Empirical power', fontsize=14)
    ax.figure.tight_layout()
    plt.show()

"""Simulate data and test the plot function."""
nl, sl = 90, 30
deltas = np.linspace(0.2, 1.8, 9)
alpha = 0.05
N = 25
n_iter = 10

epwr = np.zeros([len(deltas), n_iter])

for i, delta in enumerate(deltas):
    print('Effect size %1.3f' % delta)
    for j in np.arange(0, n_iter):
        X, X_tstats, X_raw, Y_raw = square_grid_model(nl, sl, N, delta,
                                                      equal_var=True)
        Y_tst = lsu(X.flatten(), q=alpha)
        Y_tst = Y_tst.reshape(nl, nl)
        tp, fp, tn, fn = grid_model_counts(Y_tst, nl, sl)
        epwr[i, j] = empirical_power(tp, sl ** 2)

epwr = np.mean(epwr, axis=1)

sns.set_style('darkgrid')
fig = plt.figure(figsize=(8, 5))
ax = fig.add_subplot(111)
plot_power(deltas, epwr, ax=ax)
