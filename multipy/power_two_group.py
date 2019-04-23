# -*- encoding: utf-8 -*-
"""Function for visualizing empirical power as a function of effect size
in the spatial two-group model.

Author: Tuomas Puoliväli
Email: tuomas.puolivali@helsinki.fi
Last modified: 23th April 2019
License: Revised 3-clause BSD

WARNING: There is unfinished code and only partial testing has been
         performed.
"""

from data import square_grid_model

from fdr import lsu # default correction method

import matplotlib.pyplot as plt

import numpy as np

# from permutation import tfr_permutation_test

from scipy.optimize import curve_fit

import seaborn as sns

from util import empirical_power, grid_model_counts, logistic_function

def plot_power(effect_sizes, empirical_power, ax=None):
    """Function for plotting empirical power as a function of effect size.

    Input arguments:
    ================
    effect_sizes : ndarray [n_effect_sizes, ]
        The evaluated effect sizes.

    empirical_power : ndarray[n_effect_sizes, ]
        The empirical power at each effect size.

    ax : Axes
        (Optional) Axes to plot to. If not specified, a new figure with new
        axes will be created.
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

def two_group_model_power(deltas, method, nl=90, sl=30, alpha=0.05, N=25,
                          n_iter=20, verbose=True):
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
            # TODO: q-value method returns a tuple with the first element
            # containing the decision.
            # Y = tfr_permutation_test(X_raw, Y_raw, alpha=alpha,
            #                          n_permutations=100, threshold=1)
            Y = method(X.flatten(), alpha)
            Y = Y.reshape(nl, nl)
            tp, _, _, fn = grid_model_counts(Y, nl, sl)
            pwr[i, j] = empirical_power(tp, tp+fn)

    return np.mean(pwr, axis=1)

def simulate_two_group_model(effect_sizes=np.linspace(0.2, 2.4, 12),
                             method=lsu):
    """Function for performing simulations using the two-group model and
    visualizing the results.

    Input arguments:
    ================
    effect_sizes : ndarray [n_effect_sizes, ]
        The tested effect sizes (Cohen's d's). The default range is from
        d = 0.2 to 2.4.

    method : function
        The applied correction method. The default value is the
        Benjamini-Hochberg FDR method.
    """

    """Estimate power at each effect size."""
    pwr = two_group_model_power(deltas=effect_sizes, method=method)

    """Visualize the results."""
    sns.set_style('darkgrid')
    fig = plt.figure(figsize=(8, 5))
    ax1 = fig.add_subplot(111)
    ax1.set_title('Method: %s' % method.__name__)
    plot_power(effect_sizes, pwr, ax=ax1)
    fig.tight_layout()
    plt.show()

