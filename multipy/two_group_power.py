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
# from rft import rft_2d

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

    Output arguments:
    =================
    pwr : ndarray [n_deltas, n_iter]
        The power at each tested effect size at each iteration.

    fpr : ndarray [n_deltas, n_iter]
        The corresponding false positive rates.
    """

    """Allocate memory for the results."""
    n_deltas = len(deltas)
    pwr, fpr = np.zeros([n_deltas, n_iter]), np.zeros([n_deltas, n_iter])

    """Simulate data at each effect size and compute empirical power."""
    for i, delta in enumerate(deltas):
        if (verbose):
            print('Effect size: %1.3f' % delta)
        for j in np.arange(0, n_iter):
            # NOTE: output arguments 1-4 needed for permutation testing.
            # NOTE: output arguments 1-2 needed for RFT based testing.
            X = square_grid_model(nl, sl, N, delta, equal_var=True)[0]

            # TODO: q-value method returns a tuple with the first element
            # containing the decision.
            # Y = tfr_permutation_test(X_raw, Y_raw, alpha=alpha,
            #                          n_permutations=100, threshold=1)
            # Y = rft_2d(T, fwhm=3, alpha=alpha, verbose=True)[0]
            Y = method(X.flatten(), alpha)
            Y = Y.reshape(nl, nl)
            tp, fp, _, fn = grid_model_counts(Y, nl, sl)
            pwr[i, j] = empirical_power(tp, tp+fn)
            fpr[i, j] = float(fp) / float(nl ** 2 - sl ** 2)

    return np.mean(pwr, axis=1), np.mean(fpr, axis=1)

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
    pwr, fpr = two_group_model_power(deltas=effect_sizes, method=method)

    """Visualize the results."""
    # Power
    sns.set_style('darkgrid')
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    ax.set_title('Method: %s' % method.__name__)
    plot_power(effect_sizes, pwr, ax=ax)
    fig.tight_layout()
    plt.show()

    # False positive rate
    del fig
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    ax.set_title('Method: %s' % method.__name__)
    plot_power(effect_sizes, np.log10(fpr), ax=ax)
    ax.set_ylim([-4, 0]) # = p-value range 0.0001 to 0.1.
    fig.tight_layout()
    plt.show()

def simulate_two_group_example(delta=0.8, nl=90, sl=30, N=25):
    """Function for performing a simulation using the two-group model and
    visualizing the t-values.

    Input arguments:
    ================
    delta : float
        Effect size (Cohen's d).
    nl, sl : int
        The sizes of the signal and noise regions.
    N : int
        Sample size in each of the two groups.
    """
    pvals, tstats = square_grid_model(nl, sl, N, delta,
                                      equal_var=True)[0:2]
    sns.set_style('white')
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    im = ax.imshow(tstats, origin='lower', interpolation='none',
                   aspect='auto', cmap='gray')
    color_bar = fig.colorbar(im)
    color_bar.set_label('T-statistic')
    # ax.set_title('Effect size = %1.3f' % delta)
    fig.tight_layout()
    plt.show()
