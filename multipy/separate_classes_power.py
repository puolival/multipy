# -*- coding: utf-8 -*-
"""Programs for computing and visualizing empirical power as a function of
effect size in the spatial separate-classes model.

Author: Tuomas Puoliväli
Email: tuomas.puolivali@helsinki.fi
Last modified: 24th April 2019
License: Revised 3-clause BSD

WARNING: There is unfinished code and only partial testing has been
         performed.

"""
from data import spatial_separate_classes_model

from fdr import lsu, tst

import matplotlib.pyplot as plt

import numpy as np

from reproducibility import fdr_rvalue, fwer_replicability

from scipy.optimize import curve_fit

import seaborn as sns

from util import (empirical_power, separate_classes_model_counts,
                  logistic_function)

from viz import plot_separate_classes_model

def separate_data(X, c1, c2):
    """Function for diving the dataset X into two separate classes.

    Input arguments:
    ================
    X : ndarray [n_rows, n_cols]
        The entire dataset having 'n_rows' rows and 'n_cols' columns.
    c1, c2 : ndarray [4, ]
        The coordinates of the top left and bottom right corners of
        the two classes.

        More specifically, c1[0:2] are the x-coordinates and
        c1[2:4] are the y-coordinates.

    Output arguments:
    =================
    X_c1, X_c2 : ndarray
        The separated datasets.
    """

    """Get the classes' coordinates."""
    c1_x1, c1_x2, c1_y1, c1_y2 = c1
    c2_x1, c2_x2, c2_y1, c2_y2 = c2

    """Separate the dataset into two parts."""
    X_c1 = X[c1_y1:c1_y2, c1_x1:c1_x2]
    X_c2 = X[c2_y1:c2_y2, c2_x1:c2_x2]

    return X_c1, X_c2

def separate_classes_model_power(deltas, n_iter=20, alpha=0.05, nl=45,
                                 sl=15, method=lsu,
                                 single_analysis=False):
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

    method : function
        The applied correction method.

    single_analysis : bool
        A flag for deciding whether to perform a single combined analysis
        or two separate analyses.
    """
    # TODO: make a proper function
    n_deltas = len(deltas)
    pwr = np.zeros([n_deltas, n_deltas, n_iter])

    for ind in np.ndindex(n_deltas, n_deltas, n_iter):
        delta1, delta2 = deltas[ind[0]], deltas[ind[1]]
        X = spatial_separate_classes_model(delta1, delta2)[0]

        """Perform the multiple testing using either a single combined
        analysis or two separate analyses. If two analyses are performed,
        it is assumed that the separate classes are known a priori by
        the experiments. (However, it would be interesting to also test
        happens when this assumption is incorrect.)
        """
        if (single_analysis == True):
            Y = method(X.flatten(), alpha)
            Y = Y.reshape([nl, 2*nl])
        else:
            X_c1, X_c2 = separate_data(X, [0, nl, 0, nl],
                                          [nl, 2*nl, 0, nl])
            Y1, Y2 = (method(X_c1.flatten(), alpha),
                      method(X_c2.flatten(), alpha))
            Y1, Y2 = Y1.reshape([nl, nl]), Y2.reshape([nl, nl])
            Y = np.hstack([Y1, Y2])

        """Compute empirical power."""
        tp, _, _, fn = separate_classes_model_counts(Y, nl, sl)
        pwr[ind] = empirical_power(tp, 2*sl**2)

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
    im = ax.imshow(pwr, origin='lower', cmap='viridis', interpolation='none',
                   vmin=0, vmax=1)
    ax.grid(False)

    # Only display every other <x/y>tick.
    n_effect_sizes = len(effect_sizes)
    ax.set_xticks(np.arange(0, n_effect_sizes, 2))
    ax.set_yticks(np.arange(0, n_effect_sizes, 2))
    ax.set_xticklabels(effect_sizes[0::2])
    ax.set_yticklabels(effect_sizes[0::2])

    ax.set_xlabel('Effect size $\Delta_1$')
    ax.set_ylabel('Effect size $\Delta_2$')
    return fig, im

def simulate_separate_classes_model(method):
    """Function for performing simulations using the separate-classes model
    and visualizing the results.

    Input arguments:
    ================
    method : function
        The applied correction method.
    """

    """Compute empirical power at the chosen effect sizes using the chosen
    multiple testing method."""
    single_analysis = True
    effect_sizes = np.linspace(0.2, 2.4, 12)
    pwr = separate_classes_model_power(effect_sizes, method=method,
                                       single_analysis=single_analysis)

    """Visualize the results."""
    fig, im = plot_separate_classes_model_power(effect_sizes, pwr)
    fig.axes[0].set_title('Method: %s' % method.__name__)
    fig.colorbar(im)
    fig.tight_layout()
    fig.axes[0].grid(False)
    plt.show()

def simulate_single_separate_analyses():
    """Function for simulating data using the separate-classes model and
    comparing the performance of single and separate analyses."""

    effect_sizes = np.linspace(0.2, 2.4, 12)
    n_iter = 5
    alpha = 0.05
    nl, sl = 45, 15
    method = lsu

    pwr1 = separate_classes_model_power(deltas=effect_sizes, n_iter=n_iter,
                                        alpha=alpha, nl=nl, sl=sl, method=method,
                                        single_analysis=True)
    pwr2 = separate_classes_model_power(deltas=effect_sizes, n_iter=n_iter,
                                        alpha=alpha, nl=nl, sl=sl, method=method,
                                        single_analysis=False)

    """Visualize the results."""
    sns.set_style('white')
    fig = plt.figure(figsize=(8, 5))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.imshow(pwr1, origin='lower', vmin=0, vmax=1)
    ax2.imshow(pwr2, origin='lower', vmin=0, vmax=1)

    #ax.plot(effect_sizes, pwr1[6, :], 'k')
    #ax.plot(effect_sizes, pwr2[6, :], 'r')

    ax1.set_xlabel('Effect size')
    ax1.set_ylabel('Power')

    fig.tight_layout()
    plt.show()


