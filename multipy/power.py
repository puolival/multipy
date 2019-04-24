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
                  grid_model_counts, separate_classes_model_counts,
                  logistic_function)

from viz import plot_separate_classes_model

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
