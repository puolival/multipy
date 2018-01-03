# -*- coding: utf-8 -*-
"""The Storey-Tibshirani q-value method.

This program code is part of the MultiPy (Multiple Hypothesis Testing in
Python) package.

Author: Tuomas Puoliväli
Last modified: 28th December 2017
Email: tuomas.puolivali@helsinki.fi, puolival@gmail.com
License: Revised 3-clause BSD
Source: https://github.com/puolival/multipy/blob/master/qvalue.py

References:

[1] Storey JD, Tibshirani R (2003): Statistical significance for genomewide
    studies. The Proceedings of the National Academy of the United States of
    America 100(16):9440-9445. DOI: 10.1073/pnas.1530509100

WARNING: These functions have not been entirely validated yet.

"""

import matplotlib.pyplot as plt

import numpy as np
from numpy.random import normal

from scipy.interpolate import UnivariateSpline
from scipy.stats import ttest_ind

import seaborn as sb

def _make_test_data(n_tests, sample_size):
    """Make some test data."""
    X = normal(loc=2.5, scale=4., size=(n_tests, sample_size))
    Y = normal(loc=4.0, scale=4., size=(n_tests, sample_size))
    stat, p = ttest_ind(X, Y, axis=1, equal_var=True)
    return stat, p

def _plot_pi0_fit(kappa, pik, cs):
    """Make a diagnostic plot of the estimate of pi0."""
    fig = plt.figure(facecolor='white', edgecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(kappa, pik, '.', markersize=10)
    ax.plot(np.linspace(0, 1, 100), cs(np.linspace(0, 1, 100)),
            '-', markersize=10)
    ax.set_xlabel('$\lambda$')
    ax.set_ylabel('$\pi_{0}(\lambda)$')
    plt.show()

def plot_qvalue_diagnostics(stats, pvals, qvals, show_plot=True):
    """Visualize q-values similar to Storey and Tibshirani [1].

    Input arguments:
    stats     - Test statistics (e.g. Student's t-values)
    pvals     - P-values corresponding to the test statistics
    qvals     - Q-values corresponding to the p-values
    show_plot - A flag for deciding whether the figure should be opened
                immediately after being drawn.

    Output arguments:
    fig       - The drawn figure.
    """
    # Sort the p-values into ascending order for improved visualization.
    stat_sort_ind, pval_sort_ind = np.argsort(stats), np.argsort(pvals)
    thresholds = np.arange(0., 0.1, 0.001)
    significant_tests = np.asarray([np.sum(qvals<t) for t in thresholds])

    fig = plt.figure(figsize=(8, 6))
    fig.subplots_adjust(top=0.95, bottom=0.1, left=0.09, right=0.96)

    """Show q-values corresponding to the test statistics."""
    ax1 = fig.add_subplot(221)
    ax1.plot(stats[stat_sort_ind], qvals[stat_sort_ind])
    ax1.set_xlabel('Test statistic')
    ax1.set_ylabel('Q-value')

    """Show q-values corresponding to the p-values."""
    ax2 = fig.add_subplot(222)
    ax2.plot(pvals[pval_sort_ind], qvals[pval_sort_ind])
    ax2.set_xlabel('P-value')
    ax2.set_ylabel('Q-value')

    """Show the number of significant tests as a function of q-value
    threshold."""
    ax3 = fig.add_subplot(223)
    ax3.plot(thresholds, significant_tests)
    ax3.set_xlabel('Q-value')
    ax3.set_ylabel('Number of significant tests')

    """Show the number of expected false positives as a function of the
    number of significant tests."""
    ax4 = fig.add_subplot(224)
    ax4.plot(significant_tests, thresholds*significant_tests)
    ax4.set_xlabel('Number of significant tests')
    ax4.set_ylabel('Number of expected false positives')

    """Return the drawn figure."""
    if (show_plot):
        plt.show() # Open the figure.
    return fig


def qvalues(pvals, threshold=0.05, verbose=True):
    """Function for estimating q-values from p-values using the
    Storey-Tibshirani method [1].

    Input arguments:
    pvals       - P-values corresponding to a family of hypotheses.
    threshold   - Threshold for deciding which q-values are significant.

    Output arguments:
    significant - An array of flags indicating which p-values are significant.
    qvals       - Q-values corresponding to the p-values.
    """

    """Count the p-values. Find indices for sorting the p-values into
    ascending order and for reversing the order back to original."""
    m, pvals = len(pvals), np.asarray(pvals)
    ind = np.argsort(pvals)
    rev_ind = np.argsort(ind)
    pvals = pvals[ind]

    # Estimate proportion of features that are truly null.
    kappa = np.arange(0, 0.96, 0.01)
    pik = [sum(pvals > k) / (m*(1-k)) for k in kappa]
    cs = UnivariateSpline(kappa, pik, k=3, s=None, ext=0)
    pi0 = float(cs(1.))
    print 'The estimated proportion of truly null features is %.3f' % pi0

    # Sanity check
    # TODO: check whether orig. paper has recommendations how to handle
    if (pi0 < 0 or pi0 > 1):
        pi0 = 1
        if (verbose):
            print 'The proportion was not in [0, 1] and was set as 1.'

    # Compute the q-values.
    qvals = np.zeros(np.shape(pvals))
    qvals[-1] = pi0*pvals[-1]
    for i in xrange(m-2, -1, -1):
        qvals[i] = min(pi0*m*pvals[i]/float(i+1), qvals[i+1])

    # Test which p-values are significant.
    significant = np.zeros(np.shape(pvals), dtype='bool')
    significant[ind] = qvals<threshold

    """Order the q-values according to the original order of the p-values."""
    qvals = qvals[rev_ind]
    return significant, qvals
