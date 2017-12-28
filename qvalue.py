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

from scipy import interpolate
from scipy.stats import ttest_ind

import seaborn as sb

def _plot_statistic_qvalue(stats, stat_name, qvals):
    """Plot q-values as a function of the underlying test statistics.
    stats     - The test statistics, e.g. Student's t's
    stat_name - Name of the test statistic, e.g. t for Student's t test
    qvals     - The q-values corresponding to the test statistics.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ind = np.argsort(stats)
    ax.plot(stats[ind], qvals, '.', markersize=10)
    ax.set_xlabel(stat_name)
    ax.set_ylabel('Q-value')
    plt.show()

def _make_test_data(n_tests, sample_size):
    """Make some test data."""
    X = normal(loc=2.5, scale=4., size=(n_tests, sample_size))
    Y = normal(loc=4.0, scale=4., size=(n_tests, sample_size))
    stat, p = ttest_ind(X, Y, axis=1, equal_var=True)
    return stat, p

def _plot_pval_hist(pvals):
    """Plot a probability density histogram of the given p-values."""
    fig = plt.figure(facecolor='white', edgecolor='white')
    ax = fig.add_subplot(111)
    ax.hist(pvals, range=(0., 1.), bins=20, normed=True)
    ax.hlines(1., xmin=0, xmax=1, colors='gray', linestyles='dotted')
    ax.set_xlabel('P-value')
    ax.set_ylabel('Probability density')
    ax.legend(['Uniform density'])
    plt.show()

def _plot_pval_qval_scatter(pvals, qvals):
    """Make a scatter plot of the p-values and q-values."""
    fig = plt.figure(facecolor='white', edgecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(pvals, qvals, '.', markersize=10)
    ax.set_xlabel('P-value')
    ax.set_ylabel('Q-value')
    plt.show()

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

def _plot_qvalue_significant_features(qvals):
    """Plot the number of significant features as a function of q-value
    threshold."""
    fig = plt.figure(facecolor='white', edgecolor='white')
    ax = fig.add_subplot(111)
    thresholds = np.arange(0., 0.1, 0.001)
    significant_features = [np.sum(qvals<t) for t in thresholds]
    ax.plot(thresholds, significant_features, '.', markersize=10)
    ax.set_xlabel('Q-value')
    ax.set_ylabel('Number of significant features')
    plt.show()

def qvals(pvals):
    """Function for estimating q-values from p-values."""
    # Count and sort the p-values into ascending order.
    m, pvals = len(pvals), np.sort(pvals)

    # Estimate proportion of features that are truly null.
    kappa = np.arange(0, 0.96, 0.01)
    pik = [sum(pvals > k) / (m*(1-k)) for k in kappa]
    cs = interpolate.UnivariateSpline(kappa, pik, k=3, s=None, ext=0)
    pi0 = float(cs(1.))
    print 'The estimated proportion of truly null features is %.3f' % pi0

    # Sanity check
    if pi0 < 0 or pi0 > 1:
        pi0 = 1
        print 'The proportion was not in [0, 1] and was set as 1.'

    # Calculate the q-values.
    qvals = np.zeros(np.shape(pvals))
    qvals[-1] = pi0*pvals[-1]
    for i in xrange(m-2, -1, -1):
        qvals[i] = min(pi0*m*pvals[i]/float(i+1), qvals[i+1])

    # Print number of significant features
    print 'There are %d significant features' % np.sum(qvals<0.05)

    # Make plots
    _plot_pval_hist(pvals)
    _plot_pval_qval_scatter(pvals, qvals)
    _plot_pi0_fit(kappa, pik, cs)
    _plot_qvalue_significant_features(qvals)
    _plot_statistic_qvalue(stats, 't', qvals)

stats, pvals = _make_test_data(1000, 20)

qvals(pvals)

