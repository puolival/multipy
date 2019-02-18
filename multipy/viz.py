# -*- coding: utf-8 -*-
"""Functions for visualizing P-values and Q-values, making diagnostic plots,
and showing permutation distributions etc.

This program code is part of the MultiPy (Multiple Hypothesis Testing in
Python) package.

Author: Tuomas Puoliväli (tuomas.puolivali@helsinki.fi)
Last modified: 27th July 2018.
License: Revised 3-clause BSD
Source: https://github.com/puolival/multipy/blob/master/viz.py

References:

[1] Storey JD, Tibshirani R (2003): Statistical significance for genomewide
    studies. The Proceedings of the National Academy of the United States of
    America 100(16):9440-9445. DOI: 10.1073/pnas.1530509100

[2] Bennett CM, Wolford GL, Miller MB (2009): The principled control of
    false positives in neuroimaging. Social Cognitive and Affective
    Neuroscience 4(4):417-422.

WARNING: These functions have not been entirely validated yet.

"""

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import numpy as np

import seaborn as sns

def plot_pval_hist(pvals, hist_bins=1e2, show_plot=True):
    """Plot a simple density histogram of P-values.

    Input arguments:
    pvals      - The visualized P-values.
    hist_bins  - Number of histogram bins.
    """
    # Keep empty space at minimum.
    sns.set_style('darkgrid')
    fig = plt.figure(figsize=(6, 4))
    plt.subplots_adjust(top=0.925, bottom=0.125, left=0.105, right=0.950)

    """Plot the p-value density histogram for the whole data range."""
    ax1 = fig.add_subplot(111)
    sns.distplot(pvals, bins=hist_bins, rug=True, kde=False)

    """P-values are in the range [0, 1] so limit the drawing area
    accordingly. Label the axes etc."""
    ax1.set_xlim([-0.05, 1.05])
    ax1.set_xlabel('P-value')
    ax1.set_ylabel('Density')
    if (show_plot):
        plt.show()
    return fig

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

    sns.set_style('darkgrid')
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

def plot_qvalue_pi0_fit(kappa, pik, cs, show_plot=True):
    """Make a diagnostic plot of the estimate of pi0."""
    sns.set_style('darkgrid')
    fig = plt.figure(facecolor='white', edgecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(kappa, pik, '.', markersize=10)
    ax.plot(np.linspace(0, 1, 100), cs(np.linspace(0, 1, 100)),
            '-', markersize=10)
    ax.set_xlabel('$\lambda$')
    ax.set_ylabel('$\pi_{0}(\lambda)$')
    if (show_plot):
        plt.show()
    return fig

def plot_permutation_distribution(stat, ref_stat, show_plot=True):
    """Initialize the plot."""
    sns.set_style('darkgrid')
    fig = plt.figure(figsize=(8, 6), facecolor='white', edgecolor='white')
    ax = fig.add_subplot(111)

    """Plot the permutation distribution."""
    sns.distplot(stat, hist=True, kde=True, norm_hist=True, ax=ax)

    """Plot the reference value."""
    ax.plot([ref_stat, ref_stat], [0, ax.dataLim.ymax], '-', color='g')

    """Label the axes etc."""
    ax.set_xlabel('Test statistic')
    ax.set_ylabel('Density')
    ax.set_title('Permutation distribution')
    # Location 1 is the upper right position.
    # TODO: why the reference value needs to be given first for the legend?
    ax.legend(['Reference value', 'Permutation distribution'], loc=1)
    fig.tight_layout()

    """Return and show the plot."""
    if (show_plot):
        plt.show()
    return fig

def plot_permutation_result_1d(X, Y, significant, t, clusters):
    """Function for visualizing permutation test results."""
    X_mean, Y_mean = np.mean(X, axis=0), np.mean(Y, axis=0)

    sns.set_style('darkgrid')
    fig = plt.figure()
    ax = fig.add_subplot(111)

    """Plot locations of significant clusters."""
    significant_clusters = np.unique(clusters[significant[:, 0]])
    for c in significant_clusters:
        # Finding bounding box
        x1, x2 = t[clusters == c][0], t[clusters == c][-1]
        y1, y2 = (np.min([np.min(X_mean), np.min(Y_mean)]),
                  np.max([np.max(X_mean), np.max(Y_mean)]))
        # Indicate locations with rectangles.
        ax.patches.append(Rectangle((x1, y1), np.abs(x2-x1), np.abs(y2-y1),
                          transform=ax.transData, alpha=0.25, linewidth=0))

    """Plot the analyzed data."""
    ax.plot(t, X_mean, '-', linewidth=1.5)
    ax.plot(t, Y_mean, '-', linewidth=1.5)

    """Label the axes."""
    if (t is None):
        ax.set_xlabel('Time (samples)')
    else:
        ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')

    fig.tight_layout()
    plt.show()

def plot_grid_model(X, nl, sl):
    """Function for visualizing square grid model data similar
    to Bennett and colleagues [2].

    Input arguments:
    X : ndarray
        Array of booleans indicating which p-values were declared
        significant.
    nl : int
        The side length of the noise region.
    sl : int
        The side length of the signal region.
    """
    sns.set_style('white')
    fig = plt.figure(figsize=(5, 5), facecolor='white')
    ax = fig.add_subplot(111, axisbg='black')

    """Plot indices of p-values declared significant."""
    for i, j in np.ndindex(nl, nl):
        if (X[i, j]):
            ax.plot(i, j, 'w.')

    """Plot the signal region borders."""
    d = (nl-sl) // 2
    ax.plot([d, d], [d+sl, d], 'g-', linewidth=1.5, alpha=0.75)
    ax.plot([d+sl, d], [d+sl, d+sl], 'g-', linewidth=1.5, alpha=0.75)
    ax.plot([d+sl, d+sl], [d, d+sl], 'g-', linewidth=1.5, alpha=0.75)
    ax.plot([d+sl, d], [d, d], 'g-', linewidth=1.5, alpha=0.75)

    """Reduce the amount of unnecessary empty space."""
    ax.set_xlim([0, nl])
    ax.set_ylim([0, nl])
    fig.tight_layout()
    return fig

def plot_grid_model_pvals(X, P, nl, sl):
    """Function for visualizing square grid model data similar
    to Bennett and colleagues [2].

    Input arguments:
    nl : int
        The side length of the noise region.
    sl : int
        The side length of the signal region.
    """
    sns.set_style('white')
    fig = plt.figure(figsize=(5, 5), facecolor='white')
    ax = fig.add_subplot(111, axisbg='black')

    """Plot indices of p-values declared significant."""
    for i, j in np.ndindex(nl, nl):
        if (X[i, j]):
            if (P[i, j] < 0.001):
                ax.plot(i, j, 'r.')
            elif (P[i, j] < 0.01):
                ax.plot(i, j, 'g.')
            else:
                ax.plot(i, j, 'b.')

    """Plot the signal region borders."""
    d = (nl-sl) // 2
    ax.plot([d, d], [d+sl, d], 'g-', linewidth=1.5, alpha=0.75)
    ax.plot([d+sl, d], [d+sl, d+sl], 'g-', linewidth=1.5, alpha=0.75)
    ax.plot([d+sl, d+sl], [d, d+sl], 'g-', linewidth=1.5, alpha=0.75)
    ax.plot([d+sl, d], [d, d], 'g-', linewidth=1.5, alpha=0.75)

    """Reduce the amount of unnecessary empty space."""
    ax.set_xlim([0, nl])
    ax.set_ylim([0, nl])
    fig.tight_layout()
    return fig

