# -*- coding: utf-8 -*-
"""Functions for visualizing P-values.

This program code is part of the MultiPy (Multiple Hypothesis Testing in
Python) package.

Author: Tuomas Puoliväli (tuomas.puolivali@helsinki.fi)
Last modified: 27th December 2017.
License: Revised 3-clause BSD
Source: https://github.com/puolival/multipy/blob/master/viz.py

WARNING: These functions have not been entirely validated yet.

"""

import matplotlib.pyplot as plt

import seaborn as sb

def plot_pval_hist(pvals, hist_bins=1e2):
    """Plot a simple density histogram of P-values.

    Input arguments:
    pvals      - The visualized P-values.
    hist_bins  - Number of histogram bins.
    """
    # Keep empty space at minimum.
    fig = plt.figure(figsize=(6, 4))
    plt.subplots_adjust(top=0.925, bottom=0.125, left=0.105, right=0.950)

    """Plot the p-value density histogram for the whole data range."""
    ax1 = fig.add_subplot(111)
    sb.distplot(pvals, bins=hist_bins, rug=True, kde=False)

    """P-values are in the range [0, 1] so limit the drawing area
    accordingly. Label the axes etc."""
    ax1.set_xlim([-0.05, 1.05])
    ax1.set_xlabel('P-value')
    ax1.set_ylabel('Density')
    return fig

