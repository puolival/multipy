# -*- coding: utf-8 -*-
"""Functions for controlling the FDR using permutation tests.

Author: Tuomas Puoliväli
Email: tuomas.puolivali@helsinki.fi
Last modified: 14th March 2018
License: 3-clause BSD

References:

[1] Maris E, Oostenveld R (2007): Nonparametric statistical testing of EEG-
    and MEG-data. Journal of Neuroscience Methods 164(1):177-190.
"""

import matplotlib.pyplot as plt

import numpy as np
from numpy import arange
from numpy.random import normal, permutation

from scipy.stats import ttest_ind

import seaborn as sns

def _cluster_by_adjacency(sel_samples):
    """Function for clustering selected samples based on temporal adjacency.

    Input arguments:
    sel_samples - A vector of booleans indicating which samples have been
                  selected.

    Output arguments:
    clusters    - A vector of cluster numbers indicating to which cluster
                  each sample belongs to. The cluster number zero corresponds
                  to samples that were not selected.
    """
    clusters = np.zeros(len(sel_samples), dtype='int')
    j = 1 # Next cluster number.
    for i, s in enumerate(sel_samples):
        if (s == True):
            clusters[i] = j
        else:
            # Update the cluster number at temporal discontinuities.
            if (i > 0 and sel_samples[i-1] == True):
                j += 1
    return clusters

def _cluster_stat(tstat, clusters):
    """Function for computing the cluster mass statistic."""
    n_clusters = np.max(clusters)
    if (n_clusters == 0):
        return np.nan
    else:
        cstat = [np.sum(tstat[clusters == i]) for i in
                                              arange(1, n_clusters+1)]
        return np.max(cstat)

def permutation_test(X, Y):
    """Find the number of samples and variables in the two sets of data."""
    (n_samples_x, n_vars_x), (n_samples_y, n_vars_y) = np.shape(X), np.shape(Y)
    n_total = n_samples_x + n_samples_y

    threshold = 1
    n_permutations = 1000

    ref_tstat, _ = ttest_ind(X, Y)
    ref_cstat = _cluster_stat(ref_tstat,
                              _cluster_by_adjacency(ref_tstat > threshold))

    """Combine the two sets of data."""
    Z = np.vstack([X, Y])
    cmax = np.zeros(n_permutations)

    for p in arange(0, n_permutations):
        """Do the random partition."""
        permuted_ind = permutation(arange(n_total))
        T, U = Z[permuted_ind[0:n_samples_x], :], Z[permuted_ind[n_samples_x:], :]

        tstat, _ = ttest_ind(T, U)
        sel = tstat > threshold
        clusters = _cluster_by_adjacency(sel)

        """Compute cluster-level statistics."""
        cmax[p] = _cluster_stat(tstat, clusters)

    cmax = cmax[~np.isnan(cmax)]
    pval = np.sum(cmax > ref_cstat) / n_permutations

"""Generate some test data."""
X = normal(loc=2, scale=1, size=(20, 300))
Y = normal(loc=0, scale=1, size=(20, 300))

permutation_test(X, Y)
