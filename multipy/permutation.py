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
    """Function for computing the cluster mass statistic.

    Input arguments:
    tstat    - T-statistic for each variable.
    clusters - A vector of cluster numbers grouping the t-statistics as
               returned by the function _cluster_by_adjacency.

    Output arguments:
    cstat    - The cluster mass statistic.
    """
    n_clusters = np.max(clusters)
    if (n_clusters == 0):
        return np.nan
    else:
        cstat = [np.sum(tstat[clusters == i]) for i in
                                              arange(1, n_clusters+1)]
        return np.max(cstat)

def permutation_test(X, Y, n_permutations=1000, threshold=1, tail='both'):
    """Permutation test for a significant difference between two sets of data.

    Input arguments:
    X              - An N_1-by-m matrix (NumPy ndarray) where N is the number
                     of samples and m is the number of variables. The first
                     group of data.
    Y              - An N_2-by-m matrix. The second group of data.
    n_permutations - Number of permutations. Consider how accurate p-value
                     is needed.
    threshold      -
    tail           - Only two-sided tests are supported for the moment (i.e.
                     tail = 'both').

    Output arguments:
    pval           - Permutation test p-value.
    """

    """Find the number of samples in each of the two sets of data."""
    (n_samples_x, _), (n_samples_y, _) = np.shape(X), np.shape(Y)
    n_total = n_samples_x + n_samples_y

    """Do an independent two-sample t-test for each variable using the
    original (unpermuted) data. Compute the cluster mass statistic that is
    compared to the permutation distribution."""
    ref_tstat, _ = ttest_ind(X, Y)
    ref_clusters = _cluster_by_adjacency(ref_tstat > threshold)
    ref_cstat = _cluster_stat(ref_tstat, ref_clusters)

    """Combine the two sets of data."""
    Z = np.vstack([X, Y])

    """Do the permutations."""
    cstat = np.zeros(n_permutations)

    for p in arange(0, n_permutations):
        """Do a random partition of the data."""
        permuted_ind = permutation(arange(n_total))
        T, U = Z[permuted_ind[0:n_samples_x], :], Z[permuted_ind[n_samples_x:], :]

        tstat, _ = ttest_ind(T, U)
        sel = tstat > threshold
        clusters = _cluster_by_adjacency(sel)

        """Compute cluster-level statistics."""
        cstat[p] = _cluster_stat(tstat, clusters)

    cstat = cstat[~np.isnan(cstat)]
    pval = np.sum(cstat > ref_cstat) / n_permutations

    return pval, cstat, ref_cstat

"""Generate some test data."""
X = normal(loc=2, scale=1, size=(20, 300))
Y = normal(loc=0, scale=1, size=(20, 300))

pval, cstat, ref_cstat = permutation_test(X, Y)
