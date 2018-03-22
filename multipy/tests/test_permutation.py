# -*- coding: utf-8 -*-
"""Functions for testing permutation test routines.

Last modified: 22th March 2018.
Author: Tuomas Puoliväli
Email: tuomas.puolivali@helsinki.fi
License: Revised 3-clause BSD
"""

import sys
sys.path.append('../')

import numpy as np

import unittest

from permutation import _cluster_by_adjacency

class TestPermutation(unittest.TestCase):

    def test_cluster_by_adjacency_one_cluster(self):
        """Test clustering by adjacency with only one cluster."""
        x = np.array([1, 1, 1, 1, 1])
        clusters = _cluster_by_adjacency(x)
        self.assertTrue(np.all(clusters == 1))

    def test_cluster_by_adjacency_no_clusters(self):
        """Test clustering by adjacency with no clusters."""
        x = np.zeros([5, 1])
        clusters = _cluster_by_adjacency(x)
        self.assertTrue(np.all(clusters == 0))

    def test_cluster_by_adjacency_general(self):
        """Test clustering by adjacency for a simple general test case."""
        x = np.array([1, 1, 1, 0, 0, 1, 1, 0, 1])
        clusters = _cluster_by_adjacency(x)
        # Check that the numbering is correct for each cluster.
        self.assertTrue(np.all(clusters[0:3] == 1))
        self.assertTrue(np.all(clusters[5:7] == 2))
        self.assertTrue(clusters[8] == 3)

if __name__ == '__main__':
    unittest.main()

