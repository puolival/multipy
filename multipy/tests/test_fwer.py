# -*- coding: utf-8 -*-
"""Functions for testing the procedures for controlling the family-wise
error rate (FWER).

Last modified: 5th November 2017
Author: Tuomas Puoliväli
Email: tuomas.puolivali@helsinki.fi
License: Revised 3-clause BSD
"""

import sys
sys.path.append('../')

import numpy as np

import unittest

from data import neuhaus
from fwer import bonferroni, holm_bonferroni, sidak

class TestFWER(unittest.TestCase):

    def test_bonferroni_neuhaus(self):
        """Test the Bonferroni procedure using the Neuhaus et al. data."""
        pvals = neuhaus()
        significant = bonferroni(pvals, alpha=0.05)
        """The first three tests should be significant after the
        correction."""
        for i in range(0, 3):
            self.assertTrue(significant[i])
        """The other tests should not be significant after the
        correction."""
        for i in range(3, 15):
            self.assertFalse(significant[i])

    def test_bonferroni_nonsig(self):
        """Test the Bonferroni procedure using p-values in the range
        [0.05, 1)."""
        pvals = np.asarray([0.05, 0.3, 0.5, 1.0, 0.05])
        significant = bonferroni(pvals, alpha=0.05)
        for i in range(0, len(pvals)):
            self.assertFalse(significant[i])

    def test_sidak_neuhaus(self):
        """The Sidak's procedure using data from Neuhaus et al."""
        pvals = neuhaus()
        significant = sidak(pvals)
        for i in range(0, 3):
            self.assertTrue(significant[i])
        for i in range(3, 15):
            self.assertFalse(significant[i])

    def test_sidak_nonsig(self):
        pass

    def test_holm_bonferroni_neuhaus(self):
        pass

    def test_holm_bonferroni_nonsig(self):
        pass

if __name__ == '__main__':
    unittest.main()

