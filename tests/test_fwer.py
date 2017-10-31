import sys
sys.path.append('../')

import unittest

from data import neuhaus
from fwer import bonferroni

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

if __name__ == '__main__':
    unittest.main()

