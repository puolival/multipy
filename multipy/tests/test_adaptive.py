# -*- coding: utf-8 -*-
"""Functions for testing the adaptive false discovery rate
procedures

Last modified: 5th November 2017.
Author: Tuomas Puoliväli
Email: tuomas.puolivali@helsinki.fi
License: Revised 3-clause BSD
"""

import sys
sys.path.append('../')

import numpy as np

import unittest

from data import neuhaus
from adaptive import lsu, tst, orc, abh

class TestAdaptive(unittest.TestCase):

    def test_lsu_neuhaus(self):
        pass

    def test_tst_neuhaus(self):
        pass

    def test_orc_neuhaus(self):
        pass

    def test_abh_neuhaus(self):
        pass

if __name__ == '__main__':
    unittest.main()

