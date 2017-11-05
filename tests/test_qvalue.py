# -*- coding: utf-8 -*-
"""Unit tests for the q-value procedure suggested by Storey and
Tibshirani.

Author: Tuomas Puoliväli
Email: tuomas.puolivali@helsinki.fi
Last modified: 5th November 2017.
License: Revised 3-clause BSD.
"""

import sys
sys.path.append('../')

import numpy as np

import unittest

from data import neuhaus
from adaptive import lsu, tst, orc, abh

class TestQvalue(unittest.TestCase):

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

