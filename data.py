# -*- coding: utf-8 -*-
"""Functions for generating test data.

This program code is part of the MultiPy (Multiple Hypothesis Testing in
Python) package.

Author: Tuomas Puoliväli
Email: tuomas.puolivali@helsinki.fi
License: Revised 3-clause BSD
Source: https://github.com/puolival/multipy/blob/master/data.py
Last modified: 29th December 2017.

References:

[1] Neuhaus KL, von Essen R, Tebbe U, Vogt A, Roth M, Riess M, Niederer W,
    Forycki F, Wirtzfeld A, Maeurer W, Limbourg P, Merx W, Haerten K (1992):
    Improved thrombolysis in acute myocardial infarction with front-loaded
    administration of alteplase: Results of the rt-PA-APSAC patency study
    (TAPS). Journal of the American College of Cardiology 19(5):385-391.

[2] Benjamini Y, Hochberg Y (1995): Controlling the false discovery rate:
    A practical and powerful approach to multiple testing. Journal of Royal
    Statistical Society. Series B (Methodological): 57(1):289-300.
"""

import numpy as np
from numpy.random import permutation

def neuhaus(permute=False):
    """Function that returns the Neuhaus et al. data that was re-analyzed in
    the classic Benjamini & Hochberg (1995) FDR paper.

    Input arguments:
    permute - If true, the p-values are returned in random order. If false,
              the p-values are returned in ascending order.
    """
    pvals = np.array([0.0001, 0.0004, 0.0019, 0.0095, 0.0201, 0.0278, 0.0298,
                      0.0344, 0.0459, 0.3240, 0.4262, 0.5719, 0.6528, 0.7590,
                      1.000], dtype='float')
    if (permute):
        m = len(pvals)
        pvals = pvals[permutation(m)]
    return pvals

