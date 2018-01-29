# -*- coding: utf-8 -*-
"""Functions for controlling the family-wise error rate (FWER).

This program code is part of the MultiPy (Multiple Hypothesis Testing in
Python) package.

Author: Tuomas Puoliväli (tuomas.puolivali@helsinki.fi)
Last modified: 29th January 2018.
License: Revised 3-clause BSD
Source: https://github.com/puolival/multipy/blob/master/util.py

WARNING: These functions have not been entirely validated yet.

"""

import numpy as np

def print_result(pvals, significant_pvals, sort_pvals=True, pval_digits=4):
    """Print a list of (p-value, is_significant) tuples showing which
    p-values are significant.

    Input arguments:
    pvals             - P-values corresponding to a family of hypotheses.
    significant_pvals - An array of flags indicating which p-values are
                        significant.
    sort_pvals        - Whether to sort the p-values before printing.
    pval_digits       - Number of printed digits after the decimal place.
    """
    if (sort_pvals):
        sort_ind = np.argsort(pvals)
        pvals, significant = pvals[sort_ind], significant_pvals[sort_ind]

    # Print output directly to console.
    format_str = '{:.' + str(pval_digits) + 'f}'
    print zip([format_str.format(p) for p in pvals], significant_pvals)
