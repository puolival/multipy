# -*- coding: utf-8 -*-
"""An example comparing the Bonferroni and FDR correction techniques

Author: Tuomas Puoliväli
Email: tuomas.puolivali@helsinki.fi
Last modified 30th October 2017.
"""

from adaptive import lsu
from data import neuhaus
from fwer import bonferroni

pvals = neuhaus()

print zip(pvals, bonferroni(pvals)) # 3 significant p-values
print zip(pvals, lsu(pvals)) # 4 significant p-values
