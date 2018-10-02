# -*- coding: utf-8 -*-
"""Script for visualizing the significance decision line in the
Benjamini-Hochberg procedure.

Author: Tuomas Puoliväli
Email: tuomas.puolivali@helsinki.fi
Last modified 12th February 2018
Source: https://github.com/puolival/multipy
"""

# Allow importing modules from parent directory.
import sys
sys.path.append('..')

from data import neuhaus

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

alpha = 0.05 # The chosen critical level.

pvals = neuhaus(permute=False)
n_pvals = len(pvals)
k = np.linspace(1, n_pvals, n_pvals)
sel = np.arange(0, 8)

"""Plot the data."""
sns.set_style('darkgrid')
fig = plt.figure(figsize=(8, 6))

ax = fig.add_subplot(111)
ax.plot(k[sel], pvals[sel], 'o-')
y = (alpha/n_pvals)*k + 0 # Line through the origin.
ax.plot(k[sel], y[sel], '-')
ax.legend(['P-value', 'Decision line'], loc='upper left')
ax.set_xlabel('Hypothesis')
ax.set_ylabel('P-value')
ax.set_title('Benjamini-Hochberg procedure')
ax.set_ylim([-0.01, np.max(pvals[sel])+0.01])
ax.set_xlim([0.5, np.max(sel)+1.5])

fig.tight_layout()
plt.show()
