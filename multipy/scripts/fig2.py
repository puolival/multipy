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

pvals = neuhaus()
n_pvals = len(pvals)
k = np.linspace(1, n_pvals, n_pvals)

"""Plot the data."""
fig = plt.figure(figsize=(6, 4))
fig.subplots_adjust(top=0.92, bottom=0.13, left=0.12, right=0.95)

ax = fig.add_subplot(111)
ax.plot(k, pvals, 'o-')
y = (alpha/n_pvals)*k + 0 # Line through the origin.
ax.plot(k, y, '-')
ax.legend(['P-value', 'Decision line'], loc='upper left')
ax.set_xlabel('Hypothesis')
ax.set_ylabel('P-value')
ax.set_title('Benjamini-Hochberg procedure')
ax.set_ylim([-0.05, 1.05])

plt.show()
