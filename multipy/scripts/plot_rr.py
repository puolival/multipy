# -*- encoding: utf-8 -*-
"""Script for visualizing reproducibility rates at different effect sizes
under the spatial two-group model.

Author: Tuomas Puoliväli
Email: tuomas.puolivali@helsinki.fi
Last modified: 8th February 2019
License: Revised 3-clause BSD
"""

from glob import glob

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

from os import walk
from os.path import isfile, join

"""Read results from disk."""
fpath = '/home/puolival/multipy_data'
fname_template = 'result-*.npy'
fnames = [ff for f in walk(fpath) for ff in
          glob(join(f[0], fname_template))]

data = []
for i, fname in enumerate(fnames):
    data.append(np.load(fname).flat[0]['rr'])
data = np.asarray(data)

n_methods, n_iterations, n_effect_sizes = np.shape(data)

"""Visualize the results."""

sns.set_style('darkgrid')
fig = plt.figure(figsize=(8, 7))

ax = fig.add_subplot(111)
for i in np.arange(0, n_methods):
    ax.errorbar(x=np.arange(0, 21), y=np.mean(data[i], axis=0),
                yerr=np.std(data[i], axis=0))

fig.tight_layout()
plt.show()
