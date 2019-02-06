# -*- encoding: utf-8 -*-
"""Script for simulating experiments at several different effect sizes
under the spatial two-group model.

Author: Tuomas Puoliväli
Email: tuomas.puolivali@helsinki.fi
Last modified: 5th February 2019
License: Revised 3-clause BSD
"""

# Allow importing modules from parent directory.
import sys
sys.path.append('..')

from data import square_grid_model as two_group_model

import numpy as np

"""Define settings for the simulated experiments."""
nl, sl = 90, 30
N = 25
effect_sizes = np.linspace(0.5, 1.5, 21) # 0.05 increments
n_effect_sizes = len(effect_sizes)
n_iterations = 20

"""Simulate the primary experiments."""
pvals = np.zeros([n_iterations, n_effect_sizes, nl, nl])
tstats = np.zeros(np.shape(pvals))

rvs_a = np.zeros([n_iterations, n_effect_sizes, nl, nl, N])
rvs_b = np.zeros(np.shape(rvs_a))

for i in np.arange(0, n_iterations):
    print('Simulating experiment: iteration %3d of %3d' %
          (i+1, n_iterations))
    for j, delta in enumerate(effect_sizes):
        data = two_group_model(nl=nl, sl=sl, N=N, delta=delta)
        pvals[i, j], tstats[i, j] = data[0], data[1]
        rvs_a[i, j], rvs_b[i, j] = data[2], data[3]

"""Save the simulated data to disk."""
fpath_output = '/home/puolival/multipy_data'
fname_output = fpath_output + '/primary.npy'

print('Saving data to %s' % fname_output)
np.save(fname_output, {'rvs_a': rvs_a, 'rvs_b': rvs_b, 'pvals': pvals,
                       'tstats': tstats, 'effect_sizes': effect_sizes})
print('Done.')

