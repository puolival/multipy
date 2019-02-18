# -*- encoding: utf-8 -*-
"""Script for analyzing data from the simulated primary and follow-up
experiments."""

# Allow importing modules from parent directory.
import sys
sys.path.append('..')

from fdr import lsu, tst, qvalue
from fwer import bonferroni, sidak, hochberg, holm_bonferroni
from permutation import tfr_permutation_test

import numpy as np

from repeat import fwer_replicability as repl

from util import grid_model_counts as counts

"""Load the simulated datasets."""
fpath = '/home/puolival/multipy_data'
fname_primary = fpath + '/primary.npy'
fname_followup = fname_primary.replace('primary', 'follow-up')

print('Loading simulated datasets ..')
primary_data, followup_data = (np.load(fname_primary),
                               np.load(fname_followup))
print('Done.')

# Extract p-values
pvals_pri, pvals_fol = (primary_data.flat[0]['pvals'],
                        followup_data.flat[0]['pvals'])

# Extract raw data for permutation testing
rvs_a_pri, rvs_b_pri = (primary_data.flat[0]['rvs_a'],
                        primary_data.flat[0]['rvs_b'])

rvs_a_fol, rvs_b_fol = (followup_data.flat[0]['rvs_a'],
                        followup_data.flat[0]['rvs_b'])

"""Define analysis parameters."""
n_iterations, n_effect_sizes, nl, _ = np.shape(pvals_pri)
emph_primary = 0.1
alpha = 0.05
method = qvalue
sl = 30 # TODO: save to .npy file.

"""Compute reproducibility rates."""
rr = np.zeros([n_iterations, n_effect_sizes])

for ind in np.ndindex(n_iterations, n_effect_sizes):
    print('Analysis iteration %3d' % (1+ind[0]))
    replicable = repl(pvals_pri[ind].flatten(), pvals_fol[ind].flatten(),
                      emph_primary, method, alpha)
    replicable = np.reshape(replicable, [nl, nl])
    rr[ind] = counts(replicable, nl, sl)[0]

"""Save data to disk."""
output_fpath = fpath
output_fname = output_fpath + ('/result-%s.npy' % method.__name__)

np.save(output_fname, {'rr': rr})
print('Results saved to disk.')
