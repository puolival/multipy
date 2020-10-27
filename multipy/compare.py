# -*- coding: utf-8 -*-
"""Compare different correction techniques using data generated
according to the Bennett et al. model.

NOTES:

[1] For the random field theory (RFT) approach, we have smoothed the data
at FWHM = 30. This value corresponds to an estimate of 9 resels, which
matches the underlying parameters of the simulation. Therefore, the
results may be optimistic compared to scenarios where no such information
is available. An expanded simulation could include different ways of
estimating the FWHM parameter from the data.

Author: Tuomas Puoliväli
Email: tuomas.puolivali@helsinki.fi
Last modified: 11th September 2018
License: Revised 3-clause BSD

TODO: The process of applying each method and visualizing the results could
be replaced with a more concise iterative structure.

References:

[1] Bennett CM, Wolford GL, Miller MB (2009): The principled control of
    false positives in neuroimaging. Social Cognitive and Affective
    Neuroscience 4(4):417-422.
"""

import numpy as np

from fdr import lsu, qvalue, tst
from fwer import sidak, holm_bonferroni, hochberg
from data import square_grid_model
from permutation import tfr_permutation_test
from rft import rft_2d
from util import grid_model_counts, roc
from viz import plot_grid_model

"""Generate the test data."""
nl, sl = 90, 30
N, delta = 25, 0.7
X, X_tstats, X_raw, Y_raw = square_grid_model(nl, sl, N, delta,
                                              equal_var=True)
alpha = 0.05

"""Apply each correction technique to the generated dataset."""
Y_sidak = sidak(X.flatten(), alpha=alpha)
Y_sidak = Y_sidak.reshape(nl, nl)

Y_fdr = lsu(X.flatten(), q=alpha)
Y_fdr = Y_fdr.reshape(nl, nl)

Y_qvalue, _ = qvalue(X.flatten(), threshold=alpha)
Y_qvalue = Y_qvalue.reshape(nl, nl)

Y_rft, Y_smooth, _ = rft_2d(X_tstats, fwhm=30, alpha=alpha, verbose=True)
# No reshape needed since already in correct form.

Y_tst = tst(X.flatten(), q=alpha)
Y_tst = Y_tst.reshape(nl, nl)

Y_permutation = tfr_permutation_test(X_raw, Y_raw, n_permutations=100,
                                     alpha=alpha, threshold=1)

Y_holm = holm_bonferroni(X.flatten(), alpha=alpha)
Y_holm = Y_holm.reshape(nl, nl)

Y_hochberg = hochberg(X.flatten(), alpha=alpha)
Y_hochberg = Y_hochberg.reshape(nl, nl)

"""Visualize the results."""
fig_nocor = plot_grid_model(X<alpha, nl, sl)
t_nocor = 'Uncorrected %1.3f %1.3f' % roc(grid_model_counts(X<alpha, nl, sl))
fig_nocor.axes[0].set_title(t_nocor)

fig_sidak = plot_grid_model(Y_sidak, nl, sl)
t_sidak = 'Sidak %1.3f %1.3f' % roc(grid_model_counts(Y_sidak, nl, sl))
fig_sidak.axes[0].set_title(t_sidak)

fig_fdr = plot_grid_model(Y_fdr, nl, sl)
t_fdr = 'FDR %1.3f %1.3f' % roc(grid_model_counts(Y_fdr, nl, sl))
fig_fdr.axes[0].set_title(t_fdr)

fig_qvalue = plot_grid_model(Y_qvalue, nl, sl)
t_qvalue = 'Q-value %1.3f %1.3f' % roc(grid_model_counts(Y_qvalue, nl, sl))
fig_qvalue.axes[0].set_title(t_qvalue)

fig_rft = plot_grid_model(Y_rft, nl, sl)
t_rft = 'RFT %1.3f %1.3f' % roc(grid_model_counts(Y_rft, nl, sl))
fig_rft.axes[0].set_title(t_rft)

fig_tst = plot_grid_model(Y_tst, nl, sl)
t_tst = 'Two-stage procedure %1.3f %1.3f' % roc(grid_model_counts(Y_tst, nl, sl))
fig_tst.axes[0].set_title(t_tst)

fig_permutation = plot_grid_model(Y_permutation, nl, sl)
t_permutation = 'Permutation %1.3f %1.3f' % roc(grid_model_counts(Y_permutation, nl, sl))
fig_permutation.axes[0].set_title(t_permutation)

fig_holm = plot_grid_model(Y_holm, nl, sl)
t_holm = 'Holm-Bonferroni %1.3f %1.3f' % roc(grid_model_counts(Y_holm, nl, sl))
fig_holm.axes[0].set_title(t_holm)

fig_hochberg = plot_grid_model(Y_hochberg, nl, sl)
t_hochberg = 'Hochberg %1.3f %1.3f' % roc(grid_model_counts(Y_hochberg, nl, sl))
fig_hochberg.axes[0].set_title(t_hochberg)

import matplotlib.pyplot as plt
plt.show()

"""Save the data for later processing and checks."""

"""Save the t-statistics, random variates, and p-values to make it
possible to reproduce the results."""

output_path = '/home/local/puolival/multipy-material/simulation-data'
output_fname_x = output_path + '/x.npy'
output_fname_x_raw = output_path + '/x_raw.npy'
output_fname_y_raw = output_path + '/y_raw.npy'
output_fname_x_tstats = output_path + '/x_tstats.npy'

np.save(output_fname_x, X)
np.save(output_fname_x_raw, X_raw)
np.save(output_fname_y_raw, Y_raw)
np.save(output_fname_x_tstats, X_tstats)

"""Save also the simulation parameters."""
output_fname_params = output_path + '/params.npy'

np.save(output_fname_params, {'nl': nl, 'sl': sl,
                              'N': N, 'delta': delta})
