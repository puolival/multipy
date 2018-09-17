# -*- encoding: utf-8 -*-
"""Compare different correction techniques using data generated
according to the Bennett et al. model.

Author: Tuomas Puoliväli
Email: tuomas.puolivali@helsinki.fi
Last modified: 11th September 2018
License: Revised 3-clause BSD

References:

[1] Bennett CM, Wolford GL, Miller MB (2009): The principled control of
    false positives in neuroimaging. Social Cognitive and Affective
    Neuroscience 4(4):417-422.
"""

from data import square_grid_model

from fdr import lsu, qvalue, tst
from fwer import sidak, holm_bonferroni, hochberg
from rft import rft_2d
from permutation import tfr_permutation_test

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

Y_rft, _, _ = rft_2d(X_tstats, fwhm=30, alpha=alpha, verbose=True)
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
fig_nocor.axes[0].set_title('Uncorrected')

fig_sidak = plot_grid_model(Y_sidak, nl, sl)
fig_sidak.axes[0].set_title('Sidak')

fig_fdr = plot_grid_model(Y_fdr, nl, sl)
fig_fdr.axes[0].set_title('FDR')

fig_qvalue = plot_grid_model(Y_qvalue, nl, sl)
fig_qvalue.axes[0].set_title('Q-value')

fig_rft = plot_grid_model(Y_rft, nl, sl)
fig_rft.axes[0].set_title('RFT')

fig_tst = plot_grid_model(Y_tst, nl, sl)
fig_tst.axes[0].set_title('Two-stage procedure')

fig_permutation = plot_grid_model(Y_permutation, nl, sl)
fig_permutation.axes[0].set_title('Permutation')

fig_holm = plot_grid_model(Y_holm, nl, sl)
fig_holm.axes[0].set_title('Holm-Bonferroni')

fig_hochberg = plot_grid_model(Y_hochberg, nl, sl)
fig_hochberg.axes[0].set_title('Hochberg')

import matplotlib.pyplot as plt
plt.show()
