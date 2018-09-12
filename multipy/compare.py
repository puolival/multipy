# -*- encoding: utf-8 -*-
"""Compare different correction techniques using data generated
according to the Bennett et al. model.

Author: Tuomas Puoliväli
Email: tuomas.puolivali@helsinki.fi
Last modified: 11th September 2018
License: Revised 3-clause BSD
"""

from data import square_grid_model

from fdr import lsu, qvalue
from fwer import sidak

from viz import plot_grid_model

"""Generate the test data."""
nl, sl = 100, 60
N, delta = 25, 0.5
X = square_grid_model(nl, sl, N, delta)
alpha = 0.05

"""Apply each correction technique to the generated dataset."""
Y_sidak = sidak(X.flatten(), alpha=alpha)
Y_sidak = Y_sidak.reshape(nl, nl)

Y_fdr = lsu(X.flatten(), q=alpha)
Y_fdr = Y_fdr.reshape(nl, nl)

Y_qvalue, _ = qvalue(X.flatten(), threshold=alpha)
Y_qvalue = Y_qvalue.reshape(nl, nl)

"""Visualize the results."""
fig_nocor = plot_grid_model(X<alpha, nl, sl)
fig_nocor.axes[0].set_title('Uncorrected')

fig_sidak = plot_grid_model(Y_sidak, nl, sl)
fig_sidak.axes[0].set_title('Sidak')

fig_fdr = plot_grid_model(Y_fdr, nl, sl)
fig_fdr.axes[0].set_title('FDR')

fig_qvalue = plot_grid_model(Y_qvalue, nl, sl)
fig_qvalue.axes[0].set_title('Q-value')

import matplotlib.pyplot as plt
plt.show()
