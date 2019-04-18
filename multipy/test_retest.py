# -*- encoding: utf-8 -*-
"""Evaluate test-retest reliability as a function of effect size in the
spatial two-group model.

Author: Tuomas Puoliväli
Email: tuomas.puolivali@helsinki.fi
Last modified: 20th November 2018
License: Revised 3-clause BSD
"""

import matplotlib.pyplot as plt

import numpy as np

from fdr import lsu
from data import square_grid_model

from scipy.stats import norm

import seaborn as sns

"""The simulation settings."""
nl, sl = 90, 30
N = 25
alpha = 0.05
d = (nl-sl) // 2
n_iterations = 20
deltas = np.linspace(0.5, 1.5, 20)
n_deltas = len(deltas)

"""Estimate reproducibility and true positive rate in simulated test-retest
experiments using the two-group model."""
n_reproducible = np.zeros([n_iterations, n_deltas])
n_tp = np.zeros([2, n_iterations, n_deltas]) # test and retest conditions

for i, delta in enumerate(deltas):
    print('Estimating reproducibility at effect size d=%1.3f' % delta)
    for j in np.arange(0, n_iterations):
        """Generate test and retest data."""
        X_test = square_grid_model(nl, sl, N, delta, equal_var=True)[0]
        X_retest = square_grid_model(nl, sl, N, delta, equal_var=True)[0]

        """Correct the statistics for multiple testing."""
        Y_test = lsu(X_test.flatten(), alpha)
        Y_test = Y_test.reshape(nl, nl)

        Y_retest = lsu(X_retest.flatten(), alpha)
        Y_retest = Y_retest.reshape(nl, nl)

        """Count how many discoveries were reproducible."""
        n_reproducible[j, i] = np.sum(Y_retest[d:(nl-d), d:(nl-d)]
                                      [Y_test[d:(nl-d), d:(nl-d)]])

        """Count the number of true positives in test and retest sets."""
        n_tp[0, j, i] = np.sum(Y_test[d:(nl-d), d:(nl-d)])
        n_tp[1, j, i] = np.sum(Y_retest[d:(nl-d), d:(nl-d)])

"""Visualize the results."""
z = norm.ppf(0.995) # 99% confidence interval

sns.set_style('darkgrid')
fig = plt.figure(figsize=(8, 7))

ax = fig.add_subplot(111)

"""Compute TPR confidence intervals."""
yerr_tp1 = (z * np.std(n_tp[0, :] / (sl ** 2), axis=0) /
             np.sqrt(n_iterations))
yerr_tp2 = (z * np.std(n_tp[1, :] / (sl ** 2), axis=0) /
             np.sqrt(n_iterations))

"""Plot the TPR and reproducibility data."""
ax.errorbar(x=deltas, y=np.mean(n_tp[0, :], axis=0) / (sl ** 2),
            yerr=yerr_tp1)
ax.errorbar(x=deltas, y=np.mean(n_tp[1, :], axis=0) / (sl ** 2),
            yerr=yerr_tp2)
ax.errorbar(x=deltas, y=np.mean(n_reproducible, axis=0) / (sl ** 2),
            yerr=z * np.std(n_reproducible / (sl ** 2), axis=0) /
            np.sqrt(n_iterations))

"""Label the axes etc."""
ax.set_xlabel("Effect size (Cohen's d)")
ax.set_ylabel('True positive or reproducibility rate')
ax.set_xlim([deltas[0]-0.05, deltas[-1]+0.05])
ax.set_ylim([-0.05, 1.05])
ax.legend(['Test', 'Retest', 'Reproducibility'], loc=4)

fig.tight_layout()
plt.show()
