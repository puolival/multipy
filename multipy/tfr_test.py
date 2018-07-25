""".

Author: Tuomas Puoliväli
Email: tuomas.puolivali@helsinki.fi
Last modified: 25th July 2018
License: Revised 3-clause BSD
Source: https://github.com/puolival/multipy/blob/master/tfr_test.py
"""

from data import two_group_model
from fdr import lsu

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

fs = 200
n_iter = 1000

power_all, power_separate = [], []

for i in np.arange(0, n_iter):
    """Generate new simulated data."""
    m_a, m_g = int(5*fs), int(10*fs)
    alpha_tstats, alpha_pvals = two_group_model(N=25, m=m_a, pi0=0.7, delta=1.0)
    gamma_tstats, gamma_pvals = two_group_model(N=25, m=m_g, pi0=0.9, delta=0.7)

    """Combine the two datasets."""
    tstats = np.hstack([alpha_tstats, gamma_tstats])
    pvals = np.hstack([alpha_pvals, alpha_pvals])

    """Apply FDR on the combined dataset."""
    significant = lsu(pvals)
    power_all.append(
        (np.sum(significant[int(m_a*0.7):m_a]) +
         np.sum(significant[m_a+int(m_g*0.9):m_g])) / (m_a*0.3+m_g*0.1)
    )

    """Apply FDR separately to the two parts."""
    alpha_significant = lsu(alpha_pvals)
    gamma_significant = lsu(gamma_pvals)

    power_separate.append(
        (np.sum(alpha_significant[int(m_a*0.7):m_a]) +
         np.sum(gamma_significant[int(m_g*0.9):m_g])) /
        np.sum(m_a*0.3+m_g*0.1)
    )

    print('iteration %d' % i)

"""Visualize the results."""
sns.set_style('darkgrid')
fig = plt.figure(figsize=(8, 6))

ax = fig.add_subplot(111)
sns.distplot(power_all, ax=ax)
sns.distplot(power_separate, ax=ax)

ax.set_xlabel('Power')
ax.set_ylabel('Density')

fig.tight_layout()
plt.show()
