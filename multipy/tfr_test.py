# -*- encoding: utf8 -*-
"""Script for testing Efron's separate classes model using parameters that
one could have in a typical time-frequency M/EEG data analysis situation.

Author: Tuomas Puoliväli
Email: tuomas.puolivali@helsinki.fi
Last modified: 30th July 2018
License: Revised 3-clause BSD
Source: https://github.com/puolival/multipy/blob/master/tfr_test.py
"""

from data import two_group_model
from fdr import lsu

import matplotlib.pyplot as plt

import numpy as np

from scipy.stats import norm

import seaborn as sns

fs = 200
n_iter = 1000

A_pi0, B_pi0 = 0.75, 0.9
A_delta, B_delta = 1.2, 0.7
A_m, B_m = 3*fs, 8*fs

power_all, power_separate = [], []

for i in np.arange(0, n_iter):
    """Draw random numbers from two different mixture distributions."""
    _, A_pvals = two_group_model(N=25, m=A_m, pi0=A_pi0, delta=A_delta)
    _, B_pvals = two_group_model(N=25, m=B_m, pi0=B_pi0, delta=B_delta)

    """Combine the separate classes and perform one FDR."""
    all_pvals = np.hstack([A_pvals, B_pvals])
    significant = lsu(all_pvals)

    """Compute power of the procedure, i.e. Pr(reject H0 | H1 is true)."""
    A_rejections = np.sum(significant[int(A_m*A_pi0):A_m])
    B_rejections = np.sum(significant[A_m+int(B_m*B_pi0):B_m])

    power_all.append((A_rejections+B_rejections) /
                    ((1-A_pi0)*A_m + (1-B_pi0)*B_m))

    """Apply FDR separately to the two parts and perform a similar
    power calculation."""
    A_significant, B_significant = lsu(A_pvals), lsu(B_pvals)

    A_rejections = np.sum(A_significant[int(A_m*A_pi0):A_m])
    B_rejections = np.sum(B_significant[int(B_m*B_pi0):B_m])

    power_separate.append((A_rejections+B_rejections) /
                          ((1-A_pi0)*A_m + (1-B_pi0)*B_m))

    print('iteration %4d' % i)

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

"""Plot densities."""
x = np.linspace(-3, 4, 100)

p_a0 = norm.pdf(x, loc=0.0, scale=1)
p_a1 = norm.pdf(x, loc=A_delta, scale=1)
p_b0 = norm.pdf(x, loc=0.0, scale=1)
p_b1 = norm.pdf(x, loc=B_delta, scale=1)

sns.set_style('darkgrid')
fig = plt.figure(figsize=(3, 8))

ax = fig.add_subplot(111)

ax.plot(x, 0.75+p_a0, '-')
ax.plot(x, 1.50+p_a1, '-')
ax.plot(x, 2.25+p_b0, '-')
ax.plot(x, 3.00+p_b1, '-')

fig.tight_layout()
plt.show()

