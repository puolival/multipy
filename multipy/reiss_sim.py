# -*- coding: utf-8 -*-
"""Script for reproducing the result in Figure 2. of Reiss et al. (2012)

Author: Tuomas Puoliväli
Email: tuomas.puolivali@helsinki.fi
Last modified: 7th November 2017

References:

[1] Reiss PT, Schwartzman A, Lu F, Huang L, Proal E (2012): Paradoxical
results of adaptive false discovery rate procedures in neuroimaging studies.
NeuroImage 63(4):1833-1840.
"""

from data import two_group_model

from fdr import lsu, tst

import matplotlib.pyplot as plt

import numpy as np
from numpy.random import normal

from scipy.stats import ttest_ind

import seaborn as sb

"""Reproduce the simulation in Figure 2. panel C."""
m = 100
n_iter = 99
delta = np.arange(0., 1.6, 0.1)
n_rejected_lsu, n_rejected_tst = (np.ndarray([len(delta), n_iter]),
                                  np.ndarray([len(delta), n_iter]))
n_rejected_nc = np.ndarray([len(delta), n_iter])
for i in range(0, n_iter):
    for j, d in enumerate(delta):
        _, pvals = two_group_model(N=20, m=m, pi0=0, delta=d)
        n_rejected_lsu[j, i] = np.sum(lsu(pvals))
        n_rejected_tst[j, i] = np.sum(tst(pvals))
        n_rejected_nc[j, i] = np.sum(pvals<0.05)
    print 'iteration %3d' % i

"""Visualize the results (i.e. the proportion of rejected null hypotheses
as a function of the effect size.)"""
fig = plt.figure(figsize=(6, 4))
fig.subplots_adjust(bottom=0.15)

ax = fig.add_subplot(111)
# lsu
ax.errorbar(x=delta, y=np.mean(n_rejected_lsu/float(m), axis=1),
            yerr=np.std(n_rejected_lsu/float(m), axis=1))
# tst
ax.errorbar(x=delta, y=np.mean(n_rejected_tst/float(m), axis=1),
            yerr=np.std(n_rejected_tst/float(m), axis=1))
# orig. pvals
ax.errorbar(x=delta, y=np.mean(n_rejected_nc/float(m), axis=1),
            yerr=np.std(n_rejected_nc/float(m), axis=1))
ax.legend(['fdr', 'tst', 'uncorrected'])
ax.set_xlabel('Effect size $\delta$')
ax.set_ylabel('Proportion rejected')

plt.show()

"""Reproduce Figure 2. panel A in Reiss et al."""
pi = np.arange(0, 1.1, 0.1)

n_rejected_lsu, n_rejected_tst = (np.ndarray([len(pi), n_iter]),
                                  np.ndarray([len(pi), n_iter]))
for i in range(0, n_iter):
    for j, p in enumerate(pi):
        _, pvals = two_group_model(N=20, m=m, pi0=p, delta=0.7)
        n_rejected_lsu[j, i] = np.sum(lsu(pvals))
        n_rejected_tst[j, i] = np.sum(tst(pvals))
    print 'iteration %3d' % i

"""Visualize the result."""
fig = plt.figure()

ax = fig.add_subplot(111)
# lsu
ax.errorbar(x=pi, y=np.mean(n_rejected_lsu/float(m), axis=1),
            yerr=np.std(n_rejected_lsu/float(m), axis=1))
# tst
ax.errorbar(x=pi, y=np.mean(n_rejected_tst/float(m), axis=1),
            yerr=np.std(n_rejected_tst/float(m), axis=1))
ax.legend(['fdr', 'tst'])
ax.set_xlabel('Null proportion $\pi_{0}$')
ax.set_ylabel('Proportion rejected')

plt.show()

