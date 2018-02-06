import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

n_tests = np.arange(1, 200)

"""Compute Pr(at least one false positive) in m tests."""
alpha = np.asarray([0.05, 0.01, 0.001])
n_alpha = len(alpha)

pr_1fp = np.zeros([n_alpha, len(n_tests)])
for i, a in enumerate(alpha):
    pr_1fp[i, :] = 1 - (1 - a) ** n_tests

"""Plot the data."""
fig = plt.figure(figsize=(6, 4))
fig.subplots_adjust(top=0.92, bottom=0.13, left=0.12, right=0.95)

ax = fig.add_subplot(111)
ax.plot(n_tests, pr_1fp.T)
ax.set_xlabel('Number of tests')
ax.set_ylabel('$\mathbb{P}$ (at least one false positive$)$')
ax.set_ylim([0-0.04, 1+0.04])
ax.set_xlim([n_tests[0]-5, n_tests[-1]+5])
ax.legend(alpha)

plt.show()
