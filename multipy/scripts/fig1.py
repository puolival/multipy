import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

n_tests = np.arange(1, 40)

"""Compute Pr(at least one false positive) in m tests."""
alpha = 0.05
pr_1fp = 1 - (1 - alpha) ** n_tests

"""Plot the data."""
fig = plt.figure(figsize=(6, 4))
fig.subplots_adjust(top=0.92, bottom=0.13, left=0.12, right=0.95)

ax = fig.add_subplot(111)
ax.plot(n_tests, pr_1fp)
ax.set_xlabel('Number of comparisons')
ax.set_ylabel('$\mathbb{P}($at least one false positive$)$')
ax.set_ylim([0, 1])
ax.set_xlim([1, n_tests[-1]])

plt.show()
