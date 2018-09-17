# -*- encoding: utf-8 -*-
"""Functions for controlling family-wise error rate (FWER) using random
field theory (RFT) techniques and drawing diagnostic plots.

Author: Tuomas Puoliväli
Email: tuomas.puolivali@helsinki.fi
Last modified: 26th July 2018
License: Revised 3-clause BSD
Source: https://github.com/puolival/multipy/blob/master/rft.py

References:

Brett M, Penny W, Kiebel S (2003): An introduction to random field theory.
URL: https://www.fil.ion.ucl.ac.uk/spm/doc/books/hbf2/pdfs/Ch14.pdf

Worsley KJ, Evans AC, Marrett S, Neelin P (1992): A three-dimensional
statistical analysis for CBF activation studies in human brain. Journal of
Cerebral Blood Flow and Metabolism 12:900-918.

WARNING: These functions have not been entirely validated yet.
"""

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

from skimage.filters import gaussian as gaussian_filter
from skimage.measure import label

def _n_resels(X, fwhm):
    """Function for estimating the number of resolution elements or resels.

    Input arguments:
    ================
    X : ndarray of floats
      Two-dimensional data array containing the analyzed data.
    fwhm : float
      Size of the smoothing kernel measured as full-width at half maximum
      (FWHM).
    """

    """Estimate the number of resolution elements. Here the idea is to simply
    compute how many FWHM sized blocsk are needed to cover the entire area of
    X. TODO: Replace with a better (proper?) method."""
    nx, ny = np.shape(X)
    R = float(nx * ny) / float(fwhm ** 2)
    return R

def _expected_ec_2d(R, Z):
    """Function for computing the expected value of the Euler characteristic
    of a Gaussian field.

    Input arguments:
    ================
    R : float
      The number of resels or resolution elements.
    Z : float
      The applied Z-score threshold.
    """
    EC = R * (4*np.log(2)) * (2*np.pi)**(-3/2.) * Z*np.exp(-1/2.*Z**2)
    return EC

def _threshold(X, Z):
    """Function for thresholding smoothed data. The data is z-scored before
    thresholding.

    Input arguments:
    ================
    X : ndarray of floats
      The thresholded array.
    Z : float
      The Z-score threshold.
    """
    Y = np.zeros(np.shape(X))
    # Z-score X
    X = X - np.mean(X)
    X = X / np.std(X)
    # Threshold
    Y[X > Z] = 1
    return Y

def _ec_2d(X):
    """Function for computing the empirical Euler characteristic of a given
    thresholded data array.

    Input arguments:
    ================
    Y : ndarray of floats
      The thresholded image. Ones correspond to activated regions.

    Output arguments:
    =================
    ec : float
      The empirical Euler characteristic.
    """
    # TODO: check for holes in the activated regions.
    _, ec = label(X, neighbors=4, background=0, return_num=True,
                  connectivity=1)
    return ec

def plot_ec(X, fwhm, Z_low=0, Z_high=5):
    """Plot expected and empirical Euler characteristics (ECs) for a range
    of different Z-scores.

    Input arguments:
    ================
    X : ndarray of floats
      The analyzed two-dimensional data.
    fwhm : float
      Width of the spatial smoothing kernel described as full-width at
      half maximum (FWHM).
    Z_low, z_hig : float, float
      The lowest and highest z-scores for which to compute the EC.
    """

    """Estimate number of resels."""
    R = _n_resels(X, fwhm)

    """Compute the expected Euler characteristic for Z-scores in the
    desired range."""
    Z = np.linspace(Z_low, Z_high, 100)
    expected_ec = np.asarray([_expected_ec_2d(R, z) for z in Z])

    """Compute empirical Euler characteristics for the same Z-scores."""
    empirical_ec = np.zeros(np.shape(Z))
    for i, z in enumerate(Z):
        Y = _smooth(X, fwhm)
        Yt = _threshold(Y, z)
        empirical_ec[i] = _ec_2d(Yt)

    """Plot the data."""
    sns.set_style('darkgrid')
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)

    ax.plot(Z, expected_ec, linewidth=2)
    ax.plot(Z, empirical_ec, linewidth=2)

    ax.set_xlabel('Z-score threshold')
    ax.set_ylabel('Euler characteristic')
    ax.legend(['Expected', 'Empirical'])

    fig.tight_layout()
    plt.show()

def plot_expected_ec(R, Z_low=0, Z_high=5):
    """Function for drawing a graph of the expected Euler characteristic
    as a function of Z-score threshold.

    Input arguments:
    ================
    R : float
      The number of resels or resolution elements.
    Z_low : float
      The lowest z-score for which to plot the expected Euler characteristic
      E[EC].
    Z_high : float
      The highest z-score for which to plot E[EC].
    """

    """Compute the expected Euler characteristic for Z-scores in the
    desired range."""
    Z = np.linspace(Z_low, Z_high, 100)
    EC = np.asarray([_expected_ec_2d(R, z) for z in Z])

    """Draw the plot."""
    sns.set_style('darkgrid')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(Z, EC)

    ax.set_xlabel('Z-score threshold')
    ax.set_ylabel('Expected Euler characteristic')
    ax.set_xlim([np.min(Z)-0.1, np.max(Z)+0.1])
    # +- 2%
    c = 2*np.max(EC)/100
    ax.set_ylim([-c, np.max(EC)+c])

    fig.tight_layout()
    plt.show()

def _smooth(X, fwhm):
    """Function for spatial smoothing using a Gaussian kernel.

    Input arguments:
    ================
    X : ndarray of floats
      The analyzed two-dimensional data.
    fwhm : float
      The width of the smoothing kernel described as full-width at half
      maximum (FWHM).
    """

    """Compute standard deviation corresponding to the given
    full-width at half maximum value."""
    sd = fwhm / np.sqrt(8*np.log(2))

    """Smooth the data."""
    # TODO: consider which filter mode should be used
    Y = gaussian_filter(X, sigma=sd, mode='nearest',
                        multichannel=False, preserve_range=True)
    return Y

def rft_2d(X, fwhm, alpha=0.05, verbose=True):
    """Function for controlling the FWER using random field theory (RFT) when
    the analyzed data is two-dimensional (e.g. time-frequency or single slice
    anatomical data).

    Input arguments:
    ================
    X : ndarray of floats
      The analyzed two-dimensional data array.
    fwhm : float
      Size of the smoothing kernel measured as full-width at half maximum
      (FWMH).
    alpha : float
      The desired critical level. The conventional 0.05 level is used as
      the default value.
    verbose : bool
      Whether to print data on intermediate estimation results.

    Output arguments:
    =================
    X_thr : ndarray of floats
      An array with with equal dimensions to X indicating which elements are
      statistically significant. Ones and zeros correspond to significant and
      non-significant element respectively.
    X_smooth : ndarray of floats
      The analyzed data after spatial smoothing.
    ec : float
      The empirical Euler characteristic.
    """

    """Estimate the number of resolution elements."""
    R = _n_resels(X, fwhm)
    if (verbose):
        print('The estimated number of resels is %d' % R)

    """Find z-score threshold that gives the chosen family-wise error
    rate."""
    Z = np.linspace(2, 7, 501)
    expected_ec = _expected_ec_2d(R, Z)
    z_threshold = Z[expected_ec < alpha][0]
    if (verbose):
        print('The Z-score threshold for FWER of %1.3f is %1.3f' %
              (alpha, z_threshold))

    """Smooth, z-score, and threshold the data array."""
    X_smooth = _smooth(X, fwhm=fwhm)
    X_thr = _threshold(X_smooth, Z=z_threshold)

    """Compute the empirical Euler characteristic and find significant
    elements."""
    ec = _ec_2d(X_thr)
    if (verbose):
        print('The empirical Euler characteristic is %d' % ec)

    return X_thr, X_smooth, ec

def plot_rft_2d(X, X_smooth, X_significant):
    """Function for visualizing the raw and smoothed data with significant
    regions highlighted in red color.

    Input arguments:
    X : ndarray of floats
      The original analyzed data array.
    X_smooth : ndarray of floats
      The data after spatial smoothing has been applied.
    X_significant : ndarray of floats
      The data after thresholding, showing which elements are significant.

    TODO: let user provide data colormaps and plot only the boundaries of
    the thresholded regions.
    """
    sns.set_style('dark')
    fig = plt.figure(figsize=(10, 6))

    ax = fig.add_subplot(121)
    ax.imshow(X, cmap='gray', origin='lower')
    ax.imshow(X_significant, cmap='Reds', origin='lower', alpha=0.5)
    ax.set_xlabel('Pixel X-coordinate')
    ax.set_ylabel('Pixel Y-coordinate')
    ax.set_title('Raw data')

    ax = fig.add_subplot(122)
    ax.imshow(X_smooth, cmap='gray', origin='lower')
    ax.imshow(X_significant, cmap='Reds', origin='lower', alpha=0.5)
    ax.set_title('Smoothed data')

    fig.tight_layout()
    plt.show()
