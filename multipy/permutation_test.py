import matplotlib.pyplot as plt

from mne.viz import plot_evoked_topo

import numpy as np

from permutation import permutation_test
from viz import plot_permutation_distribution, plot_permutation_result_1d

import seaborn as sns

"""Test the permutation testing methods on MNE sample data."""
import mne

"""Settings."""
plot_topography = False

"""Load the sample dataset from disk."""
data_path = mne.datasets.sample.data_path()
fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'

raw = mne.io.Raw(fname, preload=True)
"""Load EOG and ECG projectors."""
eog_proj = mne.read_proj(data_path +
                     '/MEG/sample/sample_audvis_eog-proj.fif')
ecg_proj = mne.read_proj(data_path +
                     '/MEG/sample/sample_audvis_ecg-proj.fif')
raw.add_proj(eog_proj)
raw.add_proj(ecg_proj)

"""Apply the projectors."""
raw.apply_proj()

"""Epoch the data."""
events = mne.find_events(raw)
raw.pick_types(meg=True)

lh_aud = mne.Epochs(raw, events, event_id=1, tmin=-0.75, tmax=0.75,
                    baseline=(-0.75, -0.5))
rh_aud = mne.Epochs(raw, events, event_id=3, tmin=-0.75, tmax=0.75,
                    baseline=(-0.75, -0.5))

"""Compared evoked responses"""
if (plot_topography):
    lh_aud_evoked, rh_aud_evoked = lh_aud.average(), rh_aud.average()
    plot_evoked_topo([lh_aud_evoked, rh_aud_evoked],
                     color=['blue', 'red'])

# Select all epochs and time points from one channel.
ch_ind = [i for i, ch_name in enumerate(raw.info['ch_names'])
          if 'MEG 2343'in ch_name][0]
lh_data, rh_data = (lh_aud.get_data()[:, ch_ind, :],
                    rh_aud.get_data()[:, ch_ind, :])

significant, pvals, cstats, ref_cstat, ref_clusters = permutation_test(lh_data, rh_data)
plot_permutation_distribution(cstats, ref_cstat, show_plot=True)
plot_permutation_result_1d(lh_data, rh_data, significant, lh_aud.times, ref_clusters)

"""Plot trial averages."""
plot_ave = False

if (plot_ave):
    sns.set_style('darkgrid')
    fig = plt.figure()

    ax = fig.add_subplot(111)
    ax.plot(np.mean(lh_data, axis=0))
    ax.plot(np.mean(rh_data, axis=0))

    ax.set_xlabel('Time (samples)')
    ax.set_ylabel('Amplitude')

    fig.tight_layout()
    plt.show()
