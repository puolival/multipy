from permutation import permutation_test
from viz import plot_permutation_distribution

"""Test the permutation testing methods on MNE sample data."""
import mne

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

lh_aud = mne.Epochs(raw, events, event_id=1, tmin=-0.75, tmax=0.75,
                    baseline=(-0.75, -0.5))
rh_aud = mne.Epochs(raw, events, event_id=2, tmin=-0.75, tmax=0.75,
                    baseline=(-0.75, -0.5))


# Select all epochs and time points from one channel.
ch_ind = 100
lh_data, rh_data = (lh_aud.get_data()[:, ch_ind, :],
                    rh_aud.get_data()[:, ch_ind, :])

pval, cstats, ref_cstat = permutation_test(lh_data, rh_data)
print('p-value is %f' % pval)
plot_permutation_distribution(cstats, ref_cstat, show_plot=True)

