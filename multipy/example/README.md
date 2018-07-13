# Example analyses on the Open Access Series of Imaging Studies (OASIS) data

This is work in progress. Check again later on.

## Introduction

In this example, we perform mass-univariate analyzes on magnetic resonance (MR) images 
extracted from the open-access series of imaging studies (OASIS) open data initiative
(Marcus et al, 2007). These data contain T1-weighted images from 416 demented and 
non-demented participants aged 18 to 96 years, which allows investigating how age and 
age-related diseases influence brain morphology.

## Initial preparations

Install FreeSurfer from http://surfer.nmr.mgh.harvard.edu/. These data analyses were
performed using the version 6.0.0 but newer and relatively recent older versions likely
work as well. If you have not used FreeSurfer before, we suggest to try their 
<a href="http://surfer.nmr.mgh.harvard.edu/fswiki/Tutorials">tutorials</a>.

### Download the data

To download the OASIS dataset, visit http://www.oasis-brains.org/#data and 
choose the release OASIS-1, which contains magnetic resonance (MR) images 
from 416 participants aged 18 to 96 years.

### Run the qcache pipeline of recon-all

The next step is to compute voxel-level morphometric measures and align data 
from individual participants to a common space. This can be performed using 
the qcache pipeline of recon-all.

```bash
cd /home/local/puolival/multipy-testdata/oasis
find ./ -name "OAS1_*_MR1" | cut -c 3- > subject_list
```

<a href="https://github.com/puolival/multipy/blob/master/multipy/example/qcache.sh">qcache.sh</a>:
```bash
# Settings
export FREESURFER_HOME=/usr/local/freesurfer
export SUBJECTS_DIR=/home/local/puolival/multipy-testdata/oasis

MEAS=thickness
FWHM=10

# Initialize FreeSurfer.
source $FREESURFER_HOME/SetUpFreeSurfer.sh

# Run the qcache pipeline of recon-all.
cat subject_list | while read SUBJECT
do
    recon-all -subjid $SUBJECT -qcache -measure $MEAS -fwhm $FWHM \
        -no-isrunning -openmp 8
done
```

## Data analysis

### Load and prepare the demographic data for analysis

To load, process, and visualize the cortical thickness data and 
participant demographics, we use the following libraries: 
<a href="http://nipy.org/nibabel/">NiBabel</a> (version 2.2.1), <a href="https://pandas.pydata.org/">Pandas</a> (version 0.20.3), <a href="http://www.numpy.org/">NumPy</a> (version 1.10.4),
<a href="https://www.scipy.org/">SciPy</a> (version 0.17.0), and <a href="https://pysurfer.github.io/">PySurfer</a> (0.9.dev0).

```python
from multipy.fdr import lsu
from multipy.fwer import sidak

import nibabel as nib

import numpy as np

import pandas as ps

from os.path import isfile

from scipy.stats import spearmanr

from surfer import Brain
```

The participant information file includes basic demographic variables (age, gender, handedness, education level, socioeconomic status), clinical variables, and brain volume estimates. The variable names are given on the first row of the CSV file. A more detailed description of the demographic variables is available in the OASIS-1 <a href="https://www.oasis-brains.org/files/oasis_cross-sectional_facts.pdf">fact sheet</a>.

```python
"""Read subject demographics from the CSV file. In this example we only
need to know the CDR score and age of each participant."""
fpath = '/home/local/puolival/multipy-testdata/oasis'
fname_demographics = fpath + '/demographics.csv'

column_names = ['session', 'subject', 'gender', 'handedness', 'age',
                'education','ses', 'cdr', 'mmse', 'etiv', 'nwbv',
                'asf','scans']
column_types = {'session': np.str, 'age': np.int, 'cdr': np.float}

df = ps.read_csv(fname_demographics, delimiter=',', header=0,
                 names=column_names, dtype=column_types)
```

Here we only need to know each participant's age, identifier, and 
clinical dementia rating (CDR), so we discard other columns from the 
<a href="https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html">DataFrame</a> object. Participants with a CDR value greater than 0 are 
demented and the rest are healthy.
```python
"""Discard unnecessary columns. Replace NaNs with zeros. Restrict
the analysis to non-demented subjects."""
df = df.filter(column_types.keys()).fillna(0)
df = df[df.cdr == 0]
```

### Load and prepare the thickness data for analysis

```python
"""Read morphometric data."""
hemisphere, measure, smoothing = 'rh', 'thickness', 'fwhm10'

surf_data, bad_ind = [], []
for i, subj in enumerate(df.session):
    fname = (fpath + '/' + subj + '/surf/' + hemisphere + '.' + measure
             + '.' + smoothing + '.fsaverage.mgh')
    if not (isfile(fname)):
        bad_ind.append(i)
        continue
    data = nib.load(fname).get_data()[:, 0, 0]
    surf_data.append(data)

surf_data = np.asarray(surf_data)
df = df.drop(df.index[bad_ind])
```

### Compute statistics and perform correction for multiple comparisons

We are now ready for computing correlation coefficients between cortical thickness and age in each cortical voxel. The cortical thickness estimation may sometimes fail for individual voxels, so we discard samples where the estimated thickness is less than 0.1 mm, which is much less than the MR resolution.
```python
"""For each voxel, correlate thickness with age."""
_, n_voxels = np.shape(surf_data)
pvals = np.zeros([n_voxels, 1])

for i in np.arange(0, n_voxels):
    valid_ind = surf_data[:, i] > 0.1
    pvals[i] = spearmanr(df.age.values[valid_ind],
                         surf_data[valid_ind, i])[1]
    print 'voxel %6d' % i

pvals[np.isnan(pvals)] = 1
pvals = pvals[:, 0]
```

The next step is to correct the p-values for the 163810 comparisons. Here we apply the Šidák correction, which controls the family-wise error rate (FWER), and the Benjamini-Hochberg procedure, which controls the false discovery rate (FDR).
```python
fdr_sig = lsu(pvals, q=0.05)
fwr_sig = sidak(pvals, alpha=0.05)
```

### Visualize results

The last step is visualize the results on the cortical surface using PySurfer.

```python
brain = Brain(subject_id='fsaverage', hemi=hemisphere, surf='inflated',
              subjects_dir=fpath, size=640)
```

## References

Marcus DS, Wang TH, Parker J, Csernansky JG, Morris JC, Buckner RL (2007): Open access series of imaging studies (OASIS): cross-sectional MRI data in young, middle aged, nondemented, and demented older adults. *Journal of Cognitive Neuroscience* 19(9):1498–1507 (<a href="https://dash.harvard.edu/handle/1/33896768">full text</a>)
