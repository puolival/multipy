# Example analyses on the Open Access Series of Imaging Studies (OASIS) data

This is work in progress. Check again later on.

## Initial preparations

Install a recent version of FreeSurfer from http://surfer.nmr.mgh.harvard.edu/.

### Download the data

To download the OASIS dataset, visit http://www.oasis-brains.org/#data and 
choose the dataset OASIS-1, which contains magnetic resonance (MR) images 
from 416 participants aged 18 to 96 years.

### Run the qcache pipeline of recon-all

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

## References

Marcus DS, Wang TH, Parker J, Csernansky JG, Morris JC, Buckner RL (2007): Open access series of imaging studies (OASIS): cross-sectional MRI data in young, middle aged, nondemented, and demented older adults. *Journal of Cognitive Neuroscience* 19(9):1498–1507
