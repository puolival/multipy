# Example analyses on the Open Access Series of Imaging Studies (OASIS) data

This is work in progress. Check again later on.

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

## References

Marcus DS, Wang TH, Parker J, Csernansky JG, Morris JC, Buckner RL (2007): Open access series of imaging studies (OASIS): cross-sectional MRI data in young, middle aged, nondemented, and demented older adults. *Journal of Cognitive Neuroscience* 19(9):1498–1507 (<a href="https://dash.harvard.edu/handle/1/33896768">full text</a>)
