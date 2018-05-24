# Script for running the qcache pipeline of recon-all for a number
# of participants.
#
# The directory $SUBJECTS_DIR should contain a text file called
# subject_list that specifies the included participants (1 entry per line).
#
# Author: Tuomas Puoliväli
# Email: tuomas.puolivali@helsinki.fi
# Last modified: 24 May 2018

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
