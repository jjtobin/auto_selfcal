![build and test](https://github.com/psheehan/auto_selfcal/actions/workflows/run_E2E_test.yml/badge.svg)
[![codecov](https://codecov.io/github/psheehan/auto_selfcal/graph/badge.svg?token=6PLBR79LWF)](https://codecov.io/github/psheehan/auto_selfcal)

# auto_selfcal
The goal of this code is to be able to run on an a set of ALMA or VLA *_target.ms (or other MS files with the targets split out and having the same setup in each MS) files for single-pointing or msoaic data and perform self-calibration using the continuum. If a cont.dat file is present, the code will flag the non-continuum regions and perform self-calibration on the continuum only.

To run this code with a concatenated calibrated_final.ms that one might receive from the NA ARC, one must split out the groups of SPWs associated with the individual observations, selecting on SPW, such that one has a set of MSes with SPWs that are all the same. For example, if an MS has SPWs 0,1,2,3,10,11,12,13, and 0,1,2,3 are from the first observation and 10,11,12,13 are from the second observation, they should be split out as follows:
split(vis='my_concat.ms',spw'0,1,2,3',outputvis='my_ms_0_target.ms')
split(vis='my_concat.ms',spw'10,11,12,13',outputvis='my_ms_1_target.ms')
We provide a script split_calibrated_final.py to do this automatically.

This code should only be executed within the CASA 6.4+ environment. CASA versions earlier that 6.4.0 are likely to encounter problems due to a gaincal bug that was fixed in CASA 6.4.0. Current testing of this code was conducted using the CASA 6.5 and 6.6 series. We note that the continuum subtraction portions of the code requires CASA 6.5.2 to use the new uvcontsub.

Supported data:
Standard ALMA and VLA data.
Single or multiple targets.
Multi-band VLA.
ALMA spectral scans.
ALMA Mosaics

Amplitude self-calibration will be attempted after phase-only self-calibration. If amplitude self-cal is not desired, set the variable 'do_amp_selfcal' to False in the auto_selfcal.py script.

Brief instructions:
1. Create an empty directory
2. Copy into this directory the *_target.ms files that have identical setups (targets and spectral windows) to be self-calibrated (must contain only the targets desired for self-calibration and only TARGET observation intents)
3. Copy cont.dat file for all targets into directory (will be used to flag out spectral lines).
4. Copy all .py files in the cloned auto_selfcal repo into your working directory
5. Run script with mpicasa -n X casa -c auto_selfcal.py; X is the number of mpi threads to use

If serial operation is desired (without mpicasa), run with casa -c auto_selfcal.py

The script will automatically find your *_target.ms files, and determine all the parameters it needs for imaging and self-calibration of the science targets.

The script will output a file 'applycal_to_orig_MSes.py' to apply the self-calibration solutions back to the original MS such that the line data can also have the self-calibration solutions applied.

The script will output a file 'uvcontsub_orig_MSes.py' if a cont.dat file exists, excluding the same spectral regions for continuum fitting as were flagged for continuum self-calibration. Note that the file format will be different if executed in CASA versions later than 6.5.2 to account for new uvcontsub task.

When finished summaries are generated for each target in weblog/index.html. Weblog generation may fail depending on what settings are cached for the viewer. If this is a problem you can remove all your cached casaviewer settings by running 'rm -rf ~/.casa/viewer'. Then you can run casa -c regenerate_weblog.py, the regenerate_weblog.py script is included in the repository and only generates a new weblog from the already stored images and metadata pickle files.

Acknowledgements:

Certain functions to convert from LSRK to channel, S/N estimates, and tclean wrapper have their origins from the ALMA DSHARP large program reduction scripts.

The functions to parse the cont.dat file and convert to channel ranges (used the routine from above) was adapted from a function written by Patrick Sheehan for the ALMA eDisk large program
