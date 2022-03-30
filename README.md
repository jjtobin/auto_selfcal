# auto_selfcal
The goal of this code is to be able to run on an a set of ALMA or VLA *_target.ms (or other MS files with the targets split out and having the same setup in each MS) files for single-pointing data and perform self-calibration using the continuum. If a cont.dat file is present, the code will flag the non-continuum regions and perform self-calibration on the continuum only.

This code can only be executed within the CASA 6.x environment, and is only fully tested on CASA 6.4.0, 6.4.1, and 6.4.3. CASA versions earlier that 6.4.0 are likely to encounter problems due to a gaincal bug that was fixed in CASA 6.4.0.

Standard ALMA and single-band VLA data are supported as are multiple targets.
Multi-band VLA is supported.
ALMA spectral scans may work, but have not been thoroughly tested.
Mosaics have only had limited testing, but some limited functionality might work, specifically small mosaics where there is emission of the same source(s) within each pointing.

Brief instructions:
1. Create an empty directory
2. Copy into this directory the *_target.ms files that have identical setups (targets and spectral windows) to be self-calibrated (must contain only the targets desired for self-calibration)
4. Copy auto_selfcal.py and selfcal_helpers.py into this directory
5. Run script with mpicasa -n X casa -c auto_selfcal.py; X is the number of mpi threads to use

If serial operation is desired (without mpicasa), edit the auto_selfcal.py script to specify 'parallel=False' and run with casa -c auto_selfcal.py

The script will automatically find your *_target.ms files, and determine all the parameters it needs for imaging and self-calibration of the science targets.

The script will output a file 'applycal_to_orig_MSes.py' to apply the self-calibration solutions back to the original MS such that the line data can also have the self-calibration solutions applied.

When finished summaries are generated for each target in weblog/index.html. Weblog generation may fail depending on what settings are cached for the viewer. If this is a problem you can remove all your cached casaviewer settings by running 'rm -rf ~/.casa/viewer'. Then you can run casa -c regenerate_weblog.py, the regenerate_weblog.py script is included in the repository and only generates a new weblog from the already stored images and metadata pickle files.

