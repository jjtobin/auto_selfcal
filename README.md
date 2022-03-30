# auto_selfcal
The goal of this code is to be able to run on an a set of ALMA or VLA *_target.ms (or other MS files with the targets split out and having the same setup in each MS) files for single-pointing data and perform self-calibration using the continuum. If a cont.dat file is present, the code will flag the non-continuum regions and perform self-calibration on the continuum only.

This code can only be executed within the CASA 6.x environment, and is only fully tested on CASA 6.4.0, 6.4.1, and 6.4.3. CASA versions earlier that 6.4.0 are likely to encounter problems due to a gaincal bug that was fixed in CASA 6.4.0.

Multi-band VLA is supported.
Mosaics are not tested, but some limited functionality might work on a per-scan basis.
