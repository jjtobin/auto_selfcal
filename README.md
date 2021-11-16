# auto_selfcal
The goal of this code is to be able to run on an a set of ALMA or VLA *_target.ms files for single-pointing data and perform self-calibration using the continuum. If a cont.dat file is present, the code will flag the non-continuum regions and perform self-calibration on the continuum only.

Multi-band VLA is supported.
Mosaics are not tested, but some limited functionality might work on a per-scan basis.
