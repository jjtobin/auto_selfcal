![build and test](https://github.com/psheehan/auto_selfcal/actions/workflows/run_E2E_test.yml/badge.svg)
[![codecov](https://codecov.io/github/psheehan/auto_selfcal/graph/badge.svg?token=6PLBR79LWF)](https://codecov.io/github/psheehan/auto_selfcal)

auto_selfcal: Self-calibration, without the hassle!
===================================================

auto_selfcal does automatic self-calibration of ALMA and VLA with (almost) no effort from you. It can handle most forms of data, including single pointing and ALMA mosaics (VLA mosaics coming soon), ephemeris data, spectral scans, and more. See below to give it a try, or check out our more extensive documentation at https://auto-selfcal.readthedocs.io/en/latest/running.html.

Quickstart
----------

To use auto_selfcal with an existing monolithic CASA distribution:

```bash
git clone https://github.com/jjtobin/auto_selfcal.git
cd </path/to/pipeline/calibrated/*_targets.ms/files>
casa -c </path/to/auto_selfcal>/bin/auto_selfcal.py
```

Or to install into an Anaconda environment and run from a directory where pipeline-calibrated *_targets.ms files exist::

```bash
git clone https://github.com/jjtobin/auto_selfcal.git
cd auto_selfcal
conda env create -f environment.yaml
conda activate casa-6.6.5

cd </path/to/pipeline/calibrated/*_targets.ms/files>

auto_selfcal
```

Acknowledgements:
-----------------

Certain functions to convert from LSRK to channel, S/N estimates, and tclean wrapper have their origins from the ALMA DSHARP large program reduction scripts.

The functions to parse the cont.dat file and convert to channel ranges (used the routine from above) was adapted from a function written by Patrick Sheehan for the ALMA eDisk large program
