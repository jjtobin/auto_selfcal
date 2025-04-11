.. auto_selfcal documentation master file, created by
   sphinx-quickstart on Tue Jan 21 12:59:07 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

auto_selfcal: Self-calibration without the hassle!
==================================================

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: User-Guide

   installation
   data
   running

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: API

   api

This package provides tools to automatically self-calibrate your interferometric data, all you need is your data! auto_selfcal supports data that fall in the below categories:

- ALMA or VLA
- Single pointing 
- Multiple target EBs
- Multi-band 
- ALMA spectral scans
- ALMA mosaics (VLA mosaics supported but will be self-calibrated field-by-field at the moment)
- Ephemeris targets (including support for user-defined models)

Acknowledgements:

Certain functions to convert from LSRK to channel, S/N estimates, and tclean wrapper have their origins from the ALMA DSHARP large program reduction scripts.

The functions to parse the cont.dat file and convert to channel ranges (used the routine from above) was adapted from a function written by Patrick Sheehan for the ALMA eDisk large program
