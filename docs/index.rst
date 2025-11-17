.. auto_selfcal documentation master file, created by
   sphinx-quickstart on Tue Jan 21 12:59:07 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Self-calibration without the hassle!
====================================

Tired of self-calibrating your data by hand? You've come to the right place! The auto_selfcal package can automatically self-calibrate your 
(ALMA and VLA, currently) data with almost no effort from you. But don't take our word for it, just ask the `ALMA Pipeline <https://almascience.nrao.edu/processing/alma_pipeline_user_s_guide_for_release_2024-1.pdf>`_. To see how to make this work
for you, check out the rest of our documentation:

.. toctree::
   :maxdepth: 1
   :caption: User-Guide

   installation
   data
   running
   weblog

.. toctree::
   :maxdepth: 1
   :caption: API

   api
   selfcal_helpers

Acknowledging auto_selfcal
--------------------------

Love auto_selfcal and want to cite it in your paper? Please include the following citations:

<COMING SOON>

Contributing and/or Bugs
------------------------

Want to contribute? Found a bug? Please feel free to open an `issue <https://github.com/jjtobin/auto_selfcal/issues/new/choose>`_ or `pull request <https://github.com/jjtobin/auto_selfcal/compare>`_ on GitHub and the auto_selfcal team will follow up with you there.

Acknowledgements:
-----------------

Certain functions to convert from LSRK to channel, S/N estimates, and tclean wrapper have their origins from the ALMA DSHARP large program reduction scripts.

The functions to parse the cont.dat file and convert to channel ranges (used the routine from above) was adapted from a function written by Patrick Sheehan for the ALMA eDisk large program
