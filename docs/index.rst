.. auto_selfcal documentation master file, created by
   sphinx-quickstart on Tue Jan 21 12:59:07 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Self-calibration without the hassle!
====================================

Tired of self-calibrating your data by hand? You've come to the right place! The auto_selfcal package can automatically self-calibrate your 
(ALMA and VLA, currently) data with almost no effort from you. But don't take our word for it, just ask the `ALMA Pipeline <https://almascience.nrao.edu/processing/alma_pipeline_user_s_guide_for_release_2024-1.pdf>`_. To see how to make this work
for you, see our quickstart guide:

Quickstart
----------

To use auto_selfcal with an existing monolithic CASA distribution:

.. code-block:: bash

   git clone https://github.com/jjtobin/auto_selfcal.git
   cd </path/to/pipeline/calibrated/*_targets.ms/files>
   casa -c </path/to/auto_selfcal>/bin/auto_selfcal.py

Or to install into an Anaconda environment and run from a directory where pipeline-calibrated *_targets.ms files exist:

.. code-block:: bash

   pip install auto_selfcal

   cd </path/to/pipeline/calibrated/*_targets.ms/files>

   auto_selfcal

Or check out the rest of our documentation:

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

.. code-block:: bibtex

   @software{auto_selfcal,
      author       = {Patrick D. Sheehan and John J. Tobin and Rui Xue and Austen Fourkas},
      title        = {jjtobin/auto\_selfcal: v1.3.1},
      month        = nov,
      year         = 2025,
      publisher    = {Zenodo},
      version      = {v1.3.1},
      doi          = {10.5281/zenodo.17603063},
      url          = {https://doi.org/10.5281/zenodo.17603063},
   }

Contributing and/or Bugs
------------------------

Want to contribute? Found a bug? Please feel free to open an `issue <https://github.com/jjtobin/auto_selfcal/issues/new/choose>`_ or `pull request <https://github.com/jjtobin/auto_selfcal/compare>`_ on GitHub and the auto_selfcal team will follow up with you there.

Acknowledgements:
-----------------

Certain functions to convert from LSRK to channel, S/N estimates, and tclean wrapper have their origins from the ALMA DSHARP large program reduction scripts.

The functions to parse the cont.dat file and convert to channel ranges (used the routine from above) was adapted from a function written by Patrick Sheehan for the ALMA eDisk large program
