Preparing your data
===================

auto_selfcal supports data that fall in the below categories:

- ALMA or VLA
- Single pointing 
- Multiple target EBs
- Multi-band 
- ALMA spectral scans
- ALMA mosaics (VLA mosaics supported but will be self-calibrated field-by-field at the moment)
- Ephemeris targets (including support for user-defined models)

auto_selfcal expects to be operating on a collection of MSes that come from individual observation blocks (i.e., no concatenating please), 
with the science targets and science SPWs split out. At the moment, all EBs must have the same spectral window set up, and the names of the 
SPWs must match across EBs (for ALMA, this is equivalent to saying that the data must come from the MOUS level; the array configuration remains 
the same.) While a user can supply a list of MSes to auto_selfcal, if no list is provided then it will look for MS files in the current working directory
with the following naming conventions (in order; please don't mix and match):
::

    <descriptor>_targets.ms
    <descriptor>_target.ms
    <descriptor>_cont.ms
    calibrated_final.ms

To run this code with a concatenated calibrated_final.ms that one might receive from the NA ARC or with the variety of formats that might be 
expected from restores of older pipeline runs, one must split out the groups of SPWs associated with the individual observations, selecting on SPW, 
such that one has a set of MSes with SPWs that are all the same. To simplify this process, we provide a helper routine, :meth:`split_calibrated_final<auto_selfcal.split_calibrated_final>` 
that can ingest data from a variety of formats and spit out data that should be in the format that auto_selfcal expects. It can be used from Monolithic CASA via the provided auto_selfcal scripts:

.. code-block:: bash

    casa -c <path/to/auto_selfcal>/bin/split_calibrated_final.py

via the command line tool in modular CASA:

.. code-block:: bash

    auto_selfcal --action prepare_data --vislist <list of MSes>

or via the Python API (either modular CASA or pip-installed-auto_selfcal in monolithic CASA):

.. code-block:: python

    from auto_selfcal import split_calibrated_final

    split_calibrated_final(vislist=<list of MSes>)

Alternatively, as shown above, if no list of MSes is provided and a calibrated_final.ms file is present in the current working directory,
auto_selfcal will automatically attempt to split it into the expected format.

If a user has a cont.dat file from a previous pipeline run available, that file can be placed in the same location as the MS files. 
The presence of a cont.dat is not, however, required to run auto_selfcal, and if one wishes to generate a cont.dat file, there is an 
option to do so within auto_selfcal (when run in monolithic CASA with the pipeline installed).
