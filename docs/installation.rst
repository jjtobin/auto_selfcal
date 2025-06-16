Installation
============

Installing with pip
-------------------

auto_selfcal can be installed into any Python environment using the pip package manager:

    pip install auto_selfcal

Note that auto_selfcal depends on casatasks and casatools, both of which are available only in limited Python versions, so be sure to check your Python version before attempting to install. For further details, please see the `CASA compatibility matrix <https://casadocs.readthedocs.io/en/stable/notebooks/introduction.html#Compatibility>`_.

To install auto_selfcal with support for casampi:

    pip install auto_selfcal[mpi]

A note: casampi can be quite particular about versions of mpi4py and other libraries. If you are having trouble, check the included environment.yaml file, which should document a configuration that works for casa 6.6.5 (and can be used to directly create a working environment within the Anaconda package manager, see below).

Installing with conda
---------------------

To create an Anaconda environment into which auto_selfcal can be installed, we provide an Anaconda environment file that is capable of reproducing the necessary environment. To create an environment from this file, run:

    conda env create -f environment.yaml

Once the environment is created, auto_selfcal can be pip installed into it:

    conda activate casa-6.6.5

    pip install auto_selfcal

Installing into monolithic CASA
-------------------------------

Formal installation into monolithic CASA is not required, as we provide script versions of the main functionality of this package. If you do, however, want to install into monolithic CASA, see above under 'Install with pip'.

A note on CASA versions
-----------------------

There may be specific CASA version requirements matched to specific versions of auto_selfcal, but one overarching requirement is that his code should only be executed within the CASA 6.4+ environment. CASA versions earlier that 6.4.0 are likely to encounter problems due to a gaincal bug that was fixed in CASA 6.4.0. Current testing of this code was conducted using CASA 6.5.4 and CASA 6.5.6, but we fully expect this code to run properly on CASA 6.6 and should aso run properly back to CASA 6.4.0.
