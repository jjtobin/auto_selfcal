Running auto_selfcal
====================

Modular CASA
------------

To run auto_selfcal in a modular CASA environment, a command line tool is provided:

    auto_selfcal --<command line option> <argument> ...

Or, if installed with support for MPI:

    mpirun -n <N> auto_selfcal --<command line option> <argument> ...

auto_selfcal can also be imported and run within Python scripts or environments:

    .. highlight:: python

    from auto_selfcal import auto_selfcal

    auto_selfcal(<options>)

Monolithic CASA
---------------

Support remains for running auto_selfcal in a manner similar to versions 1.X within a monolithic CASA distribution. To do so, download the auto_selfcal package from here: https://github.com/jjtobin/auto_selfcal/releases and run:

    casa --nologger --nogui -c <path/to/auto_selfcal>/bin/auto_selfcal.py

Or with mpicasa:

    mpicasa -n <N> casa --nologger --nogui -c <path/to/auto_selfcal>/bin/auto_selfcal.py

Note that with these scripts there is no support for command line arguments; to change these options, edit the auto_selfcal.py file directly.
