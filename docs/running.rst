Running auto_selfcal
====================

auto_selfcal can be run in two different modes, depending on whether you are using a modular CASA environment or a monolithic CASA distribution:

Running in Monolithic CASA
--------------------------

Although the structure is a little different, support remains for running auto_selfcal in a manner similar to versions 1.X within a monolithic CASA distribution. To do so, follow the instructions for :ref:`Installing into Monolithic CASA` and run:

.. code-block:: bash

    casa --nologger --nogui -c <path/to/auto_selfcal>/bin/auto_selfcal.py

Or with mpicasa:

.. code-block:: bash

    mpicasa -n <N> casa -c <path/to/auto_selfcal>/bin/auto_selfcal.py

Additional CASA scripts that handle other aspects of auto_selfcal are also available:

.. code-block:: bash

    mpicasa -n <N> casa -c <path/to/auto_selfcal>/bin/split_calibrated_final.py
    mpicasa -n <N> casa -c <path/to/auto_selfcal>/bin/applycal_to_orig_MSes.py
    mpicasa -n <N> casa -c <path/to/auto_selfcal>/bin/uvcontsub_orig_MSes.py
    mpicasa -n <N> casa -c <path/to/auto_selfcal>/bin/regenerate_weblog.py

Note that with these scripts there is no support for command line arguments; to change these options, edit the bin/\*.py files directly. For details about the available parameters for each of these functions, see our :ref:`Top Level API`.

Running in Modular CASA
-----------------------

To run auto_selfcal in a modular CASA environment, a command line tool is provided:

.. code-block:: bash

    auto_selfcal --<command line option> <argument> ...

Or, if installed with support for MPI:

.. code-block:: bash

    mpirun -n <N> auto_selfcal --<command line option> <argument> ...

auto_selfcal can also be imported and run within Python scripts or environments:

.. code-block:: python

    from auto_selfcal import auto_selfcal

    auto_selfcal(vislist=<list of MSes>)

For a full list of command line options, run:

.. code-block:: bash

    auto_selfcal --help

or see the API documentation for the :meth:`auto_selfcal<auto_selfcal.auto_selfcal>` function.
