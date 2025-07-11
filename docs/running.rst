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

By default, the :meth:`auto_selfcal<auto_selfcal.auto_selfcal>` function does not apply the derived calibrations back to the original MS files supplied. Though this can be changed with the relevant keyword arguments, the calibrations can be applied after the fact using a script supplied with auto_selfcal:

.. code-block:: bash

    casa -c <path/to/auto_selfcal>/bin/applycal_to_orig_MSes.py

We also provide a tool to do continuum subtraction of the original MS files using the continuum ranges from cont.dat (if available):

.. code-block:: bash

    casa -c <path/to/auto_selfcal>/bin/uvcontsub_orig_MSes.py

Note that with these scripts there is no support for command line arguments; to change these options, edit the bin/\*.py files directly. For details about the available parameters for each of these functions, see our :ref:`Top Level API`.

Running in Modular CASA
-----------------------

To run auto_selfcal in a modular CASA environment, a command line tool is provided:

.. code-block:: bash

    auto_selfcal --<command line option> <argument> ...

Or, if installed with support for MPI:

.. code-block:: bash

    mpirun -n <N> auto_selfcal --<command line option> <argument> ...

For the same examples as above,

.. code-block:: bash

    mpirun -n 5 auto_selfcal --action run
    auto_selfcal --action apply
    auto_selfcal --action contsub

auto_selfcal can also be imported and run within Python scripts or environments:

.. code-block:: python

    from auto_selfcal import auto_selfcal, applycal_to_orig_MSes, uvcontsub_orig_MSes

    auto_selfcal(vislist=<list of MSes>)
    applycal_to_orig_MSes()
    uvcontsub_orig_MSes()

For a full list of command line options, run:

.. code-block:: bash

    auto_selfcal --help

or see the API documentation for the :meth:`auto_selfcal<auto_selfcal.auto_selfcal>` function.
