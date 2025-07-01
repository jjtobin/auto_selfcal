Examining the results
=====================

By default, auto_selfcal will produce a weblog that provides details about how self-calibration proceeded, including a breakdown of the before and after statistics for each solution interval attempted, reasons for failures, and more. The weblog resides within a "weblog" directory that is created by running auto_selfcal, and can be viewed within a browser by opening weblog/index.html. 

If you are running auto_selfcal on a remote system, the weblog can be viewed by SSH tunneling:

.. code-block:: bash

    ssh -L 8000:localhost:8000 hostname

setting up an HTTP Server with Python on the tunneled port:

.. code-block:: bash

    cd weblog
    python -m http.server 8000

and then to directing your browser to 'localhost:8000'.