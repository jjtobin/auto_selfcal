Examining the results
=====================

By default, auto_selfcal will produce a weblog that provides details about how self-calibration proceeded, including a breakdown of the before and after statistics for each solution interval attempted, reasons for failures, and more. The weblog resides within a "weblog" directory that is created by running auto_selfcal, and can be viewed within a browser. One specific way to view the weblog is to run

    cd weblog

    python -m http.server 8000

and then to direct your browser to 'localhost:8000'.