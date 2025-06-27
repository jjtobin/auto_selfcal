from casatools import msmetadata as msmdtool
from casatools import table as tbtool
from casatasks import split
import glob
import os
import numpy as np

msmd = msmdtool()
tb = tbtool()

def split_calibrated_final(vislist=[], overwrite=True):
    """
    Takes an input list of MS files and splits out the data into a collection of datasets that are in the format
    expected by the auto_selfcal function.

    Parameters
    ----------
    vislist : list or str, optional:
        A list of MS files to split. If a string is provided, it will be treated as a single MS file.
        If a list is provided, it should contain the names of the MS files to split.
        If vislist is empty, it will default to looking for 'calibrated_final.ms' in the current directory.
    overwrite : bool, optional:
        If True, will overwrite any existing output files. Default is True.
        If False, will skip any existing output files.
        
    Returns
    -------
    None
    """

    # Check that the vislist keyword is supplied correctly.

    if not is_iterable(vislist):
        print("Argument vislist must be a string or list-like. Exiting...")
    elif type(vislist) == str:
        vislist = [vislist]
    elif len(vislist) == 0:
        vislist = glob.glob('calibrated_final.ms')

    # Loop over the vislist and split out the relevant data.

    for vis in vislist:
        # Check whether a CORRECTED_DATA column exists. If not, use the DATA column.

        tb.open(vis)
        if "CORRECTED" in tb.colnames():
            datacolumn="corrected"
        else:
            datacolumn="data"
        tb.close()

        # Split each observation out separately, if this is a concatenated MS file.

        msmd.open(vis)
        for i in range(msmd.nobservations()):
            # Set the output filename based on the ALMA naming scheme to be similar to what the pipeline would have called it.

            outputvis = msmd.schedule(i)[1].split(" ")[1].replace("/","_").replace(":","_")+"_targets.ms"
            if os.path.exists(outputvis):
                if overwrite:
                    print(f"{outputvis} already exists, but overwrite=True, removing.")
                    os.system(f"rm -rf {outputvis}")
                else:
                    print(f"{outputvis} already exists, skipping.")
                    continue

            # msmd.scansforintent gets all of the spw observed for given scan, but we need to intersect that with only the 
            # TDM and FDM spws for that scan, in case some of the other types of spw that can exist are left over.

            output_spw = ','.join(np.intersect1d(msmd.spwsforscan(msmd.scansforintent("*OBSERVE_TARGET*", obsid=i)[0], obsid=i), \
                    np.concatenate((msmd.tdmspws(),msmd.fdmspws()))).astype(str))

            # Only take the antennas used for a scan on a relevant target from the relevant observation ID.

            output_antennas = ','.join(msmd.antennasforscan(msmd.scansforintent("*OBSERVE_TARGET*", obsid=i)[0], obsid=i).astype(str))

            # Do the split

            split(vis, outputvis=outputvis, observation=i, intent="*OBSERVE_TARGET*", spw=output_spw, antenna=output_antennas, \
                    datacolumn=datacolumn)

        msmd.close()

def is_iterable(obj):
    try:
        iter(obj)
        return True
    except TypeError:
        return False
