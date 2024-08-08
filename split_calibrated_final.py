from casatools import msmetadata as msmdtool
from casatools import table as tbtool
import glob
import os
import numpy as np

msmd = msmdtool()
tb = tbtool()

vislist = glob.glob('calibrated_final.ms')
overwrite = True

for vis in vislist:
    tb.open(vis)
    if "CORRECTED" in tb.colnames():
        datacolumn="corrected"
    else:
        datacolumn="data"
    tb.close()

    msmd.open(vis)
    for i in range(msmd.nobservations()):
        outputvis = msmd.schedule(i)[1].split(" ")[1].replace("/","_").replace(":","_")+"_targets.ms"
        if os.path.exists(outputvis):
            if overwrite:
                print(f"{outputvis} already exists, but overwrite=True, removing.")
                os.system(f"rm -rf {outputvis}")
            else:
                print(f"{outputvis} already exists, skipping.")
                continue

        split(vis, outputvis=outputvis, observation=i, intent="*OBSERVE_TARGET*", \
                spw=','.join(np.intersect1d(msmd.spwsforscan(msmd.scansforintent("*OBSERVE_TARGET*", obsid=i)[0], obsid=i), \
                np.concatenate((msmd.tdmspws(),msmd.fdmspws()))).astype(str)), \
                antenna=','.join(msmd.antennasforscan(msmd.scansforintent("*OBSERVE_TARGET*", obsid=i)[0], obsid=i).astype(str)), \
                datacolumn=datacolumn)
    msmd.close()
