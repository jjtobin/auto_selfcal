# Mac builds of CASA lack MPI and error without this try/except
try:
   from casampi.MPIEnvironment import MPIEnvironment   
   parallel=MPIEnvironment.is_mpi_enabled
except:
   parallel=False

from .auto_selfcal import auto_selfcal
from .regenerate_weblog import regenerate_weblog
from .split_calibrated_final import split_calibrated_final
import argparse

parser = argparse.ArgumentParser(
                    prog='auto_selfcal',
                    description='Run automated self-calibration on a collection of MS files.')

parser.add_argument('-a','--action', default='run')
parser.add_argument('-e','--exit', action='store_true')

args = parser.parse_args()

if args.exit:
    import sys
    sys.exit(0)

if args.action == "run":
    ##
    ## Get list of MS files in directory
    ##
    vislist=glob.glob('*_target.ms')
    if len(vislist) == 0:
       vislist=glob.glob('*_targets.ms')   # adaptation for PL2022 output
       if len(vislist)==0:
          vislist=glob.glob('*_cont.ms')   # adaptation for PL2022 output
          if len(vislist)==0:
             if len(glob.glob("calibrated_final.ms")) > 0:
                 split_calibrated_final()
             else:
                 sys.exit('No Measurement sets found in current working directory, exiting')

    auto_selfcal(vislist, parallel=parallel)

elif args.action == "regenerate_weblog":
    regenerate_weblog()
