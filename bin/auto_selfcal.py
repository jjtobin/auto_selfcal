import sys
sys.path.append(os.path.dirname(__file__)+"/..")
from auto_selfcal import auto_selfcal
import glob

# Mac builds of CASA lack MPI and error without this try/except
try:
   from casampi.MPIEnvironment import MPIEnvironment   
   parallel=MPIEnvironment.is_mpi_enabled
except:
   parallel=False


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

