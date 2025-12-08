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


vislist = [] # Edit manually, or leave and let auto_selfcal automatically detect.

auto_selfcal(vislist, parallel=parallel)

