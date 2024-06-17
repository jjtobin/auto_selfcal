import casatasks
import sys

def end_program():
    print('This version of auto_selfcal requires CASA 6.5.3 or higher to run. Please update your CASA version and try again.')
    sys.exit(0)

casaversion=casatasks.version()
if casaversion[0]>=6:
   if  casaversion[1]>=5:
      if casaversion[1]==5 and casaversion[2]<3:
         end_program()
   else:
      end_program()
else:
   end_program()


from .auto_selfcal import auto_selfcal
from .regenerate_weblog import regenerate_weblog
