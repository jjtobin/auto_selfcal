import sys
sys.path.append(os.path.dirname(__file__)+"/..")
from auto_selfcal import split_calibrated_final
import glob

vislist = [] # Edit manually, or leave and let auto_selfcal automatically detect.

split_calibrated_final(vislist, overwrite=True)
