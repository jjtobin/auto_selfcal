import pickle
import numpy as np
from scipy import stats
import glob
execfile('selfcal_helpers.py', globals())
with open('selfcal_library.pickle', 'rb') as handle:
    selfcal_library = pickle.load(handle)

with open('solints.pickle', 'rb') as handle:
    solints = pickle.load(handle)

with open('bands.pickle', 'rb') as handle:
    bands = pickle.load(handle)

generate_weblog(selfcal_library, solints, bands)
