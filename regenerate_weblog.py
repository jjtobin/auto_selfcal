import numpy as np
from scipy import stats
import glob
import sys
sys.path.append("./")
from selfcal_helpers import *
from weblog_creation import *

import pickle
with open('selfcal_library.pickle', 'rb') as handle:
    selfcal_library=pickle.load(handle)

with open('selfcal_plan.pickle', 'rb') as handle:
    selfcal_plan=pickle.load(handle)

generate_weblog(selfcal_library,selfcal_plan,directory='weblog')

# For simplicity, instead of redoing all of the weblog code, create a new selfcal_library dictionary where all of the sub-fields exist at the
# same level as the main field so that they all get their own entry in the weblog, in addition to the entry for the main field.
for target in selfcal_library:
    new_selfcal_library = {}
    new_selfcal_plan = {}
    for band in selfcal_library[target].keys():
        if selfcal_library[target][band]['obstype'] == 'mosaic':
            for fid in selfcal_library[target][band]['sub-fields']:
                if target+'_field_'+str(fid) not in new_selfcal_library:
                    new_selfcal_library[target+'_field_'+str(fid)] = {}
                    new_selfcal_plan[target+'_field_'+str(fid)] = {}
                new_selfcal_library[target+'_field_'+str(fid)][band] = selfcal_library[target][band][fid]
                new_selfcal_plan[target+'_field_'+str(fid)][band] = selfcal_plan[target][band]

    if len(new_selfcal_library) > 0:
        generate_weblog(new_selfcal_library,new_selfcal_plan,directory='weblog/'+target+'_field-by-field')


