from casatasks import flagmanager, applycal, clearcal
from .prepare_selfcal import plan_selfcal_per_solint
import numpy as np
import copy
import os

from .selfcal_helpers import sanitize_string

def prepare_cocal(selfcal_library, selfcal_plan, inf_EB_gaincal_combine, inf_EB_gaintype):
    for target in selfcal_library:
        for band in selfcal_library[target]:
            for vis in selfcal_library[target][band]['vislist']:
                if not os.path.exists(vis+'.flagversions/flags.fb_selfcal_starting_flags'):
                    flagmanager(vis=vis,mode='save',versionname='fb_selfcal_starting_flags')
                else:
                    flagmanager(vis=vis,mode='restore',versionname='fb_selfcal_starting_flags')
    
    ##
    ## For sources that self-calibration failed, try to use the inf_EB and the inf solutions from the sources that
    ## were successful.
    
    for target in selfcal_library.keys():
        for band in selfcal_library[target].keys():
            print(target, selfcal_library[target][band]["final_solint"])
    
    inf_EB_fields = {}
    inf_fields = {}
    fallback_fields = {}
    calibrators = {}
    
    # Collect the potential targets and calibrators
    for target in selfcal_library:
        for band in selfcal_library[target]:
            if band not in inf_EB_fields:
                # Initialize the lists for this band.
                inf_EB_fields[band] = []
                inf_fields[band] = []
                fallback_fields[band] = []
        
            if selfcal_library[target][band]['SC_success'] and 'fb' not in selfcal_library[target][band]['final_solint']:
                inf_EB_fields[band].append(target)
                if selfcal_library[target][band]['final_solint'] != 'inf_EB' or \
                        (selfcal_library[target][band]['final_solint'] == 'inf_EB' and 
                         'inf' not in selfcal_plan[target][band]['solints']):
                    inf_fields[band].append(target)
                else:
                    fallback_fields[band].append(target)
            else:
                fallback_fields[band].append(target)
        
                
    if len(fallback_fields[band]) > 0:
        for target in selfcal_library:
            for band in selfcal_library[target]:
                # Update the relevant lists if we are going to do a fallback mode.
                selfcal_plan[target][band]['solints'] += ["inf_EB_fb","inf_fb1","inf_fb2","inf_fb3"]
                selfcal_plan[target][band]['solmode'] += ["p","p","p","p"]
                selfcal_plan[target][band]['solint_interval'] += ["inf","inf","inf","inf"]
                
        plan_selfcal_per_solint(selfcal_library, 
                                selfcal_plan, 
                                optimize_spw_combine=False, 
                                solints=["inf_EB_fb","inf_fb1","inf_fb2","inf_fb3"])

        for target in selfcal_library:
            for band in selfcal_library[target]:
                for vis in selfcal_library[target][band]['vislist']:
                    selfcal_plan[target][band][vis]['solint_settings']["inf_EB_fb"]["preapply_solints"] = []
                    selfcal_plan[target][band][vis]['solint_settings']["inf_fb1"]["preapply_solints"] = ["inf_EB_fb"]
                    selfcal_plan[target][band][vis]['solint_settings']["inf_fb2"]["preapply_solints"] = ["inf_EB"]
                    selfcal_plan[target][band][vis]['solint_settings']["inf_fb3"]["preapply_solints"] = ["inf_EB"]

                    if selfcal_library[target][band]['SC_success']:
                       selfcal_plan[target][band][vis]['solint_settings']["inf_fb1"]["applycal_solint"] = ["inf_EB"]
                       selfcal_plan[target][band][vis]['solint_settings']["inf_fb2"]["applycal_solint"] = ["inf_EB"]
                       selfcal_plan[target][band][vis]['solint_settings']["inf_fb3"]["applycal_solint"] = ["inf_EB"]
                    else:
                       selfcal_plan[target][band][vis]['solint_settings']["inf_fb1"]["applycal_solint"] = ["inf_EB_fb"]
                       selfcal_plan[target][band][vis]['solint_settings']["inf_fb2"]["applycal_solint"] = ["inf_EB_fb"]
                       selfcal_plan[target][band][vis]['solint_settings']["inf_fb3"]["applycal_solint"] = ["inf_EB_fb"]

                    selfcal_plan[target][band][vis]['solint_settings']["inf_fb3"]["preapply_gaintable_dict"] = {}
                    for cal_target in inf_fields[band]:
                        selfcal_plan[target][band][vis]['solint_settings']["inf_fb3"]["preapply_gaintable_dict"][cal_target] = selfcal_plan[cal_target][band][vis.replace(sanitize_string(target),  sanitize_string(cal_target))]['solint_settings']['inf_EB']['accepted_gaintable']
                
                selfcal_plan[target][band]['gaincal_combine'] += [selfcal_plan[target][band]['gaincal_combine'][0], selfcal_plan[target][band]['gaincal_combine'][1], selfcal_plan[target][band]['gaincal_combine'][1], selfcal_plan[target][band]['gaincal_combine'][1]]
                selfcal_plan[target][band]['applycal_mode'] += [selfcal_plan[target][band]['applycal_mode'][0], selfcal_plan[target][band]['applycal_mode'][1], selfcal_plan[target][band]['applycal_mode'][1], selfcal_plan[target][band]['applycal_mode'][1]]
                
                calibrators[band] = [inf_EB_fields[band], inf_fields[band], inf_fields[band], inf_fields[band]]
                
                selfcal_library[target][band]["nsigma"] = np.concatenate((selfcal_library[target][band]["nsigma"],[selfcal_library[target][band]["nsigma"][0], \
                        selfcal_library[target][band]["nsigma"][1], selfcal_library[target][band]["nsigma"][1], selfcal_library[target][band]["nsigma"][1]]))
    
    print(inf_EB_fields)
    print(inf_fields)
    print(fallback_fields)
    
    ##
    ## Reset the inf_EB informational dictionaries.
    ##
    
    for target in selfcal_library:
     for band in selfcal_library[target].keys():
       # If the target had a successful inf_EB solution, no need to reset.
       if target in inf_EB_fields[band]:
           continue
    
       for vis in selfcal_library[target][band]['vislist']:
        selfcal_plan[target][band][vis]['inf_EB_gaincal_combine']=inf_EB_gaincal_combine #'scan'
        if selfcal_library[target][band]['obstype']=='mosaic':
           selfcal_plan[target][band][vis]['inf_EB_gaincal_combine']+=',field'   
        selfcal_plan[target][band][vis]['inf_EB_gaintype']=inf_EB_gaintype #G
        selfcal_plan[target][band][vis]['inf_EB_fallback_mode']='' #'scan'
    
    ## The below sets the calibrations back to what they were prior to starting the fallback mode. It should not be needed
    ## for the final version of the codue, but is used for testing.
    
    
    for target in selfcal_library:
     sani_target=sanitize_string(target)
     for band in selfcal_library[target].keys():
       if target not in fallback_fields[band]:
           continue
       if 'gaintable_final' in selfcal_library[target][band]['vislist'][0]:
          print('****************Reapplying previous solint solutions*************')
          for vis in selfcal_library[target][band]['vislist']:
             print('****************Applying '+str(selfcal_library[target][band][vis]['gaintable_final'])+' to '+target+' '+band+'*************')
             ## NOTE: should this be selfcal_starting_flags instead of fb_selfcal_starting_flags ???
             flagmanager(vis=vis,mode='delete',versionname='fb_selfcal_starting_flags_'+sani_target)
             applycal(vis=vis,\
                     gaintable=selfcal_library[target][band][vis]['gaintable_final'],\
                     interp=selfcal_library[target][band][vis]['applycal_interpolate_final'],\
                     calwt=True,spwmap=selfcal_library[target][band][vis]['spwmap_final'],\
                     applymode=selfcal_library[target][band][vis]['applycal_mode_final'],\
                     field=target,spw=selfcal_library[target][band][vis]['spws'])    
       else:            
          print('****************Removing all calibrations for '+target+' '+band+'**************')
          for vis in selfcal_library[target][band]['vislist']:
             flagmanager(vis=vis,mode='delete',versionname='fb_selfcal_starting_flags_'+sani_target)
             clearcal(vis=vis,field=target,spw=selfcal_library[target][band][vis]['spws'])

    return fallback_fields, calibrators
