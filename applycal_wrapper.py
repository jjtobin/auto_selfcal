from casatasks import applycal, clearcal, flagmanager
import numpy as np

def applycal_wrapper(vis, target, band, solint, selfcal_library, current=lambda f: False, final=lambda f: False, 
        clear=lambda f: False, restore_flags=None):

    if restore_flags != None:
        flagmanager(vis=vis,mode='restore',versionname=restore_flags)

    for fid in [fid for fid in np.intersect1d(selfcal_library['sub-fields'],
            list(selfcal_library['sub-fields-fid_map'][vis].keys())) if current(fid)]:
        applycal(vis=vis,\
                 gaintable=selfcal_library[fid][vis][solint]['gaintable'],\
                 interp=selfcal_library[fid][vis][solint]['applycal_interpolate'], calwt=False,\
                 spwmap=selfcal_library[fid][vis][solint]['spwmap'],\
                 #applymode=applymode,field=target,spw=selfcal_library[vis]['spws'])
                 applymode='calflag',field=str(selfcal_library['sub-fields-fid_map'][vis][fid]),\
                 spw=selfcal_library[vis]['spws'])

    for fid in [fid for fid in np.intersect1d(selfcal_library['sub-fields'],
            list(selfcal_library['sub-fields-fid_map'][vis].keys())) if final(fid)]:
        print('****************Applying '+str(selfcal_library[fid][vis]['gaintable_final'])+' to '+target+' field '+\
                str(fid)+' '+band+'*************')
        applycal(vis=vis,\
                gaintable=selfcal_library[fid][vis]['gaintable_final'],\
                interp=selfcal_library[fid][vis]['applycal_interpolate_final'],\
                calwt=False,spwmap=selfcal_library[fid][vis]['spwmap_final'],\
                applymode=selfcal_library[fid][vis]['applycal_mode_final'],\
                field=str(selfcal_library['sub-fields-fid_map'][vis][fid]),\
                spw=selfcal_library[vis]['spws'])    

    for fid in [fid for fid in np.intersect1d(selfcal_library['sub-fields'],
            list(selfcal_library['sub-fields-fid_map'][vis].keys())) if clear(fid)]:
        print('****************Removing all calibrations for '+target+' '+str(fid)+' '+band+'**************')
        clearcal(vis=vis,field=str(selfcal_library['sub-fields-fid_map'][vis][fid]),\
                spw=selfcal_library[vis]['spws'])
