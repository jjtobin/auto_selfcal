import numpy as np
from scipy import stats
import glob
import sys
#execfile('selfcal_helpers.py',globals())
sys.path.append("./")
from selfcal_helpers import *
from casampi.MPIEnvironment import MPIEnvironment 
parallel=MPIEnvironment.is_mpi_enabled

def gaincal_wrapper(selfcal_library, selfcal_plan, target, band, vis, solint, applymode, iteration, telescope, 
        gaincal_minsnr, gaincal_unflag_minsnr=5.0, minsnr_to_proceed=3.0, rerank_refants=False, unflag_only_lbants=False, unflag_only_lbants_onlyap=False, \
        calonly_max_flagged=0.0, second_iter_solmode="", unflag_fb_to_prev_solint=False, \
        refantmode="flex", mode="selfcal", calibrators="", gaincalibrator_dict={}, allow_gain_interpolation=False):

    sani_target=sanitize_string(target)
    ##
    ## Solve gain solutions per MS, target, solint, and band
    ##
    os.system('rm -rf '+sani_target+'_'+vis+'_'+band+'_'+solint+'_'+str(iteration)+'_'+selfcal_plan['solmode'][iteration]+'*.g')
    ##
    ## Set gaincal parameters depending on which iteration and whether to use combine=spw for inf_EB or not
    ## Defaults should assume combine='scan' and gaintpe='G' will fallback to combine='scan,spw' if too much flagging
    ## At some point remove the conditional for use_inf_EB_preapply, since there isn't a reason not to do it
    ##

    if selfcal_plan['solmode'][iteration] == 'p':
        if 'inf_EB' in solint:
           gaincal_spwmap=[]
           gaincal_preapply_gaintable=[]
           gaincal_interpolate=[]
           gaincal_gaintype=selfcal_plan[vis]['inf_EB_gaintype']
           gaincal_solmode=""
           selfcal_plan['gaincal_combine'][iteration]=selfcal_plan[vis]['inf_EB_gaincal_combine']+\
                   (",field" if "fb" in solint else "")
           if 'spw' in selfcal_plan[vis]['inf_EB_gaincal_combine']:
              applycal_spwmap=[selfcal_library[vis]['spwmap']]
              gaincal_spwmap=[selfcal_library[vis]['spwmap']]
           else:
              applycal_spwmap=[[]]
           applycal_interpolate=[selfcal_library['applycal_interp']]
           applycal_gaintable=[sani_target+'_'+vis+'_'+band+'_'+solint+'_'+str(iteration)+'_'+selfcal_plan['solmode'][iteration]+'.g']
        #elif selfcal_plan['solmode'][iteration]=='p':
        else:
           gaincal_spwmap=[]
           if mode == "cocal":
               if 'inf_EB' in selfcal_library[vis]:
                   #gaincal_preapply_gaintable[vis]=[sani_target+'_'+vis+'_'+band+'_inf_EB_0_p.g']
                   if calculate_inf_EB_fb_anyways or not selfcal_library[vis]["inf_EB"]["Pass"]:
                       previous_solint = "inf_EB_fb"
                   else:
                       previous_solint = "inf_EB"
               else:
                   #gaincal_preapply_gaintable[vis]=[sani_target+'_'+vis+'_'+band+'_inf_EB_fb_'+str(iteration-1)+'_p.g']
                   previous_solint = "inf_EB_fb"
           else:
               if selfcal_plan['solmode'][iteration]=='p':
                   previous_solint = "inf_EB"
               else:
                   previous_solint = selfcal_library['final_phase_solint']

           gaincal_preapply_gaintable=selfcal_library[vis][previous_solint]['gaintable']
           gaincal_interpolate=[selfcal_library['applycal_interp']]
           gaincal_gaintype='T' if applymode == "calflag" or second_iter_solmode == "" else "GSPLINE" if second_iter_solmode == "GSPLINE" else "G"
           gaincal_solmode = "" if applymode == "calflag" or second_iter_solmode == "GSPLINE" else second_iter_solmode
           gaincal_spwmap = selfcal_library[vis][previous_solint]['spwmap']
           """
           if 'spw' in selfcal_plan[vis]['inf_EB_gaincal_combine']:
              applycal_spwmap[vis]=[selfcal_library[vis]['spwmap'],selfcal_library[vis]['spwmap']]
              gaincal_spwmap[vis]=[selfcal_library[vis]['spwmap']]
           elif selfcal_plan[vis]['inf_EB_fallback_mode']=='spwmap':
              applycal_spwmap[vis]=selfcal_library[vis]['inf_EB']['spwmap'] + [selfcal_library[vis]['spwmap']]
              gaincal_spwmap[vis]=selfcal_library[vis]['inf_EB']['spwmap']
           else:
              applycal_spwmap[vis]=[[],selfcal_library[vis]['spwmap']]
              gaincal_spwmap[vis]=[]
           """
           # Revert back to applying the inf_EB solution if calculate_inf_EB_fb_anyways, i.e. we just use the inf_EB_fb solution
           # for gaincal.
           if mode == "cocal":
               if selfcal_library['final_solint'] == 'inf_EB' and calculate_inf_EB_fb_anyways:
                   previous_solint = "inf_EB"

           applycal_spwmap = selfcal_library[vis][previous_solint]['spwmap'] + [selfcal_library[vis]['spwmap']]
           applycal_interpolate=[selfcal_library['applycal_interp'],selfcal_library['applycal_interp']]
           applycal_gaintable = selfcal_library[vis][previous_solint]['gaintable'] + [sani_target+'_'+vis+'_'+band+'_'+solint+'_'+str(iteration)+'_'+selfcal_plan['solmode'][iteration]+'.g']
        selfcal_library[vis][solint]['gaintable']=applycal_gaintable
        selfcal_library[vis][solint]['iteration']=iteration+0
        selfcal_library[vis][solint]['spwmap']=applycal_spwmap
        selfcal_library[vis][solint]['applycal_mode']=selfcal_plan['applycal_mode'][iteration]+''
        selfcal_library[vis][solint]['applycal_interpolate']=applycal_interpolate
        selfcal_library[vis][solint]['gaincal_combine']=selfcal_plan['gaincal_combine'][iteration]+''
        selfcal_library[vis][solint]['solmode']=selfcal_plan['solmode'][iteration]+''
        for fid in np.intersect1d(selfcal_library['sub-fields-to-selfcal'],list(selfcal_library['sub-fields-fid_map'][vis].keys())):
            selfcal_library[fid][vis][solint]['gaintable']=applycal_gaintable
            selfcal_library[fid][vis][solint]['iteration']=iteration+0
            selfcal_library[fid][vis][solint]['spwmap']=applycal_spwmap
            selfcal_library[fid][vis][solint]['applycal_mode']=selfcal_plan['applycal_mode'][iteration]+''
            selfcal_library[fid][vis][solint]['applycal_interpolate']=applycal_interpolate
            selfcal_library[fid][vis][solint]['gaincal_combine']=selfcal_plan['gaincal_combine'][iteration]+''
            selfcal_library[fid][vis][solint]['solmode']=selfcal_plan['solmode'][iteration]+''

        fallback=''
        if selfcal_plan['solmode'][iteration] == 'ap':
           solnorm=True
        else:
           solnorm=False

        if gaincal_gaintype == "GSPLINE":
            splinetime = solint.replace('_EB','').replace('_ap','')
            if splinetime == "inf":
                splinetime = selfcal_library["Median_scan_time"]
            else:
                splinetime = float(splinetime[0:-1])

        if mode == "cocal":
            # Check which targets are acceptable to use as calibrators.
            targets = calibrators[band][iteration - len(selfcal_plan['solints'])]

            include_targets, include_scans = triage_calibrators(vis, target, targets)
        else:
            include_targets = str(selfcal_library['sub-fields-fid_map'][vis][0])
            include_scans = ""

        if solint == "scan_inf":
            if len(gaincalibrator_dict[vis]) > 0:
                scans = []
                intents = []
                times = []
                for t in gaincalibrator_dict[vis].keys():
                    scans += [gaincalibrator_dict[vis][t]["scans"]]
                    intents += [np.repeat(gaincalibrator_dict[vis][t]["intent"],gaincalibrator_dict[vis][t]["scans"].size)]
                    times += [gaincalibrator_dict[vis][t]["times"]]
                
                times = np.concatenate(times)
                order = np.argsort(times)
                times = times[order]
                
                scans = np.concatenate(scans)[order]
                intents = np.concatenate(intents)[order]

                is_gaincalibrator = intents == "phase"
                scans = scans[is_gaincalibrator]

                msmd.open(vis)
                include_scans = []
                for iscan in range(scans.size-1):
                    scan_group = np.intersect1d(msmd.scansforfield(target), 
                            np.array(list(range(scans[iscan]+1, scans[iscan+1])))).astype(str)
                    if scan_group.size > 0:
                        include_scans.append(",".join(scan_group))
                msmd.close()
            elif guess_scan_combine:
                msmd.open(vis)
                
                scans = msmd.scansforfield(target)

                include_scans = []
                for iscan in range(scans.size):
                    if len(include_scans) > 0:
                        if str(scans[iscan]) in include_scans[-1]:
                            continue

                    scan_group = str(scans[iscan])

                    if iscan < scans.size-1:
                        if msmd.fieldsforscan(scans[iscan+1]).size < msmd.fieldsforscan(scans[iscan]).size/3:
                            scan_group += ","+str(scans[iscan+1])

                    include_scans.append(scan_group)

                msmd.close()
            else:
                msmd.open(vis)
                include_scans = [str(scan) for scan in msmd.scansforfield(target)]
                msmd.close()
        else:
            include_scans = [include_scans]

        if mode == "cocal" and preapply_targets_own_inf_EB and "inf_fb" in solint and "inf" in selfcal_plan['solints']:
            ##
            ## If we want to pre-apply inf_EB solution from each calibrator to itself, all we do is combine all of thier
            ## individual inf tables, as these were pre-calculated in that way.
            ##
            destination_table = sani_target+'_'+vis+'_'+band+'_'+solint+'_'+str(iteration)+'_'+selfcal_plan['solmode'][iteration]+'.g'
            for t in include_targets.split(","):
                if os.path.exists(sanitize_string(t)+'_'+vis+'_'+band+'_'+solint.replace('_fb1','').replace('_fb2','').replace('_fb3','')+'_'+str(1)+'_'+selfcal_plan['solmode'][iteration]+\
                        '.pre-pass.g'):
                    table_name = sanitize_string(t)+'_'+vis+'_'+band+'_'+solint.replace('_fb1','').replace('_fb2','').replace('_fb3','')+'_'+str(1)+'_'+selfcal_plan['solmode'][iteration]+\
                            '.pre-pass.g'
                else:
                    table_name = sanitize_string(t)+'_'+vis+'_'+band+'_'+solint.replace('_fb1','').replace('_fb2','').replace('_fb3','')+'_'+str(1)+'_'+selfcal_plan['solmode'][iteration]+'.g'
                #t_final_solint = selfcal_library[t][band]["final_phase_solint"]
                #t_iteration = selfcal_library[t][band][vislist[0]][t_final_solint]["iteration"]
                #table_name = sanitize_string(t)+'_'+vis+'_'+band+'_'+t_final_solint+'_'+str(t_iteration)+'_'+solmode[band][iteration]+'.g'

                rerefant(vis, table_name, caltable="tmp0.g", refantmode="strict", refant=selfcal_library[vis]['refant'])

                tb.open("tmp0.g")
                if not os.path.exists("tmp1.g"):
                    tb.copy("tmp1.g", deep=True)
                else:
                    tb.copyrows("tmp1.g")
                tb.close()

                os.system("rm -rf tmp0.g")

            tb.open("tmp1.g")
            subt = tb.query("OBSERVATION_ID==0", sortlist="TIME,ANTENNA1")
            copyt = subt.copy(destination_table, deep=True)
            tb.close()
            subt.close()
            copyt.close()

            os.system("rm -rf tmp1.g")

            # Remove all of the scans that failed the triage above.
            tb.open(destination_table, nomodify=False)
            scans = tb.getcol("SCAN_NUMBER")

            bad_scans = np.repeat(True, scans.size)
            for scan in include_scans[0].split(","):
                bad_scans[scans == int(scan)] = False

            tb.removerows(rownrs=np.where(bad_scans)[0])
            tb.flush()
            tb.close()
        else:
            # Fields that don't have any mask in the primary beam should be removed from consideration, as their models are likely bad.
            if selfcal_library['obstype'] == 'mosaic':
                msmd.open(vis)
                include_targets = []
                remove = []
                for incl_scan in include_scans:
                    scan_targets = []
                    for fid in [selfcal_library['sub-fields-fid_map'][vis][fid] for fid in \
                            np.intersect1d(selfcal_library['sub-fields-to-gaincal'],list(selfcal_library['sub-fields-fid_map'][vis].keys()))] if incl_scan == '' else \
                            np.intersect1d(msmd.fieldsforscans(np.array(incl_scan.split(",")).astype(int)), \
                            [selfcal_library['sub-fields-fid_map'][vis][fid] for fid in \
                            numpy.intersect1d(selfcal_library['sub-fields-to-gaincal'],list(selfcal_library['sub-fields-fid_map'][vis].keys()))]):
                        # Note: because of the msmd above getting actual fids from the MS, we just need to append fid below.
                        scan_targets.append(fid)

                    if len(scan_targets) > 0:
                        include_targets.append(','.join(np.array(scan_targets).astype(str)))
                    else:
                        remove.append(incl_scan)

                for incl_scan in remove:
                    include_scans.remove(incl_scan)

                msmd.close()
            else:
                include_targets = [include_targets]

            selfcal_library[vis][solint]["include_scans"] = include_scans
            selfcal_library[vis][solint]["include_targets"] = include_targets

            selfcal_library[vis][solint]['gaincal_return'] = []
            for incl_scans, incl_targets in zip(include_scans, include_targets):
                if 'inf_EB' in solint:
                   if selfcal_library['spws_set'][vis].ndim == 1:
                      nspw_sets=1
                   else:
                      nspw_sets=selfcal_library['spws_set'][vis].shape[0]
                else: #only necessary to loop over gain cal when in inf_EB to avoid inf_EB solving for all spws
                   nspw_sets=1
                for i in range(nspw_sets):  # run gaincal on each spw set to handle spectral scans
                   if 'inf_EB' in solint:
                      if nspw_sets == 1 and selfcal_library['spws_set'][vis].ndim == 1:
                         spwselect=','.join(str(spw) for spw in selfcal_library['spws_set'][vis].tolist())
                      else:
                         spwselect=','.join(str(spw) for spw in selfcal_library['spws_set'][vis][i].tolist())
                   else:
                      spwselect=selfcal_library[vis]['spws']
                   gaincal_return_tmp = gaincal(vis=vis,\
                     caltable=sani_target+'_'+vis+'_'+band+'_'+solint+'_'+str(iteration)+'_'+selfcal_plan['solmode'][iteration]+'.g',\
                     gaintype=gaincal_gaintype, spw=spwselect,
                     refant=selfcal_library[vis]['refant'], calmode=selfcal_plan['solmode'][iteration], solnorm=solnorm if applymode=="calflag" else False,
                     solint=solint.replace('_EB','').replace('_ap','').replace('scan_','').replace('_fb1','').replace('_fb2','').replace('_fb3',''),minsnr=gaincal_minsnr if applymode == 'calflag' else max(gaincal_minsnr,gaincal_unflag_minsnr), minblperant=4,combine=selfcal_plan['gaincal_combine'][iteration],
                     field=incl_targets,scan=incl_scans,gaintable=gaincal_preapply_gaintable,spwmap=gaincal_spwmap,uvrange=selfcal_library['uvrange'],
                     interp=gaincal_interpolate, solmode=gaincal_solmode, refantmode='flex', append=os.path.exists(sani_target+'_'+vis+'_'+band+'_'+solint+'_'+str(iteration)+'_'+selfcal_plan['solmode'][iteration]+'.g'))
                   #
                   selfcal_library[vis][solint]['gaincal_return'].append(gaincal_return_tmp)
                   if 'inf_EB' not in solint:
                      break
    else:
        selfcal_library[vis][solint]['gaincal_return'] = []
        for fid in np.intersect1d(selfcal_library['sub-fields-to-selfcal'],list(selfcal_library['sub-fields-fid_map'][vis].keys())):
            gaincal_spwmap=[]
            gaincal_preapply_gaintable=selfcal_library[fid][vis][selfcal_library[fid]['final_phase_solint']]['gaintable']
            gaincal_interpolate=[selfcal_library['applycal_interp']]*len(gaincal_preapply_gaintable)
            gaincal_gaintype='T' if applymode == "calflag" or second_iter_solmode == "" else "GSPLINE" if second_iter_solmode == "GSPLINE" else "G"
            gaincal_solmode = "" if applymode == "calflag" or second_iter_solmode == "GSPLINE" else second_iter_solmode
            if 'spw' in selfcal_plan[vis]['inf_EB_gaincal_combine']:
               applycal_spwmap=[selfcal_library[fid][vis]['spwmap'],selfcal_library[fid][vis]['spwmap'],selfcal_library[fid][vis]['spwmap']]
               gaincal_spwmap=[selfcal_library[fid][vis]['spwmap'],selfcal_library[fid][vis]['spwmap']]
            elif selfcal_plan[vis]['inf_EB_fallback_mode']=='spwmap':
               applycal_spwmap=selfcal_library[fid][vis]['inf_EB']['spwmap'] + [selfcal_library[fid][vis]['spwmap'],selfcal_library[fid][vis]['spwmap']]
               gaincal_spwmap=selfcal_library[fid][vis]['inf_EB']['spwmap'] + [selfcal_library[fid][vis]['spwmap']]
            else:
               applycal_spwmap=[[],selfcal_library[fid][vis]['spwmap'],selfcal_library[fid][vis]['spwmap']]
               gaincal_spwmap=[[],selfcal_library[fid][vis]['spwmap']]
            applycal_interpolate=[selfcal_library['applycal_interp']]*len(gaincal_preapply_gaintable)+['linearPD']
            applycal_gaintable=selfcal_library[fid][vis][selfcal_library[fid]['final_phase_solint']]['gaintable']+[sani_target+'_'+vis+'_'+band+'_'+solint+'_'+str(iteration)+'_ap.g']

            selfcal_library[vis][solint]['gaintable']=applycal_gaintable
            selfcal_library[vis][solint]['iteration']=iteration+0
            selfcal_library[vis][solint]['spwmap']=applycal_spwmap
            selfcal_library[vis][solint]['applycal_mode']=selfcal_plan['applycal_mode'][iteration]+''
            selfcal_library[vis][solint]['applycal_interpolate']=applycal_interpolate
            selfcal_library[vis][solint]['gaincal_combine']=selfcal_plan['gaincal_combine'][iteration]+''
            selfcal_library[vis][solint]['solmode']=selfcal_plan['solmode'][iteration]+''
            selfcal_library[fid][vis][solint]['gaintable']=applycal_gaintable
            selfcal_library[fid][vis][solint]['iteration']=iteration+0
            selfcal_library[fid][vis][solint]['spwmap']=applycal_spwmap
            selfcal_library[fid][vis][solint]['applycal_mode']=selfcal_plan['applycal_mode'][iteration]+''
            selfcal_library[fid][vis][solint]['applycal_interpolate']=applycal_interpolate
            selfcal_library[fid][vis][solint]['gaincal_combine']=selfcal_plan['gaincal_combine'][iteration]+''
            selfcal_library[fid][vis][solint]['solmode']=selfcal_plan['solmode'][iteration]+''

            fallback=''
            if selfcal_plan['solmode'][iteration] == 'ap':
               solnorm=True
            else:
               solnorm=False

            if gaincal_gaintype == "GSPLINE":
                splinetime = solint.replace('_EB','').replace('_ap','')
                if splinetime == "inf":
                    splinetime = selfcal_library[fid]["Median_scan_time"]
                else:
                    splinetime = float(splinetime[0:-1])

            gaincal_return_tmp = gaincal(vis=vis,\
                 #caltable=sani_target+'_'+vis+'_'+band+'_'+solint+'_'+str(iteration)+'_'+selfcal_plan['solmode'][iteration]+'.g',\
                 caltable="temp.g",\
                 gaintype=gaincal_gaintype, spw=selfcal_library[fid][vis]['spws'],
                 refant=selfcal_library[vis]['refant'], calmode=selfcal_plan['solmode'][iteration], solnorm=solnorm if applymode=="calflag" else False,
                 solint=solint.replace('_EB','').replace('_ap','').replace('scan_',''),minsnr=gaincal_minsnr if applymode == 'calflag' else max(gaincal_minsnr,gaincal_unflag_minsnr), minblperant=4,combine=selfcal_plan['gaincal_combine'][iteration],
                 field=str(selfcal_library['sub-fields-fid_map'][vis][fid]),gaintable=gaincal_preapply_gaintable,spwmap=gaincal_spwmap,uvrange=selfcal_library['uvrange'],
                 #interp=gaincal_interpolate[vis], solmode=gaincal_solmode, append=os.path.exists(sani_target+'_'+vis+'_'+band+'_'+
                 #solint+'_'+str(iteration)+'_'+selfcal_plan['solmode'][iteration]+'.g'))
                 interp=gaincal_interpolate, solmode=gaincal_solmode, append=os.path.exists('temp.g'), refantmode='flex')
            selfcal_library[vis][solint]['gaincal_return'].append(gaincal_return_tmp)

        tb.open("temp.g")
        subt = tb.query("OBSERVATION_ID==0", sortlist="TIME,ANTENNA1")
        tb.close()

        subt.copy(sani_target+'_'+vis+'_'+band+'_'+solint+'_'+str(iteration)+'_'+selfcal_plan['solmode'][iteration]+'.g', deep=True)
        subt.close()

        os.system("rm -rf temp.g")

    if rerank_refants:
        selfcal_library[vis]["refant"] = rank_refants(vis, caltable=sani_target+'_'+vis+'_'+band+'_'+solint+'_'+str(iteration)+'_'+selfcal_plan['solmode'][iteration]+'.g')

        # If we are falling back to a previous solution interval on the unflagging, we need to make sure all tracks use a common 
        # reference antenna.
        if unflag_fb_to_prev_solint:
            for it, sint in enumerate(selfcal_plan['solints'][0:iteration+1]):
                if not os.path.exists(sani_target+'_'+vis+'_'+band+'_'+sint+'_'+str(it)+'_'+selfcal_plan['solmode'][it]+'.g'):
                    continue

                # If a previous iteration went through the unflagging routine, it is possible that some antennas fell back to
                # a previous solint. In that case, rerefant will flag those antennas because they can't be re-referenced with
                # a different time interval. So to be safe, we go back to the pre-pass solutions and then re-run the passing.
                # We could probably check more carefully whether this is the case to avoid having to do this... but the 
                # computing time isn't significant so it's easy just to run through again.
                if os.path.exists(sani_target+'_'+vis+'_'+band+'_'+sint+'_'+str(it)+'_'+selfcal_plan['solmode'][it]+'.pre-pass.g'):
                    rerefant(vis, sani_target+'_'+vis+'_'+band+'_'+sint+'_'+str(it)+'_'+selfcal_plan['solmode'][it]+'.pre-pass.g', \
                            refant=selfcal_library[vis]["refant"], refantmode=refantmode if 'inf_EB' not in sint else 'flex')

                    os.system("rm -rf "+sani_target+'_'+vis+'_'+band+'_'+sint+'_'+str(it)+'_'+selfcal_plan['solmode'][it]+'.g')
                    os.system("cp -r "+sani_target+'_'+vis+'_'+band+'_'+sint+'_'+str(it)+'_'+selfcal_plan['solmode'][it]+'.pre-pass.g '+\
                            sani_target+'_'+vis+'_'+band+'_'+sint+'_'+str(it)+'_'+selfcal_plan['solmode'][it]+'.g')

                    if sint == "inf_EB" and len(selfcal_library[vis][sint]["spwmap"][0]) > 0:
                        unflag_spwmap = selfcal_library[vis][sint]["spwmap"][0]
                    else:
                        unflag_spwmap = []

                    unflag_failed_antennas(vis, sani_target+'_'+vis+'_'+band+'_'+sint+'_'+str(it)+'_'+\
                            selfcal_plan['solmode'][it]+'.g', selfcal_library[vis][sint]['gaincal_return'], \
                            flagged_fraction=0.25, solnorm=solnorm, \
                            only_long_baselines=selfcal_plan['solmode'][it]=="ap" if unflag_only_lbants and \
                            unflag_only_lbants_onlyap else unflag_only_lbants, calonly_max_flagged=calonly_max_flagged, \
                            spwmap=unflag_spwmap, fb_to_prev_solint=unflag_fb_to_prev_solint, solints=selfcal_plan['solints'], iteration=it)
                else:
                    rerefant(vis, sani_target+'_'+vis+'_'+band+'_'+sint+'_'+str(it)+'_'+selfcal_plan['solmode'][it]+'.g', \
                            refant=selfcal_library[vis]["refant"], refantmode=refantmode if 'inf_EB' not in sint else 'flex')
        else:
            os.system("cp -r "+sani_target+'_'+vis+'_'+band+'_'+solint+'_'+str(iteration)+'_'+selfcal_plan['solmode'][iteration]+'.g '+\
                    sani_target+'_'+vis+'_'+band+'_'+solint+'_'+str(iteration)+'_'+selfcal_plan['solmode'][iteration]+'.pre-rerefant.g')
            rerefant(vis, sani_target+'_'+vis+'_'+band+'_'+solint+'_'+str(iteration)+'_'+selfcal_plan['solmode'][iteration]+'.g', \
                    refant=selfcal_library[vis]["refant"], refantmode=refantmode if 'inf_EB' not in solint else 'flex')

    ##
    ## default is to run without combine=spw for inf_EB, here we explicitly run a test inf_EB with combine='scan,spw' to determine
    ## the number of flagged antennas when combine='spw' then determine if it needs spwmapping or to use the gaintable with spwcombine.
    ##
    if 'inf_EB' in solint and fallback=='':
       os.system('rm -rf test_inf_EB.g')
       test_gaincal_combine='scan,spw'
       if selfcal_library['obstype']=='mosaic' or mode=="cocal":
          test_gaincal_combine+=',field'   
       test_gaincal_return = {'G':[], 'T':[]}
       for gaintype in np.unique([gaincal_gaintype,'T']):
           for i in range(selfcal_library['spws_set'][vis].shape[0]):  # run gaincal on each spw set to handle spectral scans
              if nspw_sets == 1 and selfcal_library['spws_set'][vis].ndim == 1:
                 spwselect=','.join(str(spw) for spw in selfcal_library['spws_set'][vis].tolist())
              else:
                 spwselect=','.join(str(spw) for spw in selfcal_library['spws_set'][vis][i].tolist())

              test_gaincal_return[gaintype] += [gaincal(vis=vis,\
                caltable='test_inf_EB_'+gaintype+'.g',\
                gaintype=gaintype, spw=spwselect,
                refant=selfcal_library[vis]['refant'], calmode='p', 
                solint=solint.replace('_EB','').replace('_ap','').replace('_fb1','').replace('_fb2','').replace('_fb3',''),minsnr=gaincal_minsnr if applymode == "calflag" else max(gaincal_minsnr,gaincal_unflag_minsnr), minblperant=4,combine=test_gaincal_combine,
                field=include_targets[0],scan=include_scans[0],gaintable='',spwmap=[],uvrange=selfcal_library['uvrange'], refantmode=refantmode,append=os.path.exists('test_inf_EB_'+gaintype+'.g'))]
       spwlist=selfcal_library[vis]['spws'].split(',')
       fallback,map_index,spwmap,applycal_spwmap_inf_EB=analyze_inf_EB_flagging(selfcal_library,band,spwlist,sani_target+'_'+vis+'_'+band+'_'+solint+'_'+str(iteration)+'_'+selfcal_plan['solmode'][iteration]+'.g',vis,target,'test_inf_EB_'+gaincal_gaintype+'.g',selfcal_library['spectral_scan'],telescope, selfcal_plan['solint_snr_per_spw'], minsnr_to_proceed,'test_inf_EB_T.g' if gaincal_gaintype=='G' else None)

       selfcal_plan[vis]['inf_EB_fallback_mode']=fallback+''
       print(solint,fallback,applycal_spwmap_inf_EB)
       if fallback != '':
          if 'combinespw' in fallback:
             gaincal_spwmap=[selfcal_library[vis]['spwmap']]
             selfcal_plan['gaincal_combine'][iteration]='scan,spw'
             selfcal_plan[vis]['inf_EB_gaincal_combine']='scan,spw'
             applycal_spwmap=[selfcal_library[vis]['spwmap']]
             os.system('rm -rf           '+sani_target+'_'+vis+'_'+band+'_'+solint+'_'+str(iteration)+'_'+selfcal_plan['solmode'][iteration]+'.g')
             for gaintype in np.unique([gaincal_gaintype,'T']):
                os.system('cp -r test_inf_EB_'+gaintype+'.g '+sani_target+'_'+vis+'_'+band+'_'+solint+'_'+str(iteration)+'_'+selfcal_plan['solmode'][iteration]+'.gaintype'+gaintype+'.g')
             if fallback == 'combinespw':
                 gaincal_gaintype = 'G'
             else:
                 gaincal_gaintype = 'T'
             os.system('mv test_inf_EB_'+gaincal_gaintype+'.g '+sani_target+'_'+vis+'_'+band+'_'+solint+'_'+str(iteration)+'_'+selfcal_plan['solmode'][iteration]+'.g')
             selfcal_library[vis][solint]['gaincal_return'] = test_gaincal_return[gaincal_gaintype]

          if fallback =='spwmap':
             gaincal_spwmap=applycal_spwmap_inf_EB
             selfcal_plan[vis]['inf_EB_gaincal_combine']='scan'
             selfcal_plan['gaincal_combine'][iteration]='scan'
             applycal_spwmap=[applycal_spwmap_inf_EB]

          # Update the appropriate selfcal_library entries.
          selfcal_library[vis][solint]['spwmap']=applycal_spwmap
          selfcal_library[vis][solint]['gaincal_combine']=selfcal_plan['gaincal_combine'][iteration]+''
          for fid in np.intersect1d(selfcal_library['sub-fields-to-selfcal'],list(selfcal_library['sub-fields-fid_map'][vis].keys())):
              selfcal_library[fid][vis][solint]['spwmap']=applycal_spwmap
              selfcal_library[fid][vis][solint]['gaincal_combine']=selfcal_plan['gaincal_combine'][iteration]+''

       os.system('rm -rf test_inf_EB_*.g')               

    selfcal_library[vis][solint]['fallback']=fallback+''
    for fid in np.intersect1d(selfcal_library['sub-fields-to-selfcal'],list(selfcal_library['sub-fields-fid_map'][vis].keys())):
        selfcal_library[fid][vis][solint]['fallback']=fallback+''

    # If iteration two, try restricting to just the antennas with enough unflagged data.
    # Should we also restrict to just long baseline antennas?
    if applymode == "calonly":
        # Make a copy of the caltable before unflagging, for reference.
        os.system("cp -r "+sani_target+'_'+vis+'_'+band+'_'+solint+'_'+str(iteration)+'_'+\
                selfcal_plan['solmode'][iteration]+'.g '+sani_target+'_'+vis+'_'+band+'_'+solint+'_'+str(iteration)+'_'+\
                selfcal_plan['solmode'][iteration]+'.pre-pass.g')

        if solint == "inf_EB" and len(applycal_spwmap) > 0:
            unflag_spwmap = applycal_spwmap[0]
        else:
            unflag_spwmap = []

        selfcal_library[vis][solint]['unflag_spwmap'] = unflag_spwmap
        selfcal_library[vis][solint]['unflagged_lbs'] = True

        unflag_failed_antennas(vis, sani_target+'_'+vis+'_'+band+'_'+solint+'_'+str(iteration)+'_'+\
                selfcal_plan['solmode'][iteration]+'.g', selfcal_library[vis][solint]['gaincal_return'], flagged_fraction=0.25, solnorm=solnorm, \
                only_long_baselines=selfcal_plan['solmode'][iteration]=="ap" if unflag_only_lbants and unflag_only_lbants_onlyap else \
                unflag_only_lbants, calonly_max_flagged=calonly_max_flagged, spwmap=unflag_spwmap, \
                fb_to_prev_solint=unflag_fb_to_prev_solint, solints=selfcal_plan['solints'], iteration=iteration)

    # Do some post-gaincal cleanup for mosaics.
    if selfcal_library['obstype'] == 'mosaic' or mode == "cocal":
        os.system("cp -r "+sani_target+'_'+vis+'_'+band+'_'+solint+'_'+str(iteration)+'_'+selfcal_plan['solmode'][iteration]+'.g '+\
                sani_target+'_'+vis+'_'+band+'_'+solint+'_'+str(iteration)+'_'+selfcal_plan['solmode'][iteration]+'.pre-drop.g')
        tb.open(sani_target+'_'+vis+'_'+band+'_'+solint+'_'+str(iteration)+'_'+selfcal_plan['solmode'][iteration]+'.g', nomodify=False)
        antennas = tb.getcol("ANTENNA1")
        fields = tb.getcol("FIELD_ID")
        scans = tb.getcol("SCAN_NUMBER")
        flags = tb.getcol("FLAG")

        """
        # Flag any solutions whose S/N ratio is too small.
        snr = tb.getcol("SNR")
        flags[snr < 5.] = True
        """

        if (solint != "inf_EB" and not allow_gain_interpolation) or (allow_gain_interpolation and "inf" not in solint):
            # If a given field has > 25% of its solutions flagged then just flag the whole field because it will have too much 
            # interpolation.
            if solint == "scan_inf":
                max_n_solutions = max([(scans == scan).sum() for scan in np.unique(scans)])
                for scan in np.unique(scans):
                    scan_n_solutions = (flags[0,0,scans == scan] == False).sum()
                    if scan_n_solutions < 0.75 * max_n_solutions:
                        flags[:,:,scans == scan] = True
            else:
                n_all_flagged = np.sum([np.all(flags[:,:,antennas == ant]) for ant in np.unique(antennas)])
                max_n_solutions = max([(fields == fid).sum() for fid in np.unique(fields)]) - n_all_flagged
                for fid in np.unique(fields):
                    fid_n_solutions = (flags[0,0,fields == fid] == False).sum()
                    if fid_n_solutions < 0.75 * max_n_solutions:
                        flags[:,:,fields == fid] = True

        bad = np.where(np.any(flags, axis=0)[0])[0]
        tb.removerows(rownrs=bad)
        tb.flush()
        tb.close()
