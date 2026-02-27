import numpy as np
from .selfcal_helpers import *

def gaincal_wrapper(selfcal_library, selfcal_plan, target, band, vis, solint, solint_interval, applymode, iteration, 
        gaincal_minsnr, gaincal_unflag_minsnr=5.0, minsnr_to_proceed=3.0, rerank_refants=False, unflag_only_lbants=False, unflag_only_lbants_onlyap=False, 
        calonly_max_flagged=0.0, second_iter_solmode="", unflag_fb_to_prev_solint=False, \
        refantmode="flex", mode="selfcal", calibrators="", gaincalibrator_dict={}, allow_gain_interpolation=False,spectral_solution_fraction=0.3,
        guess_scan_combine=False, do_fallback_calonly=False):
    """
    This function runs gaincal for a given target, band, and solint, and updates the selfcal_library and selfcal_plan dictionaries with the results.
    It also handles the pre-application of inf_EB solutions if necessary.
    Parameters
    ----------
    selfcal_library : dict
        The selfcal_library dictionary containing information about the self-calibration process.
    selfcal_plan : dict
        The selfcal_plan dictionary containing the planned self-calibration steps.
    target : str
        The target field for which gaincal is being run.
    band : str
        The band for which gaincal is being run.
    """
    print('in gaincal_wrapper for solint', solint)
    sani_target=sanitize_string(target)
    ##
    ## Solve gain solutions per MS, target, solint, and band
    ##

    os.system('rm -rf '+sani_target+'_'+vis+'_'+band+'_'+solint+'_'+str(iteration)+'_'+selfcal_plan['solmode'][iteration]+'*.g')

    ## Reset the gaincal return dictionaries, in case this is a repeat of the current solution interval.
    for mode in selfcal_plan[vis]['solint_settings'][solint]['modes_to_attempt']:
        selfcal_plan[vis]['solint_settings'][solint]['gaincal_return_dict'][mode] = []
    ##
    ## Set gaincal parameters depending on which iteration and whether to use combine=spw for inf_EB or not
    ## Defaults should assume combine='scan' and gaintpe='G' will fallback to combine='scan,spw' if too much flagging
    ## At some point remove the conditional for use_inf_EB_preapply, since there isn't a reason not to do it
    ##
    current_solint_index=selfcal_plan['solints'].index(solint)
    selfcal_plan[vis]['solint_settings'][solint]['computed_gaintable'] = {}
    if selfcal_plan['solmode'][iteration] == 'p':
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
            # not entirely sure why this if/else is necessary might not be with new selfcal plan
            if selfcal_plan['solmode'][iteration]=='p':
                previous_solint = "inf_EB"
            else:
                previous_solint = selfcal_library['final_phase_solint']
        gaincal_spwmap=[]
        gaincal_preapply_gaintable=[]
        gaincal_interpolate=[]
        applycal_spwmap=[]
        applycal_interpolate=[]
        applycal_gaintable=[]
        for j in range(current_solint_index):
           if selfcal_plan[vis]['solint_settings'][selfcal_plan['solints'][j]]['preapply_this_gaintable']:
              gaincal_preapply_gaintable.append(selfcal_plan[vis]['solint_settings'][selfcal_plan['solints'][j]]['accepted_gaintable'])
              gaincal_spwmap.append(selfcal_plan[vis]['solint_settings'][selfcal_plan['solints'][j]]['applycal_spwmap'])
              gaincal_interpolate.append(selfcal_plan[vis]['solint_settings'][selfcal_plan['solints'][j]]['applycal_interpolate'])
              applycal_spwmap.append(selfcal_plan[vis]['solint_settings'][selfcal_plan['solints'][j]]['applycal_spwmap'])
              applycal_interpolate.append(selfcal_plan[vis]['solint_settings'][selfcal_plan['solints'][j]]['applycal_interpolate'])
              applycal_gaintable.append(selfcal_plan[vis]['solint_settings'][selfcal_plan['solints'][j]]['accepted_gaintable'])


        if solint != 'inf_EB':
            gaincal_solmode = "" if not do_fallback_calonly or second_iter_solmode == "GSPLINE" else second_iter_solmode

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

        fallback=''
        if selfcal_plan['solmode'][iteration] == 'ap':
           solnorm=True
        else:
           solnorm=False

        if mode == "cocal":
            # Check which targets are acceptable to use as calibrators.
            targets = calibrators[band][iteration - len(selfcal_plan['solints'])]

            include_targets, include_scans = triage_calibrators(vis, target, targets)
        else:
            #include_targets = str(selfcal_library['sub-fields-fid_map'][vis][0])
            include_targets = selfcal_library['bands_for_targets'][vis]['field_str']
            include_scans = ""

        if solint == "scan_inf":
            if len(gaincalibrator_dict[vis]) > 0:
                print("Determining scan_inf from calibrator scans in full MS")
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
                if selfcal_library['telescope'] == 'ALMA' or selfcal_library['telescope'] == 'ACA':
                    scan_ids_for_target = msmd.scansforfield(target)
                elif 'VLA' in selfcal_library['telescope']:
                    scan_ids_for_target=np.array([],dtype=int)
                    for fid in selfcal_library['sub-fields']:
                        if fid in selfcal_library['sub-fields-fid_map'][vis].keys():
                            field_id=selfcal_library['sub-fields-fid_map'][vis][fid]
                            scan_ids_for_target=np.append(scan_ids_for_target,msmd.scansforfield(field_id))

                    scan_ids_for_target.sort() # sort scans since they will be out of order
                include_scans = []
                for iscan in range(scans.size-1):
                    scan_group = np.intersect1d(scan_ids_for_target, 
                            np.array(list(range(scans[iscan]+1, scans[iscan+1])))).astype(str)
                    if scan_group.size > 0:
                        include_scans.append(",".join(scan_group))
                # PIPE-2741: if there are any scans before the first gain calibrator scan or after the last gain calibrator scan, catch them.
                if scans.size > 0:
                    extra_scans = scan_ids_for_target[scan_ids_for_target > max(scans)]
                    if extra_scans.size > 0:
                        include_scans.append(','.join(extra_scans.astype(str)))
                    extra_scans = scan_ids_for_target[scan_ids_for_target < min(scans)]
                    if extra_scans.size > 0:
                        include_scans.append(','.join(extra_scans.astype(str)))
                msmd.close()
            elif guess_scan_combine:
                print("Determining scan_inf from guessing where the calibrator scans were")
                msmd.open(vis)
                include_scans = []

                #to guess at scan_inf combination for VLA look for breaks in the consecutive
                #scan numbers and assume that the break is due to a calibrator scan
                #Fetch scans for scan inf by collecting the field ids and running msmd.scansforfield
                if 'VLA' in selfcal_library['telescope']:
                    scans=np.array([],dtype=int)
                    for fid in selfcal_library['sub-fields']:
                        if fid in selfcal_library['sub-fields-fid_map'][vis].keys():
                            field_id=selfcal_library['sub-fields-fid_map'][vis][fid]
                            scans=np.append(scans,msmd.scansforfield(field_id))

                    scans.sort() # sort scans since they will be out of order
                    scan_group=''
                    for iscan in range(scans.size):
                        if len(include_scans) > 0:
                            if str(scans[iscan]) in include_scans[-1]:
                                continue
                        if scan_group == '':
                            scan_group = str(int(scans[iscan]))

                        if iscan < scans.size-1:
                            if scans[iscan+1] == scans[iscan]+1:
                                scan_group += ","+str(int(scans[iscan+1]))
                            else:
                                include_scans.append(scan_group)
                                scan_group=''
                        else: #write out the last scan group to include_scans
                            if scan_group != '':
                                include_scans.append(scan_group)
                                scan_group=''

                #guess scan_inf combination by getting all the scans for targets and do a simple grouping
                if selfcal_library['telescope'] == 'ALMA' or selfcal_library['telescope'] == 'ACA':
                    scans = msmd.scansforfield(target)

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
                print("Not guessing where calibration scans are and justincluding all scans")
                msmd.open(vis)
                if selfcal_library['telescope'] == 'ALMA' or selfcal_library['telescope'] == 'ACA':
                    include_scans = [str(scan) for scan in msmd.scansforfield(target)]
                elif 'VLA' in selfcal_library['telescope']:
                    scans=np.array([],dtype=int)
                    for fid in selfcal_library['sub-fields']:
                        if fid in selfcal_library['sub-fields-fid_map'][vis].keys():
                            field_id=selfcal_library['sub-fields-fid_map'][vis][fid]
                            scans=np.append(scans,msmd.scansforfield(field_id))

                    scans.sort() # sort scans since they will be out of order      
                    include_scans = [str(scan) for scan in scans]             
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
            gaincal_solmode=""
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
            print(solint,'Include scans: ', include_scans)
            print(solint,'Include targets: ', include_targets)
            print(solint,'Modes to attempt: ',selfcal_plan[vis]['solint_settings'][solint]['modes_to_attempt'])
            for incl_scans, incl_targets in zip(include_scans, include_targets):
                for mode in selfcal_plan[vis]['solint_settings'][solint]['modes_to_attempt']:
                   print(vis,solint,mode)
                   print(selfcal_plan[vis]['solint_settings'][solint]['gaincal_combine'])
                   gaincal_combine=selfcal_plan[vis]['solint_settings'][solint]['gaincal_combine'][mode]
                   filename_append=selfcal_plan[vis]['solint_settings'][solint]['filename_append'][mode]
                   if selfcal_plan[vis]['solint_settings'][solint]['gaincal_gaintype'][mode] =='':
                       if '_EB' not in solint and '_delay' not in solint:
                           gaincal_gaintype=''
                           if not do_fallback_calonly or second_iter_solmode == "":
                               gaincal_gaintype='T'
                           else:
                               gaincal_gaintype='GSPLINE'
                           if second_iter_solmode == "GSPLINE": 
                               gaincal_gaintype='T'
                           else:
                               gaincal_gaintype= 'G'
                           if gaincal_gaintype !='':
                              selfcal_plan[vis]['solint_settings'][solint]['gaincal_gaintype'][mode]=gaincal_gaintype
                               
                           #selfcal_plan[vis]['solint_settings'][solint]['gaincal_gaintype'][mode]='T' if not do_fallback_calonly or second_iter_solmode == "" else "GSPLINE" if second_iter_solmode == "GSPLINE" else "G"

                   if selfcal_plan[vis]['solint_settings'][solint]['gaincal_gaintype'][mode] == "GSPLINE":
                       splinetime = solint.replace('_EB','').replace('_ap','')
                       if splinetime == "inf":
                           splinetime = selfcal_library["Median_scan_time"]
                       else:
                           splinetime = float(splinetime[0:-1])

                   if 'spw' in gaincal_combine:
                      if selfcal_library['spws_set'][vis].ndim == 1:
                         nspw_sets=1
                      else:
                         nspw_sets=selfcal_library['spws_set'][vis].shape[0]
                   else: #only necessary to loop over gain cal when in inf_EB to avoid inf_EB solving for all spws
                      nspw_sets=1
                   for i in range(nspw_sets):  # run gaincal on each spw set to handle spectral scans
                      if 'spw' in gaincal_combine:
                         if nspw_sets == 1 and selfcal_library['spws_set'][vis].ndim == 1:
                            spwselect=','.join(str(spw) for spw in selfcal_library['spws_set'][vis].tolist())
                         else:
                            spwselect=','.join(str(spw) for spw in selfcal_library['spws_set'][vis][i].tolist())
                      else:
                         spwselect=selfcal_library[vis]['spws']
                      gaintable_name=sani_target+'_'+vis+'_'+band+'_'+solint+'_'+str(iteration)+'_'+selfcal_plan['solmode'][iteration]+'_'+filename_append+'.g'
                      print('prior to gaincal',gaintable_name, mode)
                      if mode != 'per_bb':      
                         gcdict=call_gaincal(vis=vis, caltable=gaintable_name, gaintype=selfcal_plan[vis]['solint_settings'][solint]['gaincal_gaintype'][mode], spw=spwselect,
                                refant=selfcal_library[vis]['refant'], calmode=selfcal_plan['solmode'][iteration], solnorm=solnorm if not do_fallback_calonly else False,
                                solint=solint_interval.replace('_EB','').replace('_ap','').replace('scan_','').replace('_fb1','').replace('_fb2','').replace('_fb3',''),\
                                minsnr=gaincal_minsnr if not do_fallback_calonly else max(gaincal_minsnr,gaincal_unflag_minsnr), minblperant=4,combine=gaincal_combine,\
                                field=incl_targets,scan=incl_scans,gaintable=gaincal_preapply_gaintable,spwmap=gaincal_spwmap,uvrange=selfcal_library['uvrange'],\
                                interp=gaincal_interpolate, solmode=gaincal_solmode, refantmode='flex',\
                                append=os.path.exists(sani_target+'_'+vis+'_'+band+'_'+solint+'_'+str(iteration)+'_'+selfcal_plan['solmode'][iteration]+'_'+filename_append+'.g'))
                         selfcal_plan[vis]['solint_settings'][solint]['gaincal_return_dict'][mode].append(gcdict.copy())
                      else:
                        for baseband in selfcal_library[vis]['baseband'].keys():
                             spwselect_bb=selfcal_library[vis]['baseband'][baseband]['spwstring']
                             gcdict=call_gaincal(vis=vis, caltable=gaintable_name, gaintype=selfcal_plan[vis]['solint_settings'][solint]['gaincal_gaintype'][mode], spw=spwselect_bb,
                                  refant=selfcal_library[vis]['refant'], calmode=selfcal_plan['solmode'][iteration], solnorm=solnorm if not do_fallback_calonly else False,
                                  solint=solint_interval.replace('_EB','').replace('_ap','').replace('scan_','').replace('_fb1','').replace('_fb2','').replace('_fb3',''),\
                                  minsnr=gaincal_minsnr if not do_fallback_calonly else max(gaincal_minsnr,gaincal_unflag_minsnr), minblperant=4,combine=gaincal_combine,\
                                  field=incl_targets,scan=incl_scans,gaintable=gaincal_preapply_gaintable,spwmap=gaincal_spwmap,uvrange=selfcal_library['uvrange'],\
                                  interp=gaincal_interpolate, solmode=gaincal_solmode, refantmode='flex',\
                                  append=os.path.exists(sani_target+'_'+vis+'_'+band+'_'+solint+'_'+str(iteration)+'_'+selfcal_plan['solmode'][iteration]+'_'+filename_append+'.g'))
                             selfcal_plan[vis]['solint_settings'][solint]['gaincal_return_dict'][mode].append(gcdict.copy())
     
                   selfcal_plan[vis]['solint_settings'][solint]['computed_gaintable'][mode] = gaintable_name

                   # restricted gaincal table comparisons to only inf_EB prior to changes
                   # commenting because we want to do comparisons for other solints as well
                   #if 'inf_EB' not in solint:
                   #   break
        gaintable_prefix=sani_target+'_'+vis+'_'+band+'_'
        # assume that if there is only one mode to attempt, that it is combinespw and don't bother checking.
        if len(selfcal_plan[vis]['solint_settings'][solint]['modes_to_attempt']) >= 1 and 'delay' not in solint:
            preferred_mode,fallback,spwmap,spwmapping_for_applycal = \
                           select_best_gaincal_mode(selfcal_library,selfcal_plan,vis,gaintable_prefix,solint,spectral_solution_fraction,minsnr_to_proceed)

            if fallback=='spwmap':
                selfcal_plan[vis]['solint_settings'][solint]['spwmap_for_mode']['per_spw']=spwmapping_for_applycal.copy()

            print(preferred_mode,solint,fallback,spwmapping_for_applycal)
        if len(selfcal_plan[vis]['solint_settings'][solint]['modes_to_attempt']) >= 1 and 'delay' in solint:
             preferred_mode,fallback = \
                           select_best_delaycal_mode(selfcal_library,selfcal_plan,vis,gaintable_prefix,solint,spectral_solution_fraction,minsnr_to_proceed)
       
        elif 'delay' not in solint:
            preferred_mode='combinespw'
            fallback=''
        elif 'delay' in solint:
            preferred_mode='per_bb'
            fallback=''
        else:
            preferred_mode='combinespw'
            fallback=''
        print(preferred_mode,solint,fallback)


            
        # Update the appropriate selfcal_library entries.

        selfcal_plan[vis]['solint_settings'][solint]['final_mode']=preferred_mode+''
        selfcal_plan[vis]['solint_settings'][solint]['applycal_spwmap']=selfcal_plan[vis]['solint_settings'][solint]['spwmap_for_mode'][preferred_mode].copy()
        applycal_spwmap.append(selfcal_plan[vis]['solint_settings'][solint]['spwmap_for_mode'][preferred_mode])
        applycal_interpolate.append(selfcal_plan[vis]['solint_settings'][solint]['applycal_interpolate'])
        print(applycal_interpolate)
        #applycal_gaintable=selfcal_library[fid][vis][selfcal_library[fid]['final_phase_solint']]['gaintable']+[sani_target+'_'+vis+'_'+band+'_'+solint+'_'+str(iteration)+'_ap.g']
        applycal_gaintable.append(sani_target+'_'+vis+'_'+band+'_'+solint+'_'+str(iteration)+'_'+selfcal_plan['solmode'][iteration]+'_'+preferred_mode+'.g')
        selfcal_library[vis][solint]['final_mode']=preferred_mode+''
        selfcal_library[vis][solint]['gaintable']=applycal_gaintable.copy()
        selfcal_library[vis][solint]['iteration']=iteration+0
        selfcal_library[vis][solint]['spwmap']=applycal_spwmap.copy()
        selfcal_library[vis][solint]['applycal_mode']=selfcal_plan['applycal_mode'][iteration]+''
        selfcal_library[vis][solint]['applycal_interpolate']=applycal_interpolate.copy()
        selfcal_library[vis][solint]['solmode']=selfcal_plan['solmode'][iteration]+''
        selfcal_library[vis][solint]['gaincal_combine']=selfcal_plan[vis]['solint_settings'][solint]['gaincal_combine'][preferred_mode]+''

        # Remove per_spw and/or per_bb from subsequent solints if per_bb or combinespw are selected for a given solint
        if preferred_mode != 'per_spw':
            remove_modes(selfcal_plan,vis,current_solint_index)


        for fid in np.intersect1d(selfcal_library['sub-fields-to-selfcal'],list(selfcal_library['sub-fields-fid_map'][vis].keys())):
            selfcal_library[fid][vis][solint]['final_mode']=preferred_mode+''
            selfcal_library[fid][vis][solint]['spwmap']=applycal_spwmap.copy()
            selfcal_library[fid][vis][solint]['gaincal_combine']=selfcal_plan[vis]['solint_settings'][solint]['gaincal_combine'][preferred_mode]+''
            selfcal_library[fid][vis][solint]['gaintable']=applycal_gaintable.copy()
            selfcal_library[fid][vis][solint]['iteration']=iteration+0
            selfcal_library[fid][vis][solint]['spwmap']=applycal_spwmap.copy()
            selfcal_library[fid][vis][solint]['applycal_mode']=selfcal_plan['applycal_mode'][iteration]+''
            selfcal_library[fid][vis][solint]['applycal_interpolate']=applycal_interpolate.copy()
            selfcal_library[fid][vis][solint]['solmode']=selfcal_plan['solmode'][iteration]+''
        selfcal_plan[vis]['solint_settings'][solint]['accepted_gaintable']=sani_target+'_'+vis+'_'+band+'_'+solint+'_'+str(iteration)+'_'+selfcal_plan['solmode'][iteration]+'_'+preferred_mode+'.g'
        if 'combinespw' not in preferred_mode:   # preapply all non spwcombine gain tables
           selfcal_plan[vis]['solint_settings'][solint]['preapply_this_gaintable']=True



    else:  # this else is for ap selfcal
        os.system('rm -rf temp*.g')
        for fid in np.intersect1d(selfcal_library['sub-fields-to-selfcal'],list(selfcal_library['sub-fields-fid_map'][vis].keys())):
            gaincal_spwmap=[]
            gaincal_preapply_gaintable=[]
            gaincal_interpolate=[] 
            applycal_gaintable=[]
            applycal_interpolate=[]
            applycal_spwmap=[]
            for j in range(current_solint_index):
              if selfcal_plan[vis]['solint_settings'][selfcal_plan['solints'][j]]['preapply_this_gaintable'] and selfcal_plan[vis]['solint_settings'][selfcal_plan['solints'][j]]['solmode']=='p':
                 gaincal_preapply_gaintable.append(selfcal_plan[vis]['solint_settings'][selfcal_plan['solints'][j]]['accepted_gaintable'])
                 gaincal_spwmap.append(selfcal_plan[vis]['solint_settings'][selfcal_plan['solints'][j]]['applycal_spwmap'])
                 gaincal_interpolate.append(selfcal_plan[vis]['solint_settings'][selfcal_plan['solints'][j]]['applycal_interpolate'])
                 applycal_interpolate.append(selfcal_plan[vis]['solint_settings'][selfcal_plan['solints'][j]]['applycal_interpolate'])
                 applycal_gaintable.append(selfcal_plan[vis]['solint_settings'][selfcal_plan['solints'][j]]['accepted_gaintable'])
                 applycal_spwmap.append(selfcal_plan[vis]['solint_settings'][selfcal_plan['solints'][j]]['applycal_spwmap'])
            gaincal_solmode = "" if not do_fallback_calonly or second_iter_solmode == "GSPLINE" else second_iter_solmode

            fallback=''
            if selfcal_plan['solmode'][iteration] == 'ap':
               solnorm=True
            else:
               solnorm=False

            for mode in selfcal_plan[vis]['solint_settings'][solint]['modes_to_attempt']:
               gaincal_combine=selfcal_plan[vis]['solint_settings'][solint]['gaincal_combine'][mode]
               filename_append=selfcal_plan[vis]['solint_settings'][solint]['filename_append'][mode]


               if selfcal_plan[vis]['solint_settings'][solint]['gaincal_gaintype'][mode] =='':
                   gaincal_gaintype=''
                   if not do_fallback_calonly or second_iter_solmode == "":
                       gaincal_gaintype='T'
                   else:
                       gaincal_gaintype='GSPLINE'
                   if second_iter_solmode == "GSPLINE": 
                       gaincal_gaintype='T'
                   else:
                       gaincal_gaintype= 'G'
                   if gaincal_gaintype !='':
                      selfcal_plan[vis]['solint_settings'][solint]['gaincal_gaintype'][mode]=gaincal_gaintype
                   #selfcal_plan[vis]['solint_settings'][solint]['gaincal_gaintype'][mode]='T' if not do_fallback_calonly or second_iter_solmode == "" else "GSPLINE" if second_iter_solmode == "GSPLINE" else "G"
               if selfcal_plan[vis]['solint_settings'][solint]['gaincal_gaintype'][mode] == "GSPLINE":
                   splinetime = solint.replace('_EB','').replace('_ap','')
                   if splinetime == "inf":
                       splinetime = selfcal_library[fid]["Median_scan_time"]
                   else:
                       splinetime = float(splinetime[0:-1])

               if 'spw' in gaincal_combine:
                  if selfcal_library['spws_set'][vis].ndim == 1:
                     nspw_sets=1
                  else:
                     nspw_sets=selfcal_library['spws_set'][vis].shape[0]
               else: #only necessary to loop over gain cal when in inf_EB to avoid inf_EB solving for all spws
                  nspw_sets=1
               for i in range(nspw_sets):  # run gaincal on each spw set to handle spectral scans
                  if 'spw' in gaincal_combine:
                     if nspw_sets == 1 and selfcal_library['spws_set'][vis].ndim == 1:
                        spwselect=','.join(str(spw) for spw in selfcal_library['spws_set'][vis].tolist())
                     else:
                        spwselect=','.join(str(spw) for spw in selfcal_library['spws_set'][vis][i].tolist())
                  else:
                     spwselect=selfcal_library[vis]['spws']
                  gaintable_name='temp_'+solint+'_'+str(iteration)+'_'+selfcal_plan['solmode'][iteration]+'_'+filename_append+'.g'
                  if mode != 'per_bb':      
                     gcdict=call_gaincal(vis=vis, caltable=gaintable_name, gaintype=selfcal_plan[vis]['solint_settings'][solint]['gaincal_gaintype'][mode], spw=spwselect,
                            refant=selfcal_library[vis]['refant'], calmode=selfcal_plan['solmode'][iteration], solnorm=solnorm if not do_fallback_calonly else False,
                            solint=solint_interval.replace('_EB','').replace('_ap','').replace('scan_',''),\
                            minsnr=gaincal_minsnr if not do_fallback_calonly else max(gaincal_minsnr,gaincal_unflag_minsnr), minblperant=4,combine=gaincal_combine,\
                            field=str(selfcal_library['sub-fields-fid_map'][vis][fid]),gaintable=gaincal_preapply_gaintable,spwmap=gaincal_spwmap,uvrange=selfcal_library['uvrange'],\
                            interp=gaincal_interpolate, solmode=gaincal_solmode, refantmode='flex',\
                              append=os.path.exists(gaintable_name))
                     selfcal_plan[vis]['solint_settings'][solint]['gaincal_return_dict'][mode].append(gcdict.copy())
                  else:
                    for baseband in selfcal_library[vis]['baseband'].keys():
                         spwselect_bb=selfcal_library[vis]['baseband'][baseband]['spwstring']
                         gcdict=call_gaincal(vis=vis, caltable=gaintable_name, gaintype=selfcal_plan[vis]['solint_settings'][solint]['gaincal_gaintype'][mode], spw=spwselect_bb,
                              refant=selfcal_library[vis]['refant'], calmode=selfcal_plan['solmode'][iteration], solnorm=solnorm if not do_fallback_calonly else False,
                              solint=solint_interval.replace('_EB','').replace('_ap','').replace('scan_',''),\
                              minsnr=gaincal_minsnr if not do_fallback_calonly else max(gaincal_minsnr,gaincal_unflag_minsnr), minblperant=4,combine=gaincal_combine,\
                              field=str(selfcal_library['sub-fields-fid_map'][vis][fid]),gaintable=gaincal_preapply_gaintable,spwmap=gaincal_spwmap,uvrange=selfcal_library['uvrange'],\
                              interp=gaincal_interpolate, solmode=gaincal_solmode, refantmode='flex',\
                              append=os.path.exists(gaintable_name))
                         selfcal_plan[vis]['solint_settings'][solint]['gaincal_return_dict'][mode].append(gcdict.copy())

               selfcal_plan[vis]['solint_settings'][solint]['computed_gaintable'][mode] = gaintable_name

        gaintable_prefix='temp_'
        if len(selfcal_plan[vis]['solint_settings'][solint]['modes_to_attempt']) >= 1:
            preferred_mode,fallback,spwmap,spwmapping_for_applycal = \
                           select_best_gaincal_mode(selfcal_library,selfcal_plan,vis,gaintable_prefix,solint,spectral_solution_fraction,minsnr_to_proceed)
        else:
            preferred_mode='combinespw'
            fallback=''
        print(preferred_mode,solint,fallback)

        selfcal_plan[vis]['solint_settings'][solint]['final_mode']=preferred_mode
        selfcal_plan[vis]['solint_settings'][solint]['applycal_spwmap']=selfcal_plan[vis]['solint_settings'][solint]['spwmap_for_mode'][preferred_mode]
        applycal_spwmap.append(selfcal_plan[vis]['solint_settings'][solint]['spwmap_for_mode'][preferred_mode])

        applycal_interpolate.append(selfcal_plan[vis]['solint_settings'][solint]['applycal_interpolate'])
        #applycal_gaintable=selfcal_library[fid][vis][selfcal_library[fid]['final_phase_solint']]['gaintable']+[sani_target+'_'+vis+'_'+band+'_'+solint+'_'+str(iteration)+'_ap.g']
        applycal_gaintable.append(sani_target+'_'+vis+'_'+band+'_'+solint+'_'+str(iteration)+'_'+selfcal_plan['solmode'][iteration]+'_'+preferred_mode+'.g')


        # Update the appropriate selfcal_library entries.
        selfcal_library[vis][solint]['final_mode']=preferred_mode+''
        selfcal_library[vis][solint]['spwmap']=applycal_spwmap.copy()
        selfcal_library[vis][solint]['gaincal_combine']=selfcal_plan[vis]['solint_settings'][solint]['gaincal_combine'][preferred_mode]+''
        selfcal_library[vis][solint]['gaintable']=applycal_gaintable.copy()
        selfcal_library[vis][solint]['iteration']=iteration+0
        selfcal_library[vis][solint]['applycal_mode']=selfcal_plan['applycal_mode'][iteration]+''
        selfcal_library[vis][solint]['applycal_interpolate']=applycal_interpolate.copy()
        selfcal_library[vis][solint]['solmode']=selfcal_plan['solmode'][iteration]+''

        # Remove per_spw and/or per_bb from subsequent solints if per_bb or combinespw are selected for a given solint
        if preferred_mode != 'per_spw':
            remove_modes(selfcal_plan,vis,current_solint_index)


        for fid in np.intersect1d(selfcal_library['sub-fields-to-selfcal'],list(selfcal_library['sub-fields-fid_map'][vis].keys())):
            selfcal_library[fid][vis][solint]['final_mode']=preferred_mode+''
            selfcal_library[fid][vis][solint]['spwmap']=applycal_spwmap.copy()
            selfcal_library[fid][vis][solint]['gaincal_combine']=selfcal_plan[vis]['solint_settings'][solint]['gaincal_combine'][preferred_mode]+''
            selfcal_library[fid][vis][solint]['gaintable']=applycal_gaintable.copy()
            selfcal_library[fid][vis][solint]['iteration']=iteration+0
            selfcal_library[fid][vis][solint]['applycal_mode']=selfcal_plan['applycal_mode'][iteration]+''
            selfcal_library[fid][vis][solint]['applycal_interpolate']=applycal_interpolate.copy()
            selfcal_library[fid][vis][solint]['solmode']=selfcal_plan['solmode'][iteration]+''

        if 'combinespw' not in preferred_mode:   # preapply all non spwcombine gain tables
           selfcal_plan[vis]['solint_settings'][solint]['preapply_this_gaintable']=True
        selfcal_plan[vis]['solint_settings'][solint]['accepted_gaintable']=sani_target+'_'+vis+'_'+band+'_'+solint+'_'+str(iteration)+'_'+selfcal_plan['solmode'][iteration]+'_'+preferred_mode+'.g'

        filename_append=selfcal_plan[vis]['solint_settings'][solint]['filename_append'][preferred_mode]


        for mode in selfcal_plan[vis]['solint_settings'][solint]['modes_to_attempt']:
            ##### send chosen subtable to this routine for final copying to the gain table we want.
            tb.open('temp_'+solint+'_'+str(iteration)+'_'+selfcal_plan['solmode'][iteration]+'_'+mode+'.g')
            subt = tb.query("OBSERVATION_ID==0", sortlist="TIME,ANTENNA1")
            tb.close()

            subt.copy(sani_target+'_'+vis+'_'+band+'_'+solint+'_'+str(iteration)+'_'+selfcal_plan['solmode'][iteration]+'_'+mode+'.g', deep=True)
            subt.close()

            # commented so that we keep all the gain tables around
            # once well-tested, we might remove the tables for the non-chosen modes to avoid generating too many useless files.  
            os.system('rm -rf '+'temp_'+solint+'_'+str(iteration)+'_'+selfcal_plan['solmode'][iteration]+'_'+mode+'.g')

    if rerank_refants:
        selfcal_library[vis]["refant"] = rank_refants(vis, caltable=sani_target+'_'+vis+'_'+band+'_'+solint+'_'+str(iteration)+'_'+selfcal_plan['solmode'][iteration]+'_'+selfcal_library[vis][solint]['final_mode']+'.g')

        # If we are falling back to a previous solution interval on the unflagging, we need to make sure all tracks use a common 
        # reference antenna.
        if unflag_fb_to_prev_solint:
            for it, sint in enumerate(selfcal_plan['solints'][0:iteration+1]):
                if not os.path.exists(sani_target+'_'+vis+'_'+band+'_'+sint+'_'+str(it)+'_'+selfcal_plan['solmode'][it]+'_'+selfcal_library[vis][solint]['final_mode']+'.g'):
                    continue

                # If a previous iteration went through the unflagging routine, it is possible that some antennas fell back to
                # a previous solint. In that case, rerefant will flag those antennas because they can't be re-referenced with
                # a different time interval. So to be safe, we go back to the pre-pass solutions and then re-run the passing.
                # We could probably check more carefully whether this is the case to avoid having to do this... but the 
                # computing time isn't significant so it's easy just to run through again.
                if os.path.exists(sani_target+'_'+vis+'_'+band+'_'+sint+'_'+str(it)+'_'+selfcal_plan['solmode'][it]+'_'+selfcal_library[vis][solint]['final_mode']+'.pre-pass.g'):
                    rerefant(vis, sani_target+'_'+vis+'_'+band+'_'+sint+'_'+str(it)+'_'+selfcal_plan['solmode'][it]+'_'+selfcal_library[vis][solint]['final_mode']+'.pre-pass.g', \
                            refant=selfcal_library[vis]["refant"], refantmode=refantmode if 'inf_EB' not in sint else 'flex')

                    os.system("rm -rf "+sani_target+'_'+vis+'_'+band+'_'+sint+'_'+str(it)+'_'+selfcal_plan['solmode'][it]+'_'+selfcal_library[vis][solint]['final_mode']+'.g')
                    os.system("cp -r "+sani_target+'_'+vis+'_'+band+'_'+sint+'_'+str(it)+'_'+selfcal_plan['solmode'][it]+'_'+selfcal_library[vis][solint]['final_mode']+'.pre-pass.g '+\
                            sani_target+'_'+vis+'_'+band+'_'+sint+'_'+str(it)+'_'+selfcal_plan['solmode'][it]+'_'+selfcal_library[vis][solint]['final_mode']+'.g')

                    if sint == "inf_EB" and len(selfcal_library[vis][sint]["spwmap"][0]) > 0:
                        unflag_spwmap = selfcal_library[vis][sint]["spwmap"][0]
                    else:
                        unflag_spwmap = []

                    unflag_failed_antennas(vis, sani_target+'_'+vis+'_'+band+'_'+sint+'_'+str(it)+'_'+\
                            selfcal_plan['solmode'][it]+'_'+selfcal_library[vis][solint]['final_mode']+'.g', \
                            selfcal_plan[vis]['solint_settings'][sint]['gaincal_return_dict'][selfcal_library[vis][sint]['final_mode']], selfcal_library['telescope'], \
                            flagged_fraction=0.25, solnorm=solnorm, \
                            only_long_baselines=selfcal_plan['solmode'][it]=="ap" if unflag_only_lbants and \
                            unflag_only_lbants_onlyap else unflag_only_lbants, calonly_max_flagged=calonly_max_flagged, \
                            spwmap=unflag_spwmap, fb_to_prev_solint=unflag_fb_to_prev_solint, solints=selfcal_plan['solints'], iteration=it)
                else:
                    rerefant(vis, sani_target+'_'+vis+'_'+band+'_'+sint+'_'+str(it)+'_'+selfcal_plan['solmode'][it]+'_'+selfcal_library[vis][solint]['final_mode']+'.g', \
                            refant=selfcal_library[vis]["refant"], refantmode=refantmode if 'inf_EB' not in sint else 'flex')
        else:
            os.system("cp -r "+sani_target+'_'+vis+'_'+band+'_'+solint+'_'+str(iteration)+'_'+selfcal_plan['solmode'][iteration]+'_'+selfcal_library[vis][solint]['final_mode']+'.g '+\
                    sani_target+'_'+vis+'_'+band+'_'+solint+'_'+str(iteration)+'_'+selfcal_plan['solmode'][iteration]+'_'+selfcal_library[vis][solint]['final_mode']+'.pre-rerefant.g')
            rerefant(vis, sani_target+'_'+vis+'_'+band+'_'+solint+'_'+str(iteration)+'_'+selfcal_plan['solmode'][iteration]+'_'+selfcal_library[vis][solint]['final_mode']+'.g', \
                    refant=selfcal_library[vis]["refant"], refantmode=refantmode if 'inf_EB' not in solint else 'flex')

    selfcal_library[vis][solint]['fallback']=fallback+''
    for fid in np.intersect1d(selfcal_library['sub-fields-to-selfcal'],list(selfcal_library['sub-fields-fid_map'][vis].keys())):
        selfcal_library[fid][vis][solint]['fallback']=fallback+''

    # If iteration two, try restricting to just the antennas with enough unflagged data.
    # Should we also restrict to just long baseline antennas?
    if do_fallback_calonly:
        # Make a copy of the caltable before unflagging, for reference.
        os.system("cp -r "+sani_target+'_'+vis+'_'+band+'_'+solint+'_'+str(iteration)+'_'+\
                selfcal_plan['solmode'][iteration]+'_'+selfcal_library[vis][solint]['final_mode']+'.g '+sani_target+'_'+vis+'_'+band+'_'+solint+'_'+str(iteration)+'_'+\
                selfcal_plan['solmode'][iteration]+'_'+selfcal_library[vis][solint]['final_mode']+'.pre-pass.g')

        if solint == "inf_EB" and len(applycal_spwmap) > 0:
            unflag_spwmap = applycal_spwmap[0]
        else:
            unflag_spwmap = []

        selfcal_library[vis][solint]['unflag_spwmap'] = unflag_spwmap
        selfcal_library[vis][solint]['unflagged_lbs'] = True

        unflag_failed_antennas(vis, sani_target+'_'+vis+'_'+band+'_'+solint+'_'+str(iteration)+'_'+\
                selfcal_plan['solmode'][iteration]+'_'+selfcal_library[vis][solint]['final_mode']+'.g', selfcal_plan[vis]['solint_settings'][solint]['gaincal_return_dict'][selfcal_library[vis][solint]['final_mode']], selfcal_library['telescope'], flagged_fraction=0.25, solnorm=solnorm, \
                only_long_baselines=selfcal_plan['solmode'][iteration]=="ap" if unflag_only_lbants and unflag_only_lbants_onlyap else \
                unflag_only_lbants, calonly_max_flagged=calonly_max_flagged, spwmap=unflag_spwmap, \
                fb_to_prev_solint=unflag_fb_to_prev_solint, solints=selfcal_plan['solints'], iteration=iteration)

    # Do some post-gaincal cleanup for mosaics.
    if selfcal_library['obstype'] == 'mosaic' or mode == "cocal":
        os.system("cp -r "+sani_target+'_'+vis+'_'+band+'_'+solint+'_'+str(iteration)+'_'+selfcal_plan['solmode'][iteration]+'_'+selfcal_library[vis][solint]['final_mode']+'.g '+\
                sani_target+'_'+vis+'_'+band+'_'+solint+'_'+str(iteration)+'_'+selfcal_plan['solmode'][iteration]+'_'+selfcal_library[vis][solint]['final_mode']+'.pre-drop.g')
        tb.open(sani_target+'_'+vis+'_'+band+'_'+solint+'_'+str(iteration)+'_'+selfcal_plan['solmode'][iteration]+'_'+selfcal_library[vis][solint]['final_mode']+'.g', nomodify=False)
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


def generate_settings_for_combinespw_fallback(selfcal_library, selfcal_plan, target, band, vis, solint, iteration): 
    sani_target=sanitize_string(target)
    current_solint_index=selfcal_plan['solints'].index(solint)
    if selfcal_library['telescope'] == 'VLBA' and 'delay' in solint:  # use per_bb in place of combinespw for VLBA fall back'
        preferred_mode='per_bb'
    else:
        preferred_mode='combinespw'

    selfcal_plan[vis]['solint_settings'][solint]['final_mode'] = preferred_mode+''
    selfcal_plan[vis]['solint_settings'][solint]['preapply_this_gaintable'] = True if solint == 'inf_EB' else False
    selfcal_plan[vis]['solint_settings'][solint]['applycal_spwmap'] = selfcal_plan[vis]['solint_settings'][solint]['spwmap_for_mode'][preferred_mode]
    selfcal_plan[vis]['solint_settings'][solint]['accepted_gaintable'] = selfcal_plan[vis]['solint_settings'][solint]['computed_gaintable'][preferred_mode]

    gaincal_spwmap=[]
    gaincal_preapply_gaintable=[]
    gaincal_interpolate=[]
    applycal_spwmap=[]
    applycal_interpolate=[]
    applycal_gaintable=[]
    for j in range(current_solint_index):
        if selfcal_plan[vis]['solint_settings'][selfcal_plan['solints'][j]]['preapply_this_gaintable']:
           gaincal_preapply_gaintable.append(selfcal_plan[vis]['solint_settings'][selfcal_plan['solints'][j]]['accepted_gaintable'])
           gaincal_spwmap.append(selfcal_plan[vis]['solint_settings'][selfcal_plan['solints'][j]]['applycal_spwmap'])
           gaincal_interpolate.append(selfcal_plan[vis]['solint_settings'][selfcal_plan['solints'][j]]['applycal_interpolate'])
           applycal_spwmap.append(selfcal_plan[vis]['solint_settings'][selfcal_plan['solints'][j]]['applycal_spwmap'])
           applycal_interpolate.append(selfcal_plan[vis]['solint_settings'][selfcal_plan['solints'][j]]['applycal_interpolate'])
           applycal_gaintable.append(selfcal_plan[vis]['solint_settings'][selfcal_plan['solints'][j]]['accepted_gaintable'])
    applycal_spwmap.append(selfcal_plan[vis]['solint_settings'][solint]['spwmap_for_mode'][preferred_mode])
    applycal_interpolate.append(selfcal_plan[vis]['solint_settings'][solint]['applycal_interpolate'])
    applycal_gaintable += [sani_target+'_'+vis+'_'+band+'_'+solint+'_'+str(iteration)+'_'+selfcal_plan['solmode'][iteration]+'_'+preferred_mode+'.g']
    selfcal_library[vis][solint]['final_mode']=preferred_mode+'_fallback'
    selfcal_library[vis][solint]['gaintable']=applycal_gaintable
    selfcal_library[vis][solint]['iteration']=iteration+0
    selfcal_library[vis][solint]['spwmap']=applycal_spwmap
    selfcal_library[vis][solint]['applycal_mode']=selfcal_plan['applycal_mode'][iteration]+''
    selfcal_library[vis][solint]['applycal_interpolate']=applycal_interpolate
    selfcal_library[vis][solint]['solmode']=selfcal_plan['solmode'][iteration]+''
    selfcal_library[vis][solint]['gaincal_combine']=selfcal_plan[vis]['solint_settings'][solint]['gaincal_combine'][preferred_mode]+''
    for fid in np.intersect1d(selfcal_library['sub-fields-to-selfcal'],list(selfcal_library['sub-fields-fid_map'][vis].keys())):
        selfcal_library[fid][vis][solint]['final_mode']=preferred_mode+'_fallback'
        selfcal_library[fid][vis][solint]['spwmap']=applycal_spwmap
        selfcal_library[fid][vis][solint]['gaincal_combine']=selfcal_plan[vis]['solint_settings'][solint]['gaincal_combine'][preferred_mode]+''
        selfcal_library[fid][vis][solint]['gaintable']=applycal_gaintable
        selfcal_library[fid][vis][solint]['iteration']=iteration+0
        selfcal_library[fid][vis][solint]['spwmap']=applycal_spwmap
        selfcal_library[fid][vis][solint]['applycal_mode']=selfcal_plan['applycal_mode'][iteration]+''
        selfcal_library[fid][vis][solint]['applycal_interpolate']=applycal_interpolate
        selfcal_library[fid][vis][solint]['solmode']=selfcal_plan['solmode'][iteration]+''

def call_gaincal(vis, caltable='', append=False, minsnr=3.0, **kwargs):

    count=0
    minsnr_for_gc=minsnr+0.0
    while not os.path.exists(caltable) or append:
        if count == 1:
            minsnr_for_gc=0.0
        elif count > 1:
            break
    
        gaincal_return = gaincal(vis, caltable=caltable, minsnr=minsnr_for_gc, append=append, **kwargs)

        if append:
            break

        count += 1

    if minsnr_for_gc == 0.0:
        print('Flagging and Zeroing out gain table after re-running with minsnr = 0.0')
        flagdata(vis=caltable,mode='manual')
        gaincal_return = zero_out_gc_return_dict(gaincal_return)

    return gaincal_return
