import numpy as np
from .selfcal_helpers import *
from .mosaic_helpers import scan_inf_scan_combine

def gaincal_wrapper(selfcal_library, selfcal_plan, target, band, vis, solint, applymode, iteration, 
        gaincal_minsnr, gaincal_unflag_minsnr=5.0, minsnr_to_proceed=3.0, rerank_refants=False, unflag_only_lbants=False, unflag_only_lbants_onlyap=False, 
        calonly_max_flagged=0.0, second_iter_solmode="", unflag_fb_to_prev_solint=False, \
        refantmode="flex", mode="selfcal", calibrators="", gaincalibrator_dict={}, allow_gain_interpolation=False,spectral_solution_fraction=0.3,
        guess_scan_combine=False, do_fallback_calonly=False, preapply_targets_own_inf_EB=False):
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

    print(selfcal_plan['solmode'])
    print(iteration)
    os.system('rm -rf '+sani_target+'_'+vis+'_'+band+'_'+solint+'_'+str(iteration)+'_'+selfcal_plan['solmode'][iteration]+'*.g')

    ## Reset the gaincal return dictionaries, in case this is a repeat of the current solution interval.
    for gc_mode in selfcal_plan[vis]['solint_settings'][solint]['modes_to_attempt']:
        selfcal_plan[vis]['solint_settings'][solint]['gaincal_return_dict'][gc_mode] = []

    ##
    ## Set gaincal parameters depending on which iteration and whether to use combine=spw for inf_EB or not
    ## Defaults should assume combine='scan' and gaintpe='G' will fallback to combine='scan,spw' if too much flagging
    ## At some point remove the conditional for use_inf_EB_preapply, since there isn't a reason not to do it
    ##
    current_solint_index=selfcal_plan['solints'].index(solint)
    selfcal_plan[vis]['solint_settings'][solint]['computed_gaintable'] = {}
    if selfcal_plan['solmode'][iteration] == 'p':
        gaincal_spwmap=[]
        gaincal_preapply_gaintable=[]
        gaincal_interpolate=[]
        applycal_spwmap=[]
        applycal_interpolate=[]
        applycal_gaintable=[]

        if mode == "cocal":
            include_solints = selfcal_plan[vis]['solint_settings'][solint]['preapply_solints']
        else:
            include_solints = [selfcal_plan['solints'][j] for j in range(current_solint_index) if 
                               selfcal_plan[vis]['solint_settings'][selfcal_plan['solints'][j]]['preapply_this_gaintable']]

        for incl_solint in include_solints:
            gaincal_preapply_gaintable.append(selfcal_plan[vis]['solint_settings'][incl_solint]['accepted_gaintable'])
            gaincal_spwmap.append(selfcal_plan[vis]['solint_settings'][incl_solint]['applycal_spwmap'])
            gaincal_interpolate.append(selfcal_plan[vis]['solint_settings'][incl_solint]['applycal_interpolate'])
            
            # If we manually specify that applycal should use a different solint reference than gaincal (cocal), change
            # to that solint here.
            if "applycal_solint" in selfcal_plan[vis]['solint_settings'][incl_solint]:
                incl_solint = selfcal_plan[vis]['solint_settings'][incl_solint]["applycal_solint"]
            
            applycal_spwmap.append(selfcal_plan[vis]['solint_settings'][incl_solint]['applycal_spwmap'])
            applycal_interpolate.append(selfcal_plan[vis]['solint_settings'][incl_solint]['applycal_interpolate'])
            if solint == "inf_fb3":
                applycal_gaintable.append(selfcal_plan[vis]['solint_settings']["inf_EB_fb"]['accepted_gaintable'])
            else:
                applycal_gaintable.append(selfcal_plan[vis]['solint_settings'][incl_solint]['accepted_gaintable'])


        if solint != 'inf_EB':
            gaincal_solmode = "" if not do_fallback_calonly or second_iter_solmode == "GSPLINE" else second_iter_solmode

        fallback=''
        if selfcal_plan['solmode'][iteration] == 'ap':
           solnorm=True
        else:
           solnorm=False

        if mode == "cocal":
            # Check which targets are acceptable to use as calibrators.
            targets = calibrators[band][iteration - len(selfcal_plan['solints'])]

            include_targets, include_scans = triage_calibrators(vis, target, band, targets)
        else:
            #include_targets = str(selfcal_library['sub-fields-fid_map'][vis][0])
            include_targets = selfcal_library['bands_for_targets'][vis]['field_str']
            include_scans = ""
        print('Include targets: ', include_targets)

        if selfcal_plan[vis]['solint_settings'][solint]['sub-name'] == "scan_inf":
            include_scans = scan_inf_scan_combine(selfcal_library, vis, target, gaincalibrator_dict, guess_scan_combine=guess_scan_combine)
        else:
            include_scans = [include_scans]

        
        # Fields that don't have any mask in the primary beam should be removed from consideration, as their models are likely bad.
        gaincal_solmode=""
        if selfcal_library['obstype'] == 'mosaic':
            msmd.open(vis)
            include_targets = []
            remove = []
            for incl_scan in include_scans:
                scan_targets = []
                for fid in [selfcal_library['sub-fields-fid_map'][vis][fid] for fid in \
                        np.intersect1d(selfcal_library[vis]['sub-fields-to-gaincal'],list(selfcal_library['sub-fields-fid_map'][vis].keys()))] if incl_scan == '' else \
                        np.intersect1d(msmd.fieldsforscans(np.array(incl_scan.split(",")).astype(int)), \
                        [selfcal_library['sub-fields-fid_map'][vis][fid] for fid in \
                        numpy.intersect1d(selfcal_library[vis]['sub-fields-to-gaincal'],list(selfcal_library['sub-fields-fid_map'][vis].keys()))]):
                    # Note: because of the msmd above getting actual fids from the MS, we just need to append fid below.
                    scan_targets.append(fid)

                if len(scan_targets) > 0:
                    include_targets.append(','.join(np.array(scan_targets).astype(str)))
                else:
                    remove.append(incl_scan)

            for incl_scan in remove:
                include_scans.remove(incl_scan)

            msmd.close()
        elif solint == "inf_fb3":
            include_targets = include_targets.split(',')
            include_scans = include_scans * len(include_targets)
        else:
            include_targets = [include_targets]

        selfcal_library[vis][solint]["include_scans"] = include_scans
        selfcal_library[vis][solint]["include_targets"] = include_targets
        print(solint,'Include scans: ', include_scans)
        print(solint,'Include targets: ', include_targets)
        print(solint,'Modes to attempt: ',selfcal_plan[vis]['solint_settings'][solint]['modes_to_attempt'])
        for incl_scans, incl_targets in zip(include_scans, include_targets):
            if solint == "inf_fb3":
                gaincal_preapply_gaintable = [selfcal_plan[vis]['solint_settings']["inf_fb3"]["preapply_gaintable_dict"][incl_targets]]

            for gc_mode in selfcal_plan[vis]['solint_settings'][solint]['modes_to_attempt']:
               print(incl_targets)
               print(incl_scans)
               print(vis,solint,gc_mode)
               print(selfcal_plan[vis]['solint_settings'][solint]['gaincal_combine'])
               gaincal_combine=selfcal_plan[vis]['solint_settings'][solint]['gaincal_combine'][gc_mode]
               filename_append=selfcal_plan[vis]['solint_settings'][solint]['filename_append'][gc_mode]

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
                  print('prior to gaincal',gaintable_name, gc_mode)

                  if gc_mode != 'per_bb':      
                     gcdict=call_gaincal(vis=vis, caltable=gaintable_name, gaintype=selfcal_plan[vis]['solint_settings'][solint]['gaincal_gaintype'][gc_mode], spw=spwselect,
                            refant=selfcal_library[vis]['refant'], calmode=selfcal_plan['solmode'][iteration], solnorm=solnorm if not do_fallback_calonly else False,
                            solint=selfcal_plan[vis]['solint_settings'][solint]['interval'].replace('_EB','').replace('_ap','').replace('scan_','').replace('_fb1','').replace('_fb2','').replace('_fb3',''),\
                            minsnr=gaincal_minsnr if not do_fallback_calonly else max(gaincal_minsnr,gaincal_unflag_minsnr), minblperant=4,combine=gaincal_combine,\
                            field=incl_targets,scan=incl_scans,gaintable=gaincal_preapply_gaintable,spwmap=gaincal_spwmap,uvrange=selfcal_library['uvrange'],\
                            interp=gaincal_interpolate, solmode=gaincal_solmode, refantmode='flex',\
                            append=os.path.exists(sani_target+'_'+vis+'_'+band+'_'+solint+'_'+str(iteration)+'_'+selfcal_plan['solmode'][iteration]+'_'+filename_append+'.g'))
                     selfcal_plan[vis]['solint_settings'][solint]['gaincal_return_dict'][gc_mode].append(gcdict.copy())
                  else:
                    for baseband in selfcal_library[vis]['baseband'].keys():
                         spwselect_bb=selfcal_library[vis]['baseband'][baseband]['spwstring']
                         gcdict=call_gaincal(vis=vis, caltable=gaintable_name, gaintype=selfcal_plan[vis]['solint_settings'][solint]['gaincal_gaintype'][gc_mode], spw=spwselect_bb,
                              refant=selfcal_library[vis]['refant'], calmode=selfcal_plan['solmode'][iteration], solnorm=solnorm if not do_fallback_calonly else False,
                              solint=selfcal_plan[vis]['solint_settings'][solint]['interval'].replace('_EB','').replace('_ap','').replace('scan_','').replace('_fb1','').replace('_fb2','').replace('_fb3',''),\
                              minsnr=gaincal_minsnr if not do_fallback_calonly else max(gaincal_minsnr,gaincal_unflag_minsnr), minblperant=4,combine=gaincal_combine,\
                              field=incl_targets,scan=incl_scans,gaintable=gaincal_preapply_gaintable,spwmap=gaincal_spwmap,uvrange=selfcal_library['uvrange'],\
                              interp=gaincal_interpolate, solmode=gaincal_solmode, refantmode='flex',\
                              append=os.path.exists(sani_target+'_'+vis+'_'+band+'_'+solint+'_'+str(iteration)+'_'+selfcal_plan['solmode'][iteration]+'_'+filename_append+'.g'))
                         selfcal_plan[vis]['solint_settings'][solint]['gaincal_return_dict'][gc_mode].append(gcdict.copy())

               selfcal_plan[vis]['solint_settings'][solint]['computed_gaintable'][gc_mode] = gaintable_name

               # restricted gaincal table comparisons to only inf_EB prior to changes
               # commenting because we want to do comparisons for other solints as well
               #if 'inf_EB' not in solint:
               #   break

        gaintable_prefix=sani_target+'_'+vis+'_'+band+'_'
        # assume that if there is only one mode to attempt, that it is combinespw and don't bother checking.
        if 'd' not in solint:
            if len(selfcal_plan[vis]['solint_settings'][solint]['modes_to_attempt']) > 1:
                get_gaincalmode_flagging_stats(selfcal_library,selfcal_plan,vis,gaintable_prefix,solint)
                preferred_mode,fallback,spwmap,spwmapping_for_applycal = \
                           select_best_gaincal_mode(selfcal_library,selfcal_plan,vis,gaintable_prefix,solint,spectral_solution_fraction,minsnr_to_proceed)

                if fallback=='spwmap':
                    selfcal_plan[vis]['solint_settings'][solint]['spwmap_for_mode']['per_spw']=spwmapping_for_applycal.copy()
                print('Select best gaincal, preferred mode: {}, solint: {}, fallback: {}, spwmapping for applycal {}'.format(preferred_mode,solint,fallback,spwmapping_for_applycal))
            else:
                preferred_mode=selfcal_plan[vis]['solint_settings'][solint]['modes_to_attempt'][0]
                get_gaincalmode_flagging_stats(selfcal_library,selfcal_plan,vis,gaintable_prefix,solint)
                fallback=''
                print('Select best gaincal, preferred mode: {}, solint: {}, fallback: {}, possible modes: {}'.format(preferred_mode,\
                                                    solint,fallback,selfcal_plan[vis]['solint_settings'][solint]['modes_to_attempt']))
        else:
             if len(selfcal_plan[vis]['solint_settings'][solint]['modes_to_attempt']) >= 1: # run this even if we only have one mode to fill flagging stats
                preferred_mode,fallback = select_best_delaycal_mode(selfcal_library,selfcal_plan,vis,\
                                                                    gaintable_prefix,solint,\
                                                                    spectral_solution_fraction,minsnr_to_proceed)
             else:
                preferred_mode=selfcal_plan[vis]['solint_settings'][solint]['modes_to_attempt'][0]
                fallback=''            
             print('Select best delaycal, preferred mode: {}, solint: {}, fallback: {}'.format(preferred_mode,solint,fallback))

            
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


        for fid in np.intersect1d(selfcal_library[vis]['sub-fields-to-selfcal'],list(selfcal_library['sub-fields-fid_map'][vis].keys())):
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
        for fid in np.intersect1d(selfcal_library[vis]['sub-fields-to-selfcal'],list(selfcal_library['sub-fields-fid_map'][vis].keys())):
            gaincal_spwmap=[]
            gaincal_preapply_gaintable=[]
            gaincal_interpolate=[] 
            applycal_gaintable=[]
            applycal_interpolate=[]
            applycal_spwmap=[]
            for j in range(current_solint_index):
              if not selfcal_plan['solints'][j] in selfcal_plan[vis]['solint_settings']:
                  continue
              if selfcal_plan[vis]['solint_settings'][selfcal_plan['solints'][j]]['preapply_this_gaintable']: #allow amplitude and phase to be pre-applied
                 # and selfcal_plan[vis]['solint_settings'][selfcal_plan['solints'][j]]['solmode']=='p':
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

            for gc_mode in selfcal_plan[vis]['solint_settings'][solint]['modes_to_attempt']:
               gaincal_combine=selfcal_plan[vis]['solint_settings'][solint]['gaincal_combine'][gc_mode]
               filename_append=selfcal_plan[vis]['solint_settings'][solint]['filename_append'][gc_mode]

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
                  if gc_mode != 'per_bb':      
                     gcdict=call_gaincal(vis=vis, caltable=gaintable_name, gaintype=selfcal_plan[vis]['solint_settings'][solint]['gaincal_gaintype'][gc_mode], spw=spwselect,
                            refant=selfcal_library[vis]['refant'], calmode=selfcal_plan['solmode'][iteration], solnorm=solnorm if not do_fallback_calonly else False,
                            solint=selfcal_plan[vis]['solint_settings'][solint]['interval'].replace('_EB','').replace('_ap','').replace('scan_',''),\
                            minsnr=gaincal_minsnr if not do_fallback_calonly else max(gaincal_minsnr,gaincal_unflag_minsnr), minblperant=4,combine=gaincal_combine,\
                            field=str(selfcal_library['sub-fields-fid_map'][vis][fid]),gaintable=gaincal_preapply_gaintable,spwmap=gaincal_spwmap,uvrange=selfcal_library['uvrange'],\
                            interp=gaincal_interpolate, solmode=gaincal_solmode, refantmode='flex',\
                              append=os.path.exists(gaintable_name))
                     selfcal_plan[vis]['solint_settings'][solint]['gaincal_return_dict'][gc_mode].append(gcdict.copy())
                  else:
                    for baseband in selfcal_library[vis]['baseband'].keys():
                         spwselect_bb=selfcal_library[vis]['baseband'][baseband]['spwstring']
                         gcdict=call_gaincal(vis=vis, caltable=gaintable_name, gaintype=selfcal_plan[vis]['solint_settings'][solint]['gaincal_gaintype'][gc_mode], spw=spwselect_bb,
                              refant=selfcal_library[vis]['refant'], calmode=selfcal_plan['solmode'][iteration], solnorm=solnorm if not do_fallback_calonly else False,
                              solint=selfcal_plan[vis]['solint_settings'][solint]['interval'].replace('_EB','').replace('_ap','').replace('scan_',''),\
                              minsnr=gaincal_minsnr if not do_fallback_calonly else max(gaincal_minsnr,gaincal_unflag_minsnr), minblperant=4,combine=gaincal_combine,\
                              field=str(selfcal_library['sub-fields-fid_map'][vis][fid]),gaintable=gaincal_preapply_gaintable,spwmap=gaincal_spwmap,uvrange=selfcal_library['uvrange'],\
                              interp=gaincal_interpolate, solmode=gaincal_solmode, refantmode='flex',\
                              append=os.path.exists(gaintable_name))
                         selfcal_plan[vis]['solint_settings'][solint]['gaincal_return_dict'][gc_mode].append(gcdict.copy())

               selfcal_plan[vis]['solint_settings'][solint]['computed_gaintable'][gc_mode] = gaintable_name

        gaintable_prefix='temp_'
        if len(selfcal_plan[vis]['solint_settings'][solint]['modes_to_attempt']) > 1:
            get_gaincalmode_flagging_stats(selfcal_library,selfcal_plan,vis,gaintable_prefix,solint)
            preferred_mode,fallback,spwmap,spwmapping_for_applycal = \
                           select_best_gaincal_mode(selfcal_library,selfcal_plan,vis,gaintable_prefix,solint,spectral_solution_fraction,minsnr_to_proceed)
        else:
            preferred_mode=selfcal_plan[vis]['solint_settings'][solint]['modes_to_attempt'][0]
            if 'd' not in solint:
                get_gaincalmode_flagging_stats(selfcal_library,selfcal_plan,vis,gaintable_prefix,solint)
            fallback=''
            spwmapping_for_applycal=[]
        print('Select best gaincal, preferred mode: {}, solint: {}, fallback: {}, spwmapping for applycal {}'.format(preferred_mode,solint,fallback,spwmapping_for_applycal))

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


        for fid in np.intersect1d(selfcal_library[vis]['sub-fields-to-selfcal'],list(selfcal_library['sub-fields-fid_map'][vis].keys())):
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


        for gc_mode in selfcal_plan[vis]['solint_settings'][solint]['modes_to_attempt']:
            ##### send chosen subtable to this routine for final copying to the gain table we want.
            tb.open('temp_'+solint+'_'+str(iteration)+'_'+selfcal_plan['solmode'][iteration]+'_'+gc_mode+'.g')
            subt = tb.query("OBSERVATION_ID==0", sortlist="TIME,ANTENNA1")
            tb.close()

            subt.copy(sani_target+'_'+vis+'_'+band+'_'+solint+'_'+str(iteration)+'_'+selfcal_plan['solmode'][iteration]+'_'+gc_mode+'.g', deep=True)
            subt.close()

            # commented so that we keep all the gain tables around
            # once well-tested, we might remove the tables for the non-chosen modes to avoid generating too many useless files.  
            os.system('rm -rf '+'temp_'+solint+'_'+str(iteration)+'_'+selfcal_plan['solmode'][iteration]+'_'+gc_mode+'.g')

    if rerank_refants:
        selfcal_library[vis]["refant"] = rank_refants(vis, selfcal_library['telescope'], caltable=sani_target+'_'+vis+'_'+band+'_'+solint+'_'+str(iteration)+'_'+selfcal_plan['solmode'][iteration]+'_'+selfcal_library[vis][solint]['final_mode']+'.g')

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
    for fid in np.intersect1d(selfcal_library[vis]['sub-fields-to-selfcal'],list(selfcal_library['sub-fields-fid_map'][vis].keys())):
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
            if selfcal_plan[vis]['solint_settings'][solint]['sub-name'] == "scan_inf":
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
    if selfcal_library['telescope'] == 'VLBA' or 'd' in solint:  # use per_bb in place of combinespw for VLBA fall back'
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
    for fid in np.intersect1d(selfcal_library[vis]['sub-fields-to-selfcal'],list(selfcal_library['sub-fields-fid_map'][vis].keys())):
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
