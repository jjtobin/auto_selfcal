import numpy as np
from scipy import stats
import glob
import sys
#execfile('selfcal_helpers.py',globals())
sys.path.append("./")
from selfcal_helpers import *
# Mac builds of CASA lack MPI and error without this try/except
try:
   from casampi.MPIEnvironment import MPIEnvironment   
   parallel=MPIEnvironment.is_mpi_enabled
except:
   parallel=False

def run_selfcal(selfcal_library, target, band, solints, solint_snr, solint_snr_per_field, solint_snr_per_spw, applycal_mode, solmode, band_properties, telescope, n_ants, cellsize, imsize, \
        inf_EB_gaintype_dict, inf_EB_gaincal_combine_dict, inf_EB_fallback_mode_dict, gaincal_combine, applycal_interp, integration_time, spectral_scan, spws_set, \
        gaincal_minsnr=2.0, gaincal_unflag_minsnr=5.0, minsnr_to_proceed=3.0, delta_beam_thresh=0.05, do_amp_selfcal=True, inf_EB_gaincal_combine='scan', inf_EB_gaintype='G', \
        unflag_only_lbants=False, unflag_only_lbants_onlyap=False, calonly_max_flagged=0.0, second_iter_solmode="", unflag_fb_to_prev_solint=False, \
        rerank_refants=False, gaincalibrator_dict={}, allow_gain_interpolation=False, guess_scan_combine=False, aca_use_nfmask=False, mask='',usermodel=''):

   # If we are running this on a mosaic, we want to rerank reference antennas and have a higher gaincal_minsnr by default.


   if selfcal_library[target][band]["obstype"] == "mosaic":
       gaincal_minsnr = 2.0
       rerank_refants = True
       refantmode = "strict"
   else:
       refantmode = "flex"

   # Start looping over the solints.

   iterjump=-1   # useful if we want to jump iterations
   sani_target=sanitize_string(target)
   vislist=selfcal_library[target][band]['vislist'].copy()
   print('Starting selfcal procedure on: '+target+' '+band)
   if usermodel != '':
      print('Setting model column to user model')
      usermodel_wrapper(vislist,sani_target+'_'+band,
                     band_properties,band,telescope=telescope,nsigma=0.0, scales=[0],
                     threshold='0.0Jy',
                     savemodel='modelcolumn',parallel=parallel,cellsize=cellsize[band],imsize=imsize[band],
                     nterms=selfcal_library[target][band]['nterms'],reffreq=selfcal_library[target][band]['reffreq'],
                     field=target,spw=selfcal_library[target][band]['spws_per_vis'],uvrange=selfcal_library[target][band]['uvrange'],obstype=selfcal_library[target][band]['obstype'], resume=False, image_mosaic_fields_separately=selfcal_library[target][band]['obstype'] == 'mosaic', mosaic_field_phasecenters=selfcal_library[target][band]['sub-fields-phasecenters'], mosaic_field_fid_map=selfcal_library[target][band]['sub-fields-fid_map'], cyclefactor=selfcal_library[target][band]['cyclefactor'],mask=mask,usermodel=usermodel)

   for iteration in range(len(solints[band][target])):
      if (iterjump !=-1) and (iteration < iterjump): # allow jumping to amplitude selfcal and not need to use a while loop
         continue
      elif iteration == iterjump:
         iterjump=-1

      if 'ap' in solints[band][target][iteration] and not do_amp_selfcal:
          break

      if solint_snr[target][band][solints[band][target][iteration]] < minsnr_to_proceed and np.all([solint_snr_per_field[target][band][fid][solints[band][target][iteration]] < minsnr_to_proceed for fid in selfcal_library[target][band]['sub-fields']]):
         print('*********** estimated SNR for solint='+solints[band][target][iteration]+' too low, measured: '+str(solint_snr[target][band][solints[band][target][iteration]])+', Min SNR Required: '+str(minsnr_to_proceed)+' **************')
         if iteration > 1 and solmode[band][target][iteration] !='ap' and do_amp_selfcal:  # if a solution interval shorter than inf for phase-only SC has passed, attempt amplitude selfcal
            iterjump=solmode[band][target].index('ap') 
            print('****************Attempting amplitude selfcal*************')
            continue

         selfcal_library[target][band]['Stop_Reason']='Estimated_SNR_too_low_for_solint '+solints[band][target][iteration]
         break
      else:
         solint=solints[band][target][iteration]
         if iteration == 0:
            print('Starting with solint: '+solint)
         else:
            print('Continuing with solint: '+solint)
         os.system('rm -rf '+sani_target+'_'+band+'_'+solint+'_'+str(iteration)+'*')
         ##
         ## make images using the appropriate tclean heuristics for each telescope
         ## set threshold based on RMS of initial image and lower if value becomes lower
         ## during selfcal by resetting 'RMS_curr' after the post-applycal evaluation
         ##
         if selfcal_library[target][band]['final_solint'] != 'None':
             prev_solint = selfcal_library[target][band]['final_solint']
             prev_iteration = selfcal_library[target][band][vislist[0]][prev_solint]['iteration']

             nterms_changed = (len(glob.glob(sani_target+'_'+band+'_'+prev_solint+'_'+str(prev_iteration)+"_post.model.tt*")) < 
                    selfcal_library[target][band]['nterms'])

             if nterms_changed:
                 resume = False
             else:
                 resume = True
                 files = glob.glob(sani_target+'_'+band+'_'+prev_solint+'_'+str(prev_iteration)+"_post.*")
                 for f in files:
                     if "nearfield" in f:
                         continue
                     os.system("cp -r "+f+" "+f.replace(prev_solint+"_"+str(prev_iteration)+"_post", solint+'_'+str(iteration)))
         else:
             resume = False

         nfsnr_modifier = selfcal_library[target][band]['RMS_NF_curr'] / selfcal_library[target][band]['RMS_curr']
         #remove mask if exists from previous selfcal _post image user is specifying a mask
         if os.path.exists(sani_target+'_'+band+'_'+solint+'_'+str(iteration)+'.mask') and mask != '':
            os.system('rm -rf '+sani_target+'_'+band+'_'+solint+'_'+str(iteration)+'.mask')
         tclean_wrapper(vislist,sani_target+'_'+band+'_'+solint+'_'+str(iteration),
                     band_properties,band,telescope=telescope,nsigma=selfcal_library[target][band]['nsigma'][iteration], scales=[0],
                     threshold=str(selfcal_library[target][band]['nsigma'][iteration]*selfcal_library[target][band]['RMS_NF_curr'])+'Jy',
                     savemodel='none',parallel=parallel,cellsize=cellsize[band],imsize=imsize[band],
                     nterms=selfcal_library[target][band]['nterms'],reffreq=selfcal_library[target][band]['reffreq'],
                     field=target,spw=selfcal_library[target][band]['spws_per_vis'],uvrange=selfcal_library[target][band]['uvrange'],obstype=selfcal_library[target][band]['obstype'], nfrms_multiplier=nfsnr_modifier, resume=resume, image_mosaic_fields_separately=selfcal_library[target][band]['obstype'] == 'mosaic', mosaic_field_phasecenters=selfcal_library[target][band]['sub-fields-phasecenters'], mosaic_field_fid_map=selfcal_library[target][band]['sub-fields-fid_map'], cyclefactor=selfcal_library[target][band]['cyclefactor'],mask=mask,usermodel=usermodel)

         # Check that a mask was actually created, because if not the model will be empty and gaincal will do bad things and the 
         # code will break.
         if not checkmask(sani_target+'_'+band+'_'+solint+'_'+str(iteration)+'.image.tt0'):
             selfcal_library[target][band]['Stop_Reason'] = 'Empty model for solint '+solint
             break # breakout of loop because the model is empty and gaincal will therefore fail

         if iteration == 0:
            gaincal_preapply_gaintable={}
            gaincal_spwmap={}
            gaincal_interpolate={}
            applycal_gaintable={}
            applycal_spwmap={}
            fallback={}
            applycal_interpolate={}


         # Loop through up to two times. On the first attempt, try applymode = 'calflag' (assuming this is requested by the user). On the
         # second attempt, use applymode = 'calonly'.
         for applymode in np.unique([applycal_mode[band][target][iteration],'calonly']):
             for vis in vislist:
                ##
                ## Restore original flagging state each time before applying a new gaintable
                ##
                if os.path.exists(vis+".flagversions/flags.selfcal_starting_flags_"+sani_target):
                   flagmanager(vis=vis, mode = 'restore', versionname = 'selfcal_starting_flags_'+sani_target, comment = 'Flag states at start of reduction')
                else:
                   flagmanager(vis=vis,mode='save',versionname='selfcal_starting_flags_'+sani_target)

             # We need to redo saving the model now that we have potentially unflagged some data.
             if applymode == "calflag":
                 tclean_wrapper(vislist,sani_target+'_'+band+'_'+solint+'_'+str(iteration),
                             band_properties,band,telescope=telescope,nsigma=selfcal_library[target][band]['nsigma'][iteration], scales=[0],
                             threshold=str(selfcal_library[target][band]['nsigma'][iteration]*selfcal_library[target][band]['RMS_NF_curr'])+'Jy',
                             savemodel='modelcolumn',parallel=parallel,cellsize=cellsize[band],imsize=imsize[band],
                             nterms=selfcal_library[target][band]['nterms'],reffreq=selfcal_library[target][band]['reffreq'],
                             field=target,spw=selfcal_library[target][band]['spws_per_vis'],uvrange=selfcal_library[target][band]['uvrange'],obstype=selfcal_library[target][band]['obstype'], nfrms_multiplier=nfsnr_modifier, savemodel_only=True, cyclefactor=selfcal_library[target][band]['cyclefactor'],mask=mask,usermodel=usermodel)

             for vis in vislist:
                # Record gaincal details.
                selfcal_library[target][band][vis][solint]={}
                for fid in np.intersect1d(selfcal_library[target][band]['sub-fields-to-selfcal'],list(selfcal_library[target][band]['sub-fields-fid_map'][vis].keys())):
                    selfcal_library[target][band][fid][vis][solint]={}

             # Fields that don't have any mask in the primary beam should be removed from consideration, as their models are likely bad.
             if selfcal_library[target][band]['obstype'] == 'mosaic':
                 new_fields_to_selfcal = []
                 for fid in selfcal_library[target][band]['sub-fields-to-selfcal']:
                     os.system('rm -rf test*.mask')
                     tmp_SNR_NF,tmp_RMS_NF=estimate_near_field_SNR(sani_target+'_field_'+str(fid)+'_'+band+'_'+solint+'_'+str(iteration)+'.image.tt0', \
                             las=selfcal_library[target][band]['LAS'], mosaic_sub_field=True, save_near_field_mask=False)

                     immath(imagename=[sani_target+"_field_"+str(fid)+"_"+band+"_"+solint+"_"+str(iteration)+".image.tt0",\
                             sani_target+"_field_"+str(fid)+"_"+band+"_"+solint+"_"+str(iteration)+".pb.tt0",\
                             sani_target+"_field_"+str(fid)+"_"+band+"_"+solint+"_"+str(iteration)+".mospb.tt0"], outfile="test.mask", \
                             expr="IIF(IM0*IM1/IM2 > "+str(5*tmp_RMS_NF)+", 1., 0.)")

                     bmaj = ''.join(np.array(list(imhead(sani_target+"_field_"+str(fid)+"_"+band+"_"+solint+"_"+str(iteration)+".image.tt0", \
                             mode="get", hdkey="bmaj").values())[::-1]).astype(str))
                     bmin = ''.join(np.array(list(imhead(sani_target+"_field_"+str(fid)+"_"+band+"_"+solint+"_"+str(iteration)+".image.tt0", \
                             mode="get", hdkey="bmin").values())[::-1]).astype(str))
                     bpa = ''.join(np.array(list(imhead(sani_target+"_field_"+str(fid)+"_"+band+"_"+solint+"_"+str(iteration)+".image.tt0", \
                             mode="get", hdkey="bpa").values())[::-1]).astype(str))

                     imsmooth("test.mask", kernel="gauss", major=bmaj, minor=bmin, pa=bpa, outfile="test.smoothed.mask")

                     immath(imagename=["test.smoothed.mask",sani_target+"_field_"+str(fid)+"_"+band+"_"+solint+"_"+str(iteration)+".mask"], \
                             outfile="test.smoothed.truncated.mask", expr="IIF(IM0 > 0.01 || IM1 > 0., 1., 0.)")

                     original_intflux = get_intflux(sani_target+"_field_"+str(fid)+"_"+band+"_"+solint+"_"+str(iteration)+".image.tt0", \
                             rms=tmp_RMS_NF, maskname=sani_target+"_field_"+str(fid)+"_"+band+"_"+solint+"_"+str(iteration)+".mask", \
                             mosaic_sub_field=True)[0]
                     updated_intflux = get_intflux(sani_target+"_field_"+str(fid)+"_"+band+"_"+solint+"_"+str(iteration)+".image.tt0", \
                             rms=tmp_RMS_NF, maskname="test.smoothed.truncated.mask", mosaic_sub_field=True)[0]
                     os.system('rm -rf test*.mask')


                     if not checkmask(sani_target+'_field_'+str(fid)+'_'+band+'_'+solint+'_'+str(iteration)+'.image.tt0'):
                         print("Removing field "+str(fid)+" from "+sani_target+'_'+vis+'_'+band+'_'+solint+'_'+str(iteration)+'_'+\
                                 solmode[band][target][iteration]+'.g'+" because there is no signal within the primary beam.")
                         skip_reason = "No signal"
                     elif solint_snr_per_field[target][band][fid][solints[band][target][iteration]] < minsnr_to_proceed and solint not in ['inf_EB','scan_inf']:
                         print("Removing field "+str(fid)+" from "+sani_target+'_'+vis+'_'+band+'_'+solint+'_'+str(iteration)+'_'+\
                                 solmode[band][target][iteration]+'.g'+' because the estimated solint snr is too low.')
                         skip_reason = "Estimated SNR"
                     elif updated_intflux > 1.25 * original_intflux:
                         print("Removing field "+str(fid)+" from "+sani_target+'_'+vis+'_'+band+'_'+solint+'_'+str(iteration)+'_'+\
                                 solmode[band][target][iteration]+'.g'+" because there appears to be significant flux missing from the model.")
                         skip_reason = "Missing flux"
                     else:
                         new_fields_to_selfcal.append(fid)

                     if fid not in new_fields_to_selfcal and solint != "inf_EB" and not allow_gain_interpolation:
                         for vis in selfcal_library[target][band][fid]['vislist']:
                             #selfcal_library[target][band][fid][vis][solint]['interpolated_gains'] = True
                             #selfcal_library[target][band][fid]['Stop_Reason'] = "Gaincal solutions would be interpolated"
                             selfcal_library[target][band][fid][vis][solint]['Pass'] = "None"
                             selfcal_library[target][band][fid][vis][solint]['Fail_Reason'] = skip_reason

                 selfcal_library[target][band]['sub-fields-to-gaincal'] = new_fields_to_selfcal
                 if solint != 'inf_EB' and not allow_gain_interpolation:
                     selfcal_library[target][band]['sub-fields-to-selfcal'] = new_fields_to_selfcal
             else:
                selfcal_library[target][band]['sub-fields-to-gaincal'] = selfcal_library[target][band]['sub-fields-to-selfcal']

             for vis in vislist:
                if np.intersect1d(selfcal_library[target][band]['sub-fields-to-gaincal'],\
                        list(selfcal_library[target][band]['sub-fields-fid_map'][vis].keys())).size == 0:
                     continue
                applycal_gaintable[vis]=[]
                applycal_spwmap[vis]=[]
                applycal_interpolate[vis]=[]
                gaincal_spwmap[vis]=[]
                gaincal_interpolate[vis]=[]
                gaincal_preapply_gaintable[vis]=[]
                ##
                ## Solve gain solutions per MS, target, solint, and band
                ##
                os.system('rm -rf '+sani_target+'_'+vis+'_'+band+'_'+solint+'_'+str(iteration)+'_'+solmode[band][target][iteration]+'*.g')
                ##
                ## Set gaincal parameters depending on which iteration and whether to use combine=spw for inf_EB or not
                ## Defaults should assume combine='scan' and gaintpe='G' will fallback to combine='scan,spw' if too much flagging
                ## At some point remove the conditional for use_inf_EB_preapply, since there isn't a reason not to do it
                ##

                if solmode[band][target][iteration] == 'p':
                    if solint == 'inf_EB':
                       gaincal_spwmap[vis]=[]
                       gaincal_preapply_gaintable[vis]=[]
                       gaincal_interpolate[vis]=[]
                       gaincal_gaintype=inf_EB_gaintype_dict[target][band][vis]
                       gaincal_solmode=""
                       gaincal_combine[band][target][iteration]=inf_EB_gaincal_combine_dict[target][band][vis]
                       if 'spw' in inf_EB_gaincal_combine_dict[target][band][vis]:
                          applycal_spwmap[vis]=[selfcal_library[target][band][vis]['spwmap']]
                          gaincal_spwmap[vis]=[selfcal_library[target][band][vis]['spwmap']]
                       else:
                          applycal_spwmap[vis]=[]
                       applycal_interpolate[vis]=[applycal_interp[band]]
                       applycal_gaintable[vis]=[sani_target+'_'+vis+'_'+band+'_'+solint+'_'+str(iteration)+'_'+solmode[band][target][iteration]+'.g']
                    #elif solmode[band][target][iteration]=='p':
                    else:
                       gaincal_spwmap[vis]=[]
                       gaincal_preapply_gaintable[vis]=[sani_target+'_'+vis+'_'+band+'_inf_EB_0_p.g']
                       gaincal_interpolate[vis]=[applycal_interp[band]]
                       gaincal_gaintype='T' if applymode == "calflag" or second_iter_solmode == "" else "GSPLINE" if second_iter_solmode == "GSPLINE" else "G"
                       gaincal_solmode = "" if applymode == "calflag" or second_iter_solmode == "GSPLINE" else second_iter_solmode
                       if 'spw' in inf_EB_gaincal_combine_dict[target][band][vis]:
                          applycal_spwmap[vis]=[selfcal_library[target][band][vis]['spwmap'],selfcal_library[target][band][vis]['spwmap']]
                          gaincal_spwmap[vis]=[selfcal_library[target][band][vis]['spwmap']]
                       elif inf_EB_fallback_mode_dict[target][band][vis]=='spwmap':
                          applycal_spwmap[vis]=selfcal_library[target][band][vis]['inf_EB']['spwmap'] + [selfcal_library[target][band][vis]['spwmap']]
                          gaincal_spwmap[vis]=selfcal_library[target][band][vis]['inf_EB']['spwmap']
                       else:
                          applycal_spwmap[vis]=[[],selfcal_library[target][band][vis]['spwmap']]
                          gaincal_spwmap[vis]=[]
                       applycal_interpolate[vis]=[applycal_interp[band],applycal_interp[band]]
                       applycal_gaintable[vis]=[sani_target+'_'+vis+'_'+band+'_inf_EB_0'+'_p.g',sani_target+'_'+vis+'_'+band+'_'+solint+'_'+str(iteration)+'_p.g']
                    selfcal_library[target][band][vis][solint]['gaintable']=applycal_gaintable[vis]
                    selfcal_library[target][band][vis][solint]['iteration']=iteration+0
                    selfcal_library[target][band][vis][solint]['spwmap']=applycal_spwmap[vis]
                    selfcal_library[target][band][vis][solint]['applycal_mode']=applycal_mode[band][target][iteration]+''
                    selfcal_library[target][band][vis][solint]['applycal_interpolate']=applycal_interpolate[vis]
                    selfcal_library[target][band][vis][solint]['gaincal_combine']=gaincal_combine[band][target][iteration]+''
                    for fid in np.intersect1d(selfcal_library[target][band]['sub-fields-to-selfcal'],list(selfcal_library[target][band]['sub-fields-fid_map'][vis].keys())):
                        selfcal_library[target][band][fid][vis][solint]['gaintable']=applycal_gaintable[vis]
                        selfcal_library[target][band][fid][vis][solint]['iteration']=iteration+0
                        selfcal_library[target][band][fid][vis][solint]['spwmap']=applycal_spwmap[vis]
                        selfcal_library[target][band][fid][vis][solint]['applycal_mode']=applycal_mode[band][target][iteration]+''
                        selfcal_library[target][band][fid][vis][solint]['applycal_interpolate']=applycal_interpolate[vis]
                        selfcal_library[target][band][fid][vis][solint]['gaincal_combine']=gaincal_combine[band][target][iteration]+''

                    fallback[vis]=''
                    if solmode[band][target][iteration] == 'ap':
                       solnorm=True
                    else:
                       solnorm=False

                    if gaincal_gaintype == "GSPLINE":
                        splinetime = solint.replace('_EB','').replace('_ap','')
                        if splinetime == "inf":
                            splinetime = selfcal_library[target][band]["Median_scan_time"]
                        else:
                            splinetime = float(splinetime[0:-1])

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
                                include_scans.append(",".join(np.intersect1d(msmd.scansforfield(target), \
                                        np.array(list(range(scans[iscan]+1,scans[iscan+1])))).astype(str)))
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
                        include_scans = ['']

                    # Fields that don't have any mask in the primary beam should be removed from consideration, as their models are likely bad.
                    if selfcal_library[target][band]['obstype'] == 'mosaic':
                        msmd.open(vis)
                        include_targets = []
                        remove = []
                        for incl_scan in include_scans:
                            scan_targets = []
                            for fid in [selfcal_library[target][band]['sub-fields-fid_map'][vis][fid] for fid in \
                                    np.intersect1d(selfcal_library[target][band]['sub-fields-to-gaincal'],list(selfcal_library[target][band]['sub-fields-fid_map'][vis].keys()))] if incl_scan == '' else \
                                    np.intersect1d(msmd.fieldsforscans(np.array(incl_scan.split(",")).astype(int)), \
                                    [selfcal_library[target][band]['sub-fields-fid_map'][vis][fid] for fid in \
                                    numpy.intersect1d(selfcal_library[target][band]['sub-fields-to-gaincal'],list(selfcal_library[target][band]['sub-fields-fid_map'][vis].keys()))]):
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
                        include_targets = [str(selfcal_library[target][band]['sub-fields-fid_map'][vis][0])]

                    selfcal_library[target][band][vis][solint]["include_scans"] = include_scans
                    selfcal_library[target][band][vis][solint]["include_targets"] = include_targets

                    selfcal_library[target][band][vis][solint]['gaincal_return'] = []
                    for incl_scans, incl_targets in zip(include_scans, include_targets):
                        if solint == 'inf_EB':
                           if spws_set[band][vis].ndim == 1:
                              nspw_sets=1
                           else:
                              nspw_sets=spws_set[band][vis].shape[0]
                        else: #only necessary to loop over gain cal when in inf_EB to avoid inf_EB solving for all spws
                           nspw_sets=1
                        for i in range(nspw_sets):  # run gaincal on each spw set to handle spectral scans
                           if solint == 'inf_EB':
                              if nspw_sets == 1 and spws_set[band][vis].ndim == 1:
                                 spwselect=','.join(str(spw) for spw in spws_set[band][vis].tolist())
                              else:
                                 spwselect=','.join(str(spw) for spw in spws_set[band][vis][i].tolist())
                           else:
                              spwselect=selfcal_library[target][band][vis]['spws']
                           print('Running gaincal on '+spwselect+' for '+sani_target+'_'+vis+'_'+band+'_'+solint+'_'+str(iteration)+'_'+solmode[band][target][iteration]+'.g')
                           gaincal_return_tmp = gaincal(vis=vis,\
                             caltable=sani_target+'_'+vis+'_'+band+'_'+solint+'_'+str(iteration)+'_'+solmode[band][target][iteration]+'.g',\
                             gaintype=gaincal_gaintype, spw=spwselect,
                             refant=selfcal_library[target][band][vis]['refant'], calmode=solmode[band][target][iteration], solnorm=solnorm if applymode=="calflag" else False,
                             solint=solint.replace('_EB','').replace('_ap','').replace('scan_',''),minsnr=gaincal_minsnr if applymode == 'calflag' else max(gaincal_minsnr,gaincal_unflag_minsnr), minblperant=4,combine=gaincal_combine[band][target][iteration],
                             field=incl_targets,scan=incl_scans,gaintable=gaincal_preapply_gaintable[vis],spwmap=gaincal_spwmap[vis],uvrange=selfcal_library[target][band]['uvrange'],
                             interp=gaincal_interpolate[vis], solmode=gaincal_solmode, refantmode='flex', append=os.path.exists(sani_target+'_'+vis+'_'+band+'_'+solint+'_'+str(iteration)+'_'+solmode[band][target][iteration]+'.g'))
                           #
                           selfcal_library[target][band][vis][solint]['gaincal_return'].append(gaincal_return_tmp)
                           if solint != 'inf_EB':
                              break
                else:
                    selfcal_library[target][band][vis][solint]['gaincal_return'] = []
                    for fid in np.intersect1d(selfcal_library[target][band]['sub-fields-to-selfcal'],list(selfcal_library[target][band]['sub-fields-fid_map'][vis].keys())):
                        gaincal_spwmap[vis]=[]
                        gaincal_preapply_gaintable[vis]=selfcal_library[target][band][fid][vis][selfcal_library[target][band][fid]['final_phase_solint']]['gaintable']
                        gaincal_interpolate[vis]=[applycal_interp[band]]*len(gaincal_preapply_gaintable[vis])
                        gaincal_gaintype='T' if applymode == "calflag" or second_iter_solmode == "" else "GSPLINE" if second_iter_solmode == "GSPLINE" else "G"
                        gaincal_solmode = "" if applymode == "calflag" or second_iter_solmode == "GSPLINE" else second_iter_solmode
                        if 'spw' in inf_EB_gaincal_combine_dict[target][band][vis]:
                           applycal_spwmap[vis]=[selfcal_library[target][band][fid][vis]['spwmap'],selfcal_library[target][band][fid][vis]['spwmap'],selfcal_library[target][band][fid][vis]['spwmap']]
                           gaincal_spwmap[vis]=[selfcal_library[target][band][fid][vis]['spwmap'],selfcal_library[target][band][fid][vis]['spwmap']]
                        elif inf_EB_fallback_mode_dict[target][band][vis]=='spwmap':
                           applycal_spwmap[vis]=selfcal_library[target][band][fid][vis]['inf_EB']['spwmap'] + [selfcal_library[target][band][fid][vis]['spwmap'],selfcal_library[target][band][fid][vis]['spwmap']]
                           gaincal_spwmap[vis]=selfcal_library[target][band][fid][vis]['inf_EB']['spwmap'] + [selfcal_library[target][band][fid][vis]['spwmap']]
                        else:
                           applycal_spwmap[vis]=[[],selfcal_library[target][band][fid][vis]['spwmap'],selfcal_library[target][band][fid][vis]['spwmap']]
                           gaincal_spwmap[vis]=[[],selfcal_library[target][band][fid][vis]['spwmap']]
                        applycal_interpolate[vis]=[applycal_interp[band]]*len(gaincal_preapply_gaintable[vis])+['linearPD']
                        applycal_gaintable[vis]=selfcal_library[target][band][fid][vis][selfcal_library[target][band][fid]['final_phase_solint']]['gaintable']+[sani_target+'_'+vis+'_'+band+'_'+solint+'_'+str(iteration)+'_ap.g']

                        selfcal_library[target][band][vis][solint]['gaintable']=applycal_gaintable[vis]
                        selfcal_library[target][band][vis][solint]['iteration']=iteration+0
                        selfcal_library[target][band][vis][solint]['spwmap']=applycal_spwmap[vis]
                        selfcal_library[target][band][vis][solint]['applycal_mode']=applycal_mode[band][target][iteration]+''
                        selfcal_library[target][band][vis][solint]['applycal_interpolate']=applycal_interpolate[vis]
                        selfcal_library[target][band][vis][solint]['gaincal_combine']=gaincal_combine[band][target][iteration]+''
                        selfcal_library[target][band][fid][vis][solint]['gaintable']=applycal_gaintable[vis]
                        selfcal_library[target][band][fid][vis][solint]['iteration']=iteration+0
                        selfcal_library[target][band][fid][vis][solint]['spwmap']=applycal_spwmap[vis]
                        selfcal_library[target][band][fid][vis][solint]['applycal_mode']=applycal_mode[band][target][iteration]+''
                        selfcal_library[target][band][fid][vis][solint]['applycal_interpolate']=applycal_interpolate[vis]
                        selfcal_library[target][band][fid][vis][solint]['gaincal_combine']=gaincal_combine[band][target][iteration]+''

                        fallback[vis]=''
                        if solmode[band][target][iteration] == 'ap':
                           solnorm=True
                        else:
                           solnorm=False

                        if gaincal_gaintype == "GSPLINE":
                            splinetime = solint.replace('_EB','').replace('_ap','')
                            if splinetime == "inf":
                                splinetime = selfcal_library[target][band][fid]["Median_scan_time"]
                            else:
                                splinetime = float(splinetime[0:-1])

                        gaincal_return_tmp = gaincal(vis=vis,\
                             #caltable=sani_target+'_'+vis+'_'+band+'_'+solint+'_'+str(iteration)+'_'+solmode[band][target][iteration]+'.g',\
                             caltable="temp.g",\
                             gaintype=gaincal_gaintype, spw=selfcal_library[target][band][fid][vis]['spws'],
                             refant=selfcal_library[target][band][vis]['refant'], calmode=solmode[band][target][iteration], solnorm=solnorm if applymode=="calflag" else False,
                             solint=solint.replace('_EB','').replace('_ap','').replace('scan_',''),minsnr=gaincal_minsnr if applymode == 'calflag' else max(gaincal_minsnr,gaincal_unflag_minsnr), minblperant=4,combine=gaincal_combine[band][target][iteration],
                             field=str(selfcal_library[target][band]['sub-fields-fid_map'][vis][fid]),gaintable=gaincal_preapply_gaintable[vis],spwmap=gaincal_spwmap[vis],uvrange=selfcal_library[target][band]['uvrange'],
                             #interp=gaincal_interpolate[vis], solmode=gaincal_solmode, append=os.path.exists(sani_target+'_'+vis+'_'+band+'_'+
                             #solint+'_'+str(iteration)+'_'+solmode[band][target][iteration]+'.g'))
                             interp=gaincal_interpolate[vis], solmode=gaincal_solmode, append=os.path.exists('temp.g'), refantmode='flex')
                        selfcal_library[target][band][vis][solint]['gaincal_return'].append(gaincal_return_tmp)

                    tb.open("temp.g")
                    subt = tb.query("OBSERVATION_ID==0", sortlist="TIME,ANTENNA1")
                    tb.close()

                    subt.copy(sani_target+'_'+vis+'_'+band+'_'+solint+'_'+str(iteration)+'_'+solmode[band][target][iteration]+'.g', deep=True)
                    subt.close()

                    os.system("rm -rf temp.g")

                if rerank_refants:
                    selfcal_library[target][band][vis]["refant"] = rank_refants(vis, caltable=sani_target+'_'+vis+'_'+band+'_'+solint+'_'+str(iteration)+'_'+solmode[band][target][iteration]+'.g')

                    # If we are falling back to a previous solution interval on the unflagging, we need to make sure all tracks use a common 
                    # reference antenna.
                    if unflag_fb_to_prev_solint:
                        for it, sint in enumerate(solints[band][target][0:iteration+1]):
                            if not os.path.exists(sani_target+'_'+vis+'_'+band+'_'+sint+'_'+str(it)+'_'+solmode[band][target][it]+'.g'):
                                continue

                            # If a previous iteration went through the unflagging routine, it is possible that some antennas fell back to
                            # a previous solint. In that case, rerefant will flag those antennas because they can't be re-referenced with
                            # a different time interval. So to be safe, we go back to the pre-pass solutions and then re-run the passing.
                            # We could probably check more carefully whether this is the case to avoid having to do this... but the 
                            # computing time isn't significant so it's easy just to run through again.
                            if os.path.exists(sani_target+'_'+vis+'_'+band+'_'+sint+'_'+str(it)+'_'+solmode[band][target][it]+'.pre-pass.g'):
                                rerefant(vis, sani_target+'_'+vis+'_'+band+'_'+sint+'_'+str(it)+'_'+solmode[band][target][it]+'.pre-pass.g', \
                                        refant=selfcal_library[target][band][vis]["refant"], refantmode=refantmode if 'inf_EB' not in sint else 'flex')

                                os.system("rm -rf "+sani_target+'_'+vis+'_'+band+'_'+sint+'_'+str(it)+'_'+solmode[band][target][it]+'.g')
                                os.system("cp -r "+sani_target+'_'+vis+'_'+band+'_'+sint+'_'+str(it)+'_'+solmode[band][target][it]+'.pre-pass.g '+\
                                        sani_target+'_'+vis+'_'+band+'_'+sint+'_'+str(it)+'_'+solmode[band][target][it]+'.g')

                                if sint == "inf_EB" and len(selfcal_library[target][band][vis][sint]["spwmap"][0]) > 0:
                                    unflag_spwmap = selfcal_library[target][band][vis][sint]["spwmap"][0]
                                else:
                                    unflag_spwmap = []

                                unflag_failed_antennas(vis, sani_target+'_'+vis+'_'+band+'_'+sint+'_'+str(it)+'_'+\
                                        solmode[band][target][it]+'.g', selfcal_library[target][band][vis][sint]['gaincal_return'], flagged_fraction=0.25, solnorm=solnorm, \
                                        only_long_baselines=solmode[band][target][it]=="ap" if unflag_only_lbants and \
                                        unflag_only_lbants_onlyap else unflag_only_lbants, calonly_max_flagged=calonly_max_flagged, \
                                        spwmap=unflag_spwmap, fb_to_prev_solint=unflag_fb_to_prev_solint, solints=solints[band][target], iteration=it)
                            else:
                                rerefant(vis, sani_target+'_'+vis+'_'+band+'_'+sint+'_'+str(it)+'_'+solmode[band][target][it]+'.g', \
                                        refant=selfcal_library[target][band][vis]["refant"], refantmode=refantmode if 'inf_EB' not in sint else 'flex')
                    else:
                        os.system("cp -r "+sani_target+'_'+vis+'_'+band+'_'+solint+'_'+str(iteration)+'_'+solmode[band][target][iteration]+'.g '+\
                                sani_target+'_'+vis+'_'+band+'_'+solint+'_'+str(iteration)+'_'+solmode[band][target][iteration]+'.pre-rerefant.g')
                        rerefant(vis, sani_target+'_'+vis+'_'+band+'_'+solint+'_'+str(iteration)+'_'+solmode[band][target][iteration]+'.g', \
                                refant=selfcal_library[target][band][vis]["refant"], refantmode=refantmode if 'inf_EB' not in solint else 'flex')

                ##
                ## default is to run without combine=spw for inf_EB, here we explicitly run a test inf_EB with combine='scan,spw' to determine
                ## the number of flagged antennas when combine='spw' then determine if it needs spwmapping or to use the gaintable with spwcombine.
                ##
                if solint =='inf_EB' and fallback[vis]=='':
                   os.system('rm -rf test_inf_EB.g')
                   test_gaincal_combine='scan,spw'
                   if selfcal_library[target][band]['obstype']=='mosaic':
                      test_gaincal_combine+=',field'   
                   test_gaincal_return = {'G':[], 'T':[]}
                   for gaintype in np.unique([gaincal_gaintype,'T']):
                       for i in range(spws_set[band][vis].shape[0]):  # run gaincal on each spw set to handle spectral scans
                          if nspw_sets == 1 and spws_set[band][vis].ndim == 1:
                             spwselect=','.join(str(spw) for spw in spws_set[band][vis].tolist())
                          else:
                             spwselect=','.join(str(spw) for spw in spws_set[band][vis][i].tolist())

                          test_gaincal_return[gaintype] += [gaincal(vis=vis,\
                            caltable='test_inf_EB_'+gaintype+'.g',\
                            gaintype=gaintype, spw=spwselect,
                            refant=selfcal_library[target][band][vis]['refant'], calmode='p', 
                            solint=solint.replace('_EB','').replace('_ap',''),minsnr=gaincal_minsnr if applymode == "calflag" else max(gaincal_minsnr,gaincal_unflag_minsnr), minblperant=4,combine=test_gaincal_combine,
                            field=include_targets[0],gaintable='',spwmap=[],uvrange=selfcal_library[target][band]['uvrange'], refantmode=refantmode,append=os.path.exists('test_inf_EB_'+gaintype+'.g'))]
                   spwlist=selfcal_library[target][band][vis]['spws'].split(',')
                   fallback[vis],map_index,spwmap,applycal_spwmap_inf_EB=analyze_inf_EB_flagging(selfcal_library,band,spwlist,sani_target+'_'+vis+'_'+band+'_'+solint+'_'+str(iteration)+'_'+solmode[band][target][iteration]+'.g',vis,target,'test_inf_EB_'+gaincal_gaintype+'.g',spectral_scan,telescope, solint_snr_per_spw, minsnr_to_proceed,'test_inf_EB_T.g' if gaincal_gaintype=='G' else None)

                   inf_EB_fallback_mode_dict[target][band][vis]=fallback[vis]+''
                   print('inf_EB',fallback[vis],applycal_spwmap_inf_EB)
                   if fallback[vis] != '':
                      if 'combinespw' in fallback[vis]:
                         gaincal_spwmap[vis]=[selfcal_library[target][band][vis]['spwmap']]
                         gaincal_combine[band][target][iteration]='scan,spw'
                         inf_EB_gaincal_combine_dict[target][band][vis]='scan,spw'
                         applycal_spwmap[vis]=[selfcal_library[target][band][vis]['spwmap']]
                         os.system('rm -rf           '+sani_target+'_'+vis+'_'+band+'_'+solint+'_'+str(iteration)+'_'+solmode[band][target][iteration]+'.g')
                         for gaintype in np.unique([gaincal_gaintype,'T']):
                            os.system('cp -r test_inf_EB_'+gaintype+'.g '+sani_target+'_'+vis+'_'+band+'_'+solint+'_'+str(iteration)+'_'+solmode[band][target][iteration]+'.gaintype'+gaintype+'.g')
                         if fallback[vis] == 'combinespw':
                             gaincal_gaintype = 'G'
                         else:
                             gaincal_gaintype = 'T'
                         os.system('mv test_inf_EB_'+gaincal_gaintype+'.g '+sani_target+'_'+vis+'_'+band+'_'+solint+'_'+str(iteration)+'_'+solmode[band][target][iteration]+'.g')
                         selfcal_library[target][band][vis][solint]['gaincal_return'] = test_gaincal_return[gaincal_gaintype]
                      if fallback[vis] =='spwmap':
                         gaincal_spwmap[vis]=applycal_spwmap_inf_EB
                         inf_EB_gaincal_combine_dict[target][band][vis]='scan'
                         gaincal_combine[band][target][iteration]='scan'
                         applycal_spwmap[vis]=[applycal_spwmap_inf_EB]

                      # Update the appropriate selfcal_library entries.
                      selfcal_library[target][band][vis][solint]['spwmap']=applycal_spwmap[vis]
                      selfcal_library[target][band][vis][solint]['gaincal_combine']=gaincal_combine[band][target][iteration]+''
                      for fid in np.intersect1d(selfcal_library[target][band]['sub-fields-to-selfcal'],list(selfcal_library[target][band]['sub-fields-fid_map'][vis].keys())):
                          selfcal_library[target][band][fid][vis][solint]['spwmap']=applycal_spwmap[vis]
                          selfcal_library[target][band][fid][vis][solint]['gaincal_combine']=gaincal_combine[band][target][iteration]+''

                   os.system('rm -rf test_inf_EB_*.g')               


                # If iteration two, try restricting to just the antennas with enough unflagged data.
                # Should we also restrict to just long baseline antennas?
                if applymode == "calonly":
                    # Make a copy of the caltable before unflagging, for reference.
                    os.system("cp -r "+sani_target+'_'+vis+'_'+band+'_'+solint+'_'+str(iteration)+'_'+\
                            solmode[band][target][iteration]+'.g '+sani_target+'_'+vis+'_'+band+'_'+solint+'_'+str(iteration)+'_'+\
                            solmode[band][target][iteration]+'.pre-pass.g')

                    if solint == "inf_EB" and len(applycal_spwmap[vis]) > 0:
                        unflag_spwmap = applycal_spwmap[vis][0]
                    else:
                        unflag_spwmap = []

                    selfcal_library[target][band][vis][solint]['unflag_spwmap'] = unflag_spwmap
                    selfcal_library[target][band][vis][solint]['unflagged_lbs'] = True

                    unflag_failed_antennas(vis, sani_target+'_'+vis+'_'+band+'_'+solint+'_'+str(iteration)+'_'+\
                            solmode[band][target][iteration]+'.g', selfcal_library[target][band][vis][solint]['gaincal_return'], flagged_fraction=0.25, solnorm=solnorm, \
                            only_long_baselines=solmode[band][target][iteration]=="ap" if unflag_only_lbants and unflag_only_lbants_onlyap else \
                            unflag_only_lbants, calonly_max_flagged=calonly_max_flagged, spwmap=unflag_spwmap, \
                            fb_to_prev_solint=unflag_fb_to_prev_solint, solints=solints[band][target], iteration=iteration)

                # Do some post-gaincal cleanup for mosaics.
                if selfcal_library[target][band]['obstype'] == 'mosaic':
                    os.system("cp -r "+sani_target+'_'+vis+'_'+band+'_'+solint+'_'+str(iteration)+'_'+solmode[band][target][iteration]+'.g '+\
                            sani_target+'_'+vis+'_'+band+'_'+solint+'_'+str(iteration)+'_'+solmode[band][target][iteration]+'.pre-drop.g')
                    tb.open(sani_target+'_'+vis+'_'+band+'_'+solint+'_'+str(iteration)+'_'+solmode[band][target][iteration]+'.g', nomodify=False)
                    antennas = tb.getcol("ANTENNA1")
                    fields = tb.getcol("FIELD_ID")
                    scans = tb.getcol("SCAN_NUMBER")
                    flags = tb.getcol("FLAG")

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

                    bad = np.where(flags[0,0,:])[0]
                    tb.removerows(rownrs=bad)
                    tb.flush()
                    tb.close()

             new_fields_to_selfcal = selfcal_library[target][band]['sub-fields-to-selfcal'].copy()
             if selfcal_library[target][band]['obstype'] == 'mosaic' and ((solint != "inf_EB" and not allow_gain_interpolation) or \
                     (allow_gain_interpolation and "inf" not in solint)):
                # With gaincal done and bad fields removed from gain tables if necessary, check whether any fields should no longer be selfcal'd
                # because they have too much interpolation.
                for vis in vislist:
                    ## If an EB had no fields to gaincal on, remove all fields in that EB from being selfcal'd as there is no calibration available
                    ## in this EB.
                    if np.intersect1d(selfcal_library[target][band]['sub-fields-to-gaincal'],\
                            list(selfcal_library[target][band]['sub-fields-fid_map'][vis].keys())).size == 0:
                        for fid in np.intersect1d(new_fields_to_selfcal,list(selfcal_library[target][band]['sub-fields-fid_map'][vis].keys())):
                            new_fields_to_selfcal.remove(fid)

                            selfcal_library[target][band][fid]['Stop_Reason'] = 'No viable calibrator fields in at least 1 EB'
                            for v in selfcal_library[target][band][fid]['vislist']:
                                selfcal_library[target][band][fid][v][solint]['Pass'] = 'None'
                                if 'Fail_Reason' in selfcal_library[target][band][fid][v][solint]:
                                    selfcal_library[target][band][fid][v][solint]['Fail_Reason'] += '; '
                                else:
                                    selfcal_library[target][band][fid][v][solint]['Fail_Reason'] = ''
                                selfcal_library[target][band][fid][v][solint]['Fail_Reason'] += 'No viable fields'
                        continue
                    ## NEXT TO DO: check % of flagged solutions - DONE, see above
                    ## After that enable option for interpolation through inf - DONE
                    tb.open(sani_target+'_'+vis+'_'+band+'_'+solint+'_'+str(iteration)+'_'+solmode[band][target][iteration]+'.g')
                    fields = tb.getcol("FIELD_ID")
                    scans = tb.getcol("SCAN_NUMBER")

                    for fid in np.intersect1d(new_fields_to_selfcal,list(selfcal_library[target][band]['sub-fields-fid_map'][vis].keys())):
                        if solint == "scan_inf":
                            msmd.open(vis)
                            scans_for_field = []
                            cals_for_scan = []
                            total_cals_for_scan = []
                            for incl_scan in selfcal_library[target][band][vis][solint]['include_scans']:
                                scans_array = np.array(incl_scan.split(",")).astype(int)
                                fields_for_scans = msmd.fieldsforscans(scans_array)

                                if selfcal_library[target][band]['sub-fields-fid_map'][vis][fid] in fields_for_scans:
                                    scans_for_field.append(np.intersect1d(scans_array, np.unique(scans)))
                                    cals_for_scan.append((scans == scans_for_field[-1]).sum() if scans_for_field[-1] in scans else 0.)
                                    #total_cals_for_scan.append(len(msmd.antennasforscan(scans_for_field[-1])))
                                    total_cals_for_scan.append(len(msmd.antennanames()))

                            if sum(cals_for_scan) / sum(total_cals_for_scan) < 0.75:
                                new_fields_to_selfcal.remove(fid)

                            msmd.close()
                        else:
                            if selfcal_library[target][band]['sub-fields-fid_map'][vis][fid] not in fields:
                                new_fields_to_selfcal.remove(fid)

                        if fid not in new_fields_to_selfcal:
                            # We need to update all the EBs, not just the one that failed.
                            for v in selfcal_library[target][band][fid]['vislist']:
                                selfcal_library[target][band][fid][v][solint]['Pass'] = 'None'
                                if allow_gain_interpolation:
                                    selfcal_library[target][band][fid][v][solint]['Fail_Reason'] = 'Interpolation beyond inf'
                                else:
                                    selfcal_library[target][band][fid][v][solint]['Fail_Reason'] = 'Bad gaincal solutions'


                    tb.close()
             elif selfcal_library[target][band]['obstype'] == 'mosaic' and solint == "inf_EB":
                ## If an EB had no fields to gaincal on, remove all fields in that EB from being selfcal'd as there is no calibration available
                ## in this EB.
                for vis in vislist:
                    if np.intersect1d(selfcal_library[target][band]['sub-fields-to-gaincal'],\
                            list(selfcal_library[target][band]['sub-fields-fid_map'][vis].keys())).size == 0:
                        for fid in np.intersect1d(new_fields_to_selfcal,list(selfcal_library[target][band]['sub-fields-fid_map'][vis].keys())):
                            new_fields_to_selfcal.remove(fid)

                            selfcal_library[target][band][fid]['Stop_Reason'] = 'No viable calibrator fields for inf_EB in at least 1 EB'
                            for v in selfcal_library[target][band][fid]['vislist']:
                                selfcal_library[target][band][fid][v][solint]['Pass'] = 'None'
                                selfcal_library[target][band][fid][v][solint]['Fail_Reason'] = 'No viable inf_EB fields'

             selfcal_library[target][band]['sub-fields-to-selfcal'] = new_fields_to_selfcal

             for vis in vislist:
                ##
                ## Apply gain solutions per MS, target, solint, and band
                ##
                for fid in np.intersect1d(selfcal_library[target][band]['sub-fields'],list(selfcal_library[target][band]['sub-fields-fid_map'][vis].keys())):
                    if fid in selfcal_library[target][band]['sub-fields-to-selfcal']:
                        applycal(vis=vis,\
                                 gaintable=selfcal_library[target][band][fid][vis][solint]['gaintable'],\
                                 interp=selfcal_library[target][band][fid][vis][solint]['applycal_interpolate'], calwt=False,\
                                 spwmap=selfcal_library[target][band][fid][vis][solint]['spwmap'],\
                                 #applymode=applymode,field=target,spw=selfcal_library[target][band][vis]['spws'])
                                 applymode='calflag',field=str(selfcal_library[target][band]['sub-fields-fid_map'][vis][fid]),\
                                 spw=selfcal_library[target][band][vis]['spws'])
                    else:
                        if selfcal_library[target][band][fid]['SC_success']:
                            applycal(vis=vis,\
                                    gaintable=selfcal_library[target][band][fid][vis]['gaintable_final'],\
                                    interp=selfcal_library[target][band][fid][vis]['applycal_interpolate_final'],\
                                    calwt=False,spwmap=selfcal_library[target][band][fid][vis]['spwmap_final'],\
                                    applymode=selfcal_library[target][band][fid][vis]['applycal_mode_final'],\
                                    field=str(selfcal_library[target][band]['sub-fields-fid_map'][vis][fid]),\
                                    spw=selfcal_library[target][band][vis]['spws'])    

             ## Create post self-cal image using the model as a startmodel to evaluate how much selfcal helped
             ##

             os.system('rm -rf '+sani_target+'_'+band+'_'+solint+'_'+str(iteration)+'_post*')
             tclean_wrapper(vislist,sani_target+'_'+band+'_'+solint+'_'+str(iteration)+'_post',
                      band_properties,band,telescope=telescope,nsigma=selfcal_library[target][band]['nsigma'][iteration], scales=[0],
                      threshold=str(selfcal_library[target][band]['nsigma'][iteration]*selfcal_library[target][band]['RMS_NF_curr'])+'Jy',
                      savemodel='none',parallel=parallel,cellsize=cellsize[band],imsize=imsize[band],
                      nterms=selfcal_library[target][band]['nterms'],reffreq=selfcal_library[target][band]['reffreq'],
                      field=target,spw=selfcal_library[target][band]['spws_per_vis'],uvrange=selfcal_library[target][band]['uvrange'],obstype=selfcal_library[target][band]['obstype'], nfrms_multiplier=nfsnr_modifier, image_mosaic_fields_separately=selfcal_library[target][band]['obstype'] == 'mosaic', mosaic_field_phasecenters=selfcal_library[target][band]['sub-fields-phasecenters'], mosaic_field_fid_map=selfcal_library[target][band]['sub-fields-fid_map'], cyclefactor=selfcal_library[target][band]['cyclefactor'],mask=mask,usermodel=usermodel)

             ##
             ## Do the assessment of the post- (and pre-) selfcal images.
             ##
             print('Pre selfcal assessemnt: '+target)
             SNR,RMS=estimate_SNR(sani_target+'_'+band+'_'+solint+'_'+str(iteration)+'.image.tt0', \
                     maskname=sani_target+'_'+band+'_'+solint+'_'+str(iteration)+'_post.mask')
             if telescope !='ACA' or aca_use_nfmask:
                SNR_NF,RMS_NF=estimate_near_field_SNR(sani_target+'_'+band+'_'+solint+'_'+str(iteration)+'.image.tt0', \
                        maskname=sani_target+'_'+band+'_'+solint+'_'+str(iteration)+'_post.mask', las=selfcal_library[target][band]['LAS'])
                if RMS_NF < 0:
                    SNR_NF,RMS_NF=estimate_near_field_SNR(sani_target+'_'+band+'_'+solint+'_'+str(iteration)+'.image.tt0', \
                            maskname=sani_target+'_'+band+'_'+solint+'_'+str(iteration)+'.mask', las=selfcal_library[target][band]['LAS'])
             else:
                SNR_NF,RMS_NF=SNR,RMS

             print('Post selfcal assessemnt: '+target)
             post_SNR,post_RMS=estimate_SNR(sani_target+'_'+band+'_'+solint+'_'+str(iteration)+'_post.image.tt0')
             if telescope !='ACA' or aca_use_nfmask:
                post_SNR_NF,post_RMS_NF=estimate_near_field_SNR(sani_target+'_'+band+'_'+solint+'_'+str(iteration)+'_post.image.tt0', \
                        las=selfcal_library[target][band]['LAS'])
                if post_RMS_NF < 0:
                    post_SNR_NF,post_RMS_NF=estimate_near_field_SNR(sani_target+'_'+band+'_'+solint+'_'+str(iteration)+'_post.image.tt0', \
                            maskname=sani_target+'_'+band+'_'+solint+'_'+str(iteration)+'.mask', las=selfcal_library[target][band]['LAS'])
             else:
                post_SNR_NF,post_RMS_NF=post_SNR,post_RMS

             mosaic_SNR, mosaic_RMS, mosaic_SNR_NF, mosaic_RMS_NF = {}, {}, {}, {}
             post_mosaic_SNR, post_mosaic_RMS, post_mosaic_SNR_NF, post_mosaic_RMS_NF = {}, {}, {}, {}
             for fid in selfcal_library[target][band]['sub-fields-to-selfcal']:
                 if selfcal_library[target][band]['obstype'] == 'mosaic':
                     imagename = sani_target+'_field_'+str(fid)+'_'+band+'_'+solint+'_'+str(iteration)
                 else:
                     imagename = sani_target+'_'+band+'_'+solint+'_'+str(iteration)

                 print()
                 print('Pre selfcal assessemnt: '+target+', field '+str(fid))
                 mosaic_SNR[fid], mosaic_RMS[fid] = estimate_SNR(imagename+'.image.tt0', maskname=imagename+'_post.mask', \
                         mosaic_sub_field=selfcal_library[target][band]["obstype"]=="mosaic")
                 if telescope !='ACA' or aca_use_nfmask:
                    mosaic_SNR_NF[fid],mosaic_RMS_NF[fid]=estimate_near_field_SNR(imagename+'.image.tt0', maskname=imagename+'_post.mask', \
                            las=selfcal_library[target][band]['LAS'], mosaic_sub_field=selfcal_library[target][band]["obstype"]=="mosaic")
                    if mosaic_RMS_NF[fid] < 0:
                        mosaic_SNR_NF[fid],mosaic_RMS_NF[fid]=estimate_near_field_SNR(imagename+'.image.tt0', maskname=imagename+'.mask', \
                                las=selfcal_library[target][band]['LAS'], mosaic_sub_field=selfcal_library[target][band]["obstype"]=="mosaic")
                 else:
                    mosaic_SNR_NF[fid],mosaic_RMS_NF[fid]=mosaic_SNR[fid],mosaic_RMS[fid]

                 print('Post selfcal assessemnt: '+target+', field '+str(fid))
                 post_mosaic_SNR[fid], post_mosaic_RMS[fid] = estimate_SNR(imagename+'_post.image.tt0', \
                         mosaic_sub_field=selfcal_library[target][band]["obstype"]=="mosaic")
                 if telescope !='ACA' or aca_use_nfmask:
                    post_mosaic_SNR_NF[fid],post_mosaic_RMS_NF[fid]=estimate_near_field_SNR(imagename+'_post.image.tt0', \
                            las=selfcal_library[target][band]['LAS'], mosaic_sub_field=selfcal_library[target][band]["obstype"]=="mosaic")
                    if post_mosaic_RMS_NF[fid] < 0:
                        post_mosaic_SNR_NF[fid],post_mosaic_RMS_NF[fid]=estimate_near_field_SNR(imagename+'_post.image.tt0', \
                                maskname=imagename+'.mask', las=selfcal_library[target][band]['LAS'], \
                                mosaic_sub_field=selfcal_library[target][band]["obstype"]=="mosaic")
                 else:
                    post_mosaic_SNR_NF[fid],post_mosaic_RMS_NF[fid]=post_mosaic_SNR[fid],post_mosaic_RMS[fid]
                 print()

             # change nterms to 2 if needed based on fracbw and SNR
             if selfcal_library[target][band]['nterms'] == 1:
                 selfcal_library[target][band]['nterms']=check_image_nterms(selfcal_library[target][band]['fracbw'],post_SNR)

             for vis in vislist:
                ##
                ## record self cal results/details for this solint
                ##
                #selfcal_library[target][band][vis][solint]={}
                selfcal_library[target][band][vis][solint]['SNR_pre']=SNR.copy()
                selfcal_library[target][band][vis][solint]['RMS_pre']=RMS.copy()
                selfcal_library[target][band][vis][solint]['SNR_NF_pre']=SNR_NF.copy()
                selfcal_library[target][band][vis][solint]['RMS_NF_pre']=RMS_NF.copy()
                header=imhead(imagename=sani_target+'_'+band+'_'+solint+'_'+str(iteration)+'.image.tt0')
                selfcal_library[target][band][vis][solint]['Beam_major_pre']=header['restoringbeam']['major']['value']
                selfcal_library[target][band][vis][solint]['Beam_minor_pre']=header['restoringbeam']['minor']['value']
                selfcal_library[target][band][vis][solint]['Beam_PA_pre']=header['restoringbeam']['positionangle']['value'] 
                #selfcal_library[target][band][vis][solint]['gaintable']=applycal_gaintable[vis]
                #selfcal_library[target][band][vis][solint]['iteration']=iteration+0
                #selfcal_library[target][band][vis][solint]['spwmap']=applycal_spwmap[vis]
                #selfcal_library[target][band][vis][solint]['applycal_mode']=applycal_mode[band][target][iteration]+''
                #selfcal_library[target][band][vis][solint]['applycal_interpolate']=applycal_interpolate[vis]
                #selfcal_library[target][band][vis][solint]['gaincal_combine']=gaincal_combine[band][target][iteration]+''
                selfcal_library[target][band][vis][solint]['clean_threshold']=selfcal_library[target][band]['nsigma'][iteration]*selfcal_library[target][band]['RMS_NF_curr']
                if checkmask(sani_target+'_'+band+'_'+solint+'_'+str(iteration)+'_post.mask'):
                    selfcal_library[target][band][vis][solint]['intflux_pre'],selfcal_library[target][band][vis][solint]['e_intflux_pre']=get_intflux(sani_target+'_'+band+'_'+solint+'_'+str(iteration)+'.image.tt0',RMS,maskname=sani_target+'_'+band+'_'+solint+'_'+str(iteration)+'_post.mask')
                elif checkmask(sani_target+'_'+band+'_'+solint+'_'+str(iteration)+'.mask'):
                    selfcal_library[target][band][vis][solint]['intflux_pre'],selfcal_library[target][band][vis][solint]['e_intflux_pre']=get_intflux(sani_target+'_'+band+'_'+solint+'_'+str(iteration)+'.image.tt0',RMS,maskname=sani_target+'_'+band+'_'+solint+'_'+str(iteration)+'.mask')
                else:
                    selfcal_library[target][band][vis][solint]['intflux_pre'],selfcal_library[target][band][vis][solint]['e_intflux_pre']=-99.0,-99.0

                if vis in fallback:
                    selfcal_library[target][band][vis][solint]['fallback']=fallback[vis]+''
                else:
                    selfcal_library[target][band][vis][solint]['fallback']=''
                selfcal_library[target][band][vis][solint]['solmode']=solmode[band][target][iteration]+''
                selfcal_library[target][band][vis][solint]['SNR_post']=post_SNR.copy()
                selfcal_library[target][band][vis][solint]['RMS_post']=post_RMS.copy()
                selfcal_library[target][band][vis][solint]['SNR_NF_post']=post_SNR_NF.copy()
                selfcal_library[target][band][vis][solint]['RMS_NF_post']=post_RMS_NF.copy()
                header=imhead(imagename=sani_target+'_'+band+'_'+solint+'_'+str(iteration)+'_post.image.tt0')
                selfcal_library[target][band][vis][solint]['Beam_major_post']=header['restoringbeam']['major']['value']
                selfcal_library[target][band][vis][solint]['Beam_minor_post']=header['restoringbeam']['minor']['value']
                selfcal_library[target][band][vis][solint]['Beam_PA_post']=header['restoringbeam']['positionangle']['value'] 
                if checkmask(sani_target+'_'+band+'_'+solint+'_'+str(iteration)+'_post.mask'):
                    selfcal_library[target][band][vis][solint]['intflux_post'],selfcal_library[target][band][vis][solint]['e_intflux_post']=get_intflux(sani_target+'_'+band+'_'+solint+'_'+str(iteration)+'_post.image.tt0',post_RMS,maskname=sani_target+'_'+band+'_'+solint+'_'+str(iteration)+'_post.mask')
                elif checkmask(sani_target+'_'+band+'_'+solint+'_'+str(iteration)+'.mask'):
                    selfcal_library[target][band][vis][solint]['intflux_post'],selfcal_library[target][band][vis][solint]['e_intflux_post']=get_intflux(sani_target+'_'+band+'_'+solint+'_'+str(iteration)+'_post.image.tt0',post_RMS,maskname=sani_target+'_'+band+'_'+solint+'_'+str(iteration)+'.mask')
                else:
                    selfcal_library[target][band][vis][solint]['intflux_post'],selfcal_library[target][band][vis][solint]['e_intflux_post']=-99.0,-99.0

                for fid in np.intersect1d(selfcal_library[target][band]['sub-fields-to-selfcal'],list(selfcal_library[target][band]['sub-fields-fid_map'][vis].keys())):
                    if selfcal_library[target][band]['obstype'] == 'mosaic':
                        imagename = sani_target+'_field_'+str(fid)+'_'+band+'_'+solint+'_'+str(iteration)
                    else:
                        imagename = sani_target+'_'+band+'_'+solint+'_'+str(iteration)

                    #selfcal_library[target][band][fid][vis][solint]={}
                    selfcal_library[target][band][fid][vis][solint]['SNR_pre']=mosaic_SNR[fid].copy()
                    selfcal_library[target][band][fid][vis][solint]['RMS_pre']=mosaic_RMS[fid].copy()
                    selfcal_library[target][band][fid][vis][solint]['SNR_NF_pre']=mosaic_SNR_NF[fid].copy()
                    selfcal_library[target][band][fid][vis][solint]['RMS_NF_pre']=mosaic_RMS_NF[fid].copy()
                    header=imhead(imagename=imagename+'.image.tt0')
                    selfcal_library[target][band][fid][vis][solint]['Beam_major_pre']=header['restoringbeam']['major']['value']
                    selfcal_library[target][band][fid][vis][solint]['Beam_minor_pre']=header['restoringbeam']['minor']['value']
                    selfcal_library[target][band][fid][vis][solint]['Beam_PA_pre']=header['restoringbeam']['positionangle']['value'] 
                    #selfcal_library[target][band][fid][vis][solint]['gaintable']=applycal_gaintable[vis]
                    #selfcal_library[target][band][fid][vis][solint]['iteration']=iteration+0
                    #selfcal_library[target][band][fid][vis][solint]['spwmap']=applycal_spwmap[vis]
                    #selfcal_library[target][band][fid][vis][solint]['applycal_mode']=applycal_mode[band][target][iteration]+''
                    #selfcal_library[target][band][fid][vis][solint]['applycal_interpolate']=applycal_interpolate[vis]
                    #selfcal_library[target][band][fid][vis][solint]['gaincal_combine']=gaincal_combine[band][target][iteration]+''
                    selfcal_library[target][band][fid][vis][solint]['clean_threshold']=selfcal_library[target][band]['nsigma'][iteration]*selfcal_library[target][band]['RMS_NF_curr']
                    if checkmask(imagename=imagename+'_post.mask'):
                        selfcal_library[target][band][fid][vis][solint]['intflux_pre'],selfcal_library[target][band][fid][vis][solint]['e_intflux_pre']=get_intflux(imagename+'.image.tt0',mosaic_RMS[fid], maskname=imagename+'_post.mask', mosaic_sub_field=selfcal_library[target][band]["obstype"]=="mosaic")
                    elif checkmask(imagename=imagename+'.mask'):
                        selfcal_library[target][band][fid][vis][solint]['intflux_pre'],selfcal_library[target][band][fid][vis][solint]['e_intflux_pre']=get_intflux(imagename+'.image.tt0',mosaic_RMS[fid], maskname=imagename+'.mask', mosaic_sub_field=selfcal_library[target][band]["obstype"]=="mosaic")
                    else:
                        selfcal_library[target][band][fid][vis][solint]['intflux_pre'],selfcal_library[target][band][fid][vis][solint]['e_intflux_pre']=-99.0,-99.0
                    if vis in fallback:
                        selfcal_library[target][band][fid][vis][solint]['fallback']=fallback[vis]+''
                    else:
                        selfcal_library[target][band][fid][vis][solint]['fallback']=''
                    selfcal_library[target][band][fid][vis][solint]['solmode']=solmode[band][target][iteration]+''
                    selfcal_library[target][band][fid][vis][solint]['SNR_post']=post_mosaic_SNR[fid].copy()
                    selfcal_library[target][band][fid][vis][solint]['RMS_post']=post_mosaic_RMS[fid].copy()
                    selfcal_library[target][band][fid][vis][solint]['SNR_NF_post']=post_mosaic_SNR_NF[fid].copy()
                    selfcal_library[target][band][fid][vis][solint]['RMS_NF_post']=post_mosaic_RMS_NF[fid].copy()
                    ## Update RMS value if necessary
                    """
                    if selfcal_library[target][band][vis][solint]['RMS_post'] < selfcal_library[target][band]['RMS_curr']:
                       selfcal_library[target][band]['RMS_curr']=selfcal_library[target][band][vis][solint]['RMS_post'].copy()
                    if selfcal_library[target][band][vis][solint]['RMS_NF_post'] < selfcal_library[target][band]['RMS_NF_curr']:
                       selfcal_library[target][band]['RMS_NF_curr']=selfcal_library[target][band][vis][solint]['RMS_NF_post'].copy()
                    """
                    header=imhead(imagename=imagename+'_post.image.tt0')
                    selfcal_library[target][band][fid][vis][solint]['Beam_major_post']=header['restoringbeam']['major']['value']
                    selfcal_library[target][band][fid][vis][solint]['Beam_minor_post']=header['restoringbeam']['minor']['value']
                    selfcal_library[target][band][fid][vis][solint]['Beam_PA_post']=header['restoringbeam']['positionangle']['value'] 
                    if checkmask(imagename+'_post.mask'):
                        selfcal_library[target][band][fid][vis][solint]['intflux_post'],selfcal_library[target][band][fid][vis][solint]['e_intflux_post']=get_intflux(imagename+'_post.image.tt0',post_mosaic_RMS[fid], maskname=imagename+'_post.mask', mosaic_sub_field=selfcal_library[target][band]["obstype"]=="mosaic")
                    elif checkmask(imagename+'.mask'):
                        selfcal_library[target][band][fid][vis][solint]['intflux_post'],selfcal_library[target][band][fid][vis][solint]['e_intflux_post']=get_intflux(imagename+'_post.image.tt0',post_mosaic_RMS[fid], maskname=imagename+'.mask', mosaic_sub_field=selfcal_library[target][band]["obstype"]=="mosaic")
                    else:
                        selfcal_library[target][band][fid][vis][solint]['intflux_post'],selfcal_library[target][band][fid][vis][solint]['e_intflux_post']=-99.0,-99.0

                ## Update RMS value if necessary
                if selfcal_library[target][band][vis][solint]['RMS_post'] < selfcal_library[target][band]['RMS_curr'] and vis == vislist[-1]:
                   selfcal_library[target][band]['RMS_curr']=selfcal_library[target][band][vis][solint]['RMS_post'].copy()
                if selfcal_library[target][band][vis][solint]['RMS_NF_post'] < selfcal_library[target][band]['RMS_NF_curr'] and \
                        selfcal_library[target][band][vis][solint]['RMS_NF_post'] > 0 and vis == vislist[-1]:
                   selfcal_library[target][band]['RMS_NF_curr']=selfcal_library[target][band][vis][solint]['RMS_NF_post'].copy()

             ##
             ## compare beam relative to original image to ensure we are not incrementally changing the beam in each iteration
             ##
             beamarea_orig=selfcal_library[target][band]['Beam_major_orig']*selfcal_library[target][band]['Beam_minor_orig']
             beamarea_post=selfcal_library[target][band][vislist[0]][solint]['Beam_major_post']*selfcal_library[target][band][vislist[0]][solint]['Beam_minor_post']
             '''
             frac_delta_b_maj=np.abs((b_maj_post-selfcal_library[target]['Beam_major_orig'])/selfcal_library[target]['Beam_major_orig'])
             frac_delta_b_min=np.abs((b_min_post-selfcal_library[target]['Beam_minor_orig'])/selfcal_library[target]['Beam_minor_orig'])
             delta_b_pa=np.abs((b_pa_post-selfcal_library[target]['Beam_PA_orig']))
             '''
             delta_beamarea=(beamarea_post-beamarea_orig)/beamarea_orig
             ## 
             ## if S/N improvement, and beamarea is changing by < delta_beam_thresh, accept solutions to main calibration dictionary
             ## allow to proceed if solint was inf_EB and SNR decrease was less than 2%
             ##
             strict_field_by_field_success = []
             loose_field_by_field_success = []
             beam_field_by_field_success = []
             rms_field_by_field_success = []
             for fid in selfcal_library[target][band]['sub-fields-to-selfcal']:
                 strict_field_by_field_success += [(post_mosaic_SNR[fid] >= mosaic_SNR[fid]) and (post_mosaic_SNR_NF[fid] >= mosaic_SNR_NF[fid])]
                 loose_field_by_field_success += [((post_mosaic_SNR[fid]-mosaic_SNR[fid])/mosaic_SNR[fid] > -0.02) and \
                         ((post_mosaic_SNR_NF[fid] - mosaic_SNR_NF[fid])/mosaic_SNR_NF[fid] > -0.02)]
                 beam_field_by_field_success += [delta_beamarea < delta_beam_thresh]
                 rms_field_by_field_success = ((post_mosaic_RMS[fid] - mosaic_RMS[fid])/mosaic_RMS[fid] < 1.05 and \
                         (post_mosaic_RMS_NF[fid] - mosaic_RMS_NF[fid])/mosaic_RMS_NF[fid] < 1.05) or \
                         (((post_mosaic_RMS[fid] - mosaic_RMS[fid])/mosaic_RMS[fid] > 1.05 or \
                         (post_mosaic_RMS_NF[fid] - mosaic_RMS_NF[fid])/mosaic_RMS_NF[fid] > 1.05) and \
                         solint_snr_per_field[target][band][fid][solint] > 5)

             if solint == 'inf_EB':
                 # If any of the fields succeed in the "strict" sense, then allow for minor reductions in the evaluation quantity in other
                 # fields because there's a good chance that those are just noise being pushed around.
                 field_by_field_success = numpy.logical_and(numpy.logical_and(loose_field_by_field_success, beam_field_by_field_success), \
                         rms_field_by_field_success)
             else:
                 field_by_field_success = numpy.logical_and(numpy.logical_and(strict_field_by_field_success, beam_field_by_field_success), \
                         rms_field_by_field_success)

             # If not all fields were successful, we need to make an additional image to evaluate whether the image as a whole improved,
             # otherwise the _post image won't be exactly representative.
             if selfcal_library[target][band]['obstype'] == "mosaic" and not np.all(field_by_field_success):
                 field_by_field_success_dict = dict(zip(selfcal_library[target][band]['sub-fields-to-selfcal'], field_by_field_success))
                 print('****************Not all fields were successful, so re-applying and re-making _post image*************')
                 for vis in vislist:
                     flagmanager(vis=vis,mode='restore',versionname='selfcal_starting_flags_'+sani_target)
                     for fid in np.intersect1d(selfcal_library[target][band]['sub-fields'],list(selfcal_library[target][band]['sub-fields-fid_map'][vis].keys())):
                         if fid not in field_by_field_success_dict or not field_by_field_success_dict[fid]:
                             if selfcal_library[target][band][fid]['SC_success']:
                                 print('****************Applying '+str(selfcal_library[target][band][fid][vis]['gaintable_final'])+' to '+target+' field '+\
                                         str(fid)+' '+band+'*************')
                                 applycal(vis=vis,\
                                         gaintable=selfcal_library[target][band][fid][vis]['gaintable_final'],\
                                         interp=selfcal_library[target][band][fid][vis]['applycal_interpolate_final'],\
                                         calwt=False,spwmap=selfcal_library[target][band][fid][vis]['spwmap_final'],\
                                         applymode=selfcal_library[target][band][fid][vis]['applycal_mode_final'],\
                                         field=str(selfcal_library[target][band]['sub-fields-fid_map'][vis][fid]),\
                                         spw=selfcal_library[target][band][vis]['spws'])    
                             else:
                                 print('****************Removing all calibrations for '+target+' '+str(fid)+' '+band+'**************')
                                 clearcal(vis=vis,field=str(selfcal_library[target][band]['sub-fields-fid_map'][vis][fid]),\
                                         spw=selfcal_library[target][band][vis]['spws'])
                         else:
                             applycal(vis=vis,\
                                      gaintable=selfcal_library[target][band][fid][vis][solint]['gaintable'],\
                                      interp=selfcal_library[target][band][fid][vis][solint]['applycal_interpolate'], calwt=False,\
                                      spwmap=selfcal_library[target][band][fid][vis][solint]['spwmap'],\
                                      #applymode=applymode,field=target,spw=selfcal_library[target][band][vis]['spws'])
                                      applymode='calflag',field=str(selfcal_library[target][band]['sub-fields-fid_map'][vis][fid]),\
                                      spw=selfcal_library[target][band][vis]['spws'])

                 files = glob.glob(sani_target+'_'+band+'_'+solint+'_'+str(iteration)+"_post.*")
                 for f in files:
                     os.system("mv "+f+" "+f.replace("_post","_post_intermediate"))

                 tclean_wrapper(vislist,sani_target+'_'+band+'_'+solint+'_'+str(iteration)+'_post',
                          band_properties,band,telescope=telescope,nsigma=selfcal_library[target][band]['nsigma'][iteration], scales=[0],
                          threshold=str(selfcal_library[target][band][vislist[0]][solint]['clean_threshold'])+'Jy',
                          savemodel='none',parallel=parallel,cellsize=cellsize[band],imsize=imsize[band],
                          nterms=selfcal_library[target][band]['nterms'],reffreq=selfcal_library[target][band]['reffreq'],
                          field=target,spw=selfcal_library[target][band]['spws_per_vis'],uvrange=selfcal_library[target][band]['uvrange'],obstype=selfcal_library[target][band]['obstype'], nfrms_multiplier=nfsnr_modifier, image_mosaic_fields_separately=False, mosaic_field_phasecenters=selfcal_library[target][band]['sub-fields-phasecenters'], mosaic_field_fid_map=selfcal_library[target][band]['sub-fields-fid_map'], cyclefactor=selfcal_library[target][band]['cyclefactor'],mask=mask,usermodel=usermodel)

                 ##
                 ## Do the assessment of the post- (and pre-) selfcal images.
                 ##
                 print('Pre selfcal assessemnt: '+target)
                 SNR,RMS=estimate_SNR(sani_target+'_'+band+'_'+solint+'_'+str(iteration)+'.image.tt0', \
                         maskname=sani_target+'_'+band+'_'+solint+'_'+str(iteration)+'_post.mask')
                 if telescope !='ACA' or aca_use_nfmask:
                    SNR_NF,RMS_NF=estimate_near_field_SNR(sani_target+'_'+band+'_'+solint+'_'+str(iteration)+'.image.tt0', \
                            maskname=sani_target+'_'+band+'_'+solint+'_'+str(iteration)+'_post.mask', las=selfcal_library[target][band]['LAS'])
                    if RMS_NF < 0:
                        SNR_NF,RMS_NF=estimate_near_field_SNR(sani_target+'_'+band+'_'+solint+'_'+str(iteration)+'.image.tt0', \
                                maskname=sani_target+'_'+band+'_'+solint+'_'+str(iteration)+'.mask', las=selfcal_library[target][band]['LAS'])
                 else:
                    SNR_NF,RMS_NF=SNR,RMS

                 print('Post selfcal assessemnt: '+target)
                 post_SNR,post_RMS=estimate_SNR(sani_target+'_'+band+'_'+solint+'_'+str(iteration)+'_post.image.tt0')
                 if telescope !='ACA' or aca_use_nfmask:
                    post_SNR_NF,post_RMS_NF=estimate_near_field_SNR(sani_target+'_'+band+'_'+solint+'_'+str(iteration)+'_post.image.tt0', \
                            las=selfcal_library[target][band]['LAS'])
                    if post_RMS_NF < 0:
                        post_SNR_NF,post_RMS_NF=estimate_near_field_SNR(sani_target+'_'+band+'_'+solint+'_'+str(iteration)+'_post.image.tt0', \
                                maskname=sani_target+'_'+band+'_'+solint+'_'+str(iteration)+'.mask', las=selfcal_library[target][band]['LAS'])
                 else:
                    post_SNR_NF,post_RMS_NF=post_SNR,post_RMS

                 for vis in vislist:
                    ##
                    ## record self cal results/details for this solint
                    ##
                    #selfcal_library[target][band][vis][solint]={}
                    selfcal_library[target][band][vis][solint]['SNR_pre']=SNR.copy()
                    selfcal_library[target][band][vis][solint]['RMS_pre']=RMS.copy()
                    selfcal_library[target][band][vis][solint]['SNR_NF_pre']=SNR_NF.copy()
                    selfcal_library[target][band][vis][solint]['RMS_NF_pre']=RMS_NF.copy()
                    header=imhead(imagename=sani_target+'_'+band+'_'+solint+'_'+str(iteration)+'.image.tt0')
                    selfcal_library[target][band][vis][solint]['Beam_major_pre']=header['restoringbeam']['major']['value']
                    selfcal_library[target][band][vis][solint]['Beam_minor_pre']=header['restoringbeam']['minor']['value']
                    selfcal_library[target][band][vis][solint]['Beam_PA_pre']=header['restoringbeam']['positionangle']['value'] 
                    #selfcal_library[target][band][vis][solint]['gaintable']=applycal_gaintable[vis]
                    #selfcal_library[target][band][vis][solint]['iteration']=iteration+0
                    #selfcal_library[target][band][vis][solint]['spwmap']=applycal_spwmap[vis]
                    #selfcal_library[target][band][vis][solint]['applycal_mode']=applycal_mode[band][target][iteration]+''
                    #selfcal_library[target][band][vis][solint]['applycal_interpolate']=applycal_interpolate[vis]
                    #selfcal_library[target][band][vis][solint]['gaincal_combine']=gaincal_combine[band][target][iteration]+''
                    selfcal_library[target][band][vis][solint]['clean_threshold']=selfcal_library[target][band]['nsigma'][iteration]*selfcal_library[target][band]['RMS_NF_curr']
                    if checkmask(sani_target+'_'+band+'_'+solint+'_'+str(iteration)+'_post.mask'):
                        selfcal_library[target][band][vis][solint]['intflux_pre'],selfcal_library[target][band][vis][solint]['e_intflux_pre']=get_intflux(sani_target+'_'+band+'_'+solint+'_'+str(iteration)+'.image.tt0',RMS,maskname=sani_target+'_'+band+'_'+solint+'_'+str(iteration)+'_post.mask')
                    elif checkmask(sani_target+'_'+band+'_'+solint+'_'+str(iteration)+'.mask'):
                        selfcal_library[target][band][vis][solint]['intflux_pre'],selfcal_library[target][band][vis][solint]['e_intflux_pre']=get_intflux(sani_target+'_'+band+'_'+solint+'_'+str(iteration)+'.image.tt0',RMS,maskname=sani_target+'_'+band+'_'+solint+'_'+str(iteration)+'.mask')
                    else:
                        selfcal_library[target][band][vis][solint]['intflux_pre'],selfcal_library[target][band][vis][solint]['e_intflux_pre']=-99.0,-99.0

                    if vis in fallback:
                        selfcal_library[target][band][vis][solint]['fallback']=fallback[vis]+''
                    else:
                        selfcal_library[target][band][vis][solint]['fallback']=''
                    selfcal_library[target][band][vis][solint]['solmode']=solmode[band][target][iteration]+''
                    selfcal_library[target][band][vis][solint]['SNR_post']=post_SNR.copy()
                    selfcal_library[target][band][vis][solint]['RMS_post']=post_RMS.copy()
                    selfcal_library[target][band][vis][solint]['SNR_NF_post']=post_SNR_NF.copy()
                    selfcal_library[target][band][vis][solint]['RMS_NF_post']=post_RMS_NF.copy()
                    ## Update RMS value if necessary
                    if selfcal_library[target][band][vis][solint]['RMS_post'] < selfcal_library[target][band]['RMS_curr']:
                       selfcal_library[target][band]['RMS_curr']=selfcal_library[target][band][vis][solint]['RMS_post'].copy()
                    if selfcal_library[target][band][vis][solint]['RMS_NF_post'] < selfcal_library[target][band]['RMS_NF_curr'] and \
                            selfcal_library[target][band][vis][solint]['RMS_NF_post'] > 0:
                       selfcal_library[target][band]['RMS_NF_curr']=selfcal_library[target][band][vis][solint]['RMS_NF_post'].copy()
                    header=imhead(imagename=sani_target+'_'+band+'_'+solint+'_'+str(iteration)+'_post.image.tt0')
                    selfcal_library[target][band][vis][solint]['Beam_major_post']=header['restoringbeam']['major']['value']
                    selfcal_library[target][band][vis][solint]['Beam_minor_post']=header['restoringbeam']['minor']['value']
                    selfcal_library[target][band][vis][solint]['Beam_PA_post']=header['restoringbeam']['positionangle']['value'] 
                    if checkmask(sani_target+'_'+band+'_'+solint+'_'+str(iteration)+'_post.mask'):
                        selfcal_library[target][band][vis][solint]['intflux_post'],selfcal_library[target][band][vis][solint]['e_intflux_post']=get_intflux(sani_target+'_'+band+'_'+solint+'_'+str(iteration)+'_post.image.tt0',post_RMS,maskname=sani_target+'_'+band+'_'+solint+'_'+str(iteration)+'_post.mask')
                    elif checkmask(sani_target+'_'+band+'_'+solint+'_'+str(iteration)+'.mask'):
                        selfcal_library[target][band][vis][solint]['intflux_post'],selfcal_library[target][band][vis][solint]['e_intflux_post']=get_intflux(sani_target+'_'+band+'_'+solint+'_'+str(iteration)+'_post.image.tt0',post_RMS,maskname=sani_target+'_'+band+'_'+solint+'_'+str(iteration)+'.mask')
                    else:
                        selfcal_library[target][band][vis][solint]['intflux_post'],selfcal_library[target][band][vis][solint]['e_intflux_post']=-99.0,-99.0

             marginal_inf_EB_will_attempt_next_solint=False
             #run a pre-check as to whether a marginal inf_EB result will go on to attempt inf, if not we will fail a marginal inf_EB
             if (solint =='inf_EB') and ((((post_SNR-SNR)/SNR > -0.02) and ((post_SNR-SNR)/SNR < 0.00)) or (((post_SNR_NF - SNR_NF)/SNR_NF > -0.02) and ((post_SNR_NF - SNR_NF)/SNR_NF < 0.00))) and (delta_beamarea < delta_beam_thresh):
                if solint_snr[target][band][solints[band][target][iteration+1]] < minsnr_to_proceed and np.all([solint_snr_per_field[target][band][fid][solints[band][target][iteration+1]] < minsnr_to_proceed for fid in selfcal_library[target][band]['sub-fields']]):
                   marginal_inf_EB_will_attempt_next_solint = False
                else:
                   marginal_inf_EB_will_attempt_next_solint =  True

             RMS_change_acceptable = (post_RMS/RMS < 1.05 and post_RMS_NF/RMS_NF < 1.05) or \
                     ((post_RMS/RMS > 1.05 or post_RMS_NF/RMS_NF > 1.05) and solint_snr[target][band][solint] > 5)

             if (((post_SNR >= SNR) and (post_SNR_NF >= SNR_NF) and (delta_beamarea < delta_beam_thresh)) or ((solint =='inf_EB') and marginal_inf_EB_will_attempt_next_solint and ((post_SNR-SNR)/SNR > -0.02) and ((post_SNR_NF - SNR_NF)/SNR_NF > -0.02) and (delta_beamarea < delta_beam_thresh))) and np.any(field_by_field_success) and RMS_change_acceptable: 
                selfcal_library[target][band]['SC_success']=True
                selfcal_library[target][band]['Stop_Reason']='None'
                #keep track of whether inf_EB had a S/N decrease
                if (solint =='inf_EB') and (((post_SNR-SNR)/SNR < 0.0) or ((post_SNR_NF - SNR_NF)/SNR_NF < 0.0)):
                   selfcal_library[target][band]['inf_EB_SNR_decrease']=True
                elif (solint =='inf_EB') and (((post_SNR-SNR)/SNR > 0.0) and ((post_SNR_NF - SNR_NF)/SNR_NF > 0.0)):
                   selfcal_library[target][band]['inf_EB_SNR_decrease']=False
                for vis in vislist:
                   selfcal_library[target][band][vis]['gaintable_final']=selfcal_library[target][band][vis][solint]['gaintable']
                   selfcal_library[target][band][vis]['spwmap_final']=selfcal_library[target][band][vis][solint]['spwmap'].copy()
                   selfcal_library[target][band][vis]['applycal_mode_final']=selfcal_library[target][band][vis][solint]['applycal_mode']
                   selfcal_library[target][band][vis]['applycal_interpolate_final']=selfcal_library[target][band][vis][solint]['applycal_interpolate']
                   selfcal_library[target][band][vis]['gaincal_combine_final']=selfcal_library[target][band][vis][solint]['gaincal_combine']
                   selfcal_library[target][band][vis][solint]['Pass']=True
                   selfcal_library[target][band][vis][solint]['Fail_Reason']='None'
                if solmode[band][target][iteration]=='p':            
                   selfcal_library[target][band]['final_phase_solint']=solint
                selfcal_library[target][band]['final_solint']=solint
                selfcal_library[target][band]['final_solint_mode']=solmode[band][target][iteration]
                selfcal_library[target][band]['iteration']=iteration

                for ind, fid in enumerate(selfcal_library[target][band]['sub-fields-to-selfcal']):
                    if field_by_field_success[ind]:
                        selfcal_library[target][band][fid]['SC_success']=True
                        selfcal_library[target][band][fid]['Stop_Reason']='None'
                        if (solint =='inf_EB') and not strict_field_by_field_success[ind]:
                           selfcal_library[target][band][fid]['inf_EB_SNR_decrease']=True
                        elif (solint =='inf_EB') and strict_field_by_field_success[ind]:
                           selfcal_library[target][band][fid]['inf_EB_SNR_decrease']=False

                        for vis in selfcal_library[target][band][fid]['vislist']:
                           selfcal_library[target][band][fid][vis]['gaintable_final']=selfcal_library[target][band][fid][vis][solint]['gaintable']
                           selfcal_library[target][band][fid][vis]['spwmap_final']=selfcal_library[target][band][fid][vis][solint]['spwmap'].copy()
                           selfcal_library[target][band][fid][vis]['applycal_mode_final']=selfcal_library[target][band][fid][vis][solint]['applycal_mode']
                           selfcal_library[target][band][fid][vis]['applycal_interpolate_final']=selfcal_library[target][band][fid][vis][solint]['applycal_interpolate']
                           selfcal_library[target][band][fid][vis]['gaincal_combine_final']=selfcal_library[target][band][fid][vis][solint]['gaincal_combine']
                           selfcal_library[target][band][fid][vis][solint]['Pass']=True
                           selfcal_library[target][band][fid][vis][solint]['Fail_Reason']='None'
                        if solmode[band][target][iteration]=='p':            
                           selfcal_library[target][band][fid]['final_phase_solint']=solint
                        selfcal_library[target][band][fid]['final_solint']=solint
                        selfcal_library[target][band][fid]['final_solint_mode']=solmode[band][target][iteration]
                        selfcal_library[target][band][fid]['iteration']=iteration
                    else:
                        for vis in selfcal_library[target][band][fid]['vislist']:
                            selfcal_library[target][band][fid][vis][solint]['Pass']=False
                        if solint == 'inf_EB':
                            selfcal_library[target][band][fid]['inf_EB_SNR_decrease']=False

                # To exit out of the applymode loop.
                break
             ##
             ## If the beam area got larger, this could be because of flagging of long baseline antennas. Try with applymode = "calonly".
             ##

             elif delta_beamarea > delta_beam_thresh and applymode == "calflag":
                 print('****************************Selfcal failed**************************')
                 print('REASON: Beam change beyond '+str(delta_beam_thresh))
                 if iteration > 0: # reapply only the previous gain tables, to get rid of solutions from this selfcal round
                    print('****************Reapplying previous solint solutions*************')
                    for vis in vislist:
                       flagmanager(vis=vis,mode='restore',versionname='selfcal_starting_flags_'+sani_target)
                       for fid in np.intersect1d(selfcal_library[target][band]['sub-fields'],list(selfcal_library[target][band]['sub-fields-fid_map'][vis].keys())):
                           if selfcal_library[target][band][fid]['SC_success']:
                               print('****************Applying '+str(selfcal_library[target][band][vis]['gaintable_final'])+' to '+target+\
                                       ' field '+str(fid)+' '+band+'*************')
                               applycal(vis=vis,\
                                       gaintable=selfcal_library[target][band][fid][vis]['gaintable_final'],\
                                       interp=selfcal_library[target][band][fid][vis]['applycal_interpolate_final'],\
                                       calwt=False,spwmap=selfcal_library[target][band][fid][vis]['spwmap_final'],\
                                       applymode=selfcal_library[target][band][fid][vis]['applycal_mode_final'],\
                                       field=str(selfcal_library[target][band]['sub-fields-fid_map'][vis][fid]),\
                                       spw=selfcal_library[target][band][vis]['spws'])    
                 else:
                    for vis in vislist:
                       inf_EB_gaincal_combine_dict[target][band][vis]=inf_EB_gaincal_combine #'scan'
                       if selfcal_library[target][band]['obstype']=='mosaic':
                          inf_EB_gaincal_combine_dict[target][band][vis]+=',field'   
                       inf_EB_gaintype_dict[target][band][vis]=inf_EB_gaintype #G
                       inf_EB_fallback_mode_dict[target][band][vis]='' #'scan'
                 print('****************Attempting applymode="calonly" fallback*************')
             else:
                for vis in vislist:
                   selfcal_library[target][band][vis][solint]['Pass']=False

                for fid in selfcal_library[target][band]['sub-fields-to-selfcal']:
                    for vis in selfcal_library[target][band][fid]['vislist']:
                        selfcal_library[target][band][fid][vis][solint]['Pass']=False
                break


         ## 
         ## if S/N worsens, and/or beam area increases reject current solutions and reapply previous (or revert to origional data)
         ##

         if not selfcal_library[target][band][vislist[0]][solint]['Pass'] or (solint == 'inf_EB' and selfcal_library[target][band]['inf_EB_SNR_decrease']):
            reason=''
            if (post_SNR <= SNR):
               reason=reason+' S/N decrease'
            if (post_SNR_NF < SNR_NF):
               if reason != '':
                   reason += '; '
               reason = reason + ' NF S/N decrease'
            if (delta_beamarea > delta_beam_thresh):
               if reason !='':
                  reason=reason+'; '
               reason=reason+'Beam change beyond '+str(delta_beam_thresh)
            if (post_RMS/RMS > 1.05 and solint_snr[target][band][solint] <= 5):
               if reason != '':
                  reason=reason+'; '
               reason=reason+'RMS increase beyond 5%'
            if (post_RMS_NF/RMS_NF > 1.05 and solint_snr[target][band][solint] <= 5):
               if reason != '':
                  reason=reason+'; '
               reason=reason+'NF RMS increase beyond 5%'
            if not np.any(field_by_field_success):
                if reason != '':
                    reason=reason+'; '
                reason=reason+'All sub-fields failed'
            selfcal_library[target][band]['Stop_Reason']=reason
            for vis in vislist:
               #selfcal_library[target][band][vis][solint]['Pass']=False
               selfcal_library[target][band][vis][solint]['Fail_Reason']=reason

         mosaic_reason = {}
         new_fields_to_selfcal = []
         for fid in selfcal_library[target][band]['sub-fields-to-selfcal']:
             if not selfcal_library[target][band][fid][selfcal_library[target][band][fid]['vislist'][0]][solint]['Pass'] or \
                     (solint == "inf_EB" and selfcal_library[target][band][fid]['inf_EB_SNR_decrease']):
                 mosaic_reason[fid]=''
                 if (post_mosaic_SNR[fid] <= mosaic_SNR[fid]):
                    mosaic_reason[fid]=mosaic_reason[fid]+' SNR decrease'
                 if (post_mosaic_SNR_NF[fid] < mosaic_SNR_NF[fid]):
                    if mosaic_reason[fid] != '':
                        mosaic_reason[fid] += '; '
                    mosaic_reason[fid] = mosaic_reason[fid] + ' NF SNR decrease'
                 if (delta_beamarea > delta_beam_thresh):
                    if mosaic_reason[fid] !='':
                       mosaic_reason[fid]=mosaic_reason[fid]+'; '
                    mosaic_reason[fid]=mosaic_reason[fid]+'Beam change beyond '+str(delta_beam_thresh)
                 if (post_RMS/RMS > 1.05 and solint_snr[target][band][solint] <= 5):
                    if mosaic_reason[fid] != '':
                       mosaic_reason[fid]=mosaic_reason[fid]+'; '
                    mosaic_reason[fid]=mosaic_reason[fid]+'RMS increase beyond 5%'
                 if (post_RMS_NF/RMS_NF > 1.05 and solint_snr[target][band][solint] <= 5):
                    if mosaic_reason[fid] != '':
                       mosaic_reason[fid]=mosaic_reason[fid]+'; '
                    mosaic_reason[fid]=mosaic_reason[fid]+'NF RMS increase beyond 5%'
                 if mosaic_reason[fid] == '':
                     mosaic_reason[fid] = "Global selfcal failed"
                 selfcal_library[target][band][fid]['Stop_Reason']=mosaic_reason[fid]
                 for vis in selfcal_library[target][band][fid]['vislist']:
                    #selfcal_library[target][band][fid][vis][solint]['Pass']=False
                    selfcal_library[target][band][fid][vis][solint]['Fail_Reason']=mosaic_reason[fid]

             if selfcal_library[target][band][fid][selfcal_library[target][band][fid]['vislist'][0]][solint]['Pass']:
                 new_fields_to_selfcal.append(fid)

         # If any of the fields failed self-calibration, we need to re-apply calibrations for all fields because we need to revert flagging back
         # to the starting point.
         if np.any([selfcal_library[target][band][fid][selfcal_library[target][band][fid]['vislist'][0]][solint]['Pass'] == False for fid in \
                 selfcal_library[target][band]['sub-fields-to-selfcal']]) or len(selfcal_library[target][band]['sub-fields-to-selfcal']) < \
                 len(selfcal_library[target][band]['sub-fields']):
             print('****************Selfcal failed for some sub-fields:*************')
             for fid in selfcal_library[target][band]['sub-fields']:
                 if fid in selfcal_library[target][band]['sub-fields-to-selfcal']:
                     if selfcal_library[target][band][fid][selfcal_library[target][band][fid]['vislist'][0]][solint]['Pass'] == False:
                         print('FIELD: '+str(fid)+', REASON: '+mosaic_reason[fid])
                 else:
                     print('FIELD: '+str(fid)+', REASON: Failed earlier solint')
             print('****************Reapplying previous solint solutions where available*************')

             #if the final successful solint was inf_EB but inf_EB had a S/N decrease, don't count it as a success and revert to no selfcal
             if selfcal_library[target][band]['final_solint'] == 'inf_EB' and selfcal_library[target][band]['inf_EB_SNR_decrease']:
                selfcal_library[target][band]['SC_success']=False
                selfcal_library[target][band]['final_solint']='None'
                for vis in vislist:
                   selfcal_library[target][band][vis]['inf_EB']['Pass']=False    #  remove the success from inf_EB
                   selfcal_library[target][band][vis]['inf_EB']['Fail_Reason']+=' with no successful solints later'    #  remove the success from inf_EB
                
             # Only set the inf_EB Pass flag to False if the mosaic as a whole failed or if this is the last phase-only solint (either because it is int or
             # because the solint failed, because for mosaics we can keep trying the field as we clean deeper. If we set to False now, that wont happen.
             for fid in np.intersect1d(selfcal_library[target][band]['sub-fields'],list(selfcal_library[target][band]['sub-fields-fid_map'][vis].keys())):
                if (selfcal_library[target][band]['final_solint'] == 'inf_EB' and selfcal_library[target][band]['inf_EB_SNR_decrease']) or \
                        ((not selfcal_library[target][band][vislist[0]][solint]['Pass'] or solint == 'int') and \
                        (selfcal_library[target][band][fid]['final_solint'] == 'inf_EB' and selfcal_library[target][band][fid]['inf_EB_SNR_decrease'])):
                   selfcal_library[target][band][fid]['SC_success']=False
                   selfcal_library[target][band][fid]['final_solint']='None'
                   for vis in vislist:
                      selfcal_library[target][band][fid][vis]['inf_EB']['Pass']=False    #  remove the success from inf_EB
                      selfcal_library[target][band][fid][vis]['inf_EB']['Fail_Reason']+=' with no successful solints later'    #  remove the success from inf_EB

             for vis in vislist:
                 flagmanager(vis=vis,mode='restore',versionname='selfcal_starting_flags_'+sani_target)
                 for fid in np.intersect1d(selfcal_library[target][band]['sub-fields'],list(selfcal_library[target][band]['sub-fields-fid_map'][vis].keys())):
                     if selfcal_library[target][band][fid]['SC_success']:
                         print('****************Applying '+str(selfcal_library[target][band][fid][vis]['gaintable_final'])+' to '+target+' field '+\
                                 str(fid)+' '+band+'*************')
                         applycal(vis=vis,\
                                 gaintable=selfcal_library[target][band][fid][vis]['gaintable_final'],\
                                 interp=selfcal_library[target][band][fid][vis]['applycal_interpolate_final'],\
                                 calwt=False,spwmap=selfcal_library[target][band][fid][vis]['spwmap_final'],\
                                 applymode=selfcal_library[target][band][fid][vis]['applycal_mode_final'],\
                                 field=str(selfcal_library[target][band]['sub-fields-fid_map'][vis][fid]),\
                                 spw=selfcal_library[target][band][vis]['spws'])    
                     else:
                         print('****************Removing all calibrations for '+target+' '+str(fid)+' '+band+'**************')
                         clearcal(vis=vis,field=str(selfcal_library[target][band]['sub-fields-fid_map'][vis][fid]),\
                                 spw=selfcal_library[target][band][vis]['spws'])
                         selfcal_library[target][band]['SNR_post']=selfcal_library[target][band]['SNR_orig'].copy()
                         selfcal_library[target][band]['RMS_post']=selfcal_library[target][band]['RMS_orig'].copy()

                         for fid in selfcal_library[target][band]['sub-fields']:
                             selfcal_library[target][band][fid]['SNR_post']=selfcal_library[target][band][fid]['SNR_orig'].copy()
                             selfcal_library[target][band][fid]['RMS_post']=selfcal_library[target][band][fid]['RMS_orig'].copy()

         # If any of the sub-fields passed, and the whole mosaic passed, then we can move on to the next solint, otherwise we have to back out.
         if selfcal_library[target][band][vislist[0]][solint]['Pass'] == True and \
                 np.any([selfcal_library[target][band][fid][selfcal_library[target][band][fid]['vislist'][0]][solint]['Pass'] == True for fid in \
                 selfcal_library[target][band]['sub-fields-to-selfcal']]):
             if (iteration < len(solints[band][target])-1) and (selfcal_library[target][band][vis][solint]['SNR_post'] > \
                     selfcal_library[target][band]['SNR_orig']): #(iteration == 0) and 
                print('Updating solint = '+solints[band][target][iteration+1]+' SNR')
                print('Was: ',solint_snr[target][band][solints[band][target][iteration+1]])
                get_SNR_self_update([target],band,vislist,selfcal_library[target][band],n_ants,solint,solints[band][target][iteration+1],integration_time,solint_snr[target][band])
                print('Now: ',solint_snr[target][band][solints[band][target][iteration+1]])

                for fid in selfcal_library[target][band]['sub-fields-to-selfcal']:
                    print('Field '+str(fid)+' Was: ',solint_snr_per_field[target][band][fid][solints[band][target][iteration+1]])
                    get_SNR_self_update([target],band,vislist,selfcal_library[target][band][fid],n_ants,solint,solints[band][target][iteration+1],integration_time,solint_snr_per_field[target][band][fid])
                    print('Field '+str(fid)+' Now: ',solint_snr_per_field[target][band][fid][solints[band][target][iteration+1]])

             # If not all fields succeed for inf_EB or scan_inf/inf, depending on mosaic or single field, then don't go on to amplitude selfcal,
             # even if *some* fields succeeded.
             if iteration <= 1 and ((not np.all([selfcal_library[target][band][fid][selfcal_library[target][band][fid]['vislist'][0]][solint]['Pass'] == True for fid in \
                    selfcal_library[target][band]['sub-fields-to-selfcal']])) or len(selfcal_library[target][band]['sub-fields-to-selfcal']) < \
                    len(selfcal_library[target][band]['sub-fields'])) and do_amp_selfcal:
                 print("***** NOTE: Amplitude self-calibration turned off because not all fields succeeded at non-inf_EB phase self-calibration")
                 do_amp_selfcal = False
                
             if iteration < (len(solints[band][target])-1):
                print('****************Selfcal passed, shortening solint*************')
             else:
                print('****************Selfcal passed for Minimum solint*************')
         else:   
            print('****************Selfcal failed*************')
            print('REASON: '+reason)
            if iteration > 1 and solmode[band][target][iteration] !='ap' and do_amp_selfcal:  # if a solution interval shorter than inf for phase-only SC has passed, attempt amplitude selfcal
               iterjump=solmode[band][target].index('ap') 
               selfcal_library[target][band]['sub-fields-to-selfcal'] = selfcal_library[target][band]['sub-fields']
               print('****************Selfcal halted for phase, attempting amplitude*************')
               continue
            else:
               print('****************Aborting further self-calibration attempts for '+target+' '+band+'**************')
               break # breakout of loops of successive solints since solutions are getting worse

         # Finally, update the list of fields to be self-calibrated now that we don't need to know the list at the beginning of this solint.
         new_fields_to_selfcal = []
         for fid in selfcal_library[target][band]['sub-fields']:
             if selfcal_library[target][band][fid][selfcal_library[target][band][fid]['vislist'][0]]["inf_EB"]["Pass"]:
                 new_fields_to_selfcal.append(fid)

         selfcal_library[target][band]['sub-fields-to-selfcal'] = new_fields_to_selfcal
