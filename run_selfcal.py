import numpy as np
from scipy import stats
import glob
import sys
#execfile('selfcal_helpers.py',globals())
sys.path.append("./")
from selfcal_helpers import *
from casampi.MPIEnvironment import MPIEnvironment 
parallel=MPIEnvironment.is_mpi_enabled

def run_selfcal(selfcal_library, target, band, solints, solint_snr, solint_snr_per_field, applycal_mode, solmode, band_properties, telescope, n_ants, cellsize, imsize, \
        inf_EB_gaintype_dict, inf_EB_gaincal_combine_dict, inf_EB_fallback_mode_dict, gaincal_combine, applycal_interp, integration_time, \
        gaincal_minsnr=2.0, gaincal_unflag_minsnr=5.0, minsnr_to_proceed=3.0, delta_beam_thresh=0.05, do_amp_selfcal=True, inf_EB_gaincal_combine='scan', inf_EB_gaintype='G', \
        unflag_only_lbants=False, unflag_only_lbants_onlyap=False, calonly_max_flagged=0.0, second_iter_solmode="", unflag_fb_to_prev_solint=False, \
        rerank_refants=False, mode="selfcal", calibrators="", calculate_inf_EB_fb_anyways=False, preapply_targets_own_inf_EB=False, \
                gaincalibrator_dict={}):

   # If we are running this on a mosaic, we want to rerank reference antennas and have a higher gaincal_minsnr by default.

   if selfcal_library[target][band]["obstype"] == "mosaic":
       gaincal_minsnr = 10.0
       rerank_refants = True
       refantmode = "strict"
   elif mode == "cocal":
       rerank_refants = True
       refantmode = "strict"
   else:
       refantmode = "flex"

   # Start looping over the solints.

   iterjump=-1   # useful if we want to jump iterations
   sani_target=sanitize_string(target)

   if mode == "cocal":
        iterjump = len(solints[band]) - 4
        if selfcal_library[target][band]["SC_success"] and not calculate_inf_EB_fb_anyways:
            iterjump += 1

   vislist=selfcal_library[target][band]['vislist'].copy()

   if mode == "cocal":
       # Check whether there are suitable calibrators, otherwise skip this target/band.
       include_targets, include_scans = triage_calibrators(vislist[0], target, calibrators[band][0])
       if include_targets == "":
           print("No suitable calibrators found, skipping "+target)
           return

   for iteration in range(len(solints[band])):
      if (iterjump !=-1) and (iteration < iterjump): # allow jumping to amplitude selfcal and not need to use a while loop
         continue
      elif iteration == iterjump:
         iterjump=-1
      print("Solving for solint="+solints[band][iteration])

      # Set some cocal parameters.
      if solints[band][iteration] in ["inf_EB_fb","inf_fb1"]:
          calculate_inf_EB_fb_anyways = True
          preapply_targets_own_inf_EB = False
      elif solints[band][iteration] == "inf_fb2":
          calculate_inf_EB_fb_anyways = False
          preapply_targets_own_inf_EB = False
          # If there was not a successful inf_EB solint, then this duplicates inf_fb1 so skip
          if "inf_EB" not in selfcal_library[target][band][vislist[0]]:
              continue
          elif not selfcal_library[target][band][vislist[0]]["inf_EB"]['Pass']:
              continue
      elif solints[band][iteration] == "inf_fb3":
          calculate_inf_EB_fb_anyways = False
          preapply_targets_own_inf_EB = True
          # If there was no inf solint (e.g. because each source was observed only a single time, skip this as there are no gain tables to stick together.
          if "inf" not in solints[band]:
              continue

      if mode == "selfcal" and solint_snr[target][band][solints[band][iteration]] < minsnr_to_proceed and np.all([solint_snr_per_field[target][band][fid][solints[band][iteration]] < minsnr_to_proceed for fid in selfcal_library[target][band]['sub-fields']]):
         print('*********** estimated SNR for solint='+solints[band][iteration]+' too low, measured: '+str(solint_snr[target][band][solints[band][iteration]])+', Min SNR Required: '+str(minsnr_to_proceed)+' **************')
         if iteration > 1 and solmode[band][iteration] !='ap' and do_amp_selfcal:  # if a solution interval shorter than inf for phase-only SC has passed, attempt amplitude selfcal
            iterjump=solmode[band].index('ap') 
            print('****************Attempting amplitude selfcal*************')
            continue

         selfcal_library[target][band]['Stop_Reason']='Estimated_SNR_too_low_for_solint '+solints[band][iteration]
         break
      else:
         solint=solints[band][iteration]
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
                     os.system("cp -r "+f+" "+f.replace(prev_solint+"_"+str(prev_iteration)+"_post", solint+'_'+str(iteration)))
         else:
             resume = False

         nfsnr_modifier = selfcal_library[target][band]['RMS_NF_curr'] / selfcal_library[target][band]['RMS_curr']
         tclean_wrapper(vislist,sani_target+'_'+band+'_'+solint+'_'+str(iteration),
                     band_properties,band,telescope=telescope,nsigma=selfcal_library[target][band]['nsigma'][iteration], scales=[0],
                     threshold=str(selfcal_library[target][band]['nsigma'][iteration]*selfcal_library[target][band]['RMS_curr'])+'Jy',
                     savemodel='none',parallel=parallel,cellsize=cellsize[band],imsize=imsize[band],
                     nterms=selfcal_library[target][band]['nterms'],
                     field=target,spw=selfcal_library[target][band]['spws_per_vis'],uvrange=selfcal_library[target][band]['uvrange'],obstype=selfcal_library[target][band]['obstype'], nfrms_multiplier=nfsnr_modifier, resume=resume, image_mosaic_fields_separately=selfcal_library[target][band]['obstype'] == 'mosaic')

         if solint == "inf_EB_fb" or ("inf_fb" in solint and selfcal_library[target][band]['final_solint'] == "inf_EB") or iteration == 0:
            gaincal_preapply_gaintable={}
            gaincal_spwmap={}
            gaincal_interpolate={}
            applycal_gaintable={}
            applycal_spwmap={}
            fallback={}
            applycal_interpolate={}


         # Loop through up to two times. On the first attempt, try applymode = 'calflag' (assuming this is requested by the user). On the
         # second attempt, use applymode = 'calonly'.
         for applymode in np.unique([applycal_mode[band][iteration],'calonly']):
             for vis in vislist:
                ##
                ## Restore original flagging state each time before applying a new gaintable
                ##
                versionname = ("fb_" if mode == "cocal" else "")+'selfcal_starting_flags_'+sani_target
                if not os.path.exists(vis+".flagversions/flags."+versionname):
                   flagmanager(vis=vis,mode='save',versionname=versionname)
                elif mode == "selfcal":
                   flagmanager(vis=vis, mode = 'restore', versionname = versionname, comment = 'Flag states at start of reduction')

                if mode == "cocal":
                    flagmanager(vis=vis, mode = 'restore', versionname = 'selfcal_starting_flags', comment = 'Flag states at start of the reduction')

             # We need to redo saving the model now that we have potentially unflagged some data.
             if applymode == "calflag":
                 tclean_wrapper(vislist,sani_target+'_'+band+'_'+solint+'_'+str(iteration),
                             band_properties,band,telescope=telescope,nsigma=selfcal_library[target][band]['nsigma'][iteration], scales=[0],
                             threshold=str(selfcal_library[target][band]['nsigma'][iteration]*selfcal_library[target][band]['RMS_curr'])+'Jy',
                             savemodel='modelcolumn',parallel=parallel,cellsize=cellsize[band],imsize=imsize[band],
                             nterms=selfcal_library[target][band]['nterms'],
                             field=target,spw=selfcal_library[target][band]['spws_per_vis'],uvrange=selfcal_library[target][band]['uvrange'],obstype=selfcal_library[target][band]['obstype'], nfrms_multiplier=nfsnr_modifier, savemodel_only=True)

             for vis in vislist:
                # Record gaincal details.
                selfcal_library[target][band][vis][solint]={}
                for fid in selfcal_library[target][band]['sub-fields-to-selfcal']:
                    selfcal_library[target][band][fid][vis][solint]={}

                applycal_gaintable[vis]=[]
                applycal_spwmap[vis]=[]
                applycal_interpolate[vis]=[]
                gaincal_spwmap[vis]=[]
                gaincal_interpolate[vis]=[]
                gaincal_preapply_gaintable[vis]=[]
                ##
                ## Solve gain solutions per MS, target, solint, and band
                ##
                os.system('rm -rf '+sani_target+'_'+vis+'_'+band+'_'+solint+'_'+str(iteration)+'_'+solmode[band][iteration]+'*.g')
                ##
                ## Set gaincal parameters depending on which iteration and whether to use combine=spw for inf_EB or not
                ## Defaults should assume combine='scan' and gaintpe='G' will fallback to combine='scan,spw' if too much flagging
                ## At some point remove the conditional for use_inf_EB_preapply, since there isn't a reason not to do it
                ##

                if solmode[band][iteration] == 'p':
                    if 'inf_EB' in solint:
                       gaincal_spwmap[vis]=[]
                       gaincal_preapply_gaintable[vis]=[]
                       gaincal_interpolate[vis]=[]
                       gaincal_gaintype=inf_EB_gaintype_dict[target][band][vis]
                       gaincal_solmode=""
                       gaincal_combine[band][iteration]=inf_EB_gaincal_combine_dict[target][band][vis]+\
                               (",field" if "fb" in solint else "")
                       if 'spw' in inf_EB_gaincal_combine_dict[target][band][vis]:
                          applycal_spwmap[vis]=[selfcal_library[target][band][vis]['spwmap']]
                          gaincal_spwmap[vis]=[selfcal_library[target][band][vis]['spwmap']]
                       else:
                          applycal_spwmap[vis]=[]
                       applycal_interpolate[vis]=[applycal_interp[band]]
                       applycal_gaintable[vis]=[sani_target+'_'+vis+'_'+band+'_'+solint+'_'+str(iteration)+'_'+solmode[band][iteration]+'.g']
                    #elif solmode[band][iteration]=='p':
                    else:
                       gaincal_spwmap[vis]=[]
                       if mode == "cocal":
                           if 'inf_EB' in selfcal_library[target][band][vis]:
                               #gaincal_preapply_gaintable[vis]=[sani_target+'_'+vis+'_'+band+'_inf_EB_0_p.g']
                               if calculate_inf_EB_fb_anyways or not selfcal_library[target][band][vis]["inf_EB"]["Pass"]:
                                   previous_solint = "inf_EB_fb"
                               else:
                                   previous_solint = "inf_EB"
                           else:
                               #gaincal_preapply_gaintable[vis]=[sani_target+'_'+vis+'_'+band+'_inf_EB_fb_'+str(iteration-1)+'_p.g']
                               previous_solint = "inf_EB_fb"
                       else:
                           if solmode[band][iteration]=='p':
                               previous_solint = "inf_EB"
                           else:
                               previous_solint = selfcal_library[target][band]['final_phase_solint']

                       gaincal_preapply_gaintable[vis]=selfcal_library[target][band][vis][previous_solint]['gaintable']
                       gaincal_interpolate[vis]=[applycal_interp[band]]
                       gaincal_gaintype='T' if applymode == "calflag" or second_iter_solmode == "" else "GSPLINE" if second_iter_solmode == "GSPLINE" else "G"
                       gaincal_solmode = "" if applymode == "calflag" or second_iter_solmode == "GSPLINE" else second_iter_solmode
                       gaincal_spwmap[vis] = selfcal_library[target][band][vis][previous_solint]['spwmap']
                       """
                       if 'spw' in inf_EB_gaincal_combine_dict[target][band][vis]:
                          applycal_spwmap[vis]=[selfcal_library[target][band][vis]['spwmap'],selfcal_library[target][band][vis]['spwmap']]
                          gaincal_spwmap[vis]=[selfcal_library[target][band][vis]['spwmap']]
                       elif inf_EB_fallback_mode_dict[target][band][vis]=='spwmap':
                          applycal_spwmap[vis]=[selfcal_library[target][band][vis]['inf_EB']['spwmap'],selfcal_library[target][band][vis]['spwmap']]
                          gaincal_spwmap[vis]=selfcal_library[target][band][vis]['inf_EB']['spwmap']
                       else:
                          applycal_spwmap[vis]=[[],selfcal_library[target][band][vis]['spwmap']]
                          gaincal_spwmap[vis]=[]
                       """
                       # Revert back to applying the inf_EB solution if calculate_inf_EB_fb_anyways, i.e. we just use the inf_EB_fb solution
                       # for gaincal.
                       if mode == "cocal":
                           if selfcal_library[target][band]['final_solint'] == 'inf_EB' and calculate_inf_EB_fb_anyways:
                               previous_solint = "inf_EB"

                       applycal_spwmap[vis] = [selfcal_library[target][band][vis][previous_solint]['spwmap']] + [selfcal_library[target][band][vis]['spwmap']]
                       applycal_interpolate[vis]=[applycal_interp[band],applycal_interp[band]]
                       applycal_gaintable[vis] = selfcal_library[target][band][vis][previous_solint]['gaintable'] + [sani_target+'_'+vis+'_'+band+'_'+solint+'_'+str(iteration)+'_'+solmode[band][iteration]+'.g']
                    selfcal_library[target][band][vis][solint]['gaintable']=applycal_gaintable[vis]
                    selfcal_library[target][band][vis][solint]['iteration']=iteration+0
                    selfcal_library[target][band][vis][solint]['spwmap']=applycal_spwmap[vis]
                    selfcal_library[target][band][vis][solint]['applycal_mode']=applycal_mode[band][iteration]+''
                    selfcal_library[target][band][vis][solint]['applycal_interpolate']=applycal_interpolate[vis]
                    selfcal_library[target][band][vis][solint]['gaincal_combine']=gaincal_combine[band][iteration]+''
                    for fid in selfcal_library[target][band]['sub-fields-to-selfcal']:
                        selfcal_library[target][band][fid][vis][solint]['gaintable']=applycal_gaintable[vis]
                        selfcal_library[target][band][fid][vis][solint]['iteration']=iteration+0
                        selfcal_library[target][band][fid][vis][solint]['spwmap']=applycal_spwmap[vis]
                        selfcal_library[target][band][fid][vis][solint]['applycal_mode']=applycal_mode[band][iteration]+''
                        selfcal_library[target][band][fid][vis][solint]['applycal_interpolate']=applycal_interpolate[vis]
                        selfcal_library[target][band][fid][vis][solint]['gaincal_combine']=gaincal_combine[band][iteration]+''

                    fallback[vis]=''
                    if solmode[band][iteration] == 'ap':
                       solnorm=True
                    else:
                       solnorm=False

                    if gaincal_gaintype == "GSPLINE":
                        splinetime = solint.replace('_EB','').replace('_ap','')
                        if splinetime == "inf":
                            splinetime = selfcal_library[target][band]["Median_scan_time"]
                        else:
                            splinetime = float(splinetime[0:-1])

                    if mode == "cocal":
                        # Check which targets are acceptable to use as calibrators.
                        targets = calibrators[band][iteration - len(solints[band])]

                        include_targets, include_scans = triage_calibrators(vis, target, targets)
                    else:
                        include_targets = target
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

                            include_scans = []
                            for iscan in range(scans.size-1):
                                include_scans.append(",".join(np.array(list(range(scans[iscan]+1,scans[iscan+1]))).astype(str)))
                        else:
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
                        include_scans = [include_scans]

                    if mode == "cocal" and preapply_targets_own_inf_EB and "inf_fb" in solint and "inf" in solints[band]:
                        ##
                        ## If we want to pre-apply inf_EB solution from each calibrator to itself, all we do is combine all of thier
                        ## individual inf tables, as these were pre-calculated in that way.
                        ##
                        destination_table = sani_target+'_'+vis+'_'+band+'_'+solint+'_'+str(iteration)+'_'+solmode[band][iteration]+'.g'
                        for t in include_targets.split(","):
                            if os.path.exists(sanitize_string(t)+'_'+vis+'_'+band+'_'+solint.replace('_fb1','').replace('_fb2','').replace('_fb3','')+'_'+str(1)+'_'+solmode[band][iteration]+\
                                    '.pre-pass.g'):
                                table_name = sanitize_string(t)+'_'+vis+'_'+band+'_'+solint.replace('_fb1','').replace('_fb2','').replace('_fb3','')+'_'+str(1)+'_'+solmode[band][iteration]+\
                                        '.pre-pass.g'
                            else:
                                table_name = sanitize_string(t)+'_'+vis+'_'+band+'_'+solint.replace('_fb1','').replace('_fb2','').replace('_fb3','')+'_'+str(1)+'_'+solmode[band][iteration]+'.g'
                            #t_final_solint = selfcal_library[t][band]["final_phase_solint"]
                            #t_iteration = selfcal_library[t][band][vislist[0]][t_final_solint]["iteration"]
                            #table_name = sanitize_string(t)+'_'+vis+'_'+band+'_'+t_final_solint+'_'+str(t_iteration)+'_'+solmode[band][iteration]+'.g'

                            rerefant(vis, table_name, caltable="tmp0.g", refantmode="strict", refant=selfcal_library[target][band][vis]['refant'])

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
                        if selfcal_library[target][band]['obstype'] == 'mosaic':
                            msmd.open(vis)
                            include_targets = []
                            remove = []
                            for incl_scan in include_scans:
                                scan_targets = []
                                for fid in selfcal_library[target][band]['sub-fields'] if incl_scan == '' else \
                                        msmd.fieldsforscans(np.array(incl_scan.split(",")).astype(int)):
                                    os.system('rm -rf test*.mask')
                                    SNR_NF,RMS_NF=estimate_near_field_SNR(sani_target+'_field_'+str(fid)+'_'+band+'_'+solint+'_'+str(iteration)+'.image.tt0', \
                                            las=selfcal_library[target][band]['LAS'], mosaic_sub_field=True)

                                    immath(imagename=[sani_target+"_field_"+str(fid)+"_"+band+"_"+solint+"_"+str(iteration)+".image.tt0",\
                                            sani_target+"_field_"+str(fid)+"_"+band+"_"+solint+"_"+str(iteration)+".pb.tt0",\
                                            sani_target+"_field_"+str(fid)+"_"+band+"_"+solint+"_"+str(iteration)+".mospb.tt0"], outfile="test.mask", \
                                            expr="IIF(IM0*IM1/IM2 > "+str(5*RMS_NF)+", 1., 0.)")

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
                                            rms=RMS_NF, maskname=sani_target+"_field_"+str(fid)+"_"+band+"_"+solint+"_"+str(iteration)+".mask", \
                                            mosaic_sub_field=True)[0]
                                    updated_intflux = get_intflux(sani_target+"_field_"+str(fid)+"_"+band+"_"+solint+"_"+str(iteration)+".image.tt0", \
                                            rms=RMS_NF, maskname="test.smoothed.truncated.mask", mosaic_sub_field=True)[0]
                                    os.system('rm -rf test*.mask')


                                    if not checkmask(sani_target+'_field_'+str(fid)+'_'+band+'_'+solint+'_'+str(iteration)+'.image.tt0'):
                                        print("Removing field "+str(fid)+" from "+sani_target+'_'+vis+'_'+band+'_'+solint+'_'+str(iteration)+'_'+\
                                                solmode[band][iteration]+'.g'+" because there is no signal within the primary beam.")
                                    elif solint_snr_per_field[target][band][fid][solints[band][iteration]] < minsnr_to_proceed and solint not in ['inf_EB','scan_inf']:
                                        print("Removing field "+str(fid)+" from "+sani_target+'_'+vis+'_'+band+'_'+solint+'_'+str(iteration)+'_'+\
                                                solmode[band][iteration]+'.g'+' because the estimated solint snr is too low.')
                                    elif updated_intflux > 1.25 * original_intflux:
                                        print("Removing field "+str(fid)+" from "+sani_target+'_'+vis+'_'+band+'_'+solint+'_'+str(iteration)+'_'+\
                                                solmode[band][iteration]+'.g'+" because there appears to be significant flux missing from the model.")
                                    else:
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

                        for incl_scans, incl_targets in zip(include_scans, include_targets):
                            gaincal(vis=vis,\
                                 caltable=sani_target+'_'+vis+'_'+band+'_'+solint+'_'+str(iteration)+'_'+solmode[band][iteration]+'.g',\
                                 gaintype=gaincal_gaintype, spw=selfcal_library[target][band][vis]['spws'],
                                 refant=selfcal_library[target][band][vis]['refant'], calmode=solmode[band][iteration], solnorm=solnorm if applymode=="calflag" else False,
                                 solint=solint.replace('_EB','').replace('_ap','').replace('scan_','').replace('_fb1','').replace('_fb2','').replace('_fb3',''),minsnr=gaincal_minsnr if applymode == 'calflag' else max(gaincal_minsnr,gaincal_unflag_minsnr), minblperant=4,combine=gaincal_combine[band][iteration],
                                 field=incl_targets,scan=incl_scans,gaintable=gaincal_preapply_gaintable[vis],spwmap=gaincal_spwmap[vis],uvrange=selfcal_library[target][band]['uvrange'],
                                 interp=gaincal_interpolate[vis], solmode=gaincal_solmode, refantmode='flex', append=os.path.exists(sani_target+'_'+vis+'_'+band+'_'+solint+'_'+str(iteration)+'_'+solmode[band][iteration]+'.g'))

                else:
                    for fid in selfcal_library[target][band]['sub-fields-to-selfcal']:
                        gaincal_spwmap[vis]=[]
                        gaincal_preapply_gaintable[vis]=selfcal_library[target][band][fid][vis][selfcal_library[target][band][fid]['final_phase_solint']]['gaintable']
                        gaincal_interpolate[vis]=[applycal_interp[band]]*len(gaincal_preapply_gaintable[vis])
                        gaincal_gaintype='T' if applymode == "calflag" or second_iter_solmode == "" else "GSPLINE" if second_iter_solmode == "GSPLINE" else "G"
                        gaincal_solmode = "" if applymode == "calflag" or second_iter_solmode == "GSPLINE" else second_iter_solmode
                        if 'spw' in inf_EB_gaincal_combine_dict[target][band][vis]:
                           applycal_spwmap[vis]=[selfcal_library[target][band][fid][vis]['spwmap'],selfcal_library[target][band][fid][vis]['spwmap'],selfcal_library[target][band][fid][vis]['spwmap']]
                           gaincal_spwmap[vis]=[selfcal_library[target][band][fid][vis]['spwmap'],selfcal_library[target][band][fid][vis]['spwmap']]
                        elif inf_EB_fallback_mode_dict[target][band][vis]=='spwmap':
                           applycal_spwmap[vis]=[selfcal_library[target][band][fid][vis]['inf_EB']['spwmap'],selfcal_library[target][band][fid][vis]['spwmap'],selfcal_library[target][band][fid][vis]['spwmap']]
                           gaincal_spwmap[vis]=[selfcal_library[target][band][fid][vis]['inf_EB']['spwmap'],selfcal_library[target][band][fid][vis]['spwmap']]
                        else:
                           applycal_spwmap[vis]=[[],selfcal_library[target][band][fid][vis]['spwmap'],selfcal_library[target][band][fid][vis]['spwmap']]
                           gaincal_spwmap[vis]=[[],selfcal_library[target][band][fid][vis]['spwmap']]
                        applycal_interpolate[vis]=[applycal_interp[band]]*len(gaincal_preapply_gaintable[vis])+['linearPD']
                        applycal_gaintable[vis]=selfcal_library[target][band][fid][vis][selfcal_library[target][band][fid]['final_phase_solint']]['gaintable']+[sani_target+'_'+vis+'_'+band+'_'+solint+'_'+str(iteration)+'_ap.g']

                        selfcal_library[target][band][vis][solint]['gaintable']=applycal_gaintable[vis]
                        selfcal_library[target][band][vis][solint]['iteration']=iteration+0
                        selfcal_library[target][band][vis][solint]['spwmap']=applycal_spwmap[vis]
                        selfcal_library[target][band][vis][solint]['applycal_mode']=applycal_mode[band][iteration]+''
                        selfcal_library[target][band][vis][solint]['applycal_interpolate']=applycal_interpolate[vis]
                        selfcal_library[target][band][vis][solint]['gaincal_combine']=gaincal_combine[band][iteration]+''
                        selfcal_library[target][band][fid][vis][solint]['gaintable']=applycal_gaintable[vis]
                        selfcal_library[target][band][fid][vis][solint]['iteration']=iteration+0
                        selfcal_library[target][band][fid][vis][solint]['spwmap']=applycal_spwmap[vis]
                        selfcal_library[target][band][fid][vis][solint]['applycal_mode']=applycal_mode[band][iteration]+''
                        selfcal_library[target][band][fid][vis][solint]['applycal_interpolate']=applycal_interpolate[vis]
                        selfcal_library[target][band][fid][vis][solint]['gaincal_combine']=gaincal_combine[band][iteration]+''

                        fallback[vis]=''
                        if solmode[band][iteration] == 'ap':
                           solnorm=True
                        else:
                           solnorm=False

                        if gaincal_gaintype == "GSPLINE":
                            splinetime = solint.replace('_EB','').replace('_ap','')
                            if splinetime == "inf":
                                splinetime = selfcal_library[target][band][fid]["Median_scan_time"]
                            else:
                                splinetime = float(splinetime[0:-1])

                        gaincal(vis=vis,\
                             #caltable=sani_target+'_'+vis+'_'+band+'_'+solint+'_'+str(iteration)+'_'+solmode[band][iteration]+'.g',\
                             caltable="temp.g",\
                             gaintype=gaincal_gaintype, spw=selfcal_library[target][band][fid][vis]['spws'],
                             refant=selfcal_library[target][band][vis]['refant'], calmode=solmode[band][iteration], solnorm=solnorm if applymode=="calflag" else False,
                             solint=solint.replace('_EB','').replace('_ap','').replace('scan_',''),minsnr=gaincal_minsnr if applymode == 'calflag' else max(gaincal_minsnr,gaincal_unflag_minsnr), minblperant=4,combine=gaincal_combine[band][iteration],
                             field=str(fid),gaintable=gaincal_preapply_gaintable[vis],spwmap=gaincal_spwmap[vis],uvrange=selfcal_library[target][band]['uvrange'],
                             #interp=gaincal_interpolate[vis], solmode=gaincal_solmode, append=os.path.exists(sani_target+'_'+vis+'_'+band+'_'+
                             #solint+'_'+str(iteration)+'_'+solmode[band][iteration]+'.g'))
                             interp=gaincal_interpolate[vis], solmode=gaincal_solmode, append=os.path.exists('temp.g'), refantmode='flex')

                    tb.open("temp.g")
                    subt = tb.query("OBSERVATION_ID==0", sortlist="TIME,ANTENNA1")
                    tb.close()

                    subt.copy(sani_target+'_'+vis+'_'+band+'_'+solint+'_'+str(iteration)+'_'+solmode[band][iteration]+'.g', deep=True)
                    subt.close()

                    os.system("rm -rf temp.g")

                if rerank_refants:
                    selfcal_library[target][band][vis]["refant"] = rank_refants(vis, caltable=sani_target+'_'+vis+'_'+band+'_'+solint+'_'+str(iteration)+'_'+solmode[band][iteration]+'.g')

                    # If we are falling back to a previous solution interval on the unflagging, we need to make sure all tracks use a common 
                    # reference antenna.
                    if unflag_fb_to_prev_solint:
                        for it, sint in enumerate(solints[band][0:iteration+1]):
                            if not os.path.exists(sani_target+'_'+vis+'_'+band+'_'+sint+'_'+str(it)+'_'+solmode[band][it]+'.g'):
                                continue

                            # If a previous iteration went through the unflagging routine, it is possible that some antennas fell back to
                            # a previous solint. In that case, rerefant will flag those antennas because they can't be re-referenced with
                            # a different time interval. So to be safe, we go back to the pre-pass solutions and then re-run the passing.
                            # We could probably check more carefully whether this is the case to avoid having to do this... but the 
                            # computing time isn't significant so it's easy just to run through again.
                            if os.path.exists(sani_target+'_'+vis+'_'+band+'_'+sint+'_'+str(it)+'_'+solmode[band][it]+'.pre-pass.g'):
                                rerefant(vis, sani_target+'_'+vis+'_'+band+'_'+sint+'_'+str(it)+'_'+solmode[band][it]+'.pre-pass.g', \
                                        refant=selfcal_library[target][band][vis]["refant"], refantmode=refantmode if 'inf_EB' not in sint else 'flex')

                                os.system("rm -rf "+sani_target+'_'+vis+'_'+band+'_'+sint+'_'+str(it)+'_'+solmode[band][it]+'.g')
                                os.system("cp -r "+sani_target+'_'+vis+'_'+band+'_'+sint+'_'+str(it)+'_'+solmode[band][it]+'.pre-pass.g '+\
                                        sani_target+'_'+vis+'_'+band+'_'+sint+'_'+str(it)+'_'+solmode[band][it]+'.g')

                                if sint == "inf_EB" and len(selfcal_library[target][band][vis][sint]["spwmap"][0]) > 0:
                                    unflag_spwmap = selfcal_library[target][band][vis][sint]["spwmap"][0]
                                else:
                                    unflag_spwmap = []

                                unflag_failed_antennas(vis, sani_target+'_'+vis+'_'+band+'_'+sint+'_'+str(it)+'_'+\
                                        solmode[band][it]+'.g', flagged_fraction=0.25, solnorm=solnorm, \
                                        only_long_baselines=solmode[band][it]=="ap" if unflag_only_lbants and \
                                        unflag_only_lbants_onlyap else unflag_only_lbants, calonly_max_flagged=calonly_max_flagged, \
                                        spwmap=unflag_spwmap, fb_to_prev_solint=unflag_fb_to_prev_solint, solints=solints[band], iteration=it)
                            else:
                                rerefant(vis, sani_target+'_'+vis+'_'+band+'_'+sint+'_'+str(it)+'_'+solmode[band][it]+'.g', \
                                        refant=selfcal_library[target][band][vis]["refant"], refantmode=refantmode if 'inf_EB' not in sint else 'flex')
                    else:
                        os.system("cp -r "+sani_target+'_'+vis+'_'+band+'_'+solint+'_'+str(iteration)+'_'+solmode[band][iteration]+'.g '+\
                                sani_target+'_'+vis+'_'+band+'_'+solint+'_'+str(iteration)+'_'+solmode[band][iteration]+'.pre-rerefant.g')
                        rerefant(vis, sani_target+'_'+vis+'_'+band+'_'+solint+'_'+str(iteration)+'_'+solmode[band][iteration]+'.g', \
                                refant=selfcal_library[target][band][vis]["refant"], refantmode=refantmode if 'inf_EB' not in solint else 'flex')

                ##
                ## default is to run without combine=spw for inf_EB, here we explicitly run a test inf_EB with combine='scan,spw' to determine
                ## the number of flagged antennas when combine='spw' then determine if it needs spwmapping or to use the gaintable with spwcombine.
                ##
                if 'inf_EB' in solint and fallback[vis]=='':
                   os.system('rm -rf test_inf_EB.g')
                   test_gaincal_combine='scan,spw'
                   if selfcal_library[target][band]['obstype']=='mosaic' or mode=="cocal":
                      test_gaincal_combine+=',field'   
                   gaincal(vis=vis,\
                     caltable='test_inf_EB.g',\
                     gaintype=gaincal_gaintype, spw=selfcal_library[target][band][vis]['spws'],
                     refant=selfcal_library[target][band][vis]['refant'], calmode='p', 
                     solint=solint.replace('_EB','').replace('_ap','').replace('_fb1','').replace('_fb2','').replace('_fb3',''),minsnr=gaincal_minsnr if applymode == "calflag" else max(gaincal_minsnr,gaincal_unflag_minsnr), minblperant=4,combine=test_gaincal_combine,
                     field=include_targets[0],scan=include_scans[0],gaintable='',spwmap=[],uvrange=selfcal_library[target][band]['uvrange'], refantmode='flex') 
                   spwlist=selfcal_library[target][band][vis]['spws'].split(',')
                   fallback[vis],map_index,spwmap,applycal_spwmap_inf_EB=analyze_inf_EB_flagging(selfcal_library,band,spwlist,sani_target+'_'+vis+'_'+band+'_'+solint+'_'+str(iteration)+'_'+solmode[band][iteration]+'.g',vis,target,'test_inf_EB.g')

                   inf_EB_fallback_mode_dict[target][band][vis]=fallback[vis]+''
                   print(solint,fallback[vis],applycal_spwmap_inf_EB)
                   if fallback[vis] != '':
                      if fallback[vis] =='combinespw':
                         gaincal_spwmap[vis]=[selfcal_library[target][band][vis]['spwmap']]
                         gaincal_combine[band][iteration]='scan,spw'
                         inf_EB_gaincal_combine_dict[target][band][vis]='scan,spw'
                         applycal_spwmap[vis]=[selfcal_library[target][band][vis]['spwmap']]
                         os.system('rm -rf           '+sani_target+'_'+vis+'_'+band+'_'+solint+'_'+str(iteration)+'_'+solmode[band][iteration]+'.g')
                         os.system('mv test_inf_EB.g '+sani_target+'_'+vis+'_'+band+'_'+solint+'_'+str(iteration)+'_'+solmode[band][iteration]+'.g')
                      if fallback[vis] =='spwmap':
                         gaincal_spwmap[vis]=applycal_spwmap_inf_EB
                         inf_EB_gaincal_combine_dict[target][band][vis]='scan'
                         gaincal_combine[band][iteration]='scan'
                         applycal_spwmap[vis]=[applycal_spwmap_inf_EB]

                      # Update the appropriate selfcal_library entries.
                      selfcal_library[target][band][vis][solint]['spwmap']=applycal_spwmap[vis]
                      selfcal_library[target][band][vis][solint]['gaincal_combine']=gaincal_combine[band][iteration]+''
                      for fid in selfcal_library[target][band]['sub-fields-to-selfcal']:
                          selfcal_library[target][band][fid][vis][solint]['spwmap']=applycal_spwmap[vis]
                          selfcal_library[target][band][fid][vis][solint]['gaincal_combine']=gaincal_combine[band][iteration]+''

                   os.system('rm -rf test_inf_EB.g')               

                # If iteration two, try restricting to just the antennas with enough unflagged data.
                # Should we also restrict to just long baseline antennas?
                if applymode == "calonly":
                    # Make a copy of the caltable before unflagging, for reference.
                    os.system("cp -r "+sani_target+'_'+vis+'_'+band+'_'+solint+'_'+str(iteration)+'_'+\
                            solmode[band][iteration]+'.g '+sani_target+'_'+vis+'_'+band+'_'+solint+'_'+str(iteration)+'_'+\
                            solmode[band][iteration]+'.pre-pass.g')

                    if solint == "inf_EB" and len(applycal_spwmap[vis]) > 0:
                        unflag_spwmap = applycal_spwmap[vis][0]
                    else:
                        unflag_spwmap = []

                    unflag_failed_antennas(vis, sani_target+'_'+vis+'_'+band+'_'+solint+'_'+str(iteration)+'_'+\
                            solmode[band][iteration]+'.g', flagged_fraction=0.25, solnorm=solnorm, \
                            only_long_baselines=solmode[band][iteration]=="ap" if unflag_only_lbants and unflag_only_lbants_onlyap else \
                            unflag_only_lbants, calonly_max_flagged=calonly_max_flagged, spwmap=unflag_spwmap, \
                            fb_to_prev_solint=unflag_fb_to_prev_solint, solints=solints[band], iteration=iteration)

                # Do some post-gaincal cleanup for mosaics.
                if selfcal_library[target][band]['obstype'] == 'mosaic' or mode == "cocal":
                    tb.open(sani_target+'_'+vis+'_'+band+'_'+solint+'_'+str(iteration)+'_'+solmode[band][iteration]+'.g', nomodify=False)
                    flags = tb.getcol("FLAG")
                    """
                    # Flag any solutions whose S/N ratio is too small.
                    snr = tb.getcol("SNR")
                    flags[snr < 5.] = True
                    """
                    bad = np.where(flags[0,0,:])[0]
                    tb.removerows(rownrs=bad)
                    tb.flush()

                    # Once we get beyond the inf solint, we dont want to let fields be significantly interpolated.
                    fields = tb.getcol("FIELD_ID")
                    if 'inf' not in solint:
                        new_fields_to_selfcal = []
                    for fid in selfcal_library[target][band]['sub-fields-to-selfcal']:
                        if fid in fields:
                            if 'inf' not in solint:
                                new_fields_to_selfcal.append(fid)
                            selfcal_library[target][band][fid][vis][solint]['interpolated_gains'] = False
                        else:
                            selfcal_library[target][band][fid][vis][solint]['interpolated_gains'] = True
                            if 'inf' not in solint:
                                selfcal_library[target][band][fid]['Stop_Reason'] = "Gaincal solutions would be interpolated beyond solint = inf"
                                del selfcal_library[target][band][fid][vis][solint]

                    if 'inf' not in solint:
                        selfcal_library[target][band]['sub-fields-to-selfcal'] = new_fields_to_selfcal

                    tb.close()

             for vis in vislist:
                ##
                ## Apply gain solutions per MS, target, solint, and band
                ##
                if mode == "cocal":
                    flagmanager(vis=vis, mode = 'restore', versionname = 'fb_selfcal_starting_flags_'+sani_target, comment = 'Flag states at start of the fallback reduction')
                for fid in selfcal_library[target][band]['sub-fields']:
                    if fid in selfcal_library[target][band]['sub-fields-to-selfcal']:
                        applycal(vis=vis,\
                                 gaintable=selfcal_library[target][band][fid][vis][solint]['gaintable'],\
                                 interp=selfcal_library[target][band][fid][vis][solint]['applycal_interpolate'], calwt=True,\
                                 spwmap=selfcal_library[target][band][fid][vis][solint]['spwmap'],\
                                 #applymode=applymode,field=target,spw=selfcal_library[target][band][vis]['spws'])
                                 applymode='calflag',field=str(fid),spw=selfcal_library[target][band][vis]['spws'])
                    else:
                        if selfcal_library[target][band][fid]['SC_success']:
                            applycal(vis=vis,\
                                    gaintable=selfcal_library[target][band][fid][vis]['gaintable_final'],\
                                    interp=selfcal_library[target][band][fid][vis]['applycal_interpolate_final'],\
                                    calwt=True,spwmap=selfcal_library[target][band][fid][vis]['spwmap_final'],\
                                    applymode=selfcal_library[target][band][fid][vis]['applycal_mode_final'],\
                                    field=str(fid),spw=selfcal_library[target][band][vis]['spws'])    

             ## Create post self-cal image using the model as a startmodel to evaluate how much selfcal helped
             ##

             os.system('rm -rf '+sani_target+'_'+band+'_'+solint+'_'+str(iteration)+'_post*')
             tclean_wrapper(vislist,sani_target+'_'+band+'_'+solint+'_'+str(iteration)+'_post',
                      band_properties,band,telescope=telescope,nsigma=selfcal_library[target][band]['nsigma'][iteration], scales=[0],
                      threshold=str(selfcal_library[target][band]['nsigma'][iteration]*selfcal_library[target][band]['RMS_curr'])+'Jy',
                      savemodel='none',parallel=parallel,cellsize=cellsize[band],imsize=imsize[band],
                      nterms=selfcal_library[target][band]['nterms'],
                      field=target,spw=selfcal_library[target][band]['spws_per_vis'],uvrange=selfcal_library[target][band]['uvrange'],obstype=selfcal_library[target][band]['obstype'], nfrms_multiplier=nfsnr_modifier, image_mosaic_fields_separately=selfcal_library[target][band]['obstype'] == 'mosaic')

             ##
             ## Do the assessment of the post- (and pre-) selfcal images.
             ##
             print('Pre selfcal assessemnt: '+target)
             SNR,RMS=estimate_SNR(sani_target+'_'+band+'_'+solint+'_'+str(iteration)+'.image.tt0', \
                     maskname=sani_target+'_'+band+'_'+solint+'_'+str(iteration)+'_post.mask')
             if telescope !='ACA':
                SNR_NF,RMS_NF=estimate_near_field_SNR(sani_target+'_'+band+'_'+solint+'_'+str(iteration)+'.image.tt0', \
                        maskname=sani_target+'_'+band+'_'+solint+'_'+str(iteration)+'_post.mask', las=selfcal_library[target][band]['LAS'])
             else:
                SNR_NF,RMS_NF=SNR,RMS

             print('Post selfcal assessemnt: '+target)
             post_SNR,post_RMS=estimate_SNR(sani_target+'_'+band+'_'+solint+'_'+str(iteration)+'_post.image.tt0')
             if telescope !='ACA':
                post_SNR_NF,post_RMS_NF=estimate_near_field_SNR(sani_target+'_'+band+'_'+solint+'_'+str(iteration)+'_post.image.tt0', \
                        las=selfcal_library[target][band]['LAS'])
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
                 if telescope !='ACA':
                    mosaic_SNR_NF[fid],mosaic_RMS_NF[fid]=estimate_near_field_SNR(imagename+'.image.tt0', maskname=imagename+'_post.mask', \
                            las=selfcal_library[target][band]['LAS'], mosaic_sub_field=selfcal_library[target][band]["obstype"]=="mosaic")
                 else:
                    mosaic_SNR_NF[fid],mosaic_RMS_NF[fid]=mosaic_SNR[fid],mosaic_RMS[fid]

                 print('Post selfcal assessemnt: '+target+', field '+str(fid))
                 post_mosaic_SNR[fid], post_mosaic_RMS[fid] = estimate_SNR(imagename+'_post.image.tt0', \
                         mosaic_sub_field=selfcal_library[target][band]["obstype"]=="mosaic")
                 if telescope !='ACA':
                    post_mosaic_SNR_NF[fid],post_mosaic_RMS_NF[fid]=estimate_near_field_SNR(imagename+'_post.image.tt0', \
                            las=selfcal_library[target][band]['LAS'], mosaic_sub_field=selfcal_library[target][band]["obstype"]=="mosaic")
                 else:
                    post_mosaic_SNR_NF[fid],post_mosaic_RMS_NF[fid]=mosaic_SNR[fid],mosaic_RMS[fid]
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
                #selfcal_library[target][band][vis][solint]['applycal_mode']=applycal_mode[band][iteration]+''
                #selfcal_library[target][band][vis][solint]['applycal_interpolate']=applycal_interpolate[vis]
                #selfcal_library[target][band][vis][solint]['gaincal_combine']=gaincal_combine[band][iteration]+''
                selfcal_library[target][band][vis][solint]['clean_threshold']=selfcal_library[target][band]['nsigma'][iteration]*selfcal_library[target][band]['RMS_curr']
                selfcal_library[target][band][vis][solint]['intflux_pre'],selfcal_library[target][band][vis][solint]['e_intflux_pre']=get_intflux(sani_target+'_'+band+'_'+solint+'_'+str(iteration)+'.image.tt0',RMS)
                selfcal_library[target][band][vis][solint]['fallback']=fallback[vis]+''
                selfcal_library[target][band][vis][solint]['solmode']=solmode[band][iteration]+''
                selfcal_library[target][band][vis][solint]['SNR_post']=post_SNR.copy()
                selfcal_library[target][band][vis][solint]['RMS_post']=post_RMS.copy()
                selfcal_library[target][band][vis][solint]['SNR_NF_post']=post_SNR_NF.copy()
                selfcal_library[target][band][vis][solint]['RMS_NF_post']=post_RMS_NF.copy()
                ## Update RMS value if necessary
                if selfcal_library[target][band][vis][solint]['RMS_post'] < selfcal_library[target][band]['RMS_curr'] and "inf_EB_fb" not in solint:
                   selfcal_library[target][band]['RMS_curr']=selfcal_library[target][band][vis][solint]['RMS_post'].copy()
                if selfcal_library[target][band][vis][solint]['RMS_NF_post'] < selfcal_library[target][band]['RMS_NF_curr'] and "inf_EB_fb" not in solint:
                   selfcal_library[target][band]['RMS_NF_curr']=selfcal_library[target][band][vis][solint]['RMS_NF_post'].copy()
                header=imhead(imagename=sani_target+'_'+band+'_'+solint+'_'+str(iteration)+'_post.image.tt0')
                selfcal_library[target][band][vis][solint]['Beam_major_post']=header['restoringbeam']['major']['value']
                selfcal_library[target][band][vis][solint]['Beam_minor_post']=header['restoringbeam']['minor']['value']
                selfcal_library[target][band][vis][solint]['Beam_PA_post']=header['restoringbeam']['positionangle']['value'] 
                selfcal_library[target][band][vis][solint]['intflux_post'],selfcal_library[target][band][vis][solint]['e_intflux_post']=get_intflux(sani_target+'_'+band+'_'+solint+'_'+str(iteration)+'_post.image.tt0',post_RMS)

                for fid in selfcal_library[target][band]['sub-fields-to-selfcal']:
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
                    #selfcal_library[target][band][fid][vis][solint]['applycal_mode']=applycal_mode[band][iteration]+''
                    #selfcal_library[target][band][fid][vis][solint]['applycal_interpolate']=applycal_interpolate[vis]
                    #selfcal_library[target][band][fid][vis][solint]['gaincal_combine']=gaincal_combine[band][iteration]+''
                    selfcal_library[target][band][fid][vis][solint]['clean_threshold']=selfcal_library[target][band]['nsigma'][iteration]*selfcal_library[target][band]['RMS_curr']
                    selfcal_library[target][band][fid][vis][solint]['intflux_pre'],selfcal_library[target][band][fid][vis][solint]['e_intflux_pre']=get_intflux(imagename+'.image.tt0',mosaic_RMS[fid], mosaic_sub_field=selfcal_library[target][band]["obstype"]=="mosaic")
                    selfcal_library[target][band][fid][vis][solint]['fallback']=fallback[vis]+''
                    selfcal_library[target][band][fid][vis][solint]['solmode']=solmode[band][iteration]+''
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
                    selfcal_library[target][band][fid][vis][solint]['intflux_post'],selfcal_library[target][band][fid][vis][solint]['e_intflux_post']=get_intflux(imagename+'_post.image.tt0',post_RMS, mosaic_sub_field=selfcal_library[target][band]["obstype"]=="mosaic")

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
             for fid in selfcal_library[target][band]['sub-fields-to-selfcal']:
                 strict_field_by_field_success += [(post_mosaic_SNR[fid] >= mosaic_SNR[fid]) and (post_mosaic_SNR_NF[fid] >= mosaic_SNR_NF[fid])]
                 loose_field_by_field_success += [((post_mosaic_SNR[fid]-mosaic_SNR[fid])/mosaic_SNR[fid] > -0.02) and \
                         ((post_mosaic_SNR_NF[fid] - mosaic_SNR_NF[fid])/mosaic_SNR_NF[fid] > -0.02)]
                 beam_field_by_field_success += [delta_beamarea < delta_beam_thresh]

             if 'inf_EB' in solint or np.any(strict_field_by_field_success):
                 # If any of the fields succeed in the "strict" sense, then allow for minor reductions in the evaluation quantity in other
                 # fields because there's a good chance that those are just noise being pushed around.
                 field_by_field_success = numpy.logical_and(loose_field_by_field_success, beam_field_by_field_success)
             else:
                 field_by_field_success = numpy.logical_and(strict_field_by_field_success, beam_field_by_field_success)

             if mode == "cocal" and calculate_inf_EB_fb_anyways and solint == "inf_EB_fb" and selfcal_library[target][band]["SC_success"]:
                # Since we just want to calculate inf_EB_fb for use in inf_fb, we just want to revert to the original state and go back for inf_fb.
                print('****************Reapplying previous solint solutions*************')
                for vis in vislist:
                   print('****************Applying '+str(selfcal_library[target][band][vis]['gaintable_final'])+' to '+target+' '+band+'*************')
                   ## NOTE: should this be selfcal_starting_flags instead of fb_selfcal_starting_flags ???
                   flagmanager(vis=vis,mode='restore',versionname='fb_selfcal_starting_flags_'+sani_target)
                   applycal(vis=vis,\
                           gaintable=selfcal_library[target][band][vis]['gaintable_final'],\
                           interp=selfcal_library[target][band][vis]['applycal_interpolate_final'],\
                           calwt=True,spwmap=selfcal_library[target][band][vis]['spwmap_final'],\
                           applymode=selfcal_library[target][band][vis]['applycal_mode_final'],\
                           field=target,spw=selfcal_library[target][band][vis]['spws'])
             if (((post_SNR >= SNR) and (post_SNR_NF >= SNR_NF) and (delta_beamarea < delta_beam_thresh)) or (('inf_EB' in solint) and ((post_SNR-SNR)/SNR > -0.02) and ((post_SNR_NF - SNR_NF)/SNR_NF > -0.02) and (delta_beamarea < delta_beam_thresh))) and np.any(field_by_field_success): 

                if mode == "cocal" and calculate_inf_EB_fb_anyways and solint == "inf_EB_fb" and selfcal_library[target][band]["SC_success"]:
                    for vis in vislist:
                        selfcal_library[target][band][vis][solint]['Pass'] = True
                        selfcal_library[target][band][vis][solint]['Fail_Reason'] = 'None'
                    for ind, fid in enumerate(selfcal_library[target][band]['sub-fields-to-selfcal']):
                        for vis in vislist:
                            if field_by_field_success[ind]:
                                selfcal_library[target][band][fid][vis][solint]['Pass'] = True
                                selfcal_library[target][band][fid][vis][solint]['Fail_Reason'] = 'None'
                            else:
                                selfcal_library[target][band][fid][vis][solint]['Pass'] = False
                    break

                selfcal_library[target][band]['SC_success']=True
                selfcal_library[target][band]['Stop_Reason']='None'
                for vis in vislist:
                   selfcal_library[target][band][vis]['gaintable_final']=selfcal_library[target][band][vis][solint]['gaintable']
                   selfcal_library[target][band][vis]['spwmap_final']=selfcal_library[target][band][vis][solint]['spwmap'].copy()
                   selfcal_library[target][band][vis]['applycal_mode_final']=selfcal_library[target][band][vis][solint]['applycal_mode']
                   selfcal_library[target][band][vis]['applycal_interpolate_final']=selfcal_library[target][band][vis][solint]['applycal_interpolate']
                   selfcal_library[target][band][vis]['gaincal_combine_final']=selfcal_library[target][band][vis][solint]['gaincal_combine']
                   selfcal_library[target][band][vis][solint]['Pass']=True
                   selfcal_library[target][band][vis][solint]['Fail_Reason']='None'
                if solmode[band][iteration]=='p':            
                   selfcal_library[target][band]['final_phase_solint']=solint
                selfcal_library[target][band]['final_solint']=solint
                selfcal_library[target][band]['final_solint_mode']=solmode[band][iteration]
                selfcal_library[target][band]['iteration']=iteration

                for ind, fid in enumerate(selfcal_library[target][band]['sub-fields-to-selfcal']):
                    if field_by_field_success[ind]:
                        selfcal_library[target][band][fid]['SC_success']=True
                        selfcal_library[target][band][fid]['Stop_Reason']='None'
                        for vis in vislist:
                           selfcal_library[target][band][fid][vis]['gaintable_final']=selfcal_library[target][band][fid][vis][solint]['gaintable']
                           selfcal_library[target][band][fid][vis]['spwmap_final']=selfcal_library[target][band][fid][vis][solint]['spwmap'].copy()
                           selfcal_library[target][band][fid][vis]['applycal_mode_final']=selfcal_library[target][band][fid][vis][solint]['applycal_mode']
                           selfcal_library[target][band][fid][vis]['applycal_interpolate_final']=selfcal_library[target][band][fid][vis][solint]['applycal_interpolate']
                           selfcal_library[target][band][fid][vis]['gaincal_combine_final']=selfcal_library[target][band][fid][vis][solint]['gaincal_combine']
                           selfcal_library[target][band][fid][vis][solint]['Pass']=True
                           selfcal_library[target][band][fid][vis][solint]['Fail_Reason']='None'
                        if solmode[band][iteration]=='p':            
                           selfcal_library[target][band][fid]['final_phase_solint']=solint
                        selfcal_library[target][band][fid]['final_solint']=solint
                        selfcal_library[target][band][fid]['final_solint_mode']=solmode[band][iteration]
                        selfcal_library[target][band][fid]['iteration']=iteration
                    else:
                        for vis in vislist:
                            selfcal_library[target][band][fid][vis][solint]['Pass']=False

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
                       versionname = ("fb_" if mode == "cocal" else "")+'selfcal_starting_flags_'+sani_target
                       flagmanager(vis=vis,mode='restore',versionname=versionname)
                       for fid in selfcal_library[target][band]['sub-fields']:
                           if selfcal_library[target][band][fid]['SC_success']:
                               print('****************Applying '+str(selfcal_library[target][band][vis]['gaintable_final'])+' to '+target+\
                                       ' field '+str(fid)+' '+band+'*************')
                               applycal(vis=vis,\
                                       gaintable=selfcal_library[target][band][fid][vis]['gaintable_final'],\
                                       interp=selfcal_library[target][band][fid][vis]['applycal_interpolate_final'],\
                                       calwt=True,spwmap=selfcal_library[target][band][fid][vis]['spwmap_final'],\
                                       applymode=selfcal_library[target][band][fid][vis]['applycal_mode_final'],\
                                       field=str(fid),spw=selfcal_library[target][band][vis]['spws'])    
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
                    for vis in vislist:
                        selfcal_library[target][band][fid][vis][solint]['Pass']=False
                break


         ## 
         ## if S/N worsens, and/or beam area increases reject current solutions and reapply previous (or revert to origional data)
         ##

         if not selfcal_library[target][band][vislist[0]][solint]['Pass']:
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
            if not np.any(field_by_field_success):
                if reason != '':
                    reason=reason+'; '
                reason=reason+'All sub-fields failed'
            selfcal_library[target][band]['Stop_Reason']=reason
            for vis in vislist:
               selfcal_library[target][band][vis][solint]['Pass']=False
               selfcal_library[target][band][vis][solint]['Fail_Reason']=reason

         mosaic_reason = {}
         new_fields_to_selfcal = []
         for fid in selfcal_library[target][band]['sub-fields-to-selfcal']:
             if not selfcal_library[target][band][fid][vislist[0]][solint]['Pass']:
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
                 if mosaic_reason[fid] == '':
                     mosaic_reason[fid] = "Global selfcal failed"
                 selfcal_library[target][band][fid]['Stop_Reason']=mosaic_reason[fid]
                 for vis in vislist:
                    selfcal_library[target][band][fid][vis][solint]['Pass']=False
                    selfcal_library[target][band][fid][vis][solint]['Fail_Reason']=mosaic_reason[fid]
             else:
                 new_fields_to_selfcal.append(fid)

         # If any of the fields failed self-calibration, we need to re-apply calibrations for all fields because we need to revert flagging back
         # to the starting point.
         if np.any([selfcal_library[target][band][fid][vislist[0]][solint]['Pass'] == False for fid in \
                 selfcal_library[target][band]['sub-fields-to-selfcal']]) or len(selfcal_library[target][band]['sub-fields-to-selfcal']) < \
                 len(selfcal_library[target][band]['sub-fields']):
             print('****************Selfcal failed for some sub-fields:*************')
             for fid in selfcal_library[target][band]['sub-fields']:
                 if fid in selfcal_library[target][band]['sub-fields-to-selfcal']:
                     if selfcal_library[target][band][fid][vis][solint]['Pass'] == False:
                         print('FIELD: '+str(fid)+', REASON: '+mosaic_reason[fid])
                 else:
                     print('FIELD: '+str(fid)+', REASON: Failed earlier solint')
             print('****************Reapplying previous solint solutions where available*************')
             for vis in vislist:
                 versionname = ("fb_" if mode == "cocal" else "")+'selfcal_starting_flags_'+sani_target
                 flagmanager(vis=vis,mode='restore',versionname=versionname)
                 for fid in selfcal_library[target][band]['sub-fields']:
                     if selfcal_library[target][band][fid]['SC_success']:
                         print('****************Applying '+str(selfcal_library[target][band][fid][vis]['gaintable_final'])+' to '+target+' field '+\
                                 str(fid)+' '+band+'*************')
                         applycal(vis=vis,\
                                 gaintable=selfcal_library[target][band][fid][vis]['gaintable_final'],\
                                 interp=selfcal_library[target][band][fid][vis]['applycal_interpolate_final'],\
                                 calwt=True,spwmap=selfcal_library[target][band][fid][vis]['spwmap_final'],\
                                 applymode=selfcal_library[target][band][fid][vis]['applycal_mode_final'],\
                                 field=str(fid),spw=selfcal_library[target][band][vis]['spws'])    
                     else:
                         print('****************Removing all calibrations for '+target+' '+str(fid)+' '+band+'**************')
                         clearcal(vis=vis,field=str(fid),spw=selfcal_library[target][band][vis]['spws'])
                         selfcal_library[target][band]['SNR_post']=selfcal_library[target][band]['SNR_orig'].copy()
                         selfcal_library[target][band]['RMS_post']=selfcal_library[target][band]['RMS_orig'].copy()

                         for fid in selfcal_library[target][band]['sub-fields']:
                             selfcal_library[target][band][fid]['SNR_post']=selfcal_library[target][band][fid]['SNR_orig'].copy()
                             selfcal_library[target][band][fid]['RMS_post']=selfcal_library[target][band][fid]['RMS_orig'].copy()

         # If any of the sub-fields passed, and the whole mosaic passed, then we can move on to the next solint, otherwise we have to back out.
         if selfcal_library[target][band][vislist[0]][solint]['Pass'] == True and \
                 np.any([selfcal_library[target][band][fid][vislist[0]][solint]['Pass'] == True for fid in \
                 selfcal_library[target][band]['sub-fields-to-selfcal']]):
             if mode == "selfcal" and (iteration < len(solints[band])-1) and (selfcal_library[target][band][vis][solint]['SNR_post'] > \
                     selfcal_library[target][band]['SNR_orig']): #(iteration == 0) and 
                print('Updating solint = '+solints[band][iteration+1]+' SNR')
                print('Was: ',solint_snr[target][band][solints[band][iteration+1]])
                get_SNR_self_update([target],band,vislist,selfcal_library[target][band],n_ants,solint,solints[band][iteration+1],integration_time,solint_snr[target][band])
                print('Now: ',solint_snr[target][band][solints[band][iteration+1]])

                for fid in selfcal_library[target][band]['sub-fields-to-selfcal']:
                    print('Field '+str(fid)+' Was: ',solint_snr_per_field[target][band][fid][solints[band][iteration+1]])
                    get_SNR_self_update([target],band,vislist,selfcal_library[target][band][fid],n_ants,solint,solints[band][iteration+1],integration_time,solint_snr_per_field[target][band][fid])
                    print('FIeld '+str(fid)+' Now: ',solint_snr_per_field[target][band][fid][solints[band][iteration+1]])

             # If not all fields succeed for inf_EB or scan_inf/inf, depending on mosaic or single field, then don't go on to amplitude selfcal,
             # even if *some* fields succeeded.
             if iteration <= 1 and (not np.all([selfcal_library[target][band][fid][vislist[0]][solint]['Pass'] == True for fid in \
                    selfcal_library[target][band]['sub-fields-to-selfcal']]) or len(selfcal_library[target][band]['sub-fields-to-selfcal']) < \
                    len(selfcal_library[target][band]['sub-fields'])) and do_amp_selfcal:
                 print("***** NOTE: Amplitude self-calibration turned off because not all fields succeeded at non-inf_EB phase self-calibration")
                 do_amp_selfcal = False
                
             if iteration < (len(solints[band])-1):
                print('****************Selfcal passed, shortening solint*************')
             else:
                print('****************Selfcal passed for Minimum solint*************')
         else:   
            print('****************Selfcal failed*************')
            print('REASON: '+reason)
            if mode == "selfcal" and iteration > 1 and solmode[band][iteration] !='ap' and do_amp_selfcal:  # if a solution interval shorter than inf for phase-only SC has passed, attempt amplitude selfcal
               iterjump=solmode[band].index('ap') 
               selfcal_library[target][band]['sub-fields-to-selfcal'] = selfcal_library[target][band]['sub-fields']
               print('****************Selfcal halted for phase, attempting amplitude*************')
               continue
            elif mode == "cocal" and "inf_fb" in solint:
               print('****************Cocal failed, attempting next inf_fb option*************')
               continue
            else:
               print('****************Aborting further self-calibration attempts for '+target+' '+band+'**************')
               break # breakout of loops of successive solints since solutions are getting worse

         # Finally, update the list of fields to be self-calibrated now that we don't need to know the list at the beginning of this solint.
         selfcal_library[target][band]['sub-fields-to-selfcal'] = new_fields_to_selfcal