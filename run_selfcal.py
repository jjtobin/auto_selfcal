import numpy as np
from scipy import stats
import glob
import sys
#execfile('selfcal_helpers.py',globals())
sys.path.append("./")
from selfcal_helpers import *
from gaincal_wrapper import gaincal_wrapper
from image_analysis_helpers import *
from mosaic_helpers import *
from applycal_wrapper import applycal_wrapper
from casampi.MPIEnvironment import MPIEnvironment 
parallel=MPIEnvironment.is_mpi_enabled

def run_selfcal(selfcal_library, selfcal_plan, target, band, telescope, n_ants, \
        gaincal_minsnr=2.0, gaincal_unflag_minsnr=5.0, minsnr_to_proceed=3.0, delta_beam_thresh=0.05, do_amp_selfcal=True, inf_EB_gaincal_combine='scan', inf_EB_gaintype='G', \
        unflag_only_lbants=False, unflag_only_lbants_onlyap=False, calonly_max_flagged=0.0, second_iter_solmode="", unflag_fb_to_prev_solint=False, \
        rerank_refants=False, mode="selfcal", calibrators="", calculate_inf_EB_fb_anyways=False, preapply_targets_own_inf_EB=False, \
        gaincalibrator_dict={}, allow_gain_interpolation=False, guess_scan_combine=False, aca_use_nfmask=False):

   # If we are running this on a mosaic, we want to rerank reference antennas and have a higher gaincal_minsnr by default.

   if selfcal_library["obstype"] == "mosaic":
       gaincal_minsnr = 2.0
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
        iterjump = len(selfcal_plan['solints']) - 4
        if selfcal_library["SC_success"] and not calculate_inf_EB_fb_anyways:
            iterjump += 1

   vislist=selfcal_library['vislist'].copy()

   if mode == "cocal":
       # Check whether there are suitable calibrators, otherwise skip this target/band.
       include_targets, include_scans = triage_calibrators(vislist[0], target, calibrators[band][0])
       if include_targets == "":
           print("No suitable calibrators found, skipping "+target)
           selfcal_library['Stop_Reason'] += '; No suitable co-calibrators'
           return

   print('Starting selfcal procedure on: '+target+' '+band)
   for iteration in range(len(selfcal_plan['solints'])):
      if (iterjump !=-1) and (iteration < iterjump): # allow jumping to amplitude selfcal and not need to use a while loop
         continue
      elif iteration == iterjump:
         iterjump=-1
      print("Solving for solint="+selfcal_plan['solints'][iteration])

      # Set some cocal parameters.
      if selfcal_plan['solints'][iteration] in ["inf_EB_fb","inf_fb1"]:
          calculate_inf_EB_fb_anyways = True
          preapply_targets_own_inf_EB = False
      elif selfcal_plan['solints'][iteration] == "inf_fb2":
          calculate_inf_EB_fb_anyways = False
          preapply_targets_own_inf_EB = False
          # If there was not a successful inf_EB solint, then this duplicates inf_fb1 so skip
          if "inf_EB" not in selfcal_library[vislist[0]]:
              continue
          elif not selfcal_library[vislist[0]]["inf_EB"]['Pass']:
              continue
      elif selfcal_plan['solints'][iteration] == "inf_fb3":
          calculate_inf_EB_fb_anyways = False
          preapply_targets_own_inf_EB = True
          # If there was no inf solint (e.g. because each source was observed only a single time, skip this as there are no gain tables to stick together.
          if "inf" not in selfcal_plan['solints']:
              continue

      if 'ap' in selfcal_plan['solints'][iteration] and not do_amp_selfcal:
          break

      if mode == "selfcal" and selfcal_plan['solint_snr'][selfcal_plan['solints'][iteration]] < minsnr_to_proceed and np.all([selfcal_plan[fid]['solint_snr_per_field'][selfcal_plan['solints'][iteration]] < minsnr_to_proceed for fid in selfcal_library['sub-fields']]):
         print('*********** estimated SNR for solint='+selfcal_plan['solints'][iteration]+' too low, measured: '+str(selfcal_plan['solint_snr'][selfcal_plan['solints'][iteration]])+', Min SNR Required: '+str(minsnr_to_proceed)+' **************')
         if iteration > 1 and selfcal_plan['solmode'][iteration] !='ap' and do_amp_selfcal:  # if a solution interval shorter than inf for phase-only SC has passed, attempt amplitude selfcal
            iterjump=selfcal_plan['solmode'].index('ap') 
            print('****************Attempting amplitude selfcal*************')
            continue

         selfcal_library['Stop_Reason']='Estimated_SNR_too_low_for_solint '+selfcal_plan['solints'][iteration]
         break
      else:
         solint=selfcal_plan['solints'][iteration]
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
         if selfcal_library['final_solint'] != 'None':
             prev_solint = selfcal_library['final_solint']
             prev_iteration = selfcal_library[vislist[0]][prev_solint]['iteration']

             nterms_changed = (len(glob.glob(sani_target+'_'+band+'_'+prev_solint+'_'+str(prev_iteration)+"_post.model.tt*")) < 
                    selfcal_library['nterms'])

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

         nfsnr_modifier = selfcal_library['RMS_NF_curr'] / selfcal_library['RMS_curr']
         tclean_wrapper(selfcal_library,sani_target+'_'+band+'_'+solint+'_'+str(iteration),
                     band,telescope=telescope,nsigma=selfcal_library['nsigma'][iteration], scales=[0],
                     threshold=str(selfcal_library['nsigma'][iteration]*selfcal_library['RMS_NF_curr'])+'Jy',
                     savemodel='none',parallel=parallel,
                     field=target, nfrms_multiplier=nfsnr_modifier, resume=resume)

         # Check that a mask was actually created, because if not the model will be empty and gaincal will do bad things and the 
         # code will break.
         if not checkmask(sani_target+'_'+band+'_'+solint+'_'+str(iteration)+'.image.tt0'):
             selfcal_library['Stop_Reason'] = 'Empty model for solint '+solint
             break # breakout of loop because the model is empty and gaincal will therefore fail


         # Loop through up to two times. On the first attempt, try applymode = 'calflag' (assuming this is requested by the user). On the
         # second attempt, use applymode = 'calonly'.
         for applymode in np.unique([selfcal_plan['applycal_mode'][iteration],'calonly']):
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
                 tclean_wrapper(selfcal_library,sani_target+'_'+band+'_'+solint+'_'+str(iteration),
                             band,telescope=telescope,nsigma=selfcal_library['nsigma'][iteration], scales=[0],
                             threshold=str(selfcal_library['nsigma'][iteration]*selfcal_library['RMS_NF_curr'])+'Jy',
                             savemodel='modelcolumn',parallel=parallel,
                             field=target, nfrms_multiplier=nfsnr_modifier, savemodel_only=True)

             for vis in vislist:
                # Record gaincal details.
                selfcal_library[vis][solint]={}
                for fid in np.intersect1d(selfcal_library['sub-fields-to-selfcal'],list(selfcal_library['sub-fields-fid_map'][vis].keys())):
                    selfcal_library[fid][vis][solint]={}

             # Fields that don't have any mask in the primary beam should be removed from consideration, as their models are likely bad.
             if selfcal_library['obstype'] == 'mosaic':
                 selfcal_library['sub-fields-to-gaincal'] = evaluate_subfields_to_gaincal(selfcal_library, target, band, 
                         solint, iteration, selfcal_plan['solmode'], selfcal_plan['solints'], selfcal_plan, minsnr_to_proceed,
                         allow_gain_interpolation=allow_gain_interpolation)

                 if solint != 'inf_EB' and not allow_gain_interpolation:
                     selfcal_library['sub-fields-to-selfcal'] = selfcal_library['sub-fields-to-gaincal']
             else:
                selfcal_library['sub-fields-to-gaincal'] = selfcal_library['sub-fields-to-selfcal']

             # Calculate the complex gains
             for vis in vislist:
                if np.intersect1d(selfcal_library['sub-fields-to-gaincal'],\
                        list(selfcal_library['sub-fields-fid_map'][vis].keys())).size == 0:
                     continue

                gaincal_wrapper(selfcal_library, selfcal_plan, target, band, vis, solint, applymode, iteration, gaincal_minsnr, 
                        rerank_refants=rerank_refants, unflag_only_lbants=unflag_only_lbants, 
                        unflag_only_lbants_onlyap=unflag_only_lbants_onlyap, calonly_max_flagged=calonly_max_flagged, 
                        second_iter_solmode=second_iter_solmode, unflag_fb_to_prev_solint=unflag_fb_to_prev_solint, \
                        refantmode=refantmode, mode=mode, calibrators=calibrators, gaincalibrator_dict=gaincalibrator_dict, 
                        allow_gain_interpolation=allow_gain_interpolation)

             # With gaincal done and bad fields removed from gain tables if necessary, check whether any fields should no longer be 
             # selfcal'd because they have too much interpolation.
             if selfcal_library['obstype'] == 'mosaic':
                 selfcal_library['sub-fields-to-selfcal'] = evaluate_subfields_after_gaincal(selfcal_library, target, band, 
                         solint, iteration, selfcal_plan['solmode'], allow_gain_interpolation=allow_gain_interpolation)

             ##
             ## Apply gain solutions per MS, target, solint, and band
             ##
             for vis in vislist:
                applycal_wrapper(vis, target, band, solint, selfcal_library, 
                        current=lambda f: f in selfcal_library['sub-fields-to-selfcal'],
                        final=lambda f: f not in selfcal_library['sub-fields-to-selfcal'] and selfcal_library[f]['SC_success'],
                        restore_flags='fb_selfcal_starting_flags_'+sani_target if mode == "cocal" else None)

             ## Create post self-cal image using the model as a startmodel to evaluate how much selfcal helped
             ##

             os.system('rm -rf '+sani_target+'_'+band+'_'+solint+'_'+str(iteration)+'_post*')
             tclean_wrapper(selfcal_library,sani_target+'_'+band+'_'+solint+'_'+str(iteration)+'_post',
                      band,telescope=telescope,nsigma=selfcal_library['nsigma'][iteration], scales=[0],
                      threshold=str(selfcal_library['nsigma'][iteration]*selfcal_library['RMS_NF_curr'])+'Jy',
                      savemodel='none',parallel=parallel,
                      field=target, nfrms_multiplier=nfsnr_modifier)

             ##
             ## Do the assessment of the post- (and pre-) selfcal images.
             ##
             print('Pre selfcal assessemnt: '+target)
             SNR, RMS, SNR_NF, RMS_NF = get_image_stats(sani_target+'_'+band+'_'+solint+'_'+str(iteration)+'.image.tt0', 
                     sani_target+'_'+band+'_'+solint+'_'+str(iteration)+'_post.mask', sani_target+'_'+band+'_'+solint+'_'+str(iteration)+'.mask', 
                     selfcal_library, (telescope != 'ACA' or aca_use_nfmask), solint, 'pre')

             print('Post selfcal assessemnt: '+target)
             post_SNR, post_RMS, post_SNR_NF, post_RMS_NF = get_image_stats(sani_target+'_'+band+'_'+solint+'_'+str(iteration)+'_post.image.tt0',
                     sani_target+'_'+band+'_'+solint+'_'+str(iteration)+'_post.mask', sani_target+'_'+band+'_'+solint+'_'+str(iteration)+'.mask',
                     selfcal_library, (telescope != 'ACA' or aca_use_nfmask), solint, 'post')

             mosaic_SNR, mosaic_RMS, mosaic_SNR_NF, mosaic_RMS_NF = {}, {}, {}, {}
             post_mosaic_SNR, post_mosaic_RMS, post_mosaic_SNR_NF, post_mosaic_RMS_NF = {}, {}, {}, {}
             for fid in selfcal_library['sub-fields-to-selfcal']:
                 if selfcal_library['obstype'] == 'mosaic':
                     imagename = sani_target+'_field_'+str(fid)+'_'+band+'_'+solint+'_'+str(iteration)
                 else:
                     imagename = sani_target+'_'+band+'_'+solint+'_'+str(iteration)

                 print()
                 print('Pre selfcal assessemnt: '+target+', field '+str(fid))
                 mosaic_SNR[fid], mosaic_RMS[fid], mosaic_SNR_NF[fid], mosaic_RMS_NF[fid] = get_image_stats(imagename+'.image.tt0', 
                         imagename+'_post.mask', imagename+'.mask', selfcal_library[fid], (telescope != 'ACA' or aca_use_nfmask), solint,
                         'pre', mosaic_sub_field=selfcal_library["obstype"]=="mosaic")

                 print('Post selfcal assessemnt: '+target+', field '+str(fid))
                 post_mosaic_SNR[fid], post_mosaic_RMS[fid], post_mosaic_SNR_NF[fid], post_mosaic_RMS_NF[fid] = get_image_stats(
                         imagename+'_post.image.tt0', imagename+'_post.mask', imagename+'.mask', selfcal_library[fid], 
                         (telescope != 'ACA' or aca_use_nfmask), solint, 'post', mosaic_sub_field=selfcal_library["obstype"]=="mosaic")
                 print()

             # change nterms to 2 if needed based on fracbw and SNR
             if selfcal_library['nterms'] == 1:
                 selfcal_library['nterms']=check_image_nterms(selfcal_library['fracbw'],post_SNR)

             ##
             ## record self cal results/details for this solint
             ##
             for vis in vislist:
                selfcal_library[vis][solint]['clean_threshold']=selfcal_library['nsigma'][iteration]*selfcal_library['RMS_NF_curr']

                for fid in np.intersect1d(selfcal_library['sub-fields-to-selfcal'],list(selfcal_library['sub-fields-fid_map'][vis].keys())):
                    selfcal_library[fid][vis][solint]['clean_threshold']=selfcal_library['nsigma'][iteration]*selfcal_library['RMS_NF_curr']

                ## Update RMS value if necessary
                if selfcal_library[vis][solint]['RMS_post'] < selfcal_library['RMS_curr'] and \
                        "inf_EB_fb" not in solint and vis == vislist[-1]:
                   selfcal_library['RMS_curr']=selfcal_library[vis][solint]['RMS_post'].copy()
                if selfcal_library[vis][solint]['RMS_NF_post'] < selfcal_library['RMS_NF_curr'] and \
                        "inf_EB_fb" not in solint and selfcal_library[vis][solint]['RMS_NF_post'] > 0 and vis == vislist[-1]:
                   selfcal_library['RMS_NF_curr']=selfcal_library[vis][solint]['RMS_NF_post'].copy()

             ##
             ## compare beam relative to original image to ensure we are not incrementally changing the beam in each iteration
             ##
             beamarea_orig=selfcal_library['Beam_major_orig']*selfcal_library['Beam_minor_orig']
             beamarea_post=selfcal_library[vislist[0]][solint]['Beam_major_post']*selfcal_library[vislist[0]][solint]['Beam_minor_post']
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
             for fid in selfcal_library['sub-fields-to-selfcal']:
                 strict_field_by_field_success += [(post_mosaic_SNR[fid] >= mosaic_SNR[fid]) and (post_mosaic_SNR_NF[fid] >= mosaic_SNR_NF[fid])]
                 loose_field_by_field_success += [((post_mosaic_SNR[fid]-mosaic_SNR[fid])/mosaic_SNR[fid] > -0.02) and \
                         ((post_mosaic_SNR_NF[fid] - mosaic_SNR_NF[fid])/mosaic_SNR_NF[fid] > -0.02)]
                 beam_field_by_field_success += [delta_beamarea < delta_beam_thresh]

             if 'inf_EB' in solint:
                 # If any of the fields succeed in the "strict" sense, then allow for minor reductions in the evaluation quantity in other
                 # fields because there's a good chance that those are just noise being pushed around.
                 field_by_field_success = numpy.logical_and(loose_field_by_field_success, beam_field_by_field_success)
             else:
                 field_by_field_success = numpy.logical_and(strict_field_by_field_success, beam_field_by_field_success)

             # If not all fields were successful, we need to make an additional image to evaluate whether the image as a whole improved,
             # otherwise the _post image won't be exactly representative.
             if selfcal_library['obstype'] == "mosaic" and not np.all(field_by_field_success):
                 field_by_field_success_dict = dict(zip(selfcal_library['sub-fields-to-selfcal'], field_by_field_success))
                 print('****************Not all fields were successful, so re-applying and re-making _post image*************')
                 for vis in vislist:
                     applycal_wrapper(vis, target, band, solint, selfcal_library, 
                             current=lambda f: f in field_by_field_success_dict and field_by_field_success_dict[f],
                             final=lambda f: (f not in field_by_field_success_dict or not field_by_field_success_dict[f]) and 
                                 selfcal_library[f]['SC_success'],
                             clear=lambda f: (f not in field_by_field_success_dict or not field_by_field_success_dict[f]) and 
                                 not selfcal_library[f]['SC_success'],
                             restore_flags='selfcal_starting_flags_'+sani_target)


                 files = glob.glob(sani_target+'_'+band+'_'+solint+'_'+str(iteration)+"_post.*")
                 for f in files:
                     os.system("mv "+f+" "+f.replace("_post","_post_intermediate"))

                 tclean_wrapper(selfcal_library,sani_target+'_'+band+'_'+solint+'_'+str(iteration)+'_post',
                          band,telescope=telescope,nsigma=selfcal_library['nsigma'][iteration], scales=[0],
                          threshold=str(selfcal_library[vislist[0]][solint]['clean_threshold'])+'Jy',
                          savemodel='none',parallel=parallel,
                          field=target, nfrms_multiplier=nfsnr_modifier, image_mosaic_fields_separately=False)

                 ##
                 ## Do the assessment of the post- (and pre-) selfcal images.
                 ##
                 print('Pre selfcal assessemnt: '+target)
                 SNR, RMS, SNR_NF, RMS_NF = get_image_stats(sani_target+'_'+band+'_'+solint+'_'+str(iteration)+'.image.tt0', 
                         sani_target+'_'+band+'_'+solint+'_'+str(iteration)+'_post.mask', sani_target+'_'+band+'_'+solint+'_'+str(iteration)+'.mask', 
                         selfcal_library, (telescope != 'ACA' or aca_use_nfmask), solint, 'pre')

                 print('Post selfcal assessemnt: '+target)
                 post_SNR, post_RMS, post_SNR_NF, post_RMS_NF = get_image_stats(sani_target+'_'+band+'_'+solint+'_'+str(iteration)+'_post.image.tt0',
                         sani_target+'_'+band+'_'+solint+'_'+str(iteration)+'_post.mask', sani_target+'_'+band+'_'+solint+'_'+str(iteration)+'.mask',
                         selfcal_library, (telescope != 'ACA' or aca_use_nfmask), solint, 'post')

                 for vis in vislist:
                    ##
                    ## record self cal results/details for this solint
                    ##
                    selfcal_library[vis][solint]['clean_threshold']=selfcal_library['nsigma'][iteration]*selfcal_library['RMS_NF_curr']
                    selfcal_library[vis][solint]['solmode']=selfcal_plan['solmode'][iteration]+''
                    ## Update RMS value if necessary
                    if selfcal_library[vis][solint]['RMS_post'] < selfcal_library['RMS_curr']:
                       selfcal_library['RMS_curr']=selfcal_library[vis][solint]['RMS_post'].copy()
                    if selfcal_library[vis][solint]['RMS_NF_post'] < selfcal_library['RMS_NF_curr'] and \
                            selfcal_library[vis][solint]['RMS_NF_post'] > 0:
                       selfcal_library['RMS_NF_curr']=selfcal_library[vis][solint]['RMS_NF_post'].copy()

             if mode == "cocal" and calculate_inf_EB_fb_anyways and solint == "inf_EB_fb" and selfcal_library["SC_success"]:
                # Since we just want to calculate inf_EB_fb for use in inf_fb, we just want to revert to the original state and go back for inf_fb.
                print('****************Reapplying previous solint solutions*************')
                for vis in vislist:
                   print('****************Applying '+str(selfcal_library[vis]['gaintable_final'])+' to '+target+' '+band+'*************')
                   ## NOTE: should this be selfcal_starting_flags instead of fb_selfcal_starting_flags ???
                   flagmanager(vis=vis,mode='restore',versionname='fb_selfcal_starting_flags_'+sani_target)
                   applycal(vis=vis,\
                           gaintable=selfcal_library[vis]['gaintable_final'],\
                           interp=selfcal_library[vis]['applycal_interpolate_final'],\
                           calwt=True,spwmap=selfcal_library[vis]['spwmap_final'],\
                           applymode=selfcal_library[vis]['applycal_mode_final'],\
                           field=target,spw=selfcal_library[vis]['spws'])

             if (((post_SNR >= SNR) and (post_SNR_NF >= SNR_NF) and (delta_beamarea < delta_beam_thresh)) or (('inf_EB' in solint) and ((post_SNR-SNR)/SNR > -0.02) and ((post_SNR_NF - SNR_NF)/SNR_NF > -0.02) and (delta_beamarea < delta_beam_thresh))) and np.any(field_by_field_success): 

                if mode == "cocal" and calculate_inf_EB_fb_anyways and solint == "inf_EB_fb" and selfcal_library["SC_success"]:
                    for vis in vislist:
                        selfcal_library[vis][solint]['Pass'] = True
                        selfcal_library[vis][solint]['Fail_Reason'] = 'None'
                    for ind, fid in enumerate(selfcal_library['sub-fields-to-selfcal']):
                        for vis in vislist:
                            if field_by_field_success[ind]:
                                selfcal_library[fid][vis][solint]['Pass'] = True
                                selfcal_library[fid][vis][solint]['Fail_Reason'] = 'None'
                            else:
                                selfcal_library[fid][vis][solint]['Pass'] = False
                    break

                selfcal_library['SC_success']=True
                selfcal_library['Stop_Reason']='None'
                for vis in vislist:
                   selfcal_library[vis]['gaintable_final']=selfcal_library[vis][solint]['gaintable']
                   selfcal_library[vis]['spwmap_final']=selfcal_library[vis][solint]['spwmap'].copy()
                   selfcal_library[vis]['applycal_mode_final']=selfcal_library[vis][solint]['applycal_mode']
                   selfcal_library[vis]['applycal_interpolate_final']=selfcal_library[vis][solint]['applycal_interpolate']
                   selfcal_library[vis]['gaincal_combine_final']=selfcal_library[vis][solint]['gaincal_combine']
                   selfcal_library[vis][solint]['Pass']=True
                   selfcal_library[vis][solint]['Fail_Reason']='None'
                if selfcal_plan['solmode'][iteration]=='p':            
                   selfcal_library['final_phase_solint']=solint
                selfcal_library['final_solint']=solint
                selfcal_library['final_solint_mode']=selfcal_plan['solmode'][iteration]
                selfcal_library['iteration']=iteration

                for ind, fid in enumerate(selfcal_library['sub-fields-to-selfcal']):
                    if field_by_field_success[ind]:
                        selfcal_library[fid]['SC_success']=True
                        selfcal_library[fid]['Stop_Reason']='None'
                        for vis in selfcal_library[fid]['vislist']:
                           selfcal_library[fid][vis]['gaintable_final']=selfcal_library[fid][vis][solint]['gaintable']
                           selfcal_library[fid][vis]['spwmap_final']=selfcal_library[fid][vis][solint]['spwmap'].copy()
                           selfcal_library[fid][vis]['applycal_mode_final']=selfcal_library[fid][vis][solint]['applycal_mode']
                           selfcal_library[fid][vis]['applycal_interpolate_final']=selfcal_library[fid][vis][solint]['applycal_interpolate']
                           selfcal_library[fid][vis]['gaincal_combine_final']=selfcal_library[fid][vis][solint]['gaincal_combine']
                           selfcal_library[fid][vis][solint]['Pass']=True
                           selfcal_library[fid][vis][solint]['Fail_Reason']='None'
                        if selfcal_plan['solmode'][iteration]=='p':            
                           selfcal_library[fid]['final_phase_solint']=solint
                        selfcal_library[fid]['final_solint']=solint
                        selfcal_library[fid]['final_solint_mode']=selfcal_plan['solmode'][iteration]
                        selfcal_library[fid]['iteration']=iteration
                    else:
                        for vis in selfcal_library[fid]['vislist']:
                            selfcal_library[fid][vis][solint]['Pass']=False

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
                       applycal_wrapper(vis, target, band, solint, selfcal_library, 
                               final=lambda f: f in selfcal_library[f]['SC_success'],
                               restore_flags=("fb_" if mode == "cocal" else "")+'selfcal_starting_flags_'+sani_target)
                 else:
                    for vis in vislist:
                       selfcal_plan[vis]['inf_EB_gaincal_combine']=inf_EB_gaincal_combine #'scan'
                       if selfcal_library['obstype']=='mosaic':
                          selfcal_plan[vis]['inf_EB_gaincal_combine']+=',field'   
                       selfcal_plan[vis]['inf_EB_gaintype']=inf_EB_gaintype #G
                       selfcal_plan[vis]['inf_EB_fallback_mode']='' #'scan'
                 print('****************Attempting applymode="calonly" fallback*************')
             else:
                for vis in vislist:
                   selfcal_library[vis][solint]['Pass']=False

                for fid in selfcal_library['sub-fields-to-selfcal']:
                    for vis in selfcal_library[fid]['vislist']:
                        selfcal_library[fid][vis][solint]['Pass']=False
                break


         ## 
         ## if S/N worsens, and/or beam area increases reject current solutions and reapply previous (or revert to origional data)
         ##

         if not selfcal_library[vislist[0]][solint]['Pass']:
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
            selfcal_library['Stop_Reason']=reason
            for vis in vislist:
               selfcal_library[vis][solint]['Pass']=False
               selfcal_library[vis][solint]['Fail_Reason']=reason

         mosaic_reason = {}
         new_fields_to_selfcal = []
         for fid in selfcal_library['sub-fields-to-selfcal']:
             if not selfcal_library[fid][selfcal_library[fid]['vislist'][0]][solint]['Pass']:
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
                 selfcal_library[fid]['Stop_Reason']=mosaic_reason[fid]
                 for vis in selfcal_library[fid]['vislist']:
                    selfcal_library[fid][vis][solint]['Pass']=False
                    selfcal_library[fid][vis][solint]['Fail_Reason']=mosaic_reason[fid]
             else:
                 new_fields_to_selfcal.append(fid)

         # If any of the fields failed self-calibration, we need to re-apply calibrations for all fields because we need to revert flagging back
         # to the starting point.
         if np.any([selfcal_library[fid][selfcal_library[fid]['vislist'][0]][solint]['Pass'] == False for fid in \
                 selfcal_library['sub-fields-to-selfcal']]) or len(selfcal_library['sub-fields-to-selfcal']) < \
                 len(selfcal_library['sub-fields']):
             print('****************Selfcal failed for some sub-fields:*************')
             for fid in selfcal_library['sub-fields']:
                 if fid in selfcal_library['sub-fields-to-selfcal']:
                     if selfcal_library[fid][selfcal_library[fid]['vislist'][0]][solint]['Pass'] == False:
                         print('FIELD: '+str(fid)+', REASON: '+mosaic_reason[fid])
                 else:
                     print('FIELD: '+str(fid)+', REASON: Failed earlier solint')
             print('****************Reapplying previous solint solutions where available*************')
             for vis in vislist:
                 applycal_wrapper(vis, target, band, solint, selfcal_library, 
                         final=lambda f: selfcal_library[f]['SC_success'],
                         clear=lambda f: not selfcal_library[f]['SC_success'], 
                         restore_flags=("fb_" if mode == "cocal" else "")+'selfcal_starting_flags_'+sani_target)

                 for fid in np.intersect1d(selfcal_library['sub-fields'],list(selfcal_library['sub-fields-fid_map'][vis].keys())):
                     if not selfcal_library[fid]['SC_success']:
                         selfcal_library['SNR_post']=selfcal_library['SNR_orig'].copy()
                         selfcal_library['RMS_post']=selfcal_library['RMS_orig'].copy()

                         for fid in selfcal_library['sub-fields']:
                             selfcal_library[fid]['SNR_post']=selfcal_library[fid]['SNR_orig'].copy()
                             selfcal_library[fid]['RMS_post']=selfcal_library[fid]['RMS_orig'].copy()

         # If any of the sub-fields passed, and the whole mosaic passed, then we can move on to the next solint, otherwise we have to back out.
         if selfcal_library[vislist[0]][solint]['Pass'] == True and \
                 np.any([selfcal_library[fid][selfcal_library[fid]['vislist'][0]][solint]['Pass'] == True for fid in \
                 selfcal_library['sub-fields-to-selfcal']]):
             if mode == "selfcal" and (iteration < len(selfcal_plan['solints'])-1) and (selfcal_library[vis][solint]['SNR_post'] > \
                     selfcal_library['SNR_orig']): #(iteration == 0) and 
                print('Updating solint = '+selfcal_plan['solints'][iteration+1]+' SNR')
                print('Was: ',selfcal_plan['solint_snr'][selfcal_plan['solints'][iteration+1]])
                get_SNR_self_update(selfcal_library,n_ants,solint,selfcal_plan['solints'][iteration+1],selfcal_plan['integration_time'],
                        selfcal_plan['solint_snr'])
                print('Now: ',selfcal_plan['solint_snr'][selfcal_plan['solints'][iteration+1]])

                for fid in selfcal_library['sub-fields-to-selfcal']:
                    print('Field '+str(fid)+' Was: ',selfcal_plan[fid]['solint_snr_per_field'][selfcal_plan['solints'][iteration+1]])
                    get_SNR_self_update(selfcal_library[fid],n_ants,solint,selfcal_plan['solints'][iteration+1],
                            selfcal_plan['integration_time'],selfcal_plan[fid]['solint_snr_per_field'])
                    print('FIeld '+str(fid)+' Now: ',selfcal_plan[fid]['solint_snr_per_field'][selfcal_plan['solints'][iteration+1]])

             # If not all fields succeed for inf_EB or scan_inf/inf, depending on mosaic or single field, then don't go on to amplitude selfcal,
             # even if *some* fields succeeded.
             if iteration <= 1 and ((not np.all([selfcal_library[fid][selfcal_library[fid]['vislist'][0]][solint]['Pass'] == True for fid in \
                    selfcal_library['sub-fields-to-selfcal']])) or len(selfcal_library['sub-fields-to-selfcal']) < \
                    len(selfcal_library['sub-fields'])) and do_amp_selfcal:
                 print("***** NOTE: Amplitude self-calibration turned off because not all fields succeeded at non-inf_EB phase self-calibration")
                 do_amp_selfcal = False
                
             if iteration < (len(selfcal_plan['solints'])-1):
                print('****************Selfcal passed, shortening solint*************')
             else:
                print('****************Selfcal passed for Minimum solint*************')
         elif mode == "cocal" and solint == "inf_EB_fb" and (selfcal_library[vislist[0]]["inf_EB"]['Pass'] if "inf_EB" in \
                 selfcal_library[vislist[0]] else False):
            print('****************Selfcal failed for inf_EB_fb, skipping inf_fb1*****************')
            iterjump = selfcal_plan['solints'].index('inf_fb2')
            continue
         else:   
            print('****************Selfcal failed*************')
            print('REASON: '+reason)
            if mode == "selfcal" and iteration > 1 and selfcal_plan['solmode'][iteration] !='ap' and do_amp_selfcal:  # if a solution interval shorter than inf for phase-only SC has passed, attempt amplitude selfcal
               iterjump=selfcal_plan['solmode'].index('ap') 
               selfcal_library['sub-fields-to-selfcal'] = selfcal_library['sub-fields']
               print('****************Selfcal halted for phase, attempting amplitude*************')
               continue
            elif mode == "cocal" and "inf_fb" in solint:
               print('****************Cocal failed, attempting next inf_fb option*************')
               continue
            else:
               print('****************Aborting further self-calibration attempts for '+target+' '+band+'**************')
               break # breakout of loops of successive solints since solutions are getting worse

         # Finally, update the list of fields to be self-calibrated now that we don't need to know the list at the beginning of this solint.
         new_fields_to_selfcal = []
         for fid in selfcal_library['sub-fields']:
             if mode == "cocal":
                 if ("inf_EB" in selfcal_library[fid]['vislist'][0] and selfcal_library[fid][selfcal_library[fid]['vislist'][0]]["inf_EB"]["Pass"]) or selfcal_library[fid][selfcal_library[fid]['vislist'][0]]["inf_EB_fb"]["Pass"]:
                     new_fields_to_selfcal.append(fid)
             else:
                 if selfcal_library[fid][selfcal_library[fid]['vislist'][0]]["inf_EB"]["Pass"]:
                     new_fields_to_selfcal.append(fid)

         selfcal_library['sub-fields-to-selfcal'] = new_fields_to_selfcal
