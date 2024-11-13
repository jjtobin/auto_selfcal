#future improvements
# heuristics for switching between calonly and calflag
# heuristics to switch from combine=spw to combine=''
# switch heirarchy of selfcal_library such that solint is at a higher level than vis. makes storage of some parameters awkward since they live
#    in the per vis level instead of per solint

import numpy as np
import sys
import pickle

from .selfcal_helpers import *
from .run_selfcal import run_selfcal
from .image_analysis_helpers import *
from .weblog_creation import *
from .prepare_selfcal import prepare_selfcal, set_clean_thresholds, plan_selfcal_per_solint
from .original_ms_helpers import applycal_to_orig_MSes, uvcontsub_orig_MSes

import casatasks
from casatools import msmetadata as msmdtool

msmd = msmdtool()

def auto_selfcal(
        vislist=[], 
        spectral_average=True,
        do_amp_selfcal=True,
        
                     # input as dictionary for target name to allow support of multiple targets           
        usermask={},  # require that it is a CRTF region (CASA region format)
                     # usermask={'IRAS32':{'Band_6':'IRAS32.rgn'}, 'IRS5N':{'Band_6': 'IRS5N.rgn'}}
                     # If multiple sources and only want to use a mask for one, just specify that source.
                     # The keys for remaining sources will be filled with empty strings
                     # NOTE THE DICTIONARY HEIRARCHY HAS CHANGED FROM PREVIOUS VERSION, NOW IT IS [TARGET][BAND] INSTEAD OF [BAND][TARGET]
        
        usermodel={}, 
                     # input as dictionary for target name to allow support of multiple targets
                     # if includes .fits, assume a fits image, otherwise assume a CASA image
                     # for spectral image, input as list i.e., usermodel=['usermodel.tt0','usermodel.tt1']
                     # usermodel={'IRAS32':{'Band_6':['IRAS32-model.tt0','IRAS32-model.tt1']}, 'IRS5N':{'Band_6'['IRS5N-model.tt0','IRS5N-model.tt1']}}
                     # If multiple sources and only want to use a model for one, just specify that source.
                     # The keys for remaining sources will be filled with empty strings
                     # NOTE THE DICTIONARY HEIRARCHY HAS CHANGED FROM PREVIOUS VERSION, NOW IT IS [TARGET][BAND] INSTEAD OF [BAND][TARGET]
        
        
        inf_EB_gaincal_combine='scan',  # should we get rid of this option?
        inf_EB_gaintype='G',
        inf_EB_override=False,
        optimize_spw_combine=True,      # if False, will not attempt per spw or per baseband solutions for any solint except inf_EB
        gaincal_minsnr=2.0,
        gaincal_unflag_minsnr=5.0,
        minsnr_to_proceed=2.95,
        spectral_solution_fraction=0.25,
        delta_beam_thresh=0.05,
        apply_cal_mode_default='calflag',
        unflag_only_lbants = False,
        unflag_only_lbants_onlyap = False,
        calonly_max_flagged = 0.0,
        second_iter_solmode = "",
        unflag_fb_to_prev_solint = False,
        rerank_refants=False,
        allow_gain_interpolation=False,
        guess_scan_combine=False,
        aca_use_nfmask=False,
        allow_cocal=False,
        scale_fov=1.0,   # option to make field of view larger than the default
        rel_thresh_scaling='log10',  #can set to linear, log10, or loge (natural log)
        dividing_factor=-99.0,  # number that the peak SNR is divided by to determine first clean threshold -99.0 uses default
                               # default is 40 for <8ghz and 15.0 for all other frequencies
        check_all_spws=False,   # generate per-spw images to check phase transfer did not go poorly for narrow windows
        apply_to_target_ms=False, # apply final selfcal solutions back to the input _target.ms files
        uvcontsub_target_ms=False, # continuum subtract the input _target.ms files
        sort_targets_and_EBs=False,
        run_findcont=False,
        debug=False, 
        parallel=False,
        weblog=True,
        **kwargs):

    # Check that the vislist keyword is supplied correctly.

    if not is_iterable(vislist):
        print("Argument vislist must be a string or list-like. Exiting...")
    elif type(vislist) == str:
        vislist = [vislist]
    elif len(vislist) == 0:
        ##
        ## Get list of MS files in directory
        ##
        vislist=glob.glob('*_target.ms')
        if len(vislist) == 0:
           vislist=glob.glob('*_targets.ms')   # adaptation for PL2022 output
           if len(vislist)==0:
              vislist=glob.glob('*_cont.ms')   # adaptation for PL2022 output
              if len(vislist)==0:
                 if len(glob.glob("calibrated_final.ms")) > 0:
                     split_calibrated_final(vis=['calibrated_final.ms'])
                 else:
                     sys.exit('No Measurement sets found in current working directory, exiting')

    n_ants=get_n_ants(vislist)
    telescope=get_telescope(vislist[0])

    if run_findcont and os.path.exists("cont.dat"):
        if np.any([len(parse_contdotdat('cont.dat',target)) == 0 for target in all_targets]):
            if not os.path.exists("cont.dat.original"):
                print("Found existing cont.dat, but it is missing targets. Backing that up to cont.dat.original")
                os.system("mv cont.dat cont.dat.original")
            else:
                print("Found existing cont.dat, but it is missing targets. A backup of the original (cont.dat.original) already exists, so not backing up again.")
        elif run_findcont:
            print("cont.dat already exists and includes all targets, so running findcont is not needed. Continuing...")
            run_findcont=False

    if run_findcont:
        try:
            if 'pipeline' not in sys.modules:
                print("Pipeline found but not imported. Importing...")
                import pipeline
                pipeline.initcli()

            print("Running findcont")
            h_init()
            hifa_importdata(vis=vislist, dbservice=False)
            hif_checkproductsize(maxcubesize=60.0, maxcubelimit=70.0, maxproductsize=4000.0)
            hif_makeimlist(specmode="mfs")
            hif_findcont()
        except:
            print("\nWARNING: Cannot run findcont as the pipeline was not found. Please retry with a CASA version that includes the pipeline or start CASA with the --pipeline flag.\n")
            sys.exit(0)

    ##
    ## Get all of the relevant data from the MS files
    ##
    selfcal_library, selfcal_plan, gaincalibrator_dict = prepare_selfcal(vislist, spectral_average=spectral_average, 
            sort_targets_and_EBs=sort_targets_and_EBs, scale_fov=scale_fov, inf_EB_gaincal_combine=inf_EB_gaincal_combine, 
            inf_EB_gaintype=inf_EB_gaintype, apply_cal_mode_default=apply_cal_mode_default, do_amp_selfcal=do_amp_selfcal, 
            usermask=usermask, usermodel=usermodel,debug=debug)


    with open('selfcal_library.pickle', 'wb') as handle:
        pickle.dump(selfcal_library, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('selfcal_plan.pickle', 'wb') as handle:
        pickle.dump(selfcal_plan, handle, protocol=pickle.HIGHEST_PROTOCOL)


    ###################################################################################################
    ############################# Start Actual important stuff for selfcal ############################
    ###################################################################################################



    ##
    ## create initial images for each target to evaluate SNR and beam
    ## replicates what a preceding hif_makeimages would do
    ## Enables before/after comparison and thresholds to be calculated
    ## based on the achieved S/N in the real data
    ##
    for target in selfcal_library:
     sani_target=sanitize_string(target)
     for band in selfcal_library[target]:
       #make images using the appropriate tclean heuristics for each telescope
       # Because tclean doesn't deal in NF masks, the automask from the initial image is likely to contain a lot of noise unless
       # we can get an estimate of the NF modifier for the auto-masking thresholds. To do this, we need to create a very basic mask
       # with the dirty image. So we just use one iteration with a tiny gain so that nothing is really subtracted off.
       tclean_wrapper(selfcal_library[target][band],sani_target+'_'+band+'_dirty',
                      band,telescope=telescope,nsigma=4.0, scales=[0],
                      threshold='0.0Jy',niter=1, gain=0.00001,
                      savemodel='none',parallel=parallel,
                      field=target)

       dirty_SNR, dirty_RMS, dirty_NF_SNR, dirty_NF_RMS = get_image_stats(sani_target+'_'+band+'_dirty.image.tt0', sani_target+'_'+band+'_dirty.mask',
                '', selfcal_library[target][band], (telescope != 'ACA' or aca_use_nfmask), 'dirty', 'dirty')

       mosaic_dirty_SNR, mosaic_dirty_RMS, mosaic_dirty_NF_SNR, mosaic_dirty_NF_RMS = {}, {}, {}, {}
       for fid in selfcal_library[target][band]['sub-fields']:
           if selfcal_library[target][band]['obstype'] == 'mosaic':
               imagename = sani_target+'_field_'+str(fid)+'_'+band+'_dirty.image.tt0'
           else:
               imagename = sani_target+'_'+band+'_dirty.image.tt0'

           mosaic_dirty_SNR[fid], mosaic_dirty_RMS[fid], mosaic_dirty_NF_SNR[fid], mosaic_dirty_NF_RMS[fid] = get_image_stats(imagename, 
                   imagename.replace('image.tt0','mask'), '', selfcal_library[target][band][fid], (telescope != 'ACA' or aca_use_nfmask), 'dirty', 'dirty',
                   mosaic_sub_field=selfcal_library[target][band]["obstype"]=="mosaic")

       tclean_wrapper(selfcal_library[target][band],sani_target+'_'+band+'_initial',
                      band,telescope=telescope,nsigma=4.0, scales=[0],
                      threshold='theoretical_with_drmod',
                      savemodel='none',parallel=parallel,
                      field=target,nfrms_multiplier=dirty_NF_RMS/dirty_RMS,store_threshold='orig')

       initial_SNR, initial_RMS, initial_NF_SNR, initial_NF_RMS = get_image_stats(sani_target+'_'+band+'_initial.image.tt0', 
               sani_target+'_'+band+'_initial.mask', '', selfcal_library[target][band], (telescope != 'ACA' or aca_use_nfmask), 'orig', 'orig')

       mosaic_initial_SNR, mosaic_initial_RMS, mosaic_initial_NF_SNR, mosaic_initial_NF_RMS = {}, {}, {}, {}
       for fid in selfcal_library[target][band]['sub-fields']:
           if selfcal_library[target][band]['obstype'] == 'mosaic':
               imagename = sani_target+'_field_'+str(fid)+'_'+band+'_initial.image.tt0'
           else:
               imagename = sani_target+'_'+band+'_initial.image.tt0'

           mosaic_initial_SNR[fid], mosaic_initial_RMS[fid], mosaic_initial_NF_SNR[fid],mosaic_initial_NF_RMS[fid] = get_image_stats(imagename, 
                   imagename.replace('image.tt0','mask'), '', selfcal_library[target][band][fid], (telescope != 'ACA' or aca_use_nfmask), 'orig', 'orig',
                   mosaic_sub_field=selfcal_library[target][band]["obstype"]=="mosaic")

       if "VLA" in telescope and "clean_threshold_orig" not in selfcal_library[target][band]:
                 selfcal_library[target][band]['clean_threshold_orig']=4.0*initial_RMS

       if selfcal_library[target][band]['nterms'] == 1:  # updated nterms if needed based on S/N and fracbw
          selfcal_library[target][band]['nterms']=check_image_nterms(selfcal_library[target][band]['fracbw'],selfcal_library[target][band]['SNR_orig'])

       selfcal_library[target][band]['RMS_curr']=initial_RMS
       selfcal_library[target][band]['RMS_NF_curr']=initial_NF_RMS if initial_NF_RMS > 0 else initial_RMS

       for fid in selfcal_library[target][band]['sub-fields']:
           if selfcal_library[target][band][fid]['SNR_orig'] > 500.0:
              selfcal_library[target][band][fid]['nterms']=2

           selfcal_library[target][band][fid]['RMS_curr']=mosaic_initial_RMS[fid]
           selfcal_library[target][band][fid]['RMS_NF_curr']=mosaic_initial_NF_RMS[fid] if mosaic_initial_NF_RMS[fid] > 0 else mosaic_initial_RMS[fid]

     #update selfcal library after each
     with open('selfcal_library.pickle', 'wb') as handle:
        pickle.dump(selfcal_library, handle, protocol=pickle.HIGHEST_PROTOCOL)

    import json

    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return json.JSONEncoder.default(self, obj)

    if debug:
        print(json.dumps(selfcal_library, indent=4, cls=NpEncoder))

    ####MAKE DIRTY PER SPW IMAGES TO PROPERLY ASSESS DR MODIFIERS
    ##
    ## Make a initial image per spw images to assess overall improvement
    ##   

    if check_all_spws:
       for target in selfcal_library:
          sani_target=sanitize_string(target)
          for band in selfcal_library[target].keys():
             #potential place where diff spws for different VLA EBs could cause problems
             for spw in selfcal_library[target][band]['spw_map']:
                keylist=selfcal_library[target][band]['per_spw_stats'].keys()
                if spw not in keylist:
                   selfcal_library[target][band]['per_spw_stats'][spw]={}
                tclean_wrapper(selfcal_library[target][band],sani_target+'_'+band+'_'+str(spw)+'_dirty',
                      band,telescope=telescope,nsigma=4.0, scales=[0],
                      threshold='0.0Jy',niter=1,gain=0.00001,
                      savemodel='none',parallel=parallel,
                      field=target,spw=spw)

                dirty_SNR, dirty_RMS, dirty_per_spw_NF_SNR, dirty_per_spw_NF_RMS = get_image_stats(sani_target+'_'+band+'_'+str(spw)+
                        '_dirty.image.tt0', sani_target+'_'+band+'_'+str(spw)+'_dirty.mask','', selfcal_library[target][band], 
                        (telescope != 'ACA' or aca_use_nfmask), 'dirty', 'dirty', spw=spw)

                tclean_wrapper(selfcal_library[target][band],sani_target+'_'+band+'_'+str(spw)+'_initial',\
                           band,telescope=telescope,nsigma=4.0, threshold='theoretical_with_drmod',scales=[0],\
                           savemodel='none',parallel=parallel,\
                           field=target,datacolumn='corrected',\
                           spw=spw,nfrms_multiplier=dirty_per_spw_NF_RMS/dirty_RMS)

                per_spw_SNR, per_spw_RMS, initial_per_spw_NF_SNR, initial_per_spw_NF_RMS = get_image_stats(sani_target+'_'+band+'_'+str(spw)+
                        '_initial.image.tt0', sani_target+'_'+band+'_'+str(spw)+'_initial.mask', '', selfcal_library[target][band], 
                        (telescope != 'ACA' or aca_use_nfmask), 'orig', 'orig', spw=spw)





    ##
    ## estimate per scan/EB S/N using time on source and median scan times
    ##

    get_SNR_self(selfcal_library,selfcal_plan,n_ants,inf_EB_gaincal_combine,inf_EB_gaintype)

    ##
    ## Set clean selfcal thresholds
    ### Open question about determining the starting and progression of clean threshold for
    ### each iteration
    ### Peak S/N > 100; SNR/15 for first, successivly reduce to 3.0 sigma through each iteration?
    ### Peak S/N < 100; SNR/10.0 
    ##
    ## Switch to a sensitivity for low frequency that is based on the residuals of the initial image for the
    # first couple rounds and then switch to straight nsigma? Determine based on fraction of pixels that the # initial mask covers to judge very extended sources?

    set_clean_thresholds(selfcal_library, selfcal_plan, dividing_factor=dividing_factor, rel_thresh_scaling=rel_thresh_scaling, telescope=telescope)

    plan_selfcal_per_solint(selfcal_library, selfcal_plan,optimize_spw_combine=optimize_spw_combine)
    ##
    ## Save self-cal library
    ##

    with open('selfcal_library.pickle', 'wb') as handle:
        pickle.dump(selfcal_library, handle, protocol=pickle.HIGHEST_PROTOCOL)



    ##
    ## Begin Self-cal loops
    ##
    for target in selfcal_library:
     for band in selfcal_library[target].keys():
       run_selfcal(selfcal_library[target][band], selfcal_plan[target][band], target, band, telescope, n_ants, \
               gaincal_minsnr=gaincal_minsnr, gaincal_unflag_minsnr=gaincal_unflag_minsnr, minsnr_to_proceed=minsnr_to_proceed, delta_beam_thresh=delta_beam_thresh, do_amp_selfcal=do_amp_selfcal, \
               inf_EB_gaincal_combine=inf_EB_gaincal_combine, inf_EB_gaintype=inf_EB_gaintype, unflag_only_lbants=unflag_only_lbants, \
               unflag_only_lbants_onlyap=unflag_only_lbants_onlyap, calonly_max_flagged=calonly_max_flagged, \
               second_iter_solmode=second_iter_solmode, unflag_fb_to_prev_solint=unflag_fb_to_prev_solint, rerank_refants=rerank_refants, \
               gaincalibrator_dict=gaincalibrator_dict, allow_gain_interpolation=allow_gain_interpolation, guess_scan_combine=guess_scan_combine, \
               aca_use_nfmask=aca_use_nfmask,debug=debug,spectral_solution_fraction=spectral_solution_fraction)

    if debug:
        print(json.dumps(selfcal_library, indent=4, cls=NpEncoder))


    if allow_cocal:
        ##
        ## Save the flags following the main iteration of self-calibration since we will need to revert to the beginning for the fallback mode.
        ##
        # PS: I don't need this anymore?
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
        for band in bands:
            # Initialize the lists for this band.
            inf_EB_fields[band] = []
            inf_fields[band] = []
            fallback_fields[band] = []
        
            # Loop through and identify which sources belong where.
            for target in selfcal_library.keys():
                if selfcal_library[target][band]['SC_success'] and 'fb' not in selfcal_library[target][band]['final_solint']:
                    inf_EB_fields[band].append(target)
                    if selfcal_library[target][band]['final_solint'] != 'inf_EB':
                        inf_fields[band].append(target)
                    elif 'inf' in selfcal_plan[target][band]['solints']:
                        fallback_fields[band].append(target)
                else:
                    fallback_fields[band].append(target)
        
                # Update the relevant lists if we are going to do a fallback mode.
                if len(fallback_fields[band]) > 0:
                    selfcal_plan[target][band]['solints'] += ["inf_EB_fb","inf_fb1","inf_fb2","inf_fb3"]
                    selfcal_plan[target][band]['solmode'] += ["p","p","p","p"]
                    selfcal_plan[target][band]['gaincal_combine'] += [selfcal_plan[target][band]['gaincal_combine'][0], selfcal_plan[target][band]['gaincal_combine'][1], selfcal_plan[target][band]['gaincal_combine'][1], selfcal_plan[target][band]['gaincal_combine'][1]]
                    applycal_mode[band][target] += [applycal_mode[band][target][0], applycal_mode[band][target][1], applycal_mode[band][target][1], applycal_mode[band][target][1]]
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
         for band in solint_snr[target].keys():
           # If the target had a successful inf_EB solution, no need to reset.
           if target in inf_EB_fields[band]:
               continue
        
           for vis in selfcal_library[target][band]['vislist']:
            selfcal_plan[target][band][vis]['inf_EB_gaincal_combine']=inf_EB_gaincal_combine #'scan'
            if selfcal_library[target][band]['obstype']=='mosaic':
               selfcal_plan[target][band][vis]['inf_EB_gaincal_combine']+=',field'   
            selfcal_plan[target][band][vis]['inf_EB_gaintype']=inf_EB_gaintype #G
            selfcal_plan[target][band][vis]['inf_EB_fallback_mode']='' #'scan'
        
        
        calculate_inf_EB_fb_anyways = True
        preapply_targets_own_inf_EB = False
        
        ## The below sets the calibrations back to what they were prior to starting the fallback mode. It should not be needed
        ## for the final version of the codue, but is used for testing.
        
        
        for target in selfcal_library:
         sani_target=sanitize_string(target)
         for band in selfcal_library[target].keys():
           if target not in fallback_fields[band]:
               continue
           if 'gaintable_final' in selfcal_library[target][band][vislist[0]]:
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
        ## END
                    
        
        ##
        ## Begin fallback self-cal loops
        ##
        for target in selfcal_library:
         for band in selfcal_library[target].keys():
           if target not in fallback_fields[band]:
               continue
        
           run_selfcal(selfcal_library[target][band], selfcal_plan[target][band], target, band, telescope, n_ants, \
                   gaincal_minsnr=gaincal_minsnr, gaincal_unflag_minsnr=gaincal_unflag_minsnr, minsnr_to_proceed=minsnr_to_proceed, delta_beam_thresh=delta_beam_thresh, do_amp_selfcal=do_amp_selfcal, \
                   inf_EB_gaincal_combine=inf_EB_gaincal_combine, inf_EB_gaintype=inf_EB_gaintype, unflag_only_lbants=unflag_only_lbants, \
                   unflag_only_lbants_onlyap=unflag_only_lbants_onlyap, calonly_max_flagged=calonly_max_flagged, \
                   second_iter_solmode=second_iter_solmode, unflag_fb_to_prev_solint=unflag_fb_to_prev_solint, rerank_refants=rerank_refants, \
                   mode="cocal", calibrators=calibrators, calculate_inf_EB_fb_anyways=calculate_inf_EB_fb_anyways, \
                   preapply_targets_own_inf_EB=preapply_targets_own_inf_EB, gaincalibrator_dict=gaincalibrator_dict, allow_gain_interpolation=True)
        
        if debug:
            print(json.dumps(selfcal_library, indent=4, cls=NpEncoder))

    ##
    ## If we want to try amplitude selfcal, should we do it as a function out of the main loop or a separate loop?
    ## Mechanics are likely to be a bit more simple since I expect we'd only try a single solint=inf solution
    ##

    ##
    ## Make a final image per target to assess overall improvement
    ##
    for target in selfcal_library:
     sani_target=sanitize_string(target)
     for band in selfcal_library[target].keys():
       nfsnr_modifier = selfcal_library[target][band]['RMS_NF_curr'] / selfcal_library[target][band]['RMS_curr']
       clean_threshold = min(selfcal_library[target][band]['clean_threshold_orig'], selfcal_library[target][band]['RMS_NF_curr']*3.0)
       if selfcal_library[target][band]['clean_threshold_orig'] < selfcal_library[target][band]['RMS_NF_curr']*3.0:
           print("WARNING: The clean threshold used for the initial image was less than 3*RMS_NF_curr, using that for the final image threshold instead.")
       tclean_wrapper(selfcal_library[target][band],sani_target+'_'+band+'_final',\
                   band,telescope=telescope,nsigma=3.0, threshold=str(clean_threshold)+'Jy',scales=[0],\
                   savemodel='none',parallel=parallel,
                   field=target,datacolumn='corrected',\
                   nfrms_multiplier=nfsnr_modifier)

       final_SNR, final_RMS, final_NF_SNR, final_NF_RMS = get_image_stats(sani_target+'_'+band+'_final.image.tt0', sani_target+'_'+band+'_final.mask',
               '', selfcal_library[target][band], (telescope !='ACA' or aca_use_nfmask), 'final', 'final')

       # Calculate final image stats.
       mosaic_final_SNR, mosaic_final_RMS, mosaic_final_NF_SNR, mosaic_final_NF_RMS = {}, {}, {}, {}
       for fid in selfcal_library[target][band]['sub-fields']:
           if selfcal_library[target][band]['obstype'] == 'mosaic':
               imagename = sani_target+'_field_'+str(fid)+'_'+band+'_final.image.tt0'
           else:
               imagename = sani_target+'_'+band+'_final.image.tt0'

           mosaic_final_SNR[fid], mosaic_final_RMS[fid], mosaic_final_NF_SNR[fid],mosaic_final_NF_RMS[fid] = get_image_stats(imagename, 
                   imagename.replace('image.tt0','mask'), '', selfcal_library[target][band][fid], (telescope !='ACA' or aca_use_nfmask), 'final', 'final',
                   mosaic_sub_field=selfcal_library[target][band]["obstype"]=="mosaic")

       #recalc inital stats using final mask
       orig_final_SNR, orig_final_RMS, orig_final_NF_SNR, orig_final_NF_RMS = get_image_stats(sani_target+'_'+band+'_initial.image.tt0', 
               sani_target+'_'+band+'_final.mask', '', selfcal_library[target][band], (telescope !='ACA' or aca_use_nfmask), 'orig', 'orig')

       mosaic_final_SNR, mosaic_final_RMS, mosaic_final_NF_SNR, mosaic_final_NF_RMS = {}, {}, {}, {}
       for fid in selfcal_library[target][band]['sub-fields']:
           if selfcal_library[target][band]['obstype'] == 'mosaic':
               imagename = sani_target+'_field_'+str(fid)+'_'+band
           else:
               imagename = sani_target+'_'+band

           mosaic_final_SNR[fid], mosaic_final_RMS[fid], mosaic_final_NF_SNR[fid],mosaic_final_NF_RMS[fid] = get_image_stats(imagename+'_initial.image.tt0',
                   imagename+'_final.mask', '', selfcal_library[target][band][fid], (telescope !='ACA' or aca_use_nfmask), 'orig', 'orig',
                   mosaic_sub_field=selfcal_library[target][band]["obstype"]=="mosaic")




    if debug:
        print(json.dumps(selfcal_library, indent=4, cls=NpEncoder))

    ##
    ## Make a final image per spw images to assess overall improvement
    ##
    if check_all_spws:
       for target in selfcal_library:
          sani_target=sanitize_string(target)
          for band in selfcal_library[target].keys():
             selfcal_library[target][band]['vislist']=selfcal_library[target][band]['vislist'].copy()

             print('Generating final per-SPW images for '+target+' in '+band)
             for spw in selfcal_library[target][band]['spw_map']:
       ## omit DR modifiers here since we should have increased DR significantly
                if not os.path.exists(sani_target+'_'+band+'_'+str(spw)+'_final.image.tt0'):
                   nfsnr_modifier = selfcal_library[target][band]['RMS_NF_curr'] / selfcal_library[target][band]['RMS_curr']

                   tclean_wrapper(selfcal_library[target][band],sani_target+'_'+band+'_'+str(spw)+'_final',\
                              band,telescope=telescope,nsigma=4.0, threshold='theoretical',scales=[0],\
                              savemodel='none',parallel=parallel,\
                              field=target,datacolumn='corrected',\
                              spw=spw,nfrms_multiplier=nfsnr_modifier)

                final_per_spw_SNR, final_per_spw_RMS, final_per_spw_NF_SNR, final_per_spw_NF_RMS = get_image_stats(
                        sani_target+'_'+band+'_'+str(spw)+'_final.image.tt0', sani_target+'_'+band+'_'+str(spw)+'_final.mask',
                        '', selfcal_library[target][band], (telescope !='ACA' or aca_use_nfmask), 'final', 'final', spw=spw)

                #reccalc initial stats with final mask
                orig_final_per_spw_SNR, orig_final_per_spw_RMS, orig_final_per_spw_NF_SNR, orig_final_per_spw_NF_RMS = get_image_stats(
                        sani_target+'_'+band+'_'+str(spw)+'_initial.image.tt0', sani_target+'_'+band+'_'+str(spw)+'_final.mask',
                        '', selfcal_library[target][band], (telescope !='ACA' or aca_use_nfmask), 'orig', 'orig', spw=spw)











    ##
    ## Print final results
    ##
    for target in selfcal_library:
     for band in selfcal_library[target].keys():
       print(target+' '+band+' Summary')
       print('At least 1 successful selfcal iteration?: ', selfcal_library[target][band]['SC_success'])
       print('Final solint: ',selfcal_library[target][band]['final_solint'])
       print('Original SNR: ',selfcal_library[target][band]['SNR_orig'])
       print('Final SNR: ',selfcal_library[target][band]['SNR_final'])
       print('Original RMS: ',selfcal_library[target][band]['RMS_orig'])
       print('Final RMS: ',selfcal_library[target][band]['RMS_final'])
       #   for vis in vislist:
       #      print('Final gaintables: '+selfcal_library[target][band][vis]['gaintable'])
       #      print('Final spwmap: ',selfcal_library[target][band][vis]['spwmap'])
       #else:
       #   print('Selfcal failed on '+target+'. No solutions applied.')

       for fid in selfcal_library[target][band]['sub-fields']:
           print(target+' '+band+' field '+str(fid)+' Summary')
           print('At least 1 successful selfcal iteration?: ', selfcal_library[target][band][fid]['SC_success'])
           print('Final solint: ',selfcal_library[target][band][fid]['final_solint'])
           print('Original SNR: ',selfcal_library[target][band][fid]['SNR_orig'])
           print('Final SNR: ',selfcal_library[target][band][fid]['SNR_final'])
           print('Original RMS: ',selfcal_library[target][band][fid]['RMS_orig'])
           print('Final RMS: ',selfcal_library[target][band][fid]['RMS_final'])


    # Either apply the calibrations to the original MS files, or write a file with the relevant
    # commands to do so.

    applycal_to_orig_MSes(selfcal_library, write_only=(not apply_to_target_ms))

    # Do continuum subtraction of the original MS files.

    uvcontsub_orig_MSes(selfcal_library, write_only=(not uvcontsub_target_ms))

    #
    # Perform a check on the per-spw images to ensure they didn't lose quality in self-calibration
    #
    if check_all_spws:
       for target in selfcal_library:
          sani_target=sanitize_string(target)
          for band in selfcal_library[target].keys():
             vislist=selfcal_library[target][band]['vislist'].copy()

             for spw in selfcal_library[target][band]['spw_map']:
                delta_beamarea=compare_beams(sani_target+'_'+band+'_'+str(spw)+'_initial.image.tt0',\
                                             sani_target+'_'+band+'_'+str(spw)+'_final.image.tt0')
                delta_SNR=selfcal_library[target][band]['per_spw_stats'][spw]['SNR_final']-\
                          selfcal_library[target][band]['per_spw_stats'][spw]['SNR_orig']
                delta_RMS=selfcal_library[target][band]['per_spw_stats'][spw]['RMS_final']-\
                          selfcal_library[target][band]['per_spw_stats'][spw]['RMS_orig']
                selfcal_library[target][band]['per_spw_stats'][spw]['delta_SNR']=delta_SNR
                selfcal_library[target][band]['per_spw_stats'][spw]['delta_RMS']=delta_RMS
                selfcal_library[target][band]['per_spw_stats'][spw]['delta_beamarea']=delta_beamarea
                print(sani_target+'_'+band+'_'+str(spw),\
                      'Pre SNR: {:0.2f}, Post SNR: {:0.2f} Pre RMS: {:0.3f}, Post RMS: {:0.3f}'.format(selfcal_library[target][band]['per_spw_stats'][spw]['SNR_orig'],\
                       selfcal_library[target][band]['per_spw_stats'][spw]['SNR_final'],selfcal_library[target][band]['per_spw_stats'][spw]['RMS_orig']*1000.0,selfcal_library[target][band]['per_spw_stats'][spw]['RMS_final']*1000.0))
                if delta_SNR < 0.0:
                   print('WARNING SPW '+str(spw)+' HAS LOWER SNR POST SELFCAL')
                if delta_RMS > 0.0:
                   print('WARNING SPW '+str(spw)+' HAS HIGHER RMS POST SELFCAL')
                if delta_beamarea > 0.05:
                   print('WARNING SPW '+str(spw)+' HAS A >0.05 CHANGE IN BEAM AREA POST SELFCAL')


    ##
    ## Save final library results
    ##

    with open('selfcal_library.pickle', 'wb') as handle:
        pickle.dump(selfcal_library, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('selfcal_plan.pickle', 'wb') as handle:
        pickle.dump(selfcal_plan, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if weblog:
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


