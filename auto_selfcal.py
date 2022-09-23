# future improvements
# heuristics for switching between calonly and calflag
# heuristics to switch from combine=spw to combine=''
# switch heirarchy of selfcal_library such that solint is at a higher level than vis. makes storage of some parameters awkward since they live
#    in the per vis level instead of per solint

import glob
import os
import pickle
import sys

import numpy as np
from casampi.MPIEnvironment import MPIEnvironment
from casatasks import *

parallel = MPIEnvironment.is_mpi_enabled

try:
    # run as a Pipeline extern module
    from .selfcal_helpers import *
except ImportError as error:
    # run as a script when selfcal_helpers is in the working directory
    from selfcal_helpers import *

LOG = get_selfcal_logger(__name__)

###################################################################################################
######################## All code until line ~170 is just jumping through hoops ###################
######################## to get at metadata pipeline should have in the context ###################
#################### And it will do flagging of lines and/or spectral averaging ###################
######################## Some of this code is not elegant nor efficient ###########################
###################################################################################################


def selfcal_workflow():

    vislist, vis, all_targets, do_amp_selfcal, inf_EB_gaincal_combine, inf_EB_gaintype, gaincal_minsnr, minsnr_to_proceed, delta_beam_thresh, n_ants, telescope, rel_thresh_scaling, dividing_factor, check_all_spws, apply_to_target_ms, bands, band_properties, spwsarray, vislist_orig, spwstring_orig, cellsize, imsize, nterms, applycal_interp, selfcal_library, solints, gaincal_combine, solmode, applycal_mode, integration_time = prep_selfcal()

    ##
    # create initial images for each target to evaluate SNR and beam
    # replicates what a preceding hif_makeimages would do
    # Enables before/after comparison and thresholds to be calculated
    # based on the achieved S/N in the real data
    ##
    for target in all_targets:
        sani_target = sanitize_string(target)
        for band in selfcal_library[target].keys():
            # make images using the appropriate tclean heuristics for each telescope
            if not os.path.exists(sani_target+'_'+band+'_dirty.image.tt0'):
                tclean_wrapper(
                    vislist, sani_target + '_' + band + '_dirty', band_properties, band, telescope=telescope, nsigma=4.0,
                    scales=[0],
                    threshold='0.0Jy', niter=0, savemodel='none', parallel=parallel, cellsize=cellsize[band],
                    imsize=imsize[band],
                    nterms=nterms[band],
                    field=target, spw=selfcal_library[target][band]['spws_per_vis'],
                    uvrange=selfcal_library[target][band]['uvrange'],
                    obstype=selfcal_library[target][band]['obstype'])
            dirty_SNR, dirty_RMS = estimate_SNR(sani_target+'_'+band+'_dirty.image.tt0')
            dr_mod = 1.0
            if telescope == 'ALMA' or telescope == 'ACA':
                sensitivity = get_sensitivity(
                    vislist, selfcal_library[target][band],
                    selfcal_library[target][band][vis]['spws'],
                    spw=selfcal_library[target][band][vis]['spwsarray'],
                    imsize=imsize[band],
                    cellsize=cellsize[band])
                dr_mod = get_dr_correction(telescope, dirty_SNR*dirty_RMS, sensitivity, vislist)
                sensitivity_nomod = sensitivity.copy()
                LOG.info(f'DR modifier: {dr_mod}')
            if not os.path.exists(sani_target+'_'+band+'_initial.image.tt0'):
                if telescope == 'ALMA' or telescope == 'ACA':
                    sensitivity = sensitivity*dr_mod   # apply DR modifier
                    if band == 'Band_9' or band == 'Band_10':   # adjust for DSB noise increase
                        sensitivity = sensitivity  # *4.0  might be unnecessary with DR mods
                else:
                    sensitivity = 0.0
                tclean_wrapper(
                    vislist, sani_target + '_' + band + '_initial', band_properties, band, telescope=telescope, nsigma=4.0,
                    scales=[0],
                    threshold=str(sensitivity * 4.0) + 'Jy', savemodel='none', parallel=parallel, cellsize=cellsize
                    [band],
                    imsize=imsize[band],
                    nterms=nterms[band],
                    field=target, spw=selfcal_library[target][band]['spws_per_vis'],
                    uvrange=selfcal_library[target][band]['uvrange'],
                    obstype=selfcal_library[target][band]['obstype'])
            initial_SNR, initial_RMS = estimate_SNR(sani_target+'_'+band+'_initial.image.tt0')
            if telescope != 'ACA':
                initial_NF_SNR, initial_NF_RMS = estimate_near_field_SNR(sani_target+'_'+band+'_initial.image.tt0')
            else:
                initial_NF_SNR, initial_NF_RMS = initial_SNR, initial_RMS
            header = imhead(imagename=sani_target+'_'+band+'_initial.image.tt0')
            if telescope == 'ALMA' or telescope == 'ACA':
                selfcal_library[target][band]['theoretical_sensitivity'] = sensitivity_nomod
            if 'VLA' in telescope:
                selfcal_library[target][band]['theoretical_sensitivity'] = -99.0
            selfcal_library[target][band]['SNR_orig'] = initial_SNR
            if selfcal_library[target][band]['SNR_orig'] > 500.0:
                selfcal_library[target][band]['nterms'] = 2
            selfcal_library[target][band]['RMS_orig'] = initial_RMS
            selfcal_library[target][band]['SNR_NF_orig'] = initial_NF_SNR
            selfcal_library[target][band]['RMS_NF_orig'] = initial_NF_RMS
            selfcal_library[target][band]['RMS_curr'] = initial_RMS
            selfcal_library[target][band]['SNR_dirty'] = dirty_SNR
            selfcal_library[target][band]['RMS_dirty'] = dirty_RMS
            selfcal_library[target][band]['Beam_major_orig'] = header['restoringbeam']['major']['value']
            selfcal_library[target][band]['Beam_minor_orig'] = header['restoringbeam']['minor']['value']
            selfcal_library[target][band]['Beam_PA_orig'] = header['restoringbeam']['positionangle']['value']
            goodMask = checkmask(imagename=sani_target+'_'+band+'_initial.image.tt0')
            if goodMask:
                selfcal_library[target][band]['intflux_orig'], selfcal_library[target][band]['e_intflux_orig'] = get_intflux(
                    sani_target+'_'+band+'_initial.image.tt0', initial_RMS)
            else:
                selfcal_library[target][band]['intflux_orig'], selfcal_library[target][band]['e_intflux_orig'] = -99.0, -99.0

    # MAKE DIRTY PER SPW IMAGES TO PROPERLY ASSESS DR MODIFIERS
    ##
    # Make a initial image per spw images to assess overall improvement
    ##

    for target in all_targets:
        for band in selfcal_library[target].keys():
            selfcal_library[target][band]['per_spw_stats'] = {}
            vislist = selfcal_library[target][band]['vislist'].copy()
            spwlist = selfcal_library[target][band][vislist[0]]['spws'].split(',')
            spw_bandwidths, spw_effective_bandwidths = get_spw_bandwidth(
                vis, selfcal_library[target][band][vis]['spwsarray'], target)
            selfcal_library[target][band]['total_bandwidth'] = 0.0
            selfcal_library[target][band]['total_effective_bandwidth'] = 0.0
            if len(spw_effective_bandwidths.keys()) != len(spw_bandwidths.keys()):
                LOG.info('cont.dat does not contain all spws; falling back to total bandwidth')
                for spw in spw_bandwidths.keys():
                    if spw not in spw_effective_bandwidths.keys():
                        spw_effective_bandwidths[spw] = spw_bandwidths[spw]
            for spw in spwlist:
                keylist = selfcal_library[target][band]['per_spw_stats'].keys()
                if spw not in keylist:
                    selfcal_library[target][band]['per_spw_stats'][spw] = {}
                selfcal_library[target][band]['per_spw_stats'][spw]['effective_bandwidth'] = spw_effective_bandwidths[spw]
                selfcal_library[target][band]['per_spw_stats'][spw]['bandwidth'] = spw_bandwidths[spw]
                selfcal_library[target][band]['total_bandwidth'] += spw_bandwidths[spw]
                selfcal_library[target][band]['total_effective_bandwidth'] += spw_effective_bandwidths[spw]

    if check_all_spws:
        for target in all_targets:
            sani_target = sanitize_string(target)
            for band in selfcal_library[target].keys():
                vislist = selfcal_library[target][band]['vislist'].copy()
                spwlist = selfcal_library[target][band][vislist[0]]['spws'].split(',')
                for spw in spwlist:
                    keylist = selfcal_library[target][band]['per_spw_stats'].keys()
                    if spw not in keylist:
                        selfcal_library[target][band]['per_spw_stats'][spw] = {}
                    if not os.path.exists(sani_target+'_'+band+'_'+spw+'_dirty.image.tt0'):
                        spws_per_vis = [spw]*len(vislist)
                        tclean_wrapper(
                            vislist, sani_target + '_' + band + '_' + spw + '_dirty', band_properties, band,
                            telescope=telescope, nsigma=4.0, scales=[0],
                            threshold='0.0Jy', niter=0, savemodel='none', parallel=parallel, cellsize=cellsize[band],
                            imsize=imsize[band],
                            nterms=1, field=target, spw=spws_per_vis, uvrange=selfcal_library[target][band]
                            ['uvrange'],
                            obstype=selfcal_library[target][band]['obstype'])
                    dirty_SNR, dirty_RMS = estimate_SNR(sani_target+'_'+band+'_'+spw+'_dirty.image.tt0')
                    if not os.path.exists(sani_target+'_'+band+'_'+spw+'_initial.image.tt0'):
                        if telescope == 'ALMA' or telescope == 'ACA':
                            sensitivity = get_sensitivity(
                                vislist, selfcal_library[target][band],
                                spw, spw=np.array([int(spw)]),
                                imsize=imsize[band],
                                cellsize=cellsize[band])
                            dr_mod = 1.0
                            dr_mod = get_dr_correction(telescope, dirty_SNR*dirty_RMS, sensitivity, vislist)
                            LOG.info(f'DR modifier: {dr_mod}  SPW: {spw}')
                            sensitivity = sensitivity*dr_mod
                            if ((band == 'Band_9') or (band == 'Band_10')) and dr_mod != 1.0:   # adjust for DSB noise increase
                                sensitivity = sensitivity*4.0
                        else:
                            sensitivity = 0.0
                        spws_per_vis = [spw]*len(vislist)  # assumes all spw ids are identical in each MS file

                        tclean_wrapper(
                            vislist, sani_target + '_' + band + '_' + spw + '_initial', band_properties, band,
                            telescope=telescope, nsigma=4.0, threshold=str(sensitivity * 4.0) + 'Jy', scales=[0],
                            savemodel='none', parallel=parallel, cellsize=cellsize[band],
                            imsize=imsize[band],
                            nterms=1, field=target, datacolumn='corrected', spw=spws_per_vis,
                            uvrange=selfcal_library[target][band]['uvrange'],
                            obstype=selfcal_library[target][band]['obstype'])

                    per_spw_SNR, per_spw_RMS = estimate_SNR(sani_target+'_'+band+'_'+spw+'_initial.image.tt0')
                    if telescope != 'ACA':
                        initial_per_spw_NF_SNR, initial_per_spw_NF_RMS = estimate_near_field_SNR(
                            sani_target + '_' + band + '_' + spw + '_initial.image.tt0')
                    else:
                        initial_per_spw_NF_SNR, initial_per_spw_NF_RMS = per_spw_SNR, per_spw_RMS
                    selfcal_library[target][band]['per_spw_stats'][spw]['SNR_orig'] = per_spw_SNR
                    selfcal_library[target][band]['per_spw_stats'][spw]['RMS_orig'] = per_spw_RMS
                    selfcal_library[target][band]['per_spw_stats'][spw]['SNR_NF_orig'] = initial_per_spw_NF_SNR
                    selfcal_library[target][band]['per_spw_stats'][spw]['RMS_NF_orig'] = initial_per_spw_NF_RMS
                    goodMask = checkmask(sani_target+'_'+band+'_'+spw+'_initial.image.tt0')
                    if goodMask:
                        selfcal_library[target][band]['per_spw_stats'][spw]['intflux_orig'], selfcal_library[target][
                            band]['per_spw_stats'][spw]['e_intflux_orig'] = get_intflux(
                            sani_target + '_' + band + '_' + spw + '_initial.image.tt0', per_spw_RMS)
                    else:
                        selfcal_library[target][band]['per_spw_stats'][spw]['intflux_orig'], selfcal_library[target][
                            band]['per_spw_stats'][spw]['e_intflux_orig'] = -99.0, -99.0

    ##
    # estimate per scan/EB S/N using time on source and median scan times
    ##
    inf_EB_gaincal_combine_dict = {}  # 'scan'
    inf_EB_gaintype_dict = {}  # 'G'
    inf_EB_fallback_mode_dict = {}  # 'scan'

    solint_snr, solint_snr_per_spw = get_SNR_self(
        all_targets, bands, vislist, selfcal_library, n_ants, solints, integration_time, inf_EB_gaincal_combine,
        inf_EB_gaintype)
    minsolint_spw = 100.0
    for target in all_targets:
        inf_EB_gaincal_combine_dict[target] = {}  # 'scan'
        inf_EB_fallback_mode_dict[target] = {}  # 'scan'
        inf_EB_gaintype_dict[target] = {}  # 'G'
        for band in solint_snr[target].keys():
            inf_EB_gaincal_combine_dict[target][band] = {}
            inf_EB_gaintype_dict[target][band] = {}
            inf_EB_fallback_mode_dict[target][band] = {}
            for vis in vislist:
                inf_EB_gaincal_combine_dict[target][band][vis] = inf_EB_gaincal_combine  # 'scan'
                if selfcal_library[target][band]['obstype'] == 'mosaic':
                    inf_EB_gaincal_combine_dict[target][band][vis] += ',field'
                inf_EB_gaintype_dict[target][band][vis] = inf_EB_gaintype  # G
                inf_EB_fallback_mode_dict[target][band][vis] = ''  # 'scan'
                LOG.info('Estimated SNR per solint:')
                LOG.info(f'{target} {band}')
                for solint in solints[band]:
                    if solint == 'inf_EB':
                        LOG.info('{}: {:0.2f}'.format(solint, solint_snr[target][band][solint]))
                        """ 
                        for spw in solint_snr_per_spw[target][band][solint].keys():
                            LOG.info('{}: spw: {}: {:0.2f}, BW: {} GHz'.format(solint,spw,solint_snr_per_spw[target][band][solint][spw],selfcal_library[target][band]['per_spw_stats'][str(spw)]['effective_bandwidth']))
                            if solint_snr_per_spw[target][band][solint][spw] < minsolint_spw:
                            minsolint_spw=solint_snr_per_spw[target][band][solint][spw]
                        if minsolint_spw < 3.5 and minsolint_spw > 2.5 and inf_EB_override==False:  # if below 3.5 but above 2.5 switch to gaintype T, but leave combine=scan
                            LOG.info('Switching Gaintype to T for: '+target)
                            inf_EB_gaintype_dict[target][band]='T'
                        elif minsolint_spw < 2.5 and inf_EB_override==False:
                            LOG.info('Switching Gaincal combine to spw,scan for: '+target)
                            inf_EB_gaincal_combine_dict[target][band]='scan,spw' # if below 2.5 switch to combine=spw to avoid losing spws
                        """
                    else:
                        LOG.info('{}: {:0.2f}'.format(solint, solint_snr[target][band][solint]))

    ##
    # Set clean selfcal thresholds
    # Open question about determining the starting and progression of clean threshold for
    # each iteration
    # Peak S/N > 100; SNR/15 for first, successivly reduce to 3.0 sigma through each iteration?
    # Peak S/N < 100; SNR/10.0
    ##
    # Switch to a sensitivity for low frequency that is based on the residuals of the initial image for the
    # first couple rounds and then switch to straight nsigma? Determine based on fraction of pixels that the # initial mask covers to judge very extended sources?

    for target in all_targets:
        for band in selfcal_library[target].keys():
            if band_properties[selfcal_library[target][band]['vislist'][0]][band]['meanfreq'] < 8.0e9 and (
                    dividing_factor == -99.0):
                dividing_factor = 40.0
            elif (dividing_factor == -99.0):
                dividing_factor = 15.0
            nsigma_init = np.max([selfcal_library[target][band]['SNR_orig']/dividing_factor, 5.0]
                                 )  # restricts initial nsigma to be at least 5

            # count number of amplitude selfcal solints, repeat final clean depth of phase-only for amplitude selfcal
            n_ap_solints = sum(1 for solint in solints[band] if 'ap' in solint)
            if rel_thresh_scaling == 'loge':
                selfcal_library[target][band]['nsigma'] = np.append(
                    np.exp(np.linspace(np.log(nsigma_init),
                                       np.log(3.0),
                                       len(solints[band]) - n_ap_solints)),
                    np.array([np.exp(np.log(3.0))] * n_ap_solints))
            elif rel_thresh_scaling == 'linear':
                selfcal_library[target][band]['nsigma'] = np.append(
                    np.linspace(nsigma_init, 3.0, len(solints[band]) - n_ap_solints),
                    np.array([3.0] * n_ap_solints))
            else:  # implicitly making log10 the default
                selfcal_library[target][band]['nsigma'] = np.append(
                    10 ** np.linspace(np.log10(nsigma_init),
                                      np.log10(3.0),
                                      len(solints[band]) - n_ap_solints),
                    np.array([10 ** (np.log10(3.0))] * n_ap_solints))
            if n_ap_solints > 0:
                selfcal_library[target][band]['nsigma']
            if telescope == 'ALMA' or telescope == 'ACA':  # or ('VLA' in telescope)
                sensitivity = get_sensitivity(
                    vislist, selfcal_library[target][band],
                    selfcal_library[target][band][vis]['spws'],
                    spw=selfcal_library[target][band][vis]['spwsarray'],
                    imsize=imsize[band],
                    cellsize=cellsize[band])
                if band == 'Band_9' or band == 'Band_10':   # adjust for DSB noise increase
                    sensitivity = sensitivity*4.0
                if ('VLA' in telescope):
                    sensitivity = sensitivity*0.0  # empirical correction, VLA estimates for sensitivity have tended to be a factor of ~3 low
            else:
                sensitivity = 0.0
            selfcal_library[target][band]['thresholds'] = selfcal_library[target][band]['nsigma']*sensitivity

    ##
    # Save self-cal library
    ##
    with open('selfcal_library.pickle', 'wb') as handle:
        pickle.dump(selfcal_library, handle, protocol=pickle.HIGHEST_PROTOCOL)

    ##
    # Begin Self-cal loops
    ##
    iterjump = -1   # useful if we want to jump iterations
    for target in all_targets:
        sani_target = sanitize_string(target)
        for band in selfcal_library[target].keys():
            vislist = selfcal_library[target][band]['vislist'].copy()
            LOG.info('Starting selfcal procedure on: '+target+' '+band)
            for iteration in range(len(solints[band])):
                if (iterjump != -1) and (iteration < iterjump):  # allow jumping to amplitude selfcal and not need to use a while loop
                    continue
                elif iteration == iterjump:
                    iterjump = -1
                if solint_snr[target][band][solints[band][iteration]] < minsnr_to_proceed:
                    LOG.info(
                        '*********** estimated SNR for solint=' + solints[band][iteration] + ' too low, measured: ' +
                        str(solint_snr[target][band][solints[band][iteration]]) + ', Min SNR Required: ' +
                        str(minsnr_to_proceed) + ' **************')
                    # if a solution interval shorter than inf for phase-only SC has passed, attempt amplitude selfcal
                    if iteration > 1 and solmode[band][iteration] != 'ap' and do_amp_selfcal:
                        iterjump = solmode[band].index('ap')
                        LOG.info('****************Attempting amplitude selfcal*************')
                        continue

                    selfcal_library[target][band]['Stop_Reason'] = 'Estimated_SNR_too_low_for_solint '+solints[band][iteration]
                    break
                else:
                    solint = solints[band][iteration]
                    if iteration == 0:
                        LOG.info('Starting with solint: '+solint)
                    else:
                        LOG.info('Continuing with solint: '+solint)
                    os.system('rm -rf '+sani_target+'_'+band+'_'+solint+'_'+str(iteration)+'*')
                    ##
                    # make images using the appropriate tclean heuristics for each telescope
                    # set threshold based on RMS of initial image and lower if value becomes lower
                    # during selfcal by resetting 'RMS_curr' after the post-applycal evaluation
                    ##
                    tclean_wrapper(
                        vislist, sani_target + '_' + band + '_' + solint + '_' + str(iteration),
                        band_properties, band, telescope=telescope,
                        nsigma=selfcal_library[target][band]['nsigma'][iteration],
                        scales=[0],
                        threshold=str(
                            selfcal_library[target][band]['nsigma'][iteration] *
                            selfcal_library[target][band]['RMS_curr']) + 'Jy', savemodel='modelcolumn',
                        parallel=parallel, cellsize=cellsize[band],
                        imsize=imsize[band],
                        nterms=selfcal_library[target][band]['nterms'],
                        field=target, spw=selfcal_library[target][band]['spws_per_vis'],
                        uvrange=selfcal_library[target][band]['uvrange'],
                        obstype=selfcal_library[target][band]['obstype'])
                    LOG.info('Pre selfcal assessemnt: '+target)
                    SNR, RMS = estimate_SNR(sani_target+'_'+band+'_'+solint+'_'+str(iteration)+'.image.tt0')
                    if telescope != 'ACA':
                        SNR_NF, RMS_NF = estimate_near_field_SNR(
                            sani_target+'_'+band+'_'+solint+'_'+str(iteration)+'.image.tt0')
                    else:
                        SNR_NF, RMS_NF = SNR, RMS

                    header = imhead(imagename=sani_target+'_'+band+'_'+solint+'_'+str(iteration)+'.image.tt0')

                    if iteration == 0:
                        gaincal_preapply_gaintable = {}
                        gaincal_spwmap = {}
                        gaincal_interpolate = {}
                        applycal_gaintable = {}
                        applycal_spwmap = {}
                        fallback = {}
                        applycal_interpolate = {}
                    for vis in vislist:
                        ##
                        # Restore original flagging state each time before applying a new gaintable
                        ##
                        if os.path.exists(vis+".flagversions/flags.selfcal_starting_flags_"+sani_target):
                            flagmanager(
                                vis=vis, mode='restore', versionname='selfcal_starting_flags_' + sani_target,
                                comment='Flag states at start of reduction')
                        else:
                            flagmanager(vis=vis, mode='save', versionname='selfcal_starting_flags_'+sani_target)
                        applycal_gaintable[vis] = []
                        applycal_spwmap[vis] = []
                        applycal_interpolate[vis] = []
                        gaincal_spwmap[vis] = []
                        gaincal_interpolate[vis] = []
                        gaincal_preapply_gaintable[vis] = []
                        ##
                        # Solve gain solutions per MS, target, solint, and band
                        ##
                        os.system(
                            'rm -rf ' + sani_target + '_' + vis + '_' + band + '_' + solint + '_' + str(iteration) + '_' +
                            solmode[band][iteration] + '.g')
                        ##
                        # Set gaincal parameters depending on which iteration and whether to use combine=spw for inf_EB or not
                        # Defaults should assume combine='scan' and gaintpe='G' will fallback to combine='scan,spw' if too much flagging
                        # At some point remove the conditional for use_inf_EB_preapply, since there isn't a reason not to do it
                        ##

                        if solint == 'inf_EB':
                            gaincal_spwmap[vis] = []
                            gaincal_preapply_gaintable[vis] = []
                            gaincal_interpolate[vis] = []
                            gaincal_gaintype = inf_EB_gaintype_dict[target][band][vis]
                            gaincal_combine[band][iteration] = inf_EB_gaincal_combine_dict[target][band][vis]
                            if 'spw' in inf_EB_gaincal_combine_dict[target][band][vis]:
                                applycal_spwmap[vis] = [selfcal_library[target][band][vis]['spwmap']]
                                gaincal_spwmap[vis] = [selfcal_library[target][band][vis]['spwmap']]
                            else:
                                applycal_spwmap[vis] = []
                            applycal_interpolate[vis] = [applycal_interp[band]]
                            applycal_gaintable[vis] = [
                                sani_target + '_' + vis + '_' + band + '_' + solint + '_' + str(iteration) + '_' +
                                solmode[band][iteration] + '.g']
                        elif solmode[band][iteration] == 'p':
                            gaincal_spwmap[vis] = []
                            gaincal_preapply_gaintable[vis] = [sani_target+'_'+vis+'_'+band+'_inf_EB_0_p.g']
                            gaincal_interpolate[vis] = [applycal_interp[band]]
                            gaincal_gaintype = 'T'
                            if 'spw' in inf_EB_gaincal_combine_dict[target][band][vis]:
                                applycal_spwmap[vis] = [selfcal_library[target][band][vis]
                                                        ['spwmap'], selfcal_library[target][band][vis]['spwmap']]
                                gaincal_spwmap[vis] = [selfcal_library[target][band][vis]['spwmap']]
                            elif inf_EB_fallback_mode_dict[target][band][vis] == 'spwmap':
                                applycal_spwmap[vis] = [selfcal_library[target][band][vis]['inf_EB']
                                                        ['spwmap'], selfcal_library[target][band][vis]['spwmap']]
                                gaincal_spwmap[vis] = selfcal_library[target][band][vis]['inf_EB']['spwmap']
                            else:
                                applycal_spwmap[vis] = [[], selfcal_library[target][band][vis]['spwmap']]
                                gaincal_spwmap[vis] = []
                            applycal_interpolate[vis] = [applycal_interp[band], applycal_interp[band]]
                            applycal_gaintable[vis] = [sani_target+'_'+vis+'_'+band+'_inf_EB_0'+'_p.g',
                                                       sani_target+'_'+vis+'_'+band+'_'+solint+'_'+str(iteration)+'_p.g']
                        elif solmode[band][iteration] == 'ap':
                            gaincal_spwmap[vis] = []
                            gaincal_preapply_gaintable[vis] = selfcal_library[target][band][vis][
                                selfcal_library[target][band]['final_phase_solint']]['gaintable']
                            gaincal_interpolate[vis] = [applycal_interp[band]]*len(gaincal_preapply_gaintable[vis])
                            gaincal_gaintype = 'T'
                            if 'spw' in inf_EB_gaincal_combine_dict[target][band][vis]:
                                applycal_spwmap[vis] = [
                                    selfcal_library[target][band][vis]['spwmap'],
                                    selfcal_library[target][band][vis]['spwmap'],
                                    selfcal_library[target][band][vis]['spwmap']]
                                gaincal_spwmap[vis] = [selfcal_library[target][band][vis]
                                                       ['spwmap'], selfcal_library[target][band][vis]['spwmap']]
                            elif inf_EB_fallback_mode_dict[target][band][vis] == 'spwmap':
                                applycal_spwmap[vis] = [
                                    selfcal_library[target][band][vis]['inf_EB']['spwmap'],
                                    selfcal_library[target][band][vis]['spwmap'],
                                    selfcal_library[target][band][vis]['spwmap']]
                                gaincal_spwmap[vis] = [selfcal_library[target][band][vis]['inf_EB']
                                                       ['spwmap'], selfcal_library[target][band][vis]['spwmap']]
                            else:
                                applycal_spwmap[vis] = [[], selfcal_library[target][band][vis]
                                                        ['spwmap'], selfcal_library[target][band][vis]['spwmap']]
                                gaincal_spwmap[vis] = [[], selfcal_library[target][band][vis]['spwmap']]
                            applycal_interpolate[vis] = [applycal_interp[band]
                                                         ]*len(gaincal_preapply_gaintable[vis])+['linearPD']
                            applycal_gaintable[vis] = selfcal_library[target][band][vis][
                                selfcal_library[target][band]['final_phase_solint']]['gaintable'] + [
                                sani_target + '_' + vis + '_' + band + '_' + solint + '_' + str(iteration) + '_ap.g']
                        fallback[vis] = ''
                        if solmode[band][iteration] == 'ap':
                            solnorm = True
                        else:
                            solnorm = False
                        gaincal(
                            vis=vis, caltable=sani_target + '_' + vis + '_' + band + '_' + solint + '_' + str(iteration) + '_' +
                            solmode[band][iteration] + '.g', gaintype=gaincal_gaintype,
                            spw=selfcal_library[target][band][vis]['spws'],
                            refant=selfcal_library[target][band][vis]['refant'],
                            calmode=solmode[band][iteration],
                            solnorm=solnorm, solint=solint.replace('_EB', '').replace('_ap', ''),
                            minsnr=gaincal_minsnr, minblperant=4, combine=gaincal_combine[band][iteration],
                            field=target, gaintable=gaincal_preapply_gaintable[vis],
                            spwmap=gaincal_spwmap[vis],
                            uvrange=selfcal_library[target][band]['uvrange'],
                            interp=gaincal_interpolate[vis])
                        ##
                        # default is to run without combine=spw for inf_EB, here we explicitly run a test inf_EB with combine='scan,spw' to determine
                        # the number of flagged antennas when combine='spw' then determine if it needs spwmapping or to use the gaintable with spwcombine.
                        ##
                        if solint == 'inf_EB' and fallback[vis] == '':
                            os.system('rm -rf test_inf_EB.g')
                            test_gaincal_combine = 'scan,spw'
                            if selfcal_library[target][band]['obstype'] == 'mosaic':
                                test_gaincal_combine += ',field'
                            gaincal(
                                vis=vis, caltable='test_inf_EB.g', gaintype=gaincal_gaintype,
                                spw=selfcal_library[target][band][vis]['spws'],
                                refant=selfcal_library[target][band][vis]['refant'],
                                calmode='p', solint=solint.replace('_EB', '').replace('_ap', ''),
                                minsnr=gaincal_minsnr, minblperant=4, combine=test_gaincal_combine, field=target,
                                gaintable='', spwmap=[],
                                uvrange=selfcal_library[target][band]['uvrange'])
                            spwlist = selfcal_library[target][band][vislist[0]]['spws'].split(',')
                            fallback[vis], map_index, spwmap, applycal_spwmap_inf_EB = analyze_inf_EB_flagging(
                                selfcal_library, band, spwlist, sani_target + '_' + vis + '_' + band + '_' + solint + '_' +
                                str(iteration) + '_' + solmode[band][iteration] + '.g', vis, target, 'test_inf_EB.g')

                            inf_EB_fallback_mode_dict[target][band][vis] = fallback[vis]+''
                            LOG.info(f'inf_EB {fallback[vis]}  {applycal_spwmap_inf_EB}')
                            if fallback[vis] != '':
                                if fallback[vis] == 'combinespw':
                                    gaincal_spwmap[vis] = [selfcal_library[target][band][vis]['spwmap']]
                                    gaincal_combine[band][iteration] = 'scan,spw'
                                    inf_EB_gaincal_combine_dict[target][band][vis] = 'scan,spw'
                                    applycal_spwmap[vis] = [selfcal_library[target][band][vis]['spwmap']]
                                    os.system(
                                        'rm -rf           ' + sani_target + '_' + vis + '_' + band + '_' + solint + '_' +
                                        str(iteration) + '_' + solmode[band][iteration] + '.g')
                                    os.system(
                                        'mv test_inf_EB.g ' + sani_target + '_' + vis + '_' + band + '_' + solint + '_' +
                                        str(iteration) + '_' + solmode[band][iteration] + '.g')
                                if fallback[vis] == 'spwmap':
                                    gaincal_spwmap[vis] = applycal_spwmap_inf_EB
                                    inf_EB_gaincal_combine_dict[target][band][vis] = 'scan'
                                    gaincal_combine[band][iteration] = 'scan'
                                    applycal_spwmap[vis] = applycal_spwmap_inf_EB
                            os.system('rm -rf test_inf_EB.g')

                    for vis in vislist:
                        ##
                        # Apply gain solutions per MS, target, solint, and band
                        ##
                        applycal(
                            vis=vis, gaintable=applycal_gaintable[vis],
                            interp=applycal_interpolate[vis],
                            calwt=True, spwmap=applycal_spwmap[vis],
                            applymode=applycal_mode[band][iteration],
                            field=target, spw=selfcal_library[target][band][vis]['spws'])
                    for vis in vislist:
                        ##
                        # record self cal results/details for this solint
                        ##
                        selfcal_library[target][band][vis][solint] = {}
                        selfcal_library[target][band][vis][solint]['SNR_pre'] = SNR.copy()
                        selfcal_library[target][band][vis][solint]['RMS_pre'] = RMS.copy()
                        selfcal_library[target][band][vis][solint]['SNR_NF_pre'] = SNR_NF.copy()
                        selfcal_library[target][band][vis][solint]['RMS_NF_pre'] = RMS_NF.copy()
                        selfcal_library[target][band][vis][solint]['Beam_major_pre'] = header['restoringbeam'][
                            'major']['value']
                        selfcal_library[target][band][vis][solint]['Beam_minor_pre'] = header['restoringbeam'][
                            'minor']['value']
                        selfcal_library[target][band][vis][solint]['Beam_PA_pre'] = header['restoringbeam'][
                            'positionangle']['value']
                        selfcal_library[target][band][vis][solint]['gaintable'] = applycal_gaintable[vis]
                        selfcal_library[target][band][vis][solint]['iteration'] = iteration+0
                        selfcal_library[target][band][vis][solint]['spwmap'] = applycal_spwmap[vis]
                        selfcal_library[target][band][vis][solint]['applycal_mode'] = applycal_mode[band][iteration]+''
                        selfcal_library[target][band][vis][solint]['applycal_interpolate'] = applycal_interpolate[vis]
                        selfcal_library[target][band][vis][solint]['gaincal_combine'] = gaincal_combine[band][
                            iteration] + ''
                        selfcal_library[target][band][vis][solint]['clean_threshold'] = selfcal_library[target][band][
                            'nsigma'][iteration] * selfcal_library[target][band]['RMS_curr']
                        selfcal_library[target][band][vis][solint]['intflux_pre'], selfcal_library[target][band][
                            vis][solint]['e_intflux_pre'] = get_intflux(
                            sani_target + '_' + band + '_' + solint + '_' + str(iteration) + '.image.tt0', RMS)
                        selfcal_library[target][band][vis][solint]['fallback'] = fallback[vis]+''
                        selfcal_library[target][band][vis][solint]['solmode'] = solmode[band][iteration]+''
                    # Create post self-cal image using the model as a startmodel to evaluate how much selfcal helped
                    ##
                    if selfcal_library[target][band]['nterms'] == 1:
                        startmodel = [sani_target+'_'+band+'_'+solint+'_'+str(iteration)+'.model.tt0']
                    elif selfcal_library[target][band]['nterms'] == 2:
                        startmodel = [
                            sani_target + '_' + band + '_' + solint + '_' + str(iteration) + '.model.tt0', sani_target + '_' +
                            band + '_' + solint + '_' + str(iteration) + '.model.tt1']
                    tclean_wrapper(vislist, sani_target + '_' + band + '_' + solint + '_' + str(iteration) + '_post',
                                   band_properties, band, telescope=telescope, scales=[0],
                                   nsigma=0.0, savemodel='none', parallel=parallel, cellsize=cellsize[band],
                                   imsize=imsize[band],
                                   nterms=selfcal_library[target][band]['nterms'],
                                   niter=0, startmodel=startmodel, field=target,
                                   spw=selfcal_library[target][band]['spws_per_vis'],
                                   uvrange=selfcal_library[target][band]['uvrange'],
                                   obstype=selfcal_library[target][band]['obstype'])
                    LOG.info('Post selfcal assessemnt: '+target)
                    # copy mask for use in post-selfcal SNR measurement
                    os.system(
                        'cp -r ' + sani_target + '_' + band + '_' + solint + '_' + str(iteration) + '.mask ' + sani_target + '_' +
                        band + '_' + solint + '_' + str(iteration) + '_post.mask')
                    post_SNR, post_RMS = estimate_SNR(
                        sani_target+'_'+band+'_'+solint+'_'+str(iteration)+'_post.image.tt0')
                    if post_SNR > 500.0:  # if S/N > 500, change nterms to 2 for best performance
                        selfcal_library[target][band]['nterms'] = 2
                    if telescope != 'ACA':
                        post_SNR_NF, post_RMS_NF = estimate_near_field_SNR(
                            sani_target+'_'+band+'_'+solint+'_'+str(iteration)+'_post.image.tt0')
                    else:
                        post_SNR_NF, post_RMS_NF = post_SNR, post_RMS
                    for vis in vislist:
                        selfcal_library[target][band][vis][solint]['SNR_post'] = post_SNR.copy()
                        selfcal_library[target][band][vis][solint]['RMS_post'] = post_RMS.copy()
                        selfcal_library[target][band][vis][solint]['SNR_NF_post'] = post_SNR_NF.copy()
                        selfcal_library[target][band][vis][solint]['RMS_NF_post'] = post_RMS_NF.copy()
                        # Update RMS value if necessary
                        if selfcal_library[target][band][vis][solint]['RMS_post'] < selfcal_library[target][band][
                                'RMS_curr']:
                            selfcal_library[target][band]['RMS_curr'] = selfcal_library[target][band][vis][solint][
                                'RMS_post'].copy()
                        header = imhead(imagename=sani_target+'_'+band+'_'+solint+'_'+str(iteration)+'_post.image.tt0')
                        selfcal_library[target][band][vis][solint]['Beam_major_post'] = header['restoringbeam'][
                            'major']['value']
                        selfcal_library[target][band][vis][solint]['Beam_minor_post'] = header['restoringbeam'][
                            'minor']['value']
                        selfcal_library[target][band][vis][solint]['Beam_PA_post'] = header['restoringbeam'][
                            'positionangle']['value']
                        selfcal_library[target][band][vis][solint]['intflux_post'], selfcal_library[target][band][
                            vis][solint]['e_intflux_post'] = get_intflux(
                            sani_target + '_' + band + '_' + solint + '_' + str(iteration) + '_post.image.tt0', post_RMS)

                    ##
                    # compare beam relative to original image to ensure we are not incrementally changing the beam in each iteration
                    ##
                    beamarea_orig = selfcal_library[target][band]['Beam_major_orig'] * \
                        selfcal_library[target][band]['Beam_minor_orig']
                    beamarea_post = selfcal_library[target][band][
                        vislist[0]][solint]['Beam_major_post'] * selfcal_library[target][band][
                        vislist[0]][solint]['Beam_minor_post']
                    """
                    frac_delta_b_maj=np.abs((b_maj_post-selfcal_library[target]['Beam_major_orig'])/selfcal_library[target]['Beam_major_orig'])
                    frac_delta_b_min=np.abs((b_min_post-selfcal_library[target]['Beam_minor_orig'])/selfcal_library[target]['Beam_minor_orig'])
                    delta_b_pa=np.abs((b_pa_post-selfcal_library[target]['Beam_PA_orig']))
                    """
                    delta_beamarea = (beamarea_post-beamarea_orig)/beamarea_orig
                    ##
                    # if S/N improvement, and beamarea is changing by < delta_beam_thresh, accept solutions to main calibration dictionary
                    # allow to proceed if solint was inf_EB and SNR decrease was less than 2%
                    ##
                    if ((post_SNR >= SNR) and (delta_beamarea < delta_beam_thresh)) or ((solint == 'inf_EB') and ((post_SNR-SNR)/SNR > -0.02) and (delta_beamarea < delta_beam_thresh)):
                        selfcal_library[target][band]['SC_success'] = True
                        selfcal_library[target][band]['Stop_Reason'] = 'None'
                        for vis in vislist:
                            selfcal_library[target][band][vis]['gaintable_final'] = selfcal_library[target][band][
                                vis][solint]['gaintable']
                            selfcal_library[target][band][vis]['spwmap_final'] = selfcal_library[target][band][vis][
                                solint]['spwmap'].copy()
                            selfcal_library[target][band][vis]['applycal_mode_final'] = selfcal_library[target][band][
                                vis][solint]['applycal_mode']
                            selfcal_library[target][band][vis]['applycal_interpolate_final'] = selfcal_library[target][
                                band][vis][solint]['applycal_interpolate']
                            selfcal_library[target][band][vis]['gaincal_combine_final'] = selfcal_library[target][
                                band][vis][solint]['gaincal_combine']
                            selfcal_library[target][band][vis][solint]['Pass'] = True
                        if solmode[band][iteration] == 'p':
                            selfcal_library[target][band]['final_phase_solint'] = solint
                        selfcal_library[target][band]['final_solint'] = solint
                        selfcal_library[target][band]['final_solint_mode'] = solmode[band][iteration]
                        selfcal_library[target][band]['iteration'] = iteration
                        # (iteration == 0) and
                        if (iteration < len(solints[band])-1) and (selfcal_library[target][band][vis][solint]['SNR_post'] > selfcal_library[target][band]['SNR_orig']):
                            LOG.info('Updating solint = '+solints[band][iteration+1]+' SNR')
                            LOG.info('Was: ', solint_snr[target][band][solints[band][iteration+1]])
                            get_SNR_self_update([target], band, vislist, selfcal_library, n_ants, solint,
                                                solints[band][iteration+1], integration_time, solint_snr)
                            LOG.info(f'Now: {solint_snr[target][band][solints[band][iteration+1]]}')

                        if iteration < (len(solints[band])-1):
                            LOG.info('****************Selfcal passed, shortening solint*************')
                        else:
                            LOG.info('****************Selfcal passed for Minimum solint*************')
                    ##
                    # if S/N worsens, and/or beam area increases reject current solutions and reapply previous (or revert to origional data)
                    ##

                    else:
                        for vis in vislist:
                            selfcal_library[target][band][vis][solint]['Pass'] = False
                        reason = ''
                        if (post_SNR <= SNR):
                            reason = reason+' S/N decrease'
                        if (delta_beamarea > delta_beam_thresh):
                            if reason != '':
                                reason = reason+'; '
                            reason = reason+'Beam change beyond '+str(delta_beam_thresh)
                        selfcal_library[target][band]['Stop_Reason'] = reason
                        LOG.info('****************Selfcal failed*************')
                        LOG.info('REASON: '+reason)
                        if iteration > 0:  # reapply only the previous gain tables, to get rid of solutions from this selfcal round
                            LOG.info('****************Reapplying previous solint solutions*************')
                            for vis in vislist:
                                LOG.info(
                                    '****************Applying ' +
                                    str(selfcal_library[target][band][vis]['gaintable_final']) + ' to ' + target + ' ' +
                                    band + '*************')
                                flagmanager(vis=vis, mode='restore', versionname='selfcal_starting_flags_'+sani_target)
                                applycal(vis=vis,
                                         gaintable=selfcal_library[target][band][vis]['gaintable_final'],
                                         interp=selfcal_library[target][band][vis]['applycal_interpolate_final'],
                                         calwt=True, spwmap=selfcal_library[target][band][vis]['spwmap_final'],
                                         applymode=selfcal_library[target][band][vis]['applycal_mode_final'],
                                         field=target, spw=selfcal_library[target][band][vis]['spws'])
                        else:
                            LOG.info('****************Removing all calibrations for '+target+' '+band+'**************')
                            for vis in vislist:
                                flagmanager(vis=vis, mode='restore', versionname='selfcal_starting_flags_'+sani_target)
                                clearcal(vis=vis, field=target, spw=selfcal_library[target][band][vis]['spws'])
                                selfcal_library[target][band]['SNR_post'] = selfcal_library[target][band][
                                    'SNR_orig'].copy()
                                selfcal_library[target][band]['RMS_post'] = selfcal_library[target][band][
                                    'RMS_orig'].copy()

                        # if a solution interval shorter than inf for phase-only SC has passed, attempt amplitude selfcal
                        if iteration > 1 and solmode[band][iteration] != 'ap' and do_amp_selfcal:
                            iterjump = solmode[band].index('ap')
                            LOG.info('****************Selfcal halted for phase, attempting amplitude*************')
                            continue
                        else:
                            LOG.info(
                                '****************Aborting further self-calibration attempts for ' + target + ' ' + band +
                                '**************')
                            break  # breakout of loops of successive solints since solutions are getting worse

    ##
    # If we want to try amplitude selfcal, should we do it as a function out of the main loop or a separate loop?
    # Mechanics are likely to be a bit more simple since I expect we'd only try a single solint=inf solution
    ##

    ##
    # Make a final image per target to assess overall improvement
    ##
    for target in all_targets:
        sani_target = sanitize_string(target)
        for band in selfcal_library[target].keys():
            vislist = selfcal_library[target][band]['vislist'].copy()
            # omit DR modifiers here since we should have increased DR significantly
            if telescope == 'ALMA' or telescope == 'ACA':
                sensitivity = get_sensitivity(
                    vislist, selfcal_library[target][band],
                    selfcal_library[target][band][vis]['spws'],
                    spw=selfcal_library[target][band][vis]['spwsarray'],
                    imsize=imsize[band],
                    cellsize=cellsize[band])
                dr_mod = 1.0
                if not selfcal_library[target][band]['SC_success']:  # fetch the DR modifier if selfcal failed on source
                    dr_mod = get_dr_correction(
                        telescope, selfcal_library[target][band]['SNR_dirty'] *
                        selfcal_library[target][band]['RMS_dirty'],
                        sensitivity, vislist)
                    LOG.info(f'DR modifier: {dr_mod}')
                    sensitivity = sensitivity*dr_mod
                if ((band == 'Band_9') or (band == 'Band_10')) and dr_mod != 1.0:   # adjust for DSB noise increase
                    sensitivity = sensitivity*4.0
            else:
                sensitivity = 0.0
            tclean_wrapper(
                vislist, sani_target + '_' + band + '_final', band_properties, band, telescope=telescope, nsigma=3.0,
                threshold=str(sensitivity * 4.0) + 'Jy', scales=[0],
                savemodel='none', parallel=parallel, cellsize=cellsize[band],
                imsize=imsize[band],
                nterms=selfcal_library[target][band]['nterms'],
                field=target, datacolumn='corrected', spw=selfcal_library[target][band]['spws_per_vis'],
                uvrange=selfcal_library[target][band]['uvrange'],
                obstype=selfcal_library[target][band]['obstype'])
            final_SNR, final_RMS = estimate_SNR(sani_target+'_'+band+'_final.image.tt0')
            if telescope != 'ACA':
                final_NF_SNR, final_NF_RMS = estimate_near_field_SNR(sani_target+'_'+band+'_final.image.tt0')
            else:
                final_NF_SNR, final_NF_RMS = final_SNR, final_RMS
            selfcal_library[target][band]['SNR_final'] = final_SNR
            selfcal_library[target][band]['RMS_final'] = final_RMS
            selfcal_library[target][band]['SNR_NF_final'] = final_NF_SNR
            selfcal_library[target][band]['RMS_NF_final'] = final_NF_RMS
            header = imhead(imagename=sani_target+'_'+band+'_final.image.tt0')
            selfcal_library[target][band]['Beam_major_final'] = header['restoringbeam']['major']['value']
            selfcal_library[target][band]['Beam_minor_final'] = header['restoringbeam']['minor']['value']
            selfcal_library[target][band]['Beam_PA_final'] = header['restoringbeam']['positionangle']['value']
            # recalc inital stats using final mask
            final_SNR, final_RMS = estimate_SNR(sani_target+'_'+band+'_initial.image.tt0',
                                                maskname=sani_target+'_'+band+'_final.mask')
            if telescope != 'ACA':
                final_NF_SNR, final_NF_RMS = estimate_near_field_SNR(
                    sani_target+'_'+band+'_initial.image.tt0', maskname=sani_target+'_'+band+'_final.mask')
            else:
                final_NF_SNR, final_NF_RMS = final_SNR, final_RMS
            selfcal_library[target][band]['SNR_orig'] = final_SNR
            selfcal_library[target][band]['RMS_orig'] = final_RMS
            selfcal_library[target][band]['SNR_NF_orig'] = final_NF_SNR
            selfcal_library[target][band]['RMS_NF_orig'] = final_NF_RMS
            goodMask = checkmask(imagename=sani_target+'_'+band+'_final.image.tt0')
            if goodMask:
                selfcal_library[target][band]['intflux_final'], selfcal_library[target][band]['e_intflux_final'] = get_intflux(
                    sani_target+'_'+band+'_final.image.tt0', final_RMS)
                selfcal_library[target][band]['intflux_orig'], selfcal_library[target][band]['e_intflux_orig'] = get_intflux(
                    sani_target+'_'+band+'_initial.image.tt0', selfcal_library[target][band]['RMS_orig'], maskname=sani_target+'_'+band+'_final.mask')
            else:
                selfcal_library[target][band]['intflux_final'], selfcal_library[target][band]['e_intflux_final'] = -99.0, -99.0

    ##
    # Make a final image per spw images to assess overall improvement
    ##
    if check_all_spws:
        for target in all_targets:
            sani_target = sanitize_string(target)
            for band in selfcal_library[target].keys():
                vislist = selfcal_library[target][band]['vislist'].copy()

                spwlist = selfcal_library[target][band][vis]['spws'].split(',')
                LOG.info('Generating final per-SPW images for '+target+' in '+band)
                for spw in spwlist:
                    # omit DR modifiers here since we should have increased DR significantly
                    if not os.path.exists(sani_target+'_'+band+'_'+spw+'_final.image.tt0'):
                        if telescope == 'ALMA' or telescope == 'ACA':
                            sensitivity = get_sensitivity(
                                vislist, selfcal_library[target][band],
                                spw, spw=np.array([int(spw)]),
                                imsize=imsize[band],
                                cellsize=cellsize[band])
                            dr_mod = 1.0
                            # fetch the DR modifier if selfcal failed on source
                            if not selfcal_library[target][band]['SC_success']:
                                dr_mod = get_dr_correction(
                                    telescope, selfcal_library[target][band]['SNR_dirty'] *
                                    selfcal_library[target][band]['RMS_dirty'],
                                    sensitivity, vislist)
                            LOG.info(f'DR modifier:  {dr_mod} SPW:  {spw}')
                            sensitivity = sensitivity*dr_mod
                            if ((band == 'Band_9') or (band == 'Band_10')) and dr_mod != 1.0:   # adjust for DSB noise increase
                                sensitivity = sensitivity*4.0
                        else:
                            sensitivity = 0.0
                        spws_per_vis = [spw]*len(vislist)  # assumes all spw ids are identical in each MS file
                        tclean_wrapper(
                            vislist, sani_target + '_' + band + '_' + spw + '_final', band_properties, band,
                            telescope=telescope, nsigma=4.0, threshold=str(sensitivity * 4.0) + 'Jy', scales=[0],
                            savemodel='none', parallel=parallel, cellsize=cellsize[band],
                            imsize=imsize[band],
                            nterms=1, field=target, datacolumn='corrected', spw=spws_per_vis,
                            uvrange=selfcal_library[target][band]['uvrange'],
                            obstype=selfcal_library[target][band]['obstype'])
                    final_per_spw_SNR, final_per_spw_RMS = estimate_SNR(sani_target+'_'+band+'_'+spw+'_final.image.tt0')
                    if telescope != 'ACA':
                        final_per_spw_NF_SNR, final_per_spw_NF_RMS = estimate_near_field_SNR(
                            sani_target + '_' + band + '_' + spw + '_final.image.tt0')
                    else:
                        final_per_spw_NF_SNR, final_per_spw_NF_RMS = final_per_spw_SNR, final_per_spw_RMS

                    selfcal_library[target][band]['per_spw_stats'][spw]['SNR_final'] = final_per_spw_SNR
                    selfcal_library[target][band]['per_spw_stats'][spw]['RMS_final'] = final_per_spw_RMS
                    selfcal_library[target][band]['per_spw_stats'][spw]['SNR_NF_final'] = final_per_spw_NF_SNR
                    selfcal_library[target][band]['per_spw_stats'][spw]['RMS_NF_final'] = final_per_spw_NF_RMS
                    # reccalc initial stats with final mask
                    final_per_spw_SNR, final_per_spw_RMS = estimate_SNR(
                        sani_target+'_'+band+'_'+spw+'_initial.image.tt0', maskname=sani_target+'_'+band+'_'+spw+'_final.mask')
                    if telescope != 'ACA':
                        final_per_spw_NF_SNR, final_per_spw_NF_RMS = estimate_near_field_SNR(
                            sani_target+'_'+band+'_'+spw+'_initial.image.tt0', maskname=sani_target+'_'+band+'_'+spw+'_final.mask')
                    else:
                        final_per_spw_NF_SNR, final_per_spw_NF_RMS = final_per_spw_SNR, final_per_spw_RMS
                    selfcal_library[target][band]['per_spw_stats'][spw]['SNR_orig'] = final_per_spw_SNR
                    selfcal_library[target][band]['per_spw_stats'][spw]['RMS_orig'] = final_per_spw_RMS
                    selfcal_library[target][band]['per_spw_stats'][spw]['SNR_NF_orig'] = final_per_spw_NF_SNR
                    selfcal_library[target][band]['per_spw_stats'][spw]['RMS_NF_orig'] = final_per_spw_NF_RMS

                    goodMask = checkmask(sani_target+'_'+band+'_'+spw+'_final.image.tt0')
                    if goodMask:
                        selfcal_library[target][band]['per_spw_stats'][spw]['intflux_final'], selfcal_library[target][
                            band]['per_spw_stats'][spw]['e_intflux_final'] = get_intflux(
                            sani_target + '_' + band + '_' + spw + '_final.image.tt0', final_per_spw_RMS)
                    else:
                        selfcal_library[target][band]['per_spw_stats'][spw]['intflux_final'], selfcal_library[target][
                            band]['per_spw_stats'][spw]['e_intflux_final'] = -99.0, -99.0

    ##
    # Print final results
    ##
    for target in all_targets:
        for band in selfcal_library[target].keys():
            LOG.info(target+' '+band+' Summary')
            LOG.info(f"At least 1 successful selfcal iteration?:  {selfcal_library[target][band]['SC_success']}")
            LOG.info(f"Final solint:  {selfcal_library[target][band]['final_solint']}")
            LOG.info(f"Original SNR:  {selfcal_library[target][band]['SNR_orig']}")
            LOG.info(f"Final SNR:  {selfcal_library[target][band]['SNR_final']}")
            LOG.info(f"Original RMS:  {selfcal_library[target][band]['RMS_orig']}")
            LOG.info(f"Final RMS:  {selfcal_library[target][band]['RMS_final']}")
            #   for vis in vislist:
            #      LOG.info('Final gaintables: '+selfcal_library[target][band][vis]['gaintable'])
            #      LOG.info('Final spwmap: ',selfcal_library[target][band][vis]['spwmap'])
            # else:
            #   LOG.info('Selfcal failed on '+target+'. No solutions applied.')

    applyCalOut = open('applycal_to_orig_MSes.py', 'w')
    # apply selfcal solutions back to original ms files
    if apply_to_target_ms:
        for vis in vislist_orig:
            clearcal(vis=vis)
    for target in all_targets:
        for band in selfcal_library[target].keys():
            if selfcal_library[target][band]['SC_success']:
                for vis in vislist:
                    solint = selfcal_library[target][band]['final_solint']
                    iteration = selfcal_library[target][band][vis][solint]['iteration']
                    line = 'applycal(vis="' + vis.replace('.selfcal', '') + '",gaintable=' + str(
                        selfcal_library[target][band][vis]['gaintable_final']) + ',interp=' + str(
                        selfcal_library[target][band][vis]['applycal_interpolate_final']) + ', calwt=True,spwmap=' + str(
                        selfcal_library[target][band][vis]['spwmap_final']) + ', applymode="' + selfcal_library[target][
                        band][vis]['applycal_mode_final'] + '",field="' + target + '",spw="' + spwstring_orig + '")\n'
                    applyCalOut.writelines(line)
                    if apply_to_target_ms:
                        if os.path.exists(vis.replace('.selfcal', '')+".flagversions/flags.starting_flags"):
                            flagmanager(vis=vis.replace('.selfcal', ''),
                                        mode='restore', versionname='starting_flags',
                                        comment='Flag states at start of reduction')
                        else:
                            flagmanager(vis=vis.replace('.selfcal', ''), mode='save',
                                        versionname='before_final_applycal')
                        applycal(
                            vis=vis.replace('.selfcal', ''),
                            gaintable=selfcal_library[target][band][vis]['gaintable_final'],
                            interp=selfcal_library[target][band][vis]['applycal_interpolate_final'],
                            calwt=True, spwmap=[selfcal_library[target][band][vis]['spwmap_final']],
                            applymode=selfcal_library[target][band][vis]['applycal_mode_final'],
                            field=target, spw=spwstring_orig)

    applyCalOut.close()

    if os.path.exists("cont.dat"):
        uvcontsubOut = open('uvcontsub_orig_MSes.py', 'w')
        line = 'import os\n'
        uvcontsubOut.writelines(line)
        for target in all_targets:
            sani_target = sanitize_string(target)
            for band in selfcal_library[target].keys():
                for vis in vislist:
                    contdot_dat_flagchannels_string = flagchannels_from_contdotdat(
                        vis.replace('.selfcal', ''), target, spwsarray)[:-2]
                    line = 'uvcontsub(vis="'+vis.replace('.selfcal', '')+'",field="'+target+'", spw="'+spwstring_orig + \
                        '",fitspw="'+contdot_dat_flagchannels_string+'",excludechans=True, combine="spw")\n'
                    uvcontsubOut.writelines(line)
                    line = 'os.system("mv '+vis.replace('.selfcal', '')+'.contsub '+sani_target+'_'+vis+'.contsub")\n'
                    uvcontsubOut.writelines(line)
        uvcontsubOut.close()

    #
    # Perform a check on the per-spw images to ensure they didn't lose quality in self-calibration
    #
    if check_all_spws:
        for target in all_targets:
            sani_target = sanitize_string(target)
            for band in selfcal_library[target].keys():
                vislist = selfcal_library[target][band]['vislist'].copy()

                spwlist = selfcal_library[target][band][vis]['spws'].split(',')
                for spw in spwlist:
                    delta_beamarea = compare_beams(sani_target+'_'+band+'_'+spw+'_initial.image.tt0',
                                                   sani_target+'_'+band+'_'+spw+'_final.image.tt0')
                    delta_SNR = selfcal_library[target][band]['per_spw_stats'][spw]['SNR_final'] -\
                        selfcal_library[target][band]['per_spw_stats'][spw]['SNR_orig']
                    delta_RMS = selfcal_library[target][band]['per_spw_stats'][spw]['RMS_final'] -\
                        selfcal_library[target][band]['per_spw_stats'][spw]['RMS_orig']
                    selfcal_library[target][band]['per_spw_stats'][spw]['delta_SNR'] = delta_SNR
                    selfcal_library[target][band]['per_spw_stats'][spw]['delta_RMS'] = delta_RMS
                    selfcal_library[target][band]['per_spw_stats'][spw]['delta_beamarea'] = delta_beamarea
                    LOG.info(sani_target + '_' + band + '_' + spw+' ' +
                             'Pre SNR: {:0.2f}, Post SNR: {:0.2f} Pre RMS: {:0.3f}, Post RMS: {:0.3f}'.format(
                                 selfcal_library[target][band]['per_spw_stats'][spw]['SNR_orig'],
                                 selfcal_library[target][band]['per_spw_stats'][spw]['SNR_final'],
                                 selfcal_library[target][band]['per_spw_stats'][spw]['RMS_orig'] * 1000.0,
                                 selfcal_library[target][band]['per_spw_stats'][spw]['RMS_final'] * 1000.0))
                    if delta_SNR < 0.0:
                        LOG.info('WARNING SPW '+spw+' HAS LOWER SNR POST SELFCAL')
                    if delta_RMS > 0.0:
                        LOG.info('WARNING SPW '+spw+' HAS HIGHER RMS POST SELFCAL')
                    if delta_beamarea > 0.05:
                        LOG.info('WARNING SPW '+spw+' HAS A >0.05 CHANGE IN BEAM AREA POST SELFCAL')

    ##
    # Save final library results
    ##
    with open('selfcal_library.pickle', 'wb') as handle:
        pickle.dump(selfcal_library, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('solints.pickle', 'wb') as handle:
        pickle.dump(solints, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('bands.pickle', 'wb') as handle:
        pickle.dump(bands, handle, protocol=pickle.HIGHEST_PROTOCOL)

    generate_weblog(selfcal_library, solints, bands)


def prep_selfcal():

    LOG.info('Start the auto_selfcal workflow...')
    ##
    # Get list of MS files in directory
    ##
    vislist = glob.glob('*_targets.ms')
    if len(vislist) == 0:
        vislist = glob.glob('*_cont.ms')   # adaptation for PL2022 output
        if len(vislist) == 0:
            sys.exit('No Measurement sets found in current working directory, exiting')

    ##
    # save starting flags or restore to the starting flags
    ##
    for vis in vislist:
        if os.path.exists(vis+".flagversions/flags.starting_flags"):
            flagmanager(vis=vis, mode='restore', versionname='starting_flags',
                        comment='Flag states at start of reduction')
        else:
            flagmanager(vis=vis, mode='save', versionname='starting_flags')

    ##
    # Find targets, assumes all targets are in all ms files for simplicity and only science targets, will fail otherwise
    ##
    all_targets = fetch_targets(vislist[0])

    ##
    # Global environment variables for control of selfcal
    ##
    spectral_average = True
    do_amp_selfcal = True
    inf_EB_gaincal_combine = 'scan'
    inf_EB_gaintype = 'G'
    inf_EB_override = False
    gaincal_minsnr = 2.0
    minsnr_to_proceed = 3.0
    delta_beam_thresh = 0.05
    n_ants = get_n_ants(vislist)
    telescope = get_telescope(vislist[0])
    apply_cal_mode_default = 'calflag'
    rel_thresh_scaling = 'log10'  # can set to linear, log10, or loge (natural log)
    dividing_factor = -99.0  # number that the peak SNR is divided by to determine first clean threshold -99.0 uses default
    # default is 40 for <8ghz and 15.0 for all other frequencies
    check_all_spws = False   # generate per-spw images to check phase transfer did not go poorly for narrow windows
    apply_to_target_ms = False  # apply final selfcal solutions back to the input _target.ms files

    if 'VLA' in telescope:
        check_all_spws = False
        # inf_EB_gaincal_combine='spw,scan'
    ##
    # Import inital MS files to get relevant meta data
    ##
    listdict, bands, band_properties, scantimesdict, scanstartsdict, scanendsdict, integrationsdict,\
        integrationtimesdict, spwslist, spwstring, spwsarray, mosaic_field = importdata(vislist, all_targets, telescope)

    ##
    # flag spectral lines in MS(es) if there is a cont.dat file present
    ##
    if os.path.exists("cont.dat"):
        flag_spectral_lines(vislist, all_targets, spwsarray)

    ##
    # spectrally average ALMA or VLA data with telescope/frequency specific averaging properties
    ##
    split_to_selfcal_ms(vislist, band_properties, bands, spectral_average)

    ##
    # put flagging back at original state for originally input ms for when they are used next time
    ##
    for vis in vislist:
        if os.path.exists(vis+".flagversions/flags.before_line_flags"):
            flagmanager(vis=vis, mode='restore', versionname='before_line_flags')

    ##
    # Reimport MS(es) to self calibrate since frequency averaging and splitting may have changed it
    ##
    spwslist_orig = spwslist.copy()
    vislist_orig = vislist.copy()
    spwstring_orig = spwstring+''
    spwsarray_orig = spwsarray.copy()

    vislist = glob.glob('*selfcal.ms')
    listdict, bands, band_properties, scantimesdict, scanstartsdict, scanendsdict, integrationsdict,\
        integrationtimesdict, spwslist, spwstring, spwsarray, mosaic_field = importdata(vislist, all_targets, telescope)

    ##
    # Save/restore starting flags
    ##

    for vis in vislist:
        if os.path.exists(vis+'.flagversions/flags.selfcal_starting_flags'):
            flagmanager(vis=vis, mode='restore', versionname='selfcal_starting_flags')
        else:
            flagmanager(vis=vis, mode='save', versionname='selfcal_starting_flags')

    ##
    # set image parameters based on the visibility data properties and frequency
    ##
    cellsize = {}
    imsize = {}
    nterms = {}
    applycal_interp = {}

    for band in bands:
        cellsize[band], imsize[band], nterms[band] = get_image_parameters(vislist, telescope, band, band_properties)
        if band_properties[vislist[0]][band]['meanfreq'] > 12.0e9:
            applycal_interp[band] = 'linearPD'
        else:
            applycal_interp[band] = 'linear'

    ###################################################################################################
    ################################# End Metadata gathering for Selfcal ##############################
    ###################################################################################################

    ###################################################################################################
    ############################# Start Actual important stuff for selfcal ############################
    ###################################################################################################

    ##
    # begin setting up a selfcal_library with all relevant metadata to keep track of during selfcal
    ##
    selfcal_library = {}

    for target in all_targets:
        selfcal_library[target] = {}
        for band in bands:
            if target in scantimesdict[band][vislist[0]].keys():
                selfcal_library[target][band] = {}
            else:
                continue
            for vis in vislist:
                selfcal_library[target][band][vis] = {}
    ##
    # finds solints, starting with inf, ending with int, and tries to align
    # solints with number of integrations
    # solints reduce by factor of 2 in each self-cal interation
    # e.g., inf, max_scan_time/2.0, prev_solint/2.0, ..., int
    # starting solints will have solint the length of the entire EB to correct bulk offsets
    ##
    solints = {}
    gaincal_combine = {}
    solmode = {}
    applycal_mode = {}
    for band in bands:
        solints[band], integration_time, gaincal_combine[band], solmode[band] = get_solints_simple(
            vislist, scantimesdict[band],
            scanstartsdict[band],
            scanendsdict[band],
            integrationtimesdict[band],
            inf_EB_gaincal_combine, do_amp_selfcal=do_amp_selfcal)
        LOG.info(f'{band} {solints[band]}')
        applycal_mode[band] = [apply_cal_mode_default]*len(solints[band])

    ##
    # puts stuff in right place from other MS metadata to perform proper data selections
    # in tclean, gaincal, and applycal
    # Also gets relevant times on source to estimate SNR per EB/scan
    ##
    for target in all_targets:
        for band in selfcal_library[target].keys():
            LOG.info(f'{target} {band}')
            selfcal_library[target][band]['SC_success'] = False
            selfcal_library[target][band]['final_solint'] = 'None'
            selfcal_library[target][band]['Total_TOS'] = 0.0
            selfcal_library[target][band]['spws'] = []
            selfcal_library[target][band]['spws_per_vis'] = []
            selfcal_library[target][band]['nterms'] = nterms[band]
            selfcal_library[target][band]['vislist'] = vislist.copy()
            if mosaic_field[band][target]['mosaic']:
                selfcal_library[target][band]['obstype'] = 'mosaic'
            else:
                selfcal_library[target][band]['obstype'] = 'single-point'
            allscantimes = np.array([])
            for vis in vislist:
                selfcal_library[target][band][vis]['gaintable'] = []
                selfcal_library[target][band][vis]['TOS'] = np.sum(scantimesdict[band][vis][target])
                selfcal_library[target][band][vis]['Median_scan_time'] = np.median(scantimesdict[band][vis][target])
                allscantimes = np.append(allscantimes, scantimesdict[band][vis][target])
                selfcal_library[target][band][vis]['refant'] = rank_refants(vis)
                n_spws, minspw, spwsarray = fetch_spws([vis], [target], listdict)
                spwslist = spwsarray.tolist()
                spwstring = ','.join(str(spw) for spw in spwslist)
                selfcal_library[target][band][vis]['spws'] = band_properties[vis][band]['spwstring']
                selfcal_library[target][band][vis]['spwsarray'] = band_properties[vis][band]['spwarray']
                selfcal_library[target][band][vis]['spwlist'] = band_properties[vis][band]['spwarray'].tolist()
                selfcal_library[target][band][vis]['n_spws'] = len(selfcal_library[target][band][vis]['spwsarray'])
                selfcal_library[target][band][vis]['minspw'] = int(
                    np.min(selfcal_library[target][band][vis]['spwsarray']))
                selfcal_library[target][band][vis]['spwmap'] = [selfcal_library[target][band][
                    vis]['minspw']]*(np.max(selfcal_library[target][band][vis]['spwsarray'])+1)
                selfcal_library[target][band]['Total_TOS'] = selfcal_library[target][band][vis]['TOS'] + \
                    selfcal_library[target][band]['Total_TOS']
                selfcal_library[target][band]['spws_per_vis'].append(band_properties[vis][band]['spwstring'])
            selfcal_library[target][band]['Median_scan_time'] = np.median(allscantimes)
            selfcal_library[target][band]['uvrange'] = get_uv_range(band, band_properties, vislist)
            selfcal_library[target][band]['75thpct_uv'] = band_properties[vislist[0]][band]['75thpct_uv']
            LOG.info(selfcal_library[target][band]['uvrange'])

    ##
    ##
    ##
    for target in all_targets:
        for band in selfcal_library[target].keys():
            if selfcal_library[target][band]['Total_TOS'] == 0.0:
                selfcal_library[target].pop(band)
    return vislist, vis, all_targets, do_amp_selfcal, inf_EB_gaincal_combine, inf_EB_gaintype, gaincal_minsnr, minsnr_to_proceed, delta_beam_thresh, n_ants, telescope, rel_thresh_scaling, dividing_factor, check_all_spws, apply_to_target_ms, bands, band_properties, spwsarray, vislist_orig, spwstring_orig, cellsize, imsize, nterms, applycal_interp, selfcal_library, solints, gaincal_combine, solmode, applycal_mode, integration_time


if __name__ == '__main__':

    selfcal_workflow()
