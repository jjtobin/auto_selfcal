import logging
import os
import time

import casatools
import numpy as np
from casaplotms import plotms
from casatasks import *
from casatools import image, imager
from casatools import msmetadata as msmdtool
from casatools import table as tbtool
from casaviewer import imview
from PIL import Image



tb = tbtool()
msmd = msmdtool()
ia = image()
im = imager()


def get_selfcal_logger(loggername='auto_selfcal', loglevel='DEBUG', logfile=None):
    """Get a named logger for auto_selfcal.
    
    When auto_selfcal runs outside of Pipeline, this function is a custom Python logger object
    as constructed below.
    When auto_selfcal runs as a Pipeline "extern" module, this function directly wraps around 
    pipeline.infrastructure.get_logger
    """
    
    casalog.showconsole(onconsole=True)
    if logfile is None:
        logfile = casalog.logfile()

    format = '%(asctime)s %(levelname)s    %(module)s.%(funcName)s     %(message)s'
    datefmt = '%Y-%m-%d %H:%M:%S'
    fmt = logging.Formatter(format, datefmt)
    fmt.converter = time.gmtime

    logger = logging.getLogger(loggername)
    logger.handlers = []
    logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(loglevel)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    logfile_handler = logging.FileHandler(logfile, mode='a')
    logfile_handler.setLevel(loglevel)
    logfile_handler.setFormatter(fmt)
    logger.addHandler(logfile_handler)

    return logger


def tclean_wrapper(
        vis, imagename, band_properties, band, telescope='undefined', scales=[0],
        smallscalebias=0.6, mask='', nsigma=5.0, imsize=None, cellsize=None, interactive=False, robust=0.5, gain=0.1,
        niter=50000, cycleniter=300, uvtaper=[],
        savemodel='none', gridder='standard', sidelobethreshold=3.0, smoothfactor=1.0, noisethreshold=5.0,
        lownoisethreshold=1.5, parallel=False, nterms=1, cyclefactor=3, uvrange='', threshold='0.0Jy', phasecenter='',
        startmodel='', pblimit=0.1, pbmask=0.1, field='', datacolumn='', spw='', obstype='single-point',):
    """
    Wrapper for tclean with keywords set to values desired for the Large Program imaging
    See the CASA 6.1.1 documentation for tclean to get the definitions of all the parameters
    """
    msmd.open(vis[0])
    fieldid = msmd.fieldsforname(field)
    msmd.done()
    tb.open(vis[0]+'/FIELD')
    ephem_column = tb.getcol('EPHEMERIS_ID')
    tb.close()
    if ephem_column[fieldid[0]] != -1:
        phasecenter = 'TRACKFIELD'

    if obstype == 'mosaic':
        phasecenter = get_phasecenter(vis[0], field)

    if mask == '':
        usemask = 'auto-multithresh'
    else:
        usemask = 'user'
    if threshold != '0.0Jy':
        nsigma = 0.0
    if telescope == 'ALMA':
        sidelobethreshold = 3.0
        smoothfactor = 1.0
        noisethreshold = 5.0
        lownoisethreshold = 1.5
        cycleniter = -1
        cyclefactor = 1.0
        LOG.info(band_properties)
        if band_properties[vis[0]][band]['75thpct_uv'] > 2000.0:
            sidelobethreshold = 2.0

    if telescope == 'ACA':
        sidelobethreshold = 1.25
        smoothfactor = 1.0
        noisethreshold = 5.0
        lownoisethreshold = 2.0
        cycleniter = -1
        cyclefactor = 1.0

    elif 'VLA' in telescope:
        sidelobethreshold = 2.0
        smoothfactor = 1.0
        noisethreshold = 5.0
        lownoisethreshold = 1.5
        pblimit = -0.1
        cycleniter = -1
        cyclefactor = 3.0
        pbmask = 0.0
    wprojplanes = 1
    if band == 'EVLA_L' or band == 'EVLA_S':
        gridder = 'wproject'
        wprojplanes = -1
    if (band == 'EVLA_L' or band == 'EVLA_S') and obstype == 'mosaic':
        LOG.info('WARNING DETECTED VLA L- OR S-BAND MOSAIC; WILL USE gridder="mosaic" IGNORING W-TERM')
    if obstype == 'mosaic':
        gridder = 'mosaic'
    else:
        if gridder != 'wproject':
            gridder = 'standard'

    if gridder == 'mosaic' and startmodel != '':
        parallel = False
    for ext in ['.image*', '.mask', '.model*', '.pb*', '.psf*', '.residual*', '.sumwt*', '.gridwt*']:
        os.system('rm -rf ' + imagename + ext)
    tclean(vis=vis,
           imagename=imagename,
           field=field,
           specmode='mfs',
           deconvolver='mtmfs',
           scales=scales,
           gridder=gridder,
           weighting='briggs',
           robust=robust,
           gain=gain,
           imsize=imsize,
           cell=cellsize,
           smallscalebias=smallscalebias,  # set to CASA's default of 0.6 unless manually changed
           niter=niter,  # we want to end on the threshold
           interactive=interactive,
           nsigma=nsigma,
           cycleniter=cycleniter,
           cyclefactor=cyclefactor,
           uvtaper=uvtaper,
           savemodel='none',
           mask=mask,
           usemask=usemask,
           sidelobethreshold=sidelobethreshold,
           smoothfactor=smoothfactor,
           pbmask=pbmask,
           pblimit=pblimit,
           nterms=nterms,
           uvrange=uvrange,
           threshold=threshold,
           parallel=parallel,
           phasecenter=phasecenter,
           startmodel=startmodel,
           datacolumn=datacolumn, spw=spw, wprojplanes=wprojplanes)
    # this step is a workaround a bug in tclean that doesn't always save the model during multiscale clean. See the "Known Issues" section for CASA 5.1.1 on NRAO's website
    if savemodel == 'modelcolumn':
        LOG.info("")
        LOG.info("Running tclean a second time to save the model...")
        tclean(vis=vis,
               imagename=imagename,
               field=field,
               specmode='mfs',
               deconvolver='mtmfs',
               scales=scales,
               gridder=gridder,
               weighting='briggs',
               robust=robust,
               gain=gain,
               imsize=imsize,
               cell=cellsize,
               smallscalebias=smallscalebias,  # set to CASA's default of 0.6 unless manually changed
               niter=0,
               interactive=False,
               nsigma=0.0,
               cycleniter=cycleniter,
               cyclefactor=cyclefactor,
               uvtaper=uvtaper,
               usemask='user',
               savemodel=savemodel,
               sidelobethreshold=sidelobethreshold,
               smoothfactor=smoothfactor,
               pbmask=pbmask,
               pblimit=pblimit,
               calcres=False,
               calcpsf=False,
               nterms=nterms,
               uvrange=uvrange,
               threshold=threshold,
               parallel=False,
               phasecenter=phasecenter, spw=spw, wprojplanes=wprojplanes)


def collect_listobs_per_vis(vislist):
    listdict = {}
    for vis in vislist:
        listdict[vis] = listobs(vis)
    return listdict


def fetch_scan_times(vislist, targets, listdict):
    scantimesdict = {}
    integrationsdict = {}
    integrationtimesdict = {}
    integrationtime = np.array([])
    n_spws = np.array([])
    min_spws = np.array([])
    spwslist = np.array([])
    for vis in vislist:
        scantimesdict[vis] = {}
        integrationsdict[vis] = {}
        integrationtimesdict[vis] = {}
        keylist = list(listdict[vis].keys())
        for target in targets:
            countscans = 0
            scantimes = np.array([])
            integrations = np.array([])
            for key in keylist:
                if 'scan' in key and listdict[vis][key]['0']['FieldName'] == target:
                    countscans += 1
                    scantime = (listdict[vis][key]['0']['EndTime'] - listdict[vis][key]['0']['BeginTime'])*86400.0
                    ints_per_scan = np.round(scantime/listdict[vis][key]['0']['IntegrationTime'])
                    integrationtime = np.append(integrationtime, np.array([listdict[vis][key]['0']['IntegrationTime']]))
                    # LOG.info(f'Key: {key} {scantime}')
                    scantimes = np.append(scantimes, np.array([scantime]))
                    integrations = np.append(integrations, np.array([ints_per_scan]))
                    n_spws = np.append(len(listdict[vis][key]['0']['SpwIds']), n_spws)
                    min_spws = np.append(np.min(listdict[vis][key]['0']['SpwIds']), min_spws)
                    spwslist = np.append(listdict[vis][key]['0']['SpwIds'], spwslist)
                    # LOG.info(scantimes)

            scantimesdict[vis][target] = scantimes.copy()
            # assume each band only has a single integration time
            integrationtimesdict[vis][target] = np.median(integrationtime)
            integrationsdict[vis][target] = integrations.copy()
    if np.mean(n_spws) != np.max(n_spws):
        LOG.info('WARNING, INCONSISTENT NUMBER OF SPWS IN SCANS/MSes')
    if np.max(min_spws) != np.min(min_spws):
        LOG.info('WARNING, INCONSISTENT MINIMUM SPW IN SCANS/MSes')
    spwslist = np.unique(spwslist).astype(int)
    return scantimesdict, integrationsdict, integrationtimesdict, integrationtime, np.max(n_spws), np.min(min_spws), spwslist


def fetch_scan_times_band_aware(vislist, targets, listdict, band_properties, band):
    scantimesdict = {}
    scanstartsdict = {}
    scanendsdict = {}
    integrationsdict = {}
    integrationtimesdict = {}
    integrationtime = np.array([])
    n_spws = np.array([])
    min_spws = np.array([])
    spwslist = np.array([])
    mosaic_field = {}
    for vis in vislist:
        scantimesdict[vis] = {}
        scanstartsdict[vis] = {}
        scanendsdict[vis] = {}
        integrationsdict[vis] = {}
        integrationtimesdict[vis] = {}
        keylist = list(listdict[vis].keys())
        for target in targets:
            mosaic_field[target] = {}
            mosaic_field[target]['field_ids'] = []
            mosaic_field[target]['mosaic'] = False
            countscans = 0
            scantimes = np.array([])
            integrations = np.array([])
            scanstarts = np.array([])
            scanends = np.array([])
            for key in keylist:
                if ('scan' in key) and (listdict[vis][key]['0']['FieldName'] == target) and np.all(np.in1d(np.array(listdict[vis][key]['0']['SpwIds']), band_properties[vis][band]['spwarray'])):
                    countscans += 1
                    scantime = (listdict[vis][key]['0']['EndTime'] - listdict[vis][key]['0']['BeginTime'])*86400.0
                    ints_per_scan = np.round(scantime/listdict[vis][key]['0']['IntegrationTime'])
                    integrationtime = np.append(integrationtime, np.array([listdict[vis][key]['0']['IntegrationTime']]))
                    # LOG.info('Key: {key} {scantime}')
                    scanstarts = np.append(scanstarts, np.array([listdict[vis][key]['0']['BeginTime']]))
                    scanends = np.append(scanends, np.array([listdict[vis][key]['0']['EndTime']]))
                    scantimes = np.append(scantimes, np.array([scantime]))
                    integrations = np.append(integrations, np.array([ints_per_scan]))
                    n_spws = np.append(len(listdict[vis][key]['0']['SpwIds']), n_spws)
                    min_spws = np.append(np.min(listdict[vis][key]['0']['SpwIds']), min_spws)
                    spwslist = np.append(listdict[vis][key]['0']['SpwIds'], spwslist)
                    subscanlist = listdict[vis][key].keys()
                    for subscan in subscanlist:
                        mosaic_field[target]['field_ids'].append(listdict[vis][key][subscan]['FieldId'])

                    mosaic_field[target]['field_ids'] = list(set(mosaic_field[target]['field_ids']))
                    if len(mosaic_field[target]['field_ids']) > 1:
                        mosaic_field[target]['mosaic'] = True

                    # LOG.info(scantimes)

            scantimesdict[vis][target] = scantimes.copy()
            scanstartsdict[vis][target] = scanstarts.copy()
            scanendsdict[vis][target] = scanends.copy()
            # assume each band only has a single integration time
            integrationtimesdict[vis][target] = np.median(integrationtime)
            integrationsdict[vis][target] = integrations.copy()
    if len(n_spws) > 0:
        if np.mean(n_spws) != np.max(n_spws):
            LOG.info('WARNING, INCONSISTENT NUMBER OF SPWS IN SCANS/MSes')
        if np.max(min_spws) != np.min(min_spws):
            LOG.info('WARNING, INCONSISTENT MINIMUM SPW IN SCANS/MSes')
        spwslist = np.unique(spwslist).astype(int)
    else:
        return scantimesdict, scanstartsdict, scanendsdict, integrationsdict, integrationtimesdict, integrationtime, -99, -99, spwslist, mosaic_field
    return scantimesdict, scanstartsdict, scanendsdict, integrationsdict, integrationtimesdict, integrationtime, np.max(
        n_spws), np.min(min_spws), spwslist, mosaic_field


def fetch_spws(vislist, targets, listdict):
    scantimesdict = {}
    n_spws = np.array([])
    min_spws = np.array([])
    spwslist = np.array([])
    for vis in vislist:
        listdict[vis] = listobs(vis)
        keylist = list(listdict[vis].keys())
        for target in targets:
            countscans = 0
            for key in keylist:
                if 'scan' in key and listdict[vis][key]['0']['FieldName'] == target:
                    countscans += 1
                    n_spws = np.append(len(listdict[vis][key]['0']['SpwIds']), n_spws)
                    min_spws = np.append(np.min(listdict[vis][key]['0']['SpwIds']), min_spws)
                    spwslist = np.append(listdict[vis][key]['0']['SpwIds'], spwslist)
                    # LOG.info(scantimes)
    if len(n_spws) > 1:
        if np.mean(n_spws) != np.max(n_spws):
            LOG.info('WARNING, INCONSISTENT NUMBER OF SPWS IN SCANS/MSes')
        if np.max(min_spws) != np.min(min_spws):
            LOG.info('WARNING, INCONSISTENT MINIMUM SPW IN SCANS/MSes')
    spwslist = np.unique(spwslist).astype(int)
    if len(n_spws) == 1:
        return n_spws, min_spws, spwslist
    else:
        return np.max(n_spws), np.min(min_spws), spwslist


def fetch_scan_times_target(vislist, target, listdict):
    scantimesdict = {}
    integrationsdict = {}
    integrationtimesdict = {}
    integrationtime = np.array([])
    n_spws = np.array([])
    min_spws = np.array([])
    spwslist = np.array([])
    allscantimes = np.array([])
    for vis in vislist:
        listdict[vis] = listobs(vis)
        keylist = list(listdict[vis].keys())
        countscans = 0
        scantimes = np.array([])
        integrations = np.array([])
        for key in keylist:
            if 'scan' in key:
                if listdict[vis][key]['0']['FieldName'] == target:
                    countscans += 1
                    scantime = (listdict[vis][key]['0']['EndTime'] - listdict[vis][key]['0']['BeginTime'])*86400.0
                    scantimes = np.append(scantimes, np.array([scantime]))

        allscantimes = np.append(allscantimes, scantimes)

    return allscantimes

# deprecated function


def get_common_intervals(vis, integrationsdict, integrationtime):
    allintegrations = np.array([])

    # for vis in vislist:
    allintegrations = np.append(allintegrations, integrationsdict)

    unique_integrations = np.unique(allintegrations)
    common_multiples = np.array([])
    common_multiple = True
    for i in range(1, int(np.max(unique_integrations))):
        for number in unique_integrations:
            multiple = number/i
            #LOG.info(f'{multiple} {number} {i} {multiple.is_integer()}')
            if multiple.is_integer():
                common_multiple = True
            else:
                common_multiple = False
                break
        if common_multiple:
            common_multiples = np.append(common_multiples, np.array([i]))
        common_multiple = True
    solints = []
    for multiple in common_multiples:
        solint = '{:0.2f}s'.format(multiple*integrationtime)
        solints.append(solint)
    return common_multiples, solints

# deprecated function


def get_solints_vla(vis, scantimesdict, integrationtime):
    allscantimes = np.array([])

    # for vis in vislist: # use if we put all scan times from all MSes into single array
    # mix of short and long baseline data could have differing integration times and hence solints
    allscantimes = np.append(allscantimes, scantimesdict)

    medianscantime = np.median(allscantimes)
    integrations_per_scan = np.round(medianscantime/integrationtime)
    non_integer_multiple = True
    i = 0
    while non_integer_multiple:
        integrations_per_scan = integrations_per_scan+i
        integrations_per_scan_div4 = integrations_per_scan/4.0
        LOG.info(f'{integrations_per_scan} {integrations_per_scan_div4} {i}')
        if integrations_per_scan_div4.is_integer():
            non_integer_multiple = False
            n_ints_increment = i
        else:
            i += 1

    max_integrations_per_sol = integrations_per_scan
    LOG.info(f'Max integrations per solution  {max_integrations_per_sol} {n_ints_increment}')
    common_multiples = np.array([])

    for i in range(1, int(max_integrations_per_sol)):
        multiple = max_integrations_per_sol/i
        # LOG.info(f'{multiple} {number} {i} {multiple.is_integer()}')
        if multiple.is_integer():
            common_multiple = True
        else:
            common_multiple = False
        if common_multiple:
            common_multiples = np.append(common_multiples, np.array([i]))

    solints = []
    for multiple in common_multiples:
        solint = '{:0.2f}s'.format(multiple*integrationtime)
        solints.append(solint)

    return solints


# actual routine used for getting solints
def get_solints_simple(
        vislist, scantimesdict, scanstartsdict, scanendsdict, integrationtimes, inf_EB_gaincal_combine, spwcombine=True,
        solint_decrement='fixed', solint_divider=2.0, n_solints=4.0, do_amp_selfcal=False):
    all_integrations = np.array([])
    all_nscans_per_obs = np.array([])
    all_time_between_scans = np.array([])
    all_times_per_obs = np.array([])
    allscantimes = np.array([])  # we put all scan times from all MSes into single array
    # mix of short and long baseline data could have differing integration times and hence solints
    # could do solints per vis file, but too complex for now at least use perhaps keep scan groups different
    # per MOUS
    nscans_per_obs = {}
    time_per_vis = {}
    time_between_scans = {}
    for vis in vislist:
        nscans_per_obs[vis] = {}
        time_between_scans[vis] = {}
        time_per_vis[vis] = 0.0
        targets = integrationtimes[vis].keys()
        earliest_start = 1.0e10
        latest_end = 0.0
        for target in targets:
            nscans_per_obs[vis][target] = len(scantimesdict[vis][target])
            allscantimes = np.append(allscantimes, scantimesdict[vis][target])
            # way to get length of an EB with multiple targets without writing new functions; I could be more clever with np.where()
            for i in range(len(scanstartsdict[vis][target])):
                if scanstartsdict[vis][target][i] < earliest_start:
                    earliest_start = scanstartsdict[vis][target][i]
                if scanendsdict[vis][target][i] > latest_end:
                    latest_end = scanstartsdict[vis][target][i]
            if np.isfinite(integrationtimes[vis][target]):
                all_integrations = np.append(all_integrations, integrationtimes[vis][target])
            all_nscans_per_obs = np.append(all_nscans_per_obs, nscans_per_obs[vis][target])
            # determine time between scans
            delta_scan = np.zeros(len(scanstartsdict[vis][target])-1)
            # scan list isn't sorted, so sort these so they're in order and we can subtract them from each other
            sortedstarts = np.sort(scanstartsdict[vis][target])
            sortedends = np.sort(scanstartsdict[vis][target])
            # delta_scan=(sortedends[:-1]-sortedstarts[1:])*86400.0*-1.0
            delta_scan = np.zeros(len(sortedends)-1)
            for i in range(len(sortedstarts)-1):
                delta_scan[i] = (sortedends[i]-sortedstarts[i+1])*86400.0*-1.0
            all_time_between_scans = np.append(all_time_between_scans, delta_scan)
        time_per_vis[vis] = (latest_end - earliest_start)*86400.0    # calculate length of EB
        all_times_per_obs = np.append(all_times_per_obs, np.array([time_per_vis[vis]]))
    integration_time = np.max(all_integrations)  # use the longest integration time from all MS files

    max_scantime = np.median(allscantimes)
    median_scantime = np.max(allscantimes)
    min_scantime = np.min(allscantimes)
    median_scans_per_obs = np.median(all_nscans_per_obs)
    median_time_per_obs = np.median(all_times_per_obs)
    median_time_between_scans = np.median(all_time_between_scans)
    LOG.info(f'median scan length: {median_scantime}')
    LOG.info(f'median time between target scans: {median_time_between_scans}')
    LOG.info(f'median scans per observation: {median_scans_per_obs}')
    LOG.info(f'median length of observation: {median_time_per_obs}')

    solints_gt_scan = np.array([])
    gaincal_combine = []

    # commented completely, no solints between inf_EB and inf
    # make solints between inf_EB and inf if more than one scan per source and scans are short
    # if median_scans_per_obs > 1 and median_scantime < 150.0:
    #   # add one solint that is meant to combine 2 short scans, otherwise go to inf_EB
    #   solint=(median_scantime*2.0+median_time_between_scans)*1.1
    #   if solint < 300.0:  # only allow solutions that are less than 5 minutes in duration
    #      solints_gt_scan=np.append(solints_gt_scan,[solint])

    # code below would make solints between inf_EB and inf by combining scans
    # sometimes worked ok, but many times selfcal would quit before solint=inf
    '''
   solint=median_time_per_obs/4.05 # divides slightly unevenly if lengths of observation are exactly equal, but better than leaving a small out of data remaining
   while solint > (median_scantime*2.0+median_time_between_scans)*1.05:      #solint should be greater than the length of time between two scans + time between to be better than inf
      solints_gt_scan=np.append(solints_gt_scan,[solint])                       # add solint to list of solints now that it is an integer number of integrations
      solint = solint/2.0  
      # LOG.info('Next solint: {solint}')                                        #divide solint by 2.0 for next solint
   '''
    LOG.info(f'{max_scantime} {integration_time}')
    if solint_decrement == 'fixed':
        solint_divider = np.round(np.exp(1.0/n_solints*np.log(max_scantime/integration_time)))
    # division never less than 2.0
    if solint_divider < 2.0:
        solint_divider = 2.0
    solints_lt_scan = np.array([])
    n_scans = len(allscantimes)
    solint = max_scantime/solint_divider
    # 1.1*integration_time will ensure that a single int will not be returned such that solint='int' can be appended to the final list.
    while solint > 1.90*integration_time:
        ints_per_solint = solint/integration_time
        if ints_per_solint.is_integer():
            solint = solint
        else:
            # calculate delta_T greater than an a fixed multile of integrations
            remainder = ints_per_solint-float(int(ints_per_solint))
            solint = solint-remainder*integration_time  # add remainder to make solint a fixed number of integrations

        ints_per_solint = float(int(ints_per_solint))
        LOG.info(f'Checking solint = {ints_per_solint*integration_time}')
        delta = test_truncated_scans(ints_per_solint, allscantimes, integration_time)
        solint = (ints_per_solint+delta)*integration_time
        if solint > 1.90*integration_time:
            # add solint to list of solints now that it is an integer number of integrations
            solints_lt_scan = np.append(solints_lt_scan, [solint])

        solint = solint/solint_divider
        # LOG.info(f'Next solint: {solint}')                                        #divide solint by 2.0 for next solint

    solints_list = []
    if len(solints_gt_scan) > 0:
        for solint in solints_gt_scan:
            solint_string = '{:0.2f}s'.format(solint)
            solints_list.append(solint_string)
            if spwcombine:
                gaincal_combine.append('spw,scan')
            else:
                gaincal_combine.append('scan')

     # insert inf_EB
    solints_list.insert(0, 'inf_EB')
    gaincal_combine.insert(0, inf_EB_gaincal_combine)

    # insert solint = inf
    if median_scans_per_obs > 1:                    # if only a single scan per target, redundant with inf_EB and do not include
        solints_list.append('inf')
        if spwcombine:
            gaincal_combine.append('spw')
        else:
            gaincal_combine.append('')

    for solint in solints_lt_scan:
        solint_string = '{:0.2f}s'.format(solint)
        solints_list.append(solint_string)
        if spwcombine:
            gaincal_combine.append('spw')
        else:
            gaincal_combine.append('')

    # append solint = int to end
    solints_list.append('int')
    if spwcombine:
        gaincal_combine.append('spw')
    else:
        gaincal_combine.append('')
    solmode_list = ['p']*len(solints_list)
    if do_amp_selfcal:
        if median_time_between_scans > 150.0 or np.isnan(median_time_between_scans):
            amp_solints_list = ['inf_ap']
            if spwcombine:
                amp_gaincal_combine = ['spw']
            else:
                amp_gaincal_combine = ['']
        else:
            amp_solints_list = ['300s_ap', 'inf_ap']
            if spwcombine:
                amp_gaincal_combine = ['scan,spw', 'spw']
            else:
                amp_gaincal_combine = ['scan', '']
        solints_list = solints_list+amp_solints_list
        gaincal_combine = gaincal_combine+amp_gaincal_combine
        solmode_list = solmode_list+['ap']*len(amp_solints_list)

    return solints_list, integration_time, gaincal_combine, solmode_list


def test_truncated_scans(ints_per_solint, allscantimes, integration_time):
    delta_ints_per_solint = [0, -1, 1, -2, 2]
    n_truncated_scans = np.zeros(len(delta_ints_per_solint))
    n_remaining_ints = np.zeros(len(delta_ints_per_solint))
    min_index = 0
    for i in range(len(delta_ints_per_solint)):
        diff_ints_per_scan = (
            (allscantimes-((ints_per_solint+delta_ints_per_solint[i])*integration_time))/integration_time)+0.5
        diff_ints_per_scan = diff_ints_per_scan.astype(int)
        trimmed_scans = ((diff_ints_per_scan > 0.0) & (diff_ints_per_scan <
                         ints_per_solint+delta_ints_per_solint[i])).nonzero()
        if len(trimmed_scans[0]) > 0:
            n_remaining_ints[i] = np.max(diff_ints_per_scan[trimmed_scans[0]])
        else:
            n_remaining_ints[i] = 0.0
        # LOG.info(f'{(ints_per_solint+delta_ints_per_solint[i])*integration_time,ints_per_solint+delta_ints_per_solint[i]}  {diff_ints_per_scan}')

        # LOG.info(f'Max ints remaining: {n_remaining_ints[i]}')
        # LOG.info(f'N truncated scans: {len(trimmed_scans[0])}')
        n_truncated_scans[i] = len(trimmed_scans[0])
        # check if there are fewer truncated scans in the current trial and if
        # if one trial has more scans left off or fewer. Favor more left off, such that remainder might be able to
        # find a solution
        # if ((i > 0) and (n_truncated_scans[i] <= n_truncated_scans[min_index]):   # if we don't care about the amount of
        # if ((i > 0) and (n_truncated_scans[i] <= n_truncated_scans[min_index]) and (n_remaining_ints[i] > n_remaining_ints[min_index])):
        if ((i > 0) and (n_truncated_scans[i] <= n_truncated_scans[min_index]) and (n_remaining_ints[i] < n_remaining_ints[min_index])):
            min_index = i
        # LOG.info(delta_ints_per_solint[min_index])
    return delta_ints_per_solint[min_index]


def fetch_targets_old(vis):
    fields = []
    listdict = listobs(vis)
    listobskeylist = listdict.keys()
    for listobskey in listobskeylist:
        if 'field_' in listobskey:
            fields.append(listdict[listobskey]['name'])
    fields = list(set(fields))  # convert to set to only get unique items
    return fields


def fetch_targets(vis):
    fields = []
    tb.open(vis+'/FIELD')
    names = list(tb.getcol('NAME'))
    tb.close()
    listdict = listobs(vis)
    listobskeylist = listdict.keys()
    for listobskey in listobskeylist:
        if 'field_' in listobskey:
            fieldnum = int(listobskey.split('_')[1])
            fields.append(names[fieldnum])
    fields = list(set(fields))  # convert to set to only get unique items
    return fields


def checkmask(imagename):
    maskImage = imagename.replace('image', 'mask').replace('.tt0', '')
    image_stats = imstat(maskImage)
    if image_stats['max'][0] == 0:
        return False
    else:
        return True


def estimate_SNR(imagename, maskname=None, verbose=True):
    MADtoRMS = 1.4826
    headerlist = imhead(imagename, mode='list')
    beammajor = headerlist['beammajor']['value']
    beamminor = headerlist['beamminor']['value']
    beampa = headerlist['beampa']['value']
    image_stats = imstat(imagename=imagename)
    if maskname is None:
        maskImage = imagename.replace('image', 'mask').replace('.tt0', '')
    else:
        maskImage = maskname
    residualImage = imagename.replace('image', 'residual')
    os.system('rm -rf temp.mask temp.residual')
    if os.path.exists(maskImage):
        os.system('cp -r '+maskImage + ' temp.mask')
        maskImage = 'temp.mask'
    os.system('cp -r '+residualImage + ' temp.residual')
    residualImage = 'temp.residual'
    if 'dirty' not in imagename:
        goodMask = checkmask(imagename)
    else:
        goodMask = False
    if os.path.exists(maskImage) and goodMask:
        ia.close()
        ia.done()
        ia.open(residualImage)
        #ia.calcmask(maskImage+" <0.5"+"&& mask("+residualImage+")",name='madpbmask0')
        ia.calcmask("'"+maskImage+"'"+" <0.5"+"&& mask("+residualImage+")", name='madpbmask0')
        mask0Stats = ia.statistics(robust=True, axes=[0, 1])
        ia.maskhandler(op='set', name='madpbmask0')
        rms = mask0Stats['medabsdevmed'][0] * MADtoRMS
        residualMean = mask0Stats['median'][0]
    else:
        residual_stats = imstat(imagename=imagename.replace('image', 'residual'), algorithm='chauvenet')
        rms = residual_stats['rms'][0]
    peak_intensity = image_stats['max'][0]
    SNR = peak_intensity/rms
    if verbose:
        LOG.info("#%s" % imagename)
        LOG.info("#Beam %.3f arcsec x %.3f arcsec (%.2f deg)" % (beammajor, beamminor, beampa))
        LOG.info("#Peak intensity of source: %.2f mJy/beam" % (peak_intensity*1000,))
        LOG.info("#rms: %.2e mJy/beam" % (rms*1000,))
        LOG.info("#Peak SNR: %.2f" % (SNR,))
    ia.close()
    ia.done()
    os.system('rm -rf temp.mask temp.residual')
    return SNR, rms


def estimate_near_field_SNR(imagename, maskname=None, verbose=True):
    MADtoRMS = 1.4826
    headerlist = imhead(imagename, mode='list')
    beammajor = headerlist['beammajor']['value']
    beamminor = headerlist['beamminor']['value']
    beampa = headerlist['beampa']['value']
    image_stats = imstat(imagename=imagename)
    if maskname is None:
        maskImage = imagename.replace('image', 'mask').replace('.tt0', '')
    else:
        maskImage = maskname
    if not os.path.exists(maskImage):
        LOG.info('Does not exist')
        return np.float64(-99.0), np.float64(-99.0)
    goodMask = checkmask(maskImage)
    if not goodMask:
        LOG.info('checkmask')
        return np.float64(-99.0), np.float64(-99.0)
    residualImage = imagename.replace('image', 'residual')
    os.system('rm -rf temp.mask temp.residual temp.border.mask temp.smooth.ceiling.mask temp.smooth.mask temp.nearfield.mask temp.big.smooth.ceiling.mask temp.big.smooth.mask')
    os.system('cp -r '+maskImage + ' temp.mask')
    os.system('cp -r '+residualImage + ' temp.residual')
    residualImage = 'temp.residual'
    maskStats = imstat(imagename='temp.mask')
    imsmooth(imagename='temp.mask', kernel='gauss', major=str(beammajor*3.0)+'arcsec',
             minor=str(beammajor*3.0)+'arcsec', pa='0deg', outfile='temp.smooth.mask')
    immath(imagename=['temp.smooth.mask'], expr='iif(IM0 > 0.1,1.0,0.0)', outfile='temp.smooth.ceiling.mask')
    imsmooth(imagename='temp.smooth.ceiling.mask', kernel='gauss', major=str(beammajor * 5.0) + 'arcsec',
             minor=str(beammajor * 5.0) + 'arcsec', pa='0deg', outfile='temp.big.smooth.mask')
    immath(imagename=['temp.big.smooth.mask'], expr='iif(IM0 > 0.1,1.0,0.0)', outfile='temp.big.smooth.ceiling.mask')
    # immath(imagename=['temp.smooth.ceiling.mask','temp.mask'],expr='((IM0-IM1)-1.0)*-1.0',outfile='temp.border.mask')
    immath(imagename=['temp.big.smooth.ceiling.mask', 'temp.smooth.ceiling.mask'],
           expr='((IM0-IM1)-1.0)*-1.0', outfile='temp.nearfield.mask')
    maskImage = 'temp.nearfield.mask'
    ia.close()
    ia.done()
    ia.open(residualImage)
    #ia.calcmask(maskImage+" <0.5"+"&& mask("+residualImage+")",name='madpbmask0')
    ia.calcmask("'"+maskImage+"'"+" <0.5"+"&& mask("+residualImage+")", name='madpbmask0')
    mask0Stats = ia.statistics(robust=True, axes=[0, 1])
    ia.maskhandler(op='set', name='madpbmask0')
    rms = mask0Stats['medabsdevmed'][0] * MADtoRMS
    residualMean = mask0Stats['median'][0]
    peak_intensity = image_stats['max'][0]
    SNR = peak_intensity/rms
    if verbose:
        LOG.info("#%s" % imagename)
        LOG.info("#Beam %.3f arcsec x %.3f arcsec (%.2f deg)" % (beammajor, beamminor, beampa))
        LOG.info("#Peak intensity of source: %.2f mJy/beam" % (peak_intensity*1000,))
        LOG.info("#Near Field rms: %.2e mJy/beam" % (rms*1000,))
        LOG.info("#Peak Near Field SNR: %.2f" % (SNR,))
    ia.close()
    ia.done()
    os.system('rm -rf temp.mask temp.residual temp.border.mask temp.smooth.ceiling.mask temp.smooth.mask temp.nearfield.mask temp.big.smooth.ceiling.mask temp.big.smooth.mask')
    return SNR, rms


def get_intflux(imagename, rms, maskname=None):
    headerlist = imhead(imagename, mode='list')
    beammajor = headerlist['beammajor']['value']
    beamminor = headerlist['beamminor']['value']
    beampa = headerlist['beampa']['value']
    cell = headerlist['cdelt2']*180.0/3.14159*3600.0
    beamarea = 3.14159*beammajor*beamminor/(4.0*np.log(2.0))
    pix_per_beam = beamarea/(cell**2)
    if maskname is None:
        maskname = imagename.replace('image.tt0', 'mask')
    imagestats = imstat(imagename=imagename, mask=maskname)
    flux = imagestats['flux'][0]
    n_beams = imagestats['npts'][0]/pix_per_beam
    e_flux = (n_beams)**0.5*rms
    return flux, e_flux


def get_n_ants(vislist):
    # Examines number of antennas in each ms file and returns the minimum number of antennas
    msmd = casatools.msmetadata()
    tb = casatools.table()
    n_ants = 50.0
    for vis in vislist:
        msmd.open(vis)
        names = msmd.antennanames()
        msmd.close()
        n_ant_vis = len(names)
        if n_ant_vis < n_ants:
            n_ants = n_ant_vis
    return n_ants


def get_ant_list(vis):
    # Examines number of antennas in each ms file and returns the minimum number of antennas
    msmd = casatools.msmetadata()
    tb = casatools.table()
    n_ants = 50.0
    msmd.open(vis)
    names = msmd.antennanames()
    msmd.close()
    return names


def rank_refants(vis):
    # Get the antenna names and offsets.

    msmd = casatools.msmetadata()
    tb = casatools.table()

    msmd.open(vis)
    names = msmd.antennanames()
    offset = [msmd.antennaoffset(name) for name in names]
    msmd.close()

    # Calculate the mean longitude and latitude.

    mean_longitude = np.mean([offset[i]["longitude offset"]
                              ['value'] for i in range(len(names))])
    mean_latitude = np.mean([offset[i]["latitude offset"]
                             ['value'] for i in range(len(names))])

    # Calculate the offsets from the center.

    offsets = [np.sqrt((offset[i]["longitude offset"]['value'] -
                        mean_longitude)**2 + (offset[i]["latitude offset"]
                                              ['value'] - mean_latitude)**2) for i in
               range(len(names))]

    # Calculate the number of flags for each antenna.

    nflags = [tb.calc('[select from '+vis+' where ANTENNA1==' +
                      str(i)+' giving  [ntrue(FLAG)]]')['0'].sum() for i in
              range(len(names))]

    # Calculate a score based on those two.

    score = [offsets[i] / max(offsets) + nflags[i] / max(nflags)
             for i in range(len(names))]

    # Print out the antenna scores.

    LOG.info("Refant list for "+vis)
    # for i in np.argsort(score):
    #    LOG.info(names[i], score[i])
    LOG.info(','.join(np.array(names)[np.argsort(score)]))
    # Return the antenna names sorted by score.

    return ','.join(np.array(names)[np.argsort(score)])


def get_SNR_self(
        all_targets, bands, vislist, selfcal_library, n_ant, solints, integration_time, inf_EB_gaincal_combine,
        inf_EB_gaintype):
    solint_snr = {}
    solint_snr_per_spw = {}
    if inf_EB_gaintype == 'G':
        polscale = 2.0
    else:
        polscale = 1.0
    for target in all_targets:
        solint_snr[target] = {}
        solint_snr_per_spw[target] = {}
        for band in selfcal_library[target].keys():
            solint_snr[target][band] = {}
            solint_snr_per_spw[target][band] = {}
            for solint in solints[band]:
                solint_snr[target][band][solint] = 0.0
                solint_snr_per_spw[target][band][solint] = {}
                if solint == 'inf_EB':
                    SNR_self_EB = np.zeros(len(vislist))
                    SNR_self_EB_spw = np.zeros(
                        [len(vislist), len(selfcal_library[target][band][vislist[0]]['spwsarray'])])
                    SNR_self_EB_spw_mean = np.zeros([len(selfcal_library[target][band][vislist[0]]['spwsarray'])])
                    SNR_self_EB_spw = {}
                    for i in range(len(vislist)):
                        SNR_self_EB[i] = selfcal_library[target][band]['SNR_orig']/((n_ant)**0.5*(
                            selfcal_library[target][band]['Total_TOS']/selfcal_library[target][band][vislist[i]]['TOS'])**0.5)
                        SNR_self_EB_spw[vislist[i]] = {}
                        for spw in selfcal_library[target][band][vislist[i]]['spwsarray']:
                            SNR_self_EB_spw[
                                vislist[i]][
                                str(spw)] = (polscale) ** -0.5 * selfcal_library[target][band]['SNR_orig'] / (
                                (n_ant - 3) ** 0.5 *
                                (selfcal_library[target][band]['Total_TOS'] /
                                 selfcal_library[target][band][vislist[i]]['TOS']) ** 0.5) * (
                                selfcal_library[target][band]['per_spw_stats'][str(spw)]['effective_bandwidth'] /
                                selfcal_library[target][band]['total_effective_bandwidth']) ** 0.5
                    for spw in selfcal_library[target][band][vislist[0]]['spwsarray']:
                        mean_SNR = 0.0
                        for j in range(len(vislist)):
                            mean_SNR += SNR_self_EB_spw[vislist[j]][str(spw)]
                        mean_SNR = mean_SNR/len(vislist)
                        solint_snr_per_spw[target][band][solint][str(spw)] = mean_SNR
                    solint_snr[target][band][solint] = np.mean(SNR_self_EB)
                    selfcal_library[target][band]['per_EB_SNR'] = np.mean(SNR_self_EB)
                elif solint == 'inf' or solint == 'inf_ap':
                    selfcal_library[target][band]['per_scan_SNR'] = selfcal_library[target][band]['SNR_orig']/((n_ant-3)**0.5*(
                        selfcal_library[target][band]['Total_TOS']/selfcal_library[target][band]['Median_scan_time'])**0.5)
                    solint_snr[target][band][solint] = selfcal_library[target][band]['per_scan_SNR']
                    for spw in selfcal_library[target][band][vislist[0]]['spwsarray']:
                        solint_snr_per_spw[target][band][solint][str(spw)] = selfcal_library[target][band][
                            'SNR_orig'] / ((n_ant - 3) ** 0.5 *
                                           (selfcal_library[target][band]['Total_TOS'] /
                                            selfcal_library[target][band]['Median_scan_time']) ** 0.5) * (
                            selfcal_library[target][band]['per_spw_stats'][str(spw)]['effective_bandwidth'] /
                            selfcal_library[target][band]['total_effective_bandwidth']) ** 0.5
                elif solint == 'int':
                    solint_snr[target][band][solint] = selfcal_library[target][band]['SNR_orig'] / \
                        ((n_ant-3)**0.5*(selfcal_library[target][band]['Total_TOS']/integration_time)**0.5)
                    for spw in selfcal_library[target][band][vislist[0]]['spwsarray']:
                        solint_snr_per_spw[target][band][solint][str(spw)] = selfcal_library[target][band][
                            'SNR_orig'] / ((n_ant - 3) ** 0.5 *
                                           (selfcal_library[target][band]['Total_TOS'] / integration_time) ** 0.5) * (
                            selfcal_library[target][band]['per_spw_stats'][str(spw)]['effective_bandwidth'] /
                            selfcal_library[target][band]['total_effective_bandwidth']) ** 0.5
                else:
                    solint_float = float(solint.replace('s', '').replace('_ap', ''))
                    solint_snr[target][band][solint] = selfcal_library[target][band]['SNR_orig'] / \
                        ((n_ant-3)**0.5*(selfcal_library[target][band]['Total_TOS']/solint_float)**0.5)
                    for spw in selfcal_library[target][band][vislist[0]]['spwsarray']:
                        solint_snr_per_spw[target][band][solint][str(spw)] = selfcal_library[target][band][
                            'SNR_orig'] / ((n_ant - 3) ** 0.5 *
                                           (selfcal_library[target][band]['Total_TOS'] / solint_float) ** 0.5) * (
                            selfcal_library[target][band]['per_spw_stats'][str(spw)]['effective_bandwidth'] /
                            selfcal_library[target][band]['total_effective_bandwidth']) ** 0.5
    return solint_snr, solint_snr_per_spw


def get_SNR_self_update(
        all_targets, band, vislist, selfcal_library, n_ant, solint_curr, solint_next, integration_time, solint_snr):
    for target in all_targets:
        if solint_next == 'inf' or solint_next == 'inf_ap':
            selfcal_library[target][band]['per_scan_SNR'] = selfcal_library[target][band][vislist[0]][solint_curr]['SNR_post']/(
                (n_ant-3)**0.5*(selfcal_library[target][band]['Total_TOS']/selfcal_library[target][band]['Median_scan_time'])**0.5)
            solint_snr[target][band][solint_next] = selfcal_library[target][band]['per_scan_SNR']
        elif solint_next == 'int':
            solint_snr[target][band][solint_next] = selfcal_library[target][band][vislist[0]][solint_curr][
                'SNR_post'] / ((n_ant - 3) ** 0.5 * (selfcal_library[target][band]['Total_TOS'] / integration_time) ** 0.5)
        else:
            solint_float = float(solint_next.replace('s', '').replace('_ap', ''))
            solint_snr[target][band][solint_next] = selfcal_library[target][band][vislist[0]][solint_curr][
                'SNR_post'] / ((n_ant - 3) ** 0.5 * (selfcal_library[target][band]['Total_TOS'] / solint_float) ** 0.5)


def get_sensitivity(vislist, selfcal_library, specmode='mfs', spwstring='', spw=[],
                    chan=0, cellsize='0.025arcsec', imsize=1600, robust=0.5, uvtaper=''):
    sensitivities = np.zeros(len(vislist))
    TOS = np.zeros(len(vislist))
    counter = 0
    scalefactor = 1.0
    for vis in vislist:
        im.open(vis)
        im.selectvis(field='', spw=spwstring)
        im.defineimage(mode=specmode, stokes='I', spw=spw, cellx=cellsize, celly=cellsize, nx=imsize, ny=imsize)
        im.weight(type='briggs', robust=robust)
        if uvtaper != '':
            if 'klambda' in uvtaper:
                uvtaper = uvtaper.replace('klambda', '')
                uvtaperflt = float(uvtaper)
                bmaj = str(206.0/uvtaperflt)+'arcsec'
                bmin = bmaj
                bpa = '0.0deg'
            if 'arcsec' in uvtaper:
                bmaj = uvtaper
                bmin = uvtaper
                bpa = '0.0deg'
            LOG.info('uvtaper: '+bmaj+' '+bmin+' '+bpa)
            im.filter(type='gaussian', bmaj=bmaj, bmin=bmin, bpa=bpa)
        try:
            sens = im.apparentsens()
        except:
            LOG.info('#')
            LOG.info('# Sensisitivity Calculation failed for '+vis)
            LOG.info('# Continuing to next MS')
            LOG.info('# Data in this spw/MS may be flagged')
            LOG.info('#')
            continue
        # LOG.info(sens)
        # LOG.info(f'{vis} Briggs Sensitivity = {sens[1]}')
        # LOG.info(f'{vis} Relative to Natural Weighting = {sens[2]}')
        sensitivities[counter] = sens[1]*scalefactor
        TOS[counter] = selfcal_library[vis]['TOS']
        counter += 1
    # estsens=np.sum(sensitivities)/float(counter)/(float(counter))**0.5
    # LOG.info(estsens)
    estsens = np.sum(sensitivities*TOS)/np.sum(TOS)
    LOG.info(f'Estimated Sensitivity: {estsens}')
    return estsens


def LSRKfreq_to_chan(msfile, field, spw, LSRKfreq, spwsarray):
    """
    Identifies the channel(s) corresponding to input LSRK frequencies. 
    Useful for choosing which channels to split out or flag if a line has been identified by the pipeline.

    Parameters
    ==========
    msfile: Name of measurement set (string)
    spw: Spectral window number (int)
    obsid: Observation ID corresponding to the selected spectral window 
    restfreq: Rest frequency in Hz (float)
    LSRKvelocity: input velocity in LSRK frame in km/s (float or array of floats)

    Returns
    =======
    Channel number most closely corresponding to input LSRK frequency.
    """
    tb.open(msfile)
    spw_col = tb.getcol('DATA_DESC_ID')
    obs_col = tb.getcol('OBSERVATION_ID')
    # work around the fact that spws in DATA_DESC_ID don't match listobs
    uniquespws = np.unique(spw_col)
    matching_index = np.where(spw == spwsarray)
    alt_spw = uniquespws[matching_index[0]]
    tb.close()
    obsid = np.unique(obs_col[np.where(spw_col == alt_spw)])

    tb.open(msfile+'/SPECTRAL_WINDOW')
    chanfreqs = tb.getcol('CHAN_FREQ', startrow=spw, nrow=1)
    tb.close()
    tb.open(msfile+'/FIELD')
    fieldnames = tb.getcol('NAME')
    tb.close()
    tb.open(msfile+'/OBSERVATION')
    obstime = np.squeeze(tb.getcol('TIME_RANGE', startrow=obsid, nrow=1))[0]
    tb.close()
    nchan = len(chanfreqs)
    ms.open(msfile)

    lsrkfreqs = ms.cvelfreqs(
        spwids=[spw],
        fieldids=int(np.where(fieldnames == field)[0][0]),
        mode='channel', nchan=nchan, obstime=str(obstime) + 's', start=0, outframe='LSRK') / 1e9
    ms.close()

    if type(LSRKfreq) == np.ndarray:
        outchans = np.zeros_like(LSRKfreq)
        for i in range(len(LSRKfreq)):
            outchans[i] = np.argmin(np.abs(lsrkfreqs - LSRKfreq[i]))
        return outchans
    else:
        return np.argmin(np.abs(lsrkfreqs - LSRKfreq))


def parse_contdotdat(contdotdat_file, target):
    """
    Parses the cont.dat file that includes line emission automatically identified by the ALMA pipeline.

    Parameters
    ==========
    msfile: Name of the cont.dat file (string)

    Returns
    =======
    Dictionary with the boundaries of the frequency range including line emission. The dictionary keys correspond to the spectral windows identified 
    in the cont.dat file, and the entries include numpy arrays with shape (nline, 2), with the 2 corresponding to min and max frequencies identified.
    """
    f = open(contdotdat_file, 'r')
    lines = f.readlines()
    f.close()

    while '\n' in lines:
        lines.remove('\n')

    contdotdat = {}
    desiredTarget = False
    for i, line in enumerate(lines):
        if 'ALL' in line:
            continue
        if 'Field' in line:
            field = line.split()[-1]
            if field == target:
                desiredTarget = True
                continue
            else:
                desiredTarget = False
                continue
        if desiredTarget == True:
            if 'SpectralWindow' in line:
                spw = int(line.split()[-1])
                contdotdat[spw] = []
            else:
                contdotdat[spw] += [line.split()[0].split("G")[0].split("~")]

    for spw in contdotdat:
        contdotdat[spw] = np.array(contdotdat[spw], dtype=float)

    return contdotdat


def flagchannels_from_contdotdat(vis, target, spwsarray):
    """
    Generates a string with the list of lines identified by the cont.dat file from the ALMA pipeline, that need to be flagged.

    Parameters
    ==========
    ms_dict: Dictionary of information about measurement set

    Returns
    =======
    String of channels to be flagged, in a format that can be passed to the spw parameter in CASA's flagdata task. 
    """
    contdotdat = parse_contdotdat('cont.dat', target)

    flagchannels_string = ''
    for j, spw in enumerate(contdotdat):
        flagchannels_string += '%d:' % (spw)

        tb.open(vis+'/SPECTRAL_WINDOW')
        nchan = tb.getcol('CHAN_FREQ', startrow=spw, nrow=1).size
        tb.close()

        chans = np.array([])
        for k in range(contdotdat[spw].shape[0]):
            LOG.info(f'{spw} {contdotdat[spw][k]}')

            chans = np.concatenate((LSRKfreq_to_chan(vis, target, spw, contdotdat[spw][k], spwsarray), chans))

            """
            if flagchannels_string == '':
                flagchannels_string+='%d:%d~%d' % (spw, np.min([chans[0], chans[1]]), np.max([chans[0], chans[1]]))
            else:
                flagchannels_string+=', %d:%d~%d' % (spw, np.min([chans[0], chans[1]]), np.max([chans[0], chans[1]]))
            """

        chans = np.sort(chans)

        flagchannels_string += '0~%d;' % (chans[0])
        for i in range(1, chans.size-1, 2):
            flagchannels_string += '%d~%d;' % (chans[i], chans[i+1])
        flagchannels_string += '%d~%d, ' % (chans[-1], nchan-1)

    LOG.info("# Flagchannels input string for %s in %s from cont.dat file: \'%s\'" % (target, vis, flagchannels_string))

    return flagchannels_string


def get_spw_chanwidths(vis, spwarray):
    widtharray = np.zeros(len(spwarray))
    bwarray = np.zeros(len(spwarray))
    nchanarray = np.zeros(len(spwarray))
    for i in range(len(spwarray)):
        tb.open(vis+'/SPECTRAL_WINDOW')
        widtharray[i] = np.abs(np.unique(tb.getcol('CHAN_WIDTH', startrow=spwarray[i], nrow=1)))
        bwarray[i] = np.abs(np.unique(tb.getcol('TOTAL_BANDWIDTH', startrow=spwarray[i], nrow=1)))
        nchanarray[i] = np.abs(np.unique(tb.getcol('NUM_CHAN', startrow=spwarray[i], nrow=1)))
        tb.close()

    return widtharray, bwarray, nchanarray


def get_spw_bandwidth(vis, spwsarray, target):
    spwbws = {}
    for spw in spwsarray:
        tb.open(vis+'/SPECTRAL_WINDOW')
        spwbws[str(spw)] = np.abs(np.unique(tb.getcol('TOTAL_BANDWIDTH', startrow=spw, nrow=1)))[
            0]/1.0e9  # put bandwidths into GHz
        tb.close()
    spweffbws = spwbws.copy()
    if os.path.exists("cont.dat"):
        spweffbws = get_spw_eff_bandwidth(vis, target)

    return spwbws, spweffbws


def get_spw_eff_bandwidth(vis, target):
    spweffbws = {}
    contdotdat = parse_contdotdat('cont.dat', target)
    for key in contdotdat.keys():
        cumulat_bw = 0.0
        for i in range(len(contdotdat[key])):
            cumulat_bw += np.abs(contdotdat[key][i][1]-contdotdat[key][i][0])
        spweffbws[str(key)] = cumulat_bw+0.0
    return spweffbws


def get_spw_chanavg(vis, widtharray, bwarray, chanarray, desiredWidth=15.625e6):
    avgarray = np.zeros(len(widtharray))
    for i in range(len(widtharray)):
        nchan = bwarray[i]/desiredWidth
        nchan = np.round(nchan)
        avgarray[i] = chanarray[i]/nchan
        if avgarray[i] < 1.0:
            avgarray[i] = 1.0
    return avgarray


def get_image_parameters(vislist, telescope, band, band_properties):
    cells = np.zeros(len(vislist))
    for i in range(len(vislist)):
        # im.open(vislist[i])
        im.selectvis(vis=vislist[i], spw=band_properties[vislist[i]][band]['spwarray'])
        adviseparams = im.advise()
        cells[i] = adviseparams[2]['value']/2.0
        im.close()
    cell = np.min(cells)
    cellsize = '{:0.3f}arcsec'.format(cell)
    nterms = 1
    if band_properties[vislist[0]][band]['fracbw'] > 0.1:
        nterms = 2
    if 'VLA' in telescope:
        fov = 45.0e9/band_properties[vislist[0]][band]['meanfreq']*60.0*1.5
        if band_properties[vislist[0]][band]['meanfreq'] < 12.0e9:
            fov = fov*2.0
    if telescope == 'ALMA':
        fov = 63.0*100.0e9/band_properties[vislist[0]][band]['meanfreq']*1.5
    if telescope == 'ACA':
        fov = 108.0*100.0e9/band_properties[vislist[0]][band]['meanfreq']*1.5
    npixels = int(np.ceil(fov/cell / 100.0)) * 100
    if npixels > 16384:
        npixels = 16384
    return cellsize, npixels, nterms


def get_mean_freq(vislist, spwsarray):
    tb.open(vislist[0]+'/SPECTRAL_WINDOW')
    freqarray = tb.getcol('REF_FREQUENCY')
    tb.close()
    meanfreq = np.mean(freqarray[spwsarray])
    minfreq = np.min(freqarray[spwsarray])
    maxfreq = np.max(freqarray[spwsarray])
    fracbw = np.abs(maxfreq-minfreq)/meanfreq
    return meanfreq, maxfreq, minfreq, fracbw


def get_desired_width(meanfreq):
    if meanfreq >= 50.0e9:
        desiredWidth = 15.625e6
    elif (meanfreq < 50.0e9) and (meanfreq >= 40.0e9):
        desiredWidth = 16.0e6
    elif (meanfreq < 40.0e9) and (meanfreq >= 26.0e9):
        desiredWidth = 8.0e6
    elif (meanfreq < 26.0e9) and (meanfreq >= 18.0e9):
        desiredWidth = 16.0e6
    elif (meanfreq < 18.0e9) and (meanfreq >= 8.0e9):
        desiredWidth = 8.0e6
    elif (meanfreq < 8.0e9) and (meanfreq >= 4.0e9):
        desiredWidth = 4.0e6
    elif (meanfreq < 4.0e9) and (meanfreq >= 2.0e9):
        desiredWidth = 4.0e6
    elif (meanfreq < 4.0e9):
        desiredWidth = 2.0e6
    return desiredWidth


def get_ALMA_bands(vislist, spwstring, spwarray):
    meanfreq, maxfreq, minfreq, fracbw = get_mean_freq(vislist, spwarray)
    observed_bands = {}
    if (meanfreq < 950.0e9) and (meanfreq >= 787.0e9):
        band = 'Band_10'
    elif (meanfreq < 720.0e9) and (meanfreq >= 602.0e9):
        band = 'Band_9'
    elif (meanfreq < 500.0e9) and (meanfreq >= 385.0e9):
        band = 'Band_8'
    elif (meanfreq < 373.0e9) and (meanfreq >= 275.0e9):
        band = 'Band_7'
    elif (meanfreq < 275.0e9) and (meanfreq >= 211.0e9):
        band = 'Band_6'
    elif (meanfreq < 211.0e9) and (meanfreq >= 163.0e9):
        band = 'Band_5'
    elif (meanfreq < 163.0e9) and (meanfreq >= 125.0e9):
        band = 'Band_4'
    elif (meanfreq < 116.0e9) and (meanfreq >= 84.0e9):
        band = 'Band_3'
    elif (meanfreq < 84.0e9) and (meanfreq >= 67.0e9):
        band = 'Band_2'
    elif (meanfreq < 50.0e9) and (meanfreq >= 30.0e9):
        band = 'Band_1'
    bands = [band]
    for vis in vislist:
        observed_bands[vis] = {}
        observed_bands[vis]['bands'] = [band]
        for band in bands:
            observed_bands[vis][band] = {}
            observed_bands[vis][band]['spwarray'] = spwarray
            observed_bands[vis][band]['spwstring'] = spwstring+''
            observed_bands[vis][band]['meanfreq'] = meanfreq
            observed_bands[vis][band]['maxfreq'] = maxfreq
            observed_bands[vis][band]['minfreq'] = minfreq
            observed_bands[vis][band]['fracbw'] = fracbw
    get_max_uvdist(vislist, observed_bands[vislist[0]]['bands'].copy(), observed_bands)
    return bands, observed_bands


def get_VLA_bands(vislist):
    observed_bands = {}
    for vis in vislist:
        observed_bands[vis] = {}
        # visheader=vishead(vis,mode='list',listitems=[])
        tb.open(vis+'/SPECTRAL_WINDOW')
        spw_names = tb.getcol('NAME')
        tb.close()
        # spw_names=visheader['spw_name'][0]
        spw_names_band = spw_names.copy()
        spw_names_bb = spw_names.copy()
        spw_names_spw = np.zeros(len(spw_names_band)).astype('int')
        for i in range(len(spw_names)):
            spw_names_band[i] = spw_names[i].split('#')[0]
            spw_names_bb[i] = spw_names[i].split('#')[1]
            spw_names_spw[i] = i
        all_bands = np.unique(spw_names_band)
        observed_bands[vis]['n_bands'] = len(all_bands)
        observed_bands[vis]['bands'] = all_bands.tolist()
        for band in all_bands:
            index = np.where(spw_names_band == band)
            if (band == 'EVLA_X') and (len(index[0]) == 2):  # ignore pointing band
                observed_bands[vis]['n_bands'] = observed_bands[vis]['n_bands']-1
                observed_bands[vis]['bands'].remove('EVLA_X')
                continue
            elif (band == 'EVLA_X') and (len(index[0]) > 2):  # ignore pointing band
                observed_bands[vis][band] = {}
                observed_bands[vis][band]['spwarray'] = spw_names_spw[index[0]]
                indices_to_remove = np.array([])
                for i in range(len(observed_bands[vis][band]['spwarray'])):
                    meanfreq, maxfreq, minfreq, fracbw = get_mean_freq(
                        [vis], np.array([observed_bands[vis][band]['spwarray'][i]]))
                    if (meanfreq == 8.332e9) or (meanfreq == 8.460e9):
                        indices_to_remove = np.append(indices_to_remove, [i])
                observed_bands[vis][band]['spwarray'] = np.delete(
                    observed_bands[vis][band]['spwarray'],
                    indices_to_remove.astype(int))
                spwslist = observed_bands[vis][band]['spwarray'].tolist()
                spwstring = ','.join(str(spw) for spw in spwslist)
                observed_bands[vis][band]['spwstring'] = spwstring+''
                observed_bands[vis][band]['meanfreq'], observed_bands[vis][band]['maxfreq'], observed_bands[vis][
                    band]['minfreq'], observed_bands[vis][band]['fracbw'] = get_mean_freq([vis],
                                                                                          observed_bands[vis][band]['spwarray'])
            elif (band == 'EVLA_C') and (len(index[0]) == 2):  # ignore pointing band
                observed_bands[vis]['n_bands'] = observed_bands[vis]['n_bands']-1
                observed_bands[vis]['bands'].remove('EVLA_C')
                continue
            elif (band == 'EVLA_C') and (len(index[0]) > 2):  # ignore pointing band
                observed_bands[vis][band] = {}
                observed_bands[vis][band]['spwarray'] = spw_names_spw[index[0]]
                indices_to_remove = np.array([])
                for i in range(len(observed_bands[vis][band]['spwarray'])):
                    meanfreq, maxfreq, minfreq, fracbw = get_mean_freq(
                        [vis], np.array([observed_bands[vis][band]['spwarray'][i]]))
                    if (meanfreq == 4.832e9) or (meanfreq == 4.960e9):
                        indices_to_remove = np.append(indices_to_remove, [i])
                observed_bands[vis][band]['spwarray'] = np.delete(
                    observed_bands[vis][band]['spwarray'],
                    indices_to_remove.astype(int))
                spwslist = observed_bands[vis][band]['spwarray'].tolist()
                spwstring = ','.join(str(spw) for spw in spwslist)
                observed_bands[vis][band]['spwstring'] = spwstring+''
                observed_bands[vis][band]['meanfreq'], observed_bands[vis][band]['maxfreq'], observed_bands[vis][
                    band]['minfreq'], observed_bands[vis][band]['fracbw'] = get_mean_freq([vis],
                                                                                          observed_bands[vis][band]['spwarray'])
            else:
                observed_bands[vis][band] = {}
                observed_bands[vis][band]['spwarray'] = spw_names_spw[index[0]]
                spwslist = observed_bands[vis][band]['spwarray'].tolist()
                spwstring = ','.join(str(spw) for spw in spwslist)
                observed_bands[vis][band]['spwstring'] = spwstring+''
                observed_bands[vis][band]['meanfreq'], observed_bands[vis][band]['maxfreq'], observed_bands[vis][
                    band]['minfreq'], observed_bands[vis][band]['fracbw'] = get_mean_freq([vis],
                                                                                          observed_bands[vis][band]['spwarray'])
    bands_match = True
    for i in range(len(vislist)):
        for j in range(i+1, len(vislist)):
            bandlist_match = (observed_bands[vislist[i]]['bands'] == observed_bands[vislist[i+1]]['bands'])
            if not bandlist_match:
                bands_match = False
    if not bands_match:
        LOG.info('WARNING: INCONSISTENT BANDS IN THE MSFILES')
    get_max_uvdist(vislist, observed_bands[vislist[0]]['bands'].copy(), observed_bands)
    return observed_bands[vislist[0]]['bands'].copy(), observed_bands


def get_telescope(vis):
    visheader = vishead(vis, mode='list', listitems=[])
    telescope = visheader['telescope'][0][0]
    if telescope == 'ALMA':
        tb.open(vis+'/ANTENNA')
        ant_diameter = np.unique(tb.getcol('DISH_DIAMETER'))[0]
        if ant_diameter == 7.0:
            telescope = 'ACA'
    return telescope


def get_dr_correction(telescope, dirty_peak, theoretical_sens, vislist):
    dirty_dynamic_range = dirty_peak/theoretical_sens
    n_dr_max = 2.5
    n_dr = 1.0
    tlimit = 2.0
    if telescope == 'ALMA':
        if dirty_dynamic_range > 150.:
            maxSciEDR = 150.0
            new_threshold = np.max([n_dr_max * theoretical_sens, dirty_peak / maxSciEDR * tlimit])
            n_dr = new_threshold/theoretical_sens
        else:
            if dirty_dynamic_range > 100.:
                n_dr = 2.5
            elif 50. < dirty_dynamic_range <= 100.:
                n_dr = 2.0
            elif 20. < dirty_dynamic_range <= 50.:
                n_dr = 1.5
            elif dirty_dynamic_range <= 20.:
                n_dr = 1.0
    if telescope == 'ACA':
        numberEBs = len(vislist)
        if numberEBs == 1:
            # single-EB 7m array datasets have limited dynamic range
            maxSciEDR = 30
            dirtyDRthreshold = 30
            n_dr_max = 2.5
        else:
            # multi-EB 7m array datasets will have better dynamic range and can be cleaned somewhat deeper
            maxSciEDR = 55
            dirtyDRthreshold = 75
            n_dr_max = 3.5

        if dirty_dynamic_range > dirtyDRthreshold:
            new_threshold = np.max([n_dr_max * theoretical_sens, dirty_peak / maxSciEDR * tlimit])
            n_dr = new_threshold/theoretical_sens
        else:
            if dirty_dynamic_range > 40.:
                n_dr = 3.0
            elif dirty_dynamic_range > 20.:
                n_dr = 2.5
            elif 10. < dirty_dynamic_range <= 20.:
                n_dr = 2.0
            elif 4. < dirty_dynamic_range <= 10.:
                n_dr = 1.5
            elif dirty_dynamic_range <= 4.:
                n_dr = 1.0
    return n_dr


def get_baseline_dist(vis):
    # Get the antenna names and offsets.

    msmd = casatools.msmetadata()

    msmd.open(vis)
    names = msmd.antennanames()
    offset = [msmd.antennaoffset(name) for name in names]
    msmd.close()
    baselines = np.array([])
    for i in range(len(offset)):
        for j in range(i+1, len(offset)):
            baseline = np.sqrt(
                (offset[i]["longitude offset"]['value'] - offset[j]["longitude offset"]['value']) ** 2 +
                (offset[i]["latitude offset"]['value'] - offset[j]["latitude offset"]['value']) ** 2)

            baselines = np.append(baselines, np.array([baseline]))
    return baselines


def get_max_uvdist(vislist, bands, band_properties):
    for band in bands:
        all_baselines = np.array([])
        for vis in vislist:
            baselines = get_baseline_dist(vis)
            all_baselines = np.append(all_baselines, baselines)
        max_baseline = np.max(all_baselines)
        baseline_75 = np.percentile(all_baselines, 75.0)
        baseline_median = np.percentile(all_baselines, 50.0)
        for vis in vislist:
            meanlam = 3.0e8/band_properties[vis][band]['meanfreq']
            max_uv_dist = max_baseline/meanlam/1000.0
            band_properties[vis][band]['maxuv'] = max_uv_dist
            band_properties[vis][band]['75thpct_uv'] = baseline_75
            band_properties[vis][band]['median_uv'] = baseline_median


def get_uv_range(band, band_properties, vislist):
    if (band == 'EVLA_C') or (band == 'EVLA_X') or (band == 'EVLA_S') or (band == 'EVLA_L'):
        n_vis = len(vislist)
        mean_max_uv = 0.0
        for vis in vislist:
            mean_max_uv += band_properties[vis][band]['maxuv']
        mean_max_uv = mean_max_uv/float(n_vis)
        min_uv = 0.05*mean_max_uv
        uvrange = '>{:0.2f}klambda'.format(min_uv)
    else:
        uvrange = ''
    return uvrange


def sanitize_string(string):
    sani_string = string.replace('-', '_').replace(' ', '_').replace('+', '_')
    sani_string = 'Target_'+sani_string
    return sani_string


def compare_beams(image1, image2):
    header_1 = imhead(image1, mode='list')
    beammajor_1 = header_1['beammajor']['value']
    beamminor_1 = header_1['beamminor']['value']
    beampa_1 = header_1['beampa']['value']

    header_2 = imhead(image2, mode='list')
    beammajor_2 = header_2['beammajor']['value']
    beamminor_2 = header_2['beamminor']['value']
    beampa_2 = header_2['beampa']['value']
    beamarea_1 = beammajor_1*beamminor_1
    beamarea_2 = beammajor_2*beamminor_2
    delta_beamarea = (beamarea_2-beamarea_1)/beamarea_1
    return delta_beamarea


def generate_weblog_old(sclib, solints, bands):
    os.system('rm -rf weblog')
    os.system('mkdir weblog')
    os.system('mkdir weblog/images')
    htmlOut = open('weblog/index.html', 'w')
    htmlOut.writelines('<html>\n')
    htmlOut.writelines('<title>SelfCal Weblog</title>\n')
    htmlOut.writelines('<head>\n')
    htmlOut.writelines('</head>\n')
    htmlOut.writelines('<body>\n')
    htmlOut.writelines('<a name="top"></a>\n')
    htmlOut.writelines('<h1>SelfCal Weblog</h1>\n')
    htmlOut.writelines('<h2>Targets:</h2>\n')
    targets = list(sclib.keys())
    for target in targets:
        htmlOut.writelines('<a href="#'+target+'">'+target+'</a><br>\n')
    htmlOut.writelines('<h2>Bands:</h2>\n')
    bands_string = ', '.join([str(elem) for elem in bands])
    htmlOut.writelines(''+bands_string+'\n')
    htmlOut.writelines('<h2>Solints to Attempt:</h2>\n')
    for band in bands:
        solints_string = ', '.join([str(elem) for elem in solints[band]])
        htmlOut.writelines('<br>'+band+': '+solints_string)

    for target in targets:
        htmlOut.writelines('<a name="'+target+'"></a>\n')
        htmlOut.writelines('<h2>'+target+' Summary</h2>\n')
        htmlOut.writelines('<a href="#top">Back to Top</a><br>\n')
        htmlOut.writelines('<a href="#'+target+'_plots">Phase vs. Time Plots</a><br>\n')
        bands_obsd = list(sclib[target].keys())

        for band in bands_obsd:
            LOG.info(f'{target} {band}')
            htmlOut.writelines('<a href="#'+target+'_'+band+'_plots">'+band+'</a><br>\n')
            htmlOut.writelines('Selfcal Success?: '+str(sclib[target][band]['SC_success'])+'<br>\n')
            keylist = sclib[target][band].keys()
            if 'Stop_Reason' not in keylist:
                htmlOut.writelines('Stop Reason: Estimated Selfcal S/N too low for solint<br><br>\n')
                if sclib[target][band]['SC_success'] == False:
                    plot_image(sanitize_string(target)+'_'+band+'_initial.image.tt0',
                               'weblog/images/'+sanitize_string(target)+'_'+band+'_initial.image.tt0.png')
                    plot_image(sanitize_string(target)+'_'+band+'_final.image.tt0',
                               'weblog/images/'+sanitize_string(target)+'_'+band+'_final.image.tt0.png')
                    htmlOut.writelines('<a href="images/' + sanitize_string(target) + '_' + band +
                                       '_initial.image.tt0.png"><img src="images/' + sanitize_string(target) + '_' + band +
                                       '_initial.image.tt0.png" ALT="pre-SC-solint image" WIDTH=400 HEIGHT=400></a>\n')
                    htmlOut.writelines('<a href="images/' + sanitize_string(target) + '_' + band +
                                       '_final.image.tt0.png"><img src="images/' + sanitize_string(target) + '_' + band +
                                       '_final.image.tt0.png" ALT="pre-SC-solint image" WIDTH=400 HEIGHT=400></a><br>\n')
                    continue
            else:
                htmlOut.writelines('Stop Reason: '+str(sclib[target][band]['Stop_Reason'])+'<br><br>\n')
                LOG.info(f"{target} {band} {sclib[target][band]['Stop_Reason']}")
                if (('Estimated_SNR_too_low_for_solint' in sclib[target][band]['Stop_Reason']) or ('Selfcal_Not_Attempted' in sclib[target][band]['Stop_Reason'])) and sclib[target][band]['final_solint'] == 'None':
                    plot_image(sanitize_string(target)+'_'+band+'_initial.image.tt0',
                               'weblog/images/'+sanitize_string(target)+'_'+band+'_initial.image.tt0.png')
                    plot_image(sanitize_string(target)+'_'+band+'_final.image.tt0',
                               'weblog/images/'+sanitize_string(target)+'_'+band+'_final.image.tt0.png')
                    htmlOut.writelines('<a href="images/' + sanitize_string(target) + '_' + band +
                                       '_initial.image.tt0.png"><img src="images/' + sanitize_string(target) + '_' + band +
                                       '_initial.image.tt0.png" ALT="pre-SC-solint image" WIDTH=400 HEIGHT=400></a>\n')
                    htmlOut.writelines('<a href="images/' + sanitize_string(target) + '_' + band +
                                       '_final.image.tt0.png"><img src="images/' + sanitize_string(target) + '_' + band +
                                       '_final.image.tt0.png" ALT="pre-SC-solint image" WIDTH=400 HEIGHT=400></a><br>\n')
                    continue

            htmlOut.writelines('Final Successful solint: '+str(sclib[target][band]['final_solint'])+'<br>\n')
            htmlOut.writelines('Final SNR: {:0.2f}'.format(
                sclib[target][band]['SNR_final'])+'<br>Initial SNR: {:0.2f}'.format(sclib[target][band]['SNR_orig'])+'<br><br>\n')
            htmlOut.writelines('Final RMS: {:0.7f}'.format(
                sclib[target][band]['RMS_final'])+' Jy/beam<br>Initial RMS: {:0.7f}'.format(sclib[target][band]['RMS_orig'])+' Jy/beam<br>\n')
            htmlOut.writelines('Final Beam: {:0.2f}"x{:0.2f}" {:0.2f} deg'.format(
                sclib[target][band]['Beam_major_final'],
                sclib[target][band]['Beam_minor_final'],
                sclib[target][band]['Beam_PA_final']) + '<br>\n')
            htmlOut.writelines('Initial Beam: {:0.2f}"x{:0.2f}" {:0.2f} deg'.format(
                sclib[target][band]['Beam_major_orig'],
                sclib[target][band]['Beam_minor_orig'],
                sclib[target][band]['Beam_PA_orig']) + '<br><br>\n')
            plot_image(sanitize_string(target)+'_'+band+'_final.image.tt0',
                       'weblog/images/'+sanitize_string(target)+'_'+band+'_final.image.tt0.png')
            image_stats = imstat(sanitize_string(target)+'_'+band+'_final.image.tt0')

            plot_image(
                sanitize_string(target) + '_' + band + '_initial.image.tt0', 'weblog/images/' + sanitize_string(target) +
                '_' + band + '_initial.image.tt0.png', min=image_stats['min'][0],
                max=image_stats['max'][0])
            os.system('rm -rf '+sanitize_string(target)+'_'+band+'_final_initial_div_final.image.tt0')
            immath(
                imagename=[sanitize_string(target) + '_' + band + '_final.image.tt0', sanitize_string(target) + '_' +
                           band + '_initial.image.tt0'],
                mode='evalexpr', expr='(IM0-IM1)/IM0', outfile=sanitize_string(target) + '_' + band +
                '_final_initial_div_final.image.tt0')
            plot_image(sanitize_string(target)+'_'+band+'_final_initial_div_final.image.tt0',
                       'weblog/images/'+sanitize_string(target)+'_'+band+'_final_initial_div_final.image.tt0.png',
                       min=-1.5, max=1.0)

            htmlOut.writelines('Initial, Final, and  Images with scales set by Final Image<br>\n')
            htmlOut.writelines('<a href="images/' + sanitize_string(target) + '_' + band +
                               '_initial.image.tt0.png"><img src="images/' + sanitize_string(target) + '_' + band +
                               '_initial.image.tt0.png" ALT="pre-SC-solint image" WIDTH=400 HEIGHT=400></a>\n')
            htmlOut.writelines('<a href="images/' + sanitize_string(target) + '_' + band +
                               '_final.image.tt0.png"><img src="images/' + sanitize_string(target) + '_' + band +
                               '_final.image.tt0.png" ALT="pre-SC-solint image" WIDTH=400 HEIGHT=400></a>\n')
            htmlOut.writelines(
                '<a href="images/' + sanitize_string(target) + '_' + band +
                '_final_initial_div_final.image.tt0.png"><img src="images/' + sanitize_string(target) + '_' + band +
                '_final_initial_div_final.image.tt0.png" ALT="pre-SC-solint image" WIDTH=400 HEIGHT=400></a><br>\n')

            if 'per_spw_stats' in sclib[target][band].keys():
                spwlist = list(sclib[target][band]['per_spw_stats'].keys())
                htmlOut.writelines('<br>Per SPW stats: <br>\n')
                for spw in spwlist:
                    htmlOut.writelines(
                        spw + ': Pre SNR: {:0.2f}, Post SNR: {:0.2f} Pre RMS: {:0.7f}, Post RMS: {:0.7f}<br>\n'.format(
                            sclib[target][band]['per_spw_stats'][spw]['SNR_orig'],
                            sclib[target][band]['per_spw_stats'][spw]['SNR_final'],
                            sclib[target][band]['per_spw_stats'][spw]['RMS_orig'],
                            sclib[target][band]['per_spw_stats'][spw]['RMS_final']))
                    if sclib[target][band]['per_spw_stats'][spw]['delta_SNR'] < 0.0:
                        htmlOut.writelines('WARNING SPW '+spw+' HAS LOWER SNR POST SELFCAL<br>')
                    if sclib[target][band]['per_spw_stats'][spw]['delta_RMS'] > 0.0:
                        htmlOut.writelines('WARNING SPW '+spw+' HAS HIGHER RMS POST SELFCAL<br>')
                    if sclib[target][band]['per_spw_stats'][spw]['delta_beamarea'] > 0.05:
                        htmlOut.writelines('WARNING SPW '+spw+' HAS A >0.05 CHANGE IN BEAM AREA POST SELFCAL<br>')

    for target in targets:
        bands_obsd = list(sclib[target].keys())
        htmlOut.writelines('<h2>'+target+' Plots</h2>\n')
        htmlOut.writelines('<a name="'+target+'_plots"></a>\n')
        for band in bands_obsd:
            htmlOut.writelines('<a name="'+target+'_'+band+'_plots"></a>\n')
            htmlOut.writelines('<h3>'+band+'</h3>\n')
            if sclib[target][band]['final_solint'] == 'None':
                final_solint_index = 0
            else:
                final_solint_index = solints[band].index(sclib[target][band]['final_solint'])

            vislist = sclib[target][band]['vislist']
            index_addition = 1
            if sclib[target][band]['final_solint'] != 'int' and sclib[target][band]['final_solint'] != 'None':
                index_addition = 2

            final_solint_to_plot = solints[band][final_solint_index+index_addition-1]
            keylist = sclib[target][band][vislist[0]].keys()
            if index_addition == 2 and final_solint_to_plot not in keylist:
                index_addition = index_addition-1

            solints_string = ''
            for i in range(final_solint_index+index_addition):
                solints_string += '<a href="#'+target+'_'+band+'_' + \
                    solints[band][i]+'_plots">'+solints[band][i]+'  </a><br>\n'

            htmlOut.writelines('<br>Solints: '+solints_string)

            for i in range(final_solint_index+index_addition):
                keylist = sclib[target][band][vislist[0]].keys()
                if solints[band][i] not in keylist:
                    continue
                htmlOut.writelines('<a name="'+target+'_'+band+'_'+solints[band][i]+'_plots"></a>\n')
                htmlOut.writelines('<h3>Solint: '+solints[band][i]+'</h3>\n')
                htmlOut.writelines('<a href="#'+target+'_'+band+'_plots">Back to Target/Band</a><br>\n')

                keylist_top = sclib[target][band].keys()
                # must select last key for pre Jan 14th runs since they only wrote pass to the last MS dictionary entry
                passed = sclib[target][band][vislist[len(vislist)-1]][solints[band][i]]['Pass']
                '''
            if (i > final_solint_index) or ('Estimated_SNR_too_low_for_solint' not in sclib[target][band]['Stop_Reason']):
               htmlOut.writelines('<h4>Passed: <font color="red">False</font></h4>\n')
            elif 'Stop_Reason' in keylist_top:
               if (i == final_solint_index) and ('Estimated_SNR_too_low_for_solint' not in sclib[target][band]['Stop_Reason']):
                    htmlOut.writelines('<h4>Passed: <font color="red">False</font></h4>\n') 
            else:
               htmlOut.writelines('<h4>Passed: <font color="blue">True</font></h4>\n')
            '''
                if passed:
                    htmlOut.writelines('<h4>Passed: <font color="blue">True</font></h4>\n')
                else:
                    htmlOut.writelines('<h4>Passed: <font color="red">False</font></h4>\n')
                htmlOut.writelines('Pre and Post Selfcal images with scales set to Post image<br>\n')
                plot_image(
                    sanitize_string(target) + '_' + band + '_' + solints[band][i] + '_' + str(i) + '_post.image.tt0',
                    'weblog/images/' + sanitize_string(target) + '_' + band + '_' + solints[band][i] + '_' + str(i) +
                    '_post.image.tt0.png')
                image_stats = imstat(sanitize_string(target)+'_'+band+'_'+solints[band][i]+'_'+str(i)+'_post.image.tt0')
                plot_image(
                    sanitize_string(target) + '_' + band + '_' + solints[band][i] + '_' + str(i) + '.image.tt0',
                    'weblog/images/' + sanitize_string(target) + '_' + band + '_' + solints[band][i] + '_' + str(i) +
                    '.image.tt0.png', min=image_stats['min'][0],
                    max=image_stats['max'][0])

                htmlOut.writelines(
                    '<a href="images/' + sanitize_string(target) + '_' + band + '_' + solints[band][i] + '_' + str(i) +
                    '.image.tt0.png"><img src="images/' + sanitize_string(target) + '_' + band + '_' + solints[band][i] +
                    '_' + str(i) + '.image.tt0.png" ALT="pre-SC-solint image" WIDTH=400 HEIGHT=400></a>\n')
                htmlOut.writelines(
                    '<a href="images/' + sanitize_string(target) + '_' + band + '_' + solints[band][i] + '_' + str(i) +
                    '_post.image.tt0.png"><img src="images/' + sanitize_string(target) + '_' + band + '_' +
                    solints[band][i] + '_' + str(i) +
                    '_post.image.tt0.png" ALT="pre-SC-solint image" WIDTH=400 HEIGHT=400></a><br>\n')
                htmlOut.writelines('Post SC SNR: {:0.2f}'.format(
                    sclib[target][band][vislist[0]][solints[band][i]]['SNR_post']) +
                    '<br>Pre SC SNR: {:0.2f}'.format(
                    sclib[target][band][vislist[0]][solints[band][i]]['SNR_pre']) + '<br><br>\n')
                htmlOut.writelines('Post SC RMS: {:0.7f}'.format(
                    sclib[target][band][vislist[0]][solints[band][i]]['RMS_post']) +
                    ' Jy/beam<br>Pre SC RMS: {:0.7f}'.format(
                    sclib[target][band][vislist[0]][solints[band][i]]['RMS_pre']) +
                    ' Jy/beam<br>\n')
                htmlOut.writelines(
                    'Post Beam: {:0.2f}"x{:0.2f}" {:0.2f} deg'.format(
                        sclib[target][band][vislist[0]][solints[band][i]]['Beam_major_post'],
                        sclib[target][band][vislist[0]][solints[band][i]]['Beam_minor_post'],
                        sclib[target][band][vislist[0]][solints[band][i]]['Beam_PA_post']) + '<br>\n')
                htmlOut.writelines(
                    'Pre Beam: {:0.2f}"x{:0.2f}" {:0.2f} deg'.format(
                        sclib[target][band][vislist[0]][solints[band][i]]['Beam_major_pre'],
                        sclib[target][band][vislist[0]][solints[band][i]]['Beam_minor_pre'],
                        sclib[target][band][vislist[0]][solints[band][i]]['Beam_PA_pre']) + '<br><br>\n')

                htmlOut.writelines('<h3>Phase vs. Time Plots:</h3>\n')

                for vis in vislist:
                    htmlOut.writelines('<h4>MS: '+vis+'</h4>\n')
                    ant_list = get_ant_list(vis)
                    gaintable = sclib[target][band][vis][solints[band][i]]['gaintable']
                    nflagged_sols, nsols = get_sols_flagged_solns(gaintable)
                    frac_flagged_sols = nflagged_sols/nsols
                    plot_ants_flagging_colored('weblog/images/plot_ants_'+gaintable+'.png', vis, gaintable)
                    htmlOut.writelines(
                        '<a href="images/plot_ants_' + gaintable + '.png"><img src="images/plot_ants_' + gaintable +
                        '.png" ALT="antenna positions with flagging plot" WIDTH=400 HEIGHT=400></a><br>\n')
                    htmlOut.writelines('N Gain solutions: {:0.0f}<br>'.format(nsols))
                    htmlOut.writelines('Flagged solutions: {:0.0f}<br>'.format(nflagged_sols))
                    htmlOut.writelines('Fraction Flagged Solutions: {:0.3f} <br>'.format(frac_flagged_sols))
                    for ant in ant_list:
                        sani_target = sanitize_string(target)
                        try:
                            plotms(
                                vis=gaintable, xaxis='time', yaxis='phase', showgui=False, xselfscale=True,
                                plotrange=[0, 0, -180, 180],
                                antenna=ant, customflaggedsymbol=True, title=ant, plotfile='weblog/images/plot_' + ant +
                                '_' + gaintable.replace('.g', '.png'),
                                overwrite=True)
                            #htmlOut.writelines('<img src="images/plot_'+ant+'_'+gaintable.replace('.g','.png')+'" ALT="gaintable antenna '+ant+'" WIDTH=200 HEIGHT=200>')
                            htmlOut.writelines(
                                '<a href="images/plot_' + ant + '_' + gaintable.replace('.g', '.png') +
                                '"><img src="images/plot_' + ant + '_' + gaintable.replace('.g', '.png') +
                                '" ALT="gaintable antenna ' + ant + '" WIDTH=200 HEIGHT=200></a>\n')
                        except:
                            continue
    htmlOut.writelines('</body>\n')
    htmlOut.writelines('</html>\n')
    htmlOut.close()


def get_sols_flagged_solns(gaintable):
    tb.open(gaintable)
    flags = tb.getcol('FLAG').squeeze()
    nsols = flags.size
    flagged_sols = np.where(flags == True)
    nflagged_sols = flagged_sols[0].size
    return nflagged_sols, nsols


def plot_ants_flagging_colored(filename, vis, gaintable):
    names, offset_x, offset_y, offsets, nflags, nunflagged, fracflagged = get_flagged_solns_per_ant(gaintable, vis)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ants_zero_flagging = np.where(fracflagged == 0.0)
    ants_lt10pct_flagging = ((fracflagged <= 0.1) & (fracflagged > 0.0)).nonzero()
    ants_lt25pct_flagging = ((fracflagged <= 0.25) & (fracflagged > 0.10)).nonzero()
    ants_lt50pct_flagging = ((fracflagged <= 0.5) & (fracflagged > 0.25)).nonzero()
    ants_lt75pct_flagging = ((fracflagged <= 0.75) & (fracflagged > 0.5)).nonzero()
    ants_gt75pct_flagging = np.where(fracflagged > 0.75)
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.scatter(offset_x[ants_zero_flagging[0]], offset_y[ants_zero_flagging[0]],
               marker='o', color='green', label='No Flagging', s=120)
    ax.scatter(offset_x[ants_lt10pct_flagging[0]], offset_y[ants_lt10pct_flagging[0]],
               marker='o', color='blue', label='<10% Flagging', s=120)
    ax.scatter(offset_x[ants_lt25pct_flagging[0]], offset_y[ants_lt25pct_flagging[0]],
               marker='o', color='yellow', label='<25% Flagging', s=120)
    ax.scatter(offset_x[ants_lt50pct_flagging[0]], offset_y[ants_lt50pct_flagging[0]],
               marker='o', color='magenta', label='<50% Flagging', s=120)
    ax.scatter(offset_x[ants_lt75pct_flagging[0]], offset_y[ants_lt75pct_flagging[0]],
               marker='o', color='cyan', label='<75% Flagging', s=120)
    ax.scatter(offset_x[ants_gt75pct_flagging[0]], offset_y[ants_gt75pct_flagging[0]],
               marker='o', color='black', label='>75% Flagging', s=120)
    ax.legend(fontsize=20)
    for i in range(len(names)):
        ax.text(offset_x[i], offset_y[i], names[i])
    ax.set_xlabel('Latitude Offset (m)', fontsize=20)
    ax.set_ylabel('Longitude Offset (m)', fontsize=20)
    ax.set_title('Antenna Positions colorized by Selfcal Flagging', fontsize=20)
    plt.savefig(filename, dpi=200.0)
    plt.close()


def plot_image(filename, outname, min=None, max=None):
    header = imhead(filename)
    size = np.max(header['shape'])
    if os.path.exists(
            filename.replace('image.tt0', 'mask')):  # if mask exists draw it as a contour, else don't use contours
        if min == None:
            imview(raster={'file': filename, 'scaling': -1, 'colorwedge': True},
                   contour={'file': filename.replace('image.tt0', 'mask'), 'levels': [1]},
                   zoom={'blc': [int(size/4), int(size/4)],
                         'trc': [int(size-size/4), int(size-size/4)]},
                   out={'file': outname, 'orient': 'landscape'})
        else:
            imview(raster={'file': filename, 'scaling': -1, 'range': [min, max], 'colorwedge': True},
                   contour={'file': filename.replace('image.tt0', 'mask'), 'levels': [1]},
                   zoom={'blc': [int(size/4), int(size/4)],
                         'trc': [int(size-size/4), int(size-size/4)]},
                   out={'file': outname, 'orient': 'landscape'})
    else:
        if min == None:
            imview(raster={'file': filename, 'scaling': -1, 'colorwedge': True},
                   zoom={'blc': [int(size/4), int(size/4)],
                         'trc': [int(size-size/4), int(size-size/4)]},
                   out={'file': outname, 'orient': 'landscape'})
        else:
            imview(raster={'file': filename, 'scaling': -1, 'range': [min, max], 'colorwedge': True},
                   zoom={'blc': [int(size/4), int(size/4)],
                         'trc': [int(size-size/4), int(size-size/4)]},
                   out={'file': outname, 'orient': 'landscape'})
    # make image square since imview makes it a strange dimension
    im = Image.open(outname)
    width, height = im.size
    if height > width:
        remainder = height-width
        trim_amount = int(remainder/2.0)
        im1 = im.crop((0, trim_amount, width-1, height-trim_amount-1))
    else:
        remainder = width-height
        trim_amount = int(remainder/2.0)
        im1 = im.crop((trim_amount, height-1, width, width-trim_amount-1, 0))
    im1.save(outname)


def get_flagged_solns_per_ant(gaintable, vis):
    # Get the antenna names and offsets.

    msmd = casatools.msmetadata()
    tb = casatools.table()

    msmd.open(vis)
    names = msmd.antennanames()
    offset = [msmd.antennaoffset(name) for name in names]
    msmd.close()

    # Calculate the mean longitude and latitude.

    mean_longitude = np.mean([offset[i]["longitude offset"]
                              ['value'] for i in range(len(names))])
    mean_latitude = np.mean([offset[i]["latitude offset"]
                             ['value'] for i in range(len(names))])

    # Calculate the offsets from the center.

    offsets = [np.sqrt((offset[i]["longitude offset"]['value'] -
                        mean_longitude)**2 + (offset[i]["latitude offset"]
                                              ['value'] - mean_latitude)**2) for i in
               range(len(names))]
    offset_y = [(offset[i]["latitude offset"]['value']) for i in
                range(len(names))]
    offset_x = [(offset[i]["longitude offset"]['value']) for i in
                range(len(names))]
    # Calculate the number of flags for each antenna.
    # gaintable='"'+gaintable+'"'
    os.system('cp -r '+gaintable.replace(' ', '\ ')+' tempgaintable.g')
    gaintable = 'tempgaintable.g'
    nflags = [tb.calc('[select from '+gaintable+' where ANTENNA1==' +
                      str(i)+' giving  [ntrue(FLAG)]]')['0'].sum() for i in
              range(len(names))]
    nunflagged = [tb.calc('[select from '+gaintable+' where ANTENNA1==' +
                          str(i)+' giving  [nfalse(FLAG)]]')['0'].sum() for i in
                  range(len(names))]
    os.system('rm -rf tempgaintable.g')
    fracflagged = np.array(nflags)/(np.array(nflags)+np.array(nunflagged))
    # Calculate a score based on those two.
    return names, np.array(offset_x), np.array(offset_y), offsets, nflags, nunflagged, fracflagged


def create_noise_histogram(imagename):
    MADtoRMS = 1.4826
    headerlist = imhead(imagename, mode='list')
    telescope = headerlist['telescope']
    beammajor = headerlist['beammajor']['value']
    beamminor = headerlist['beamminor']['value']
    beampa = headerlist['beampa']['value']
    image_stats = imstat(imagename=imagename)
    maskImage = imagename.replace('image', 'mask').replace('.tt0', '')
    residualImage = imagename.replace('image', 'residual')
    os.system('rm -rf temp.mask temp.residual')
    if os.path.exists(maskImage):
        os.system('cp -r '+maskImage + ' temp.mask')
        maskImage = 'temp.mask'
    os.system('cp -r '+residualImage + ' temp.residual')
    residualImage = 'temp.residual'
    if os.path.exists(maskImage):
        ia.close()
        ia.done()
        ia.open(residualImage)
        #ia.calcmask(maskImage+" <0.5"+"&& mask("+residualImage+")",name='madpbmask0')
        ia.calcmask("'"+maskImage+"'"+" <0.5"+"&& mask("+residualImage+")", name='madpbmask0')
        mask0Stats = ia.statistics(robust=True, axes=[0, 1])
        ia.maskhandler(op='set', name='madpbmask0')
        rms = mask0Stats['medabsdevmed'][0] * MADtoRMS
        residualMean = mask0Stats['median'][0]
        pix = np.squeeze(ia.getchunk())
        mask = np.squeeze(ia.getchunk(getmask=True))
        dimensions = mask.ndim
        if dimensions == 4:
            mask = mask[:, :, 0, 0]
        if dimensions == 3:
            mask = mask[:, :, 0]
        unmasked = (mask == True).nonzero()
        pix_unmasked = pix[unmasked]
        N, intensity = np.histogram(pix_unmasked, bins=50)
        ia.close()
        ia.done()
    elif telescope == 'ALMA':
        ia.close()
        ia.done()
        ia.open(residualImage)
        #ia.calcmask(maskImage+" <0.5"+"&& mask("+residualImage+")",name='madpbmask0')
        ia.calcmask("mask("+residualImage+")", name='madpbmask0')
        mask0Stats = ia.statistics(robust=True, axes=[0, 1])
        ia.maskhandler(op='set', name='madpbmask0')
        rms = mask0Stats['medabsdevmed'][0] * MADtoRMS
        residualMean = mask0Stats['median'][0]
        pix = np.squeeze(ia.getchunk())
        mask = np.squeeze(ia.getchunk(getmask=True))
        mask = mask[:, :, 0, 0]
        unmasked = (mask == True).nonzero()
        pix_unmasked = pix[unmasked]
        ia.close()
        ia.done()
    elif telescope == 'VLA':
        residual_stats = imstat(imagename=imagename.replace('image', 'residual'), algorithm='chauvenet')
        rms = residual_stats['rms'][0]
        ia.open(residualImage)
        pix_unmasked = np.squeeze(ia.getchunk())
        ia.close()
        ia.done()

    N, intensity = np.histogram(pix_unmasked, bins=100)
    intensity = np.diff(intensity)+intensity[:-1]
    ia.close()
    ia.done()
    os.system('rm -rf temp.mask temp.residual')
    return N, intensity, rms


def create_noise_histogram_plots(N_1, N_2, intensity_1, intensity_2, rms_1, rms_2, outfile, rms_theory=0.0):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    bins = 50.0
    max_N_1 = np.max(N_1)
    max_N_2 = np.max(N_2)
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.set_yscale('log')
    plt.ylim([0.0001, 2.0])
    ax.step(intensity_1, N_1/np.max(N_1), label='Initial Data')
    ax.step(intensity_2, N_2/np.max(N_2), label='Final Data')
    ax.plot(intensity_1, gaussian_norm(intensity_1, 0, rms_1), label='Initial Gaussian')
    ax.plot(intensity_2, gaussian_norm(intensity_2, 0, rms_2), label='Final Gaussian')
    if rms_theory != 0.0:
        ax.plot([-1.0*rms_theory, rms_theory], [0.606, 0.606], label='Theoretical Sensitivity')
    ax.legend(fontsize=20)
    ax.set_xlabel('Intensity (mJy/Beam)', fontsize=20)
    ax.set_ylabel('N', fontsize=20)
    ax.set_title('Initial vs. Final Noise (Unmasked Pixels)', fontsize=20)
    plt.savefig(outfile, dpi=200.0)
    plt.close()


def gaussian_norm(x, mean, sigma):
    gauss_dist = np.exp(-(x-mean)**2/(2*sigma**2))
    norm_gauss_dist = gauss_dist/np.max(gauss_dist)
    return norm_gauss_dist


def generate_weblog(sclib, solints, bands, outdir='weblog/'):
    from datetime import datetime
    os.system('rm -rf weblog')
    os.system('mkdir weblog')
    os.system('mkdir weblog/images')
    htmlOut = open('weblog/index.html', 'w')
    htmlOut.writelines('<html>\n')
    htmlOut.writelines('<title>SelfCal Weblog</title>\n')
    htmlOut.writelines('<head>\n')
    htmlOut.writelines('</head>\n')
    htmlOut.writelines('<body>\n')
    htmlOut.writelines('<a name="top"></a>\n')
    htmlOut.writelines('<h1>SelfCal Weblog</h1>\n')
    htmlOut.writelines('<h4>Date Executed:'+datetime.today().strftime('%Y-%m-%d')+'</h4>\n')
    htmlOut.writelines('<h2>Targets:</h2>\n')
    targets = list(sclib.keys())
    for target in targets:
        htmlOut.writelines('<a href="#'+target+'">'+target+'</a><br>\n')
    htmlOut.writelines('<h2>Bands:</h2>\n')
    bands_string = ', '.join([str(elem) for elem in bands])
    htmlOut.writelines(''+bands_string+'\n')
    htmlOut.writelines('<h2>Solints to Attempt:</h2>\n')
    for band in bands:
        solints_string = ', '.join([str(elem) for elem in solints[band]])
        htmlOut.writelines('<br>'+band+': '+solints_string)

    for target in targets:
        htmlOut.writelines('<a name="'+target+'"></a>\n')
        htmlOut.writelines('<h2>'+target+' Summary</h2>\n')
        htmlOut.writelines('<a href="#top">Back to Top</a><br>\n')
        bands_obsd = list(sclib[target].keys())

        for band in bands_obsd:
            htmlOut.writelines('<h2>'+band+'</h2>\n')
            htmlOut.writelines('<a name="'+target+'_'+band+'"></a>\n')
            htmlOut.writelines('Selfcal Success?: '+str(sclib[target][band]['SC_success'])+'<br>\n')
            keylist = sclib[target][band].keys()
            if 'Stop_Reason' not in keylist:
                htmlOut.writelines('Stop Reason: Estimated Selfcal S/N too low for solint<br><br>\n')
                if sclib[target][band]['SC_success'] == False:
                    render_summary_table(htmlOut, sclib, target, band)
                    continue
            else:
                htmlOut.writelines('Stop Reason: '+str(sclib[target][band]['Stop_Reason'])+'<br><br>\n')
                LOG.info(f"{target} {band} {sclib[target][band]['Stop_Reason']}")
                if (('Estimated_SNR_too_low_for_solint' in sclib[target][band]['Stop_Reason']) or ('Selfcal_Not_Attempted' in sclib[target][band]['Stop_Reason'])) and sclib[target][band]['final_solint'] == 'None':
                    render_summary_table(htmlOut, sclib, target, band)
                    continue
            htmlOut.writelines('Final Successful solint: '+str(sclib[target][band]['final_solint'])+'<br><br>\n')
            # Summary table for before/after SC
            render_summary_table(htmlOut, sclib, target, band)

            # Noise Summary plot
            N_initial, intensity_initial, rms_inital = create_noise_histogram(
                sanitize_string(target) + '_' + band + '_initial.image.tt0')
            N_final, intensity_final, rms_final = create_noise_histogram(
                sanitize_string(target) + '_' + band + '_final.image.tt0')
            if 'theoretical_sensitivity' in keylist:
                rms_theory = sclib[target][band]['theoretical_sensitivity']
                if rms_theory != -99.0:
                    rms_theory = sclib[target][band]['theoretical_sensitivity']
                else:
                    rms_theory = 0.0
            else:
                rms_theory = 0.0
            create_noise_histogram_plots(
                N_initial, N_final, intensity_initial, intensity_final, rms_inital, rms_final, 'weblog/images/' +
                sanitize_string(target) + '_' + band + '_noise_plot.png', rms_theory)
            htmlOut.writelines('<br>Initial vs. Final Noise Characterization<br>')
            htmlOut.writelines('<a href="images/' + sanitize_string(target) + '_' + band +
                               '_noise_plot.png"><img src="images/' + sanitize_string(target) + '_' + band +
                               '_noise_plot.png" ALT="Noise Characteristics" WIDTH=300 HEIGHT=300></a><br>\n')

            # Solint summary table
            render_selfcal_solint_summary_table(htmlOut, sclib, target, band, solints)

            # PER SPW STATS TABLE
            if 'per_spw_stats' in sclib[target][band].keys():
                render_spw_stats_summary_table(htmlOut, sclib, target, band)

    # Close main weblog file
    htmlOut.writelines('</body>\n')
    htmlOut.writelines('</html>\n')
    htmlOut.close()

    # Pages for each solint
    render_per_solint_QA_pages(sclib, solints, bands)


def render_summary_table(htmlOut, sclib, target, band):
    plot_image(sanitize_string(target)+'_'+band+'_final.image.tt0',
               'weblog/images/'+sanitize_string(target)+'_'+band+'_final.image.tt0.png')
    image_stats = imstat(sanitize_string(target)+'_'+band+'_final.image.tt0')

    plot_image(
        sanitize_string(target) + '_' + band + '_initial.image.tt0', 'weblog/images/' + sanitize_string(target) + '_' + band +
        '_initial.image.tt0.png', min=image_stats['min'][0],
        max=image_stats['max'][0])
    os.system('rm -rf ' + sanitize_string(target) + '_' + band + '_final_initial_div_final.image.tt0 ' +
              sanitize_string(target) + '_' + band + '_final_initial_div_final.temp.image.tt0')

    # Hacky way to suppress stuff outside mask in ratio images.
    immath(
        imagename=[sanitize_string(target) + '_' + band + '_final.image.tt0', sanitize_string(target) + '_' + band +
                   '_initial.image.tt0', sanitize_string(target) + '_' + band + '_final.mask'],
        mode='evalexpr', expr='((IM0-IM1)/IM0)*IM2', outfile=sanitize_string(target) + '_' + band +
        '_final_initial_div_final.temp.image.tt0')
    immath(imagename=[sanitize_string(target) + '_' + band + '_final_initial_div_final.temp.image.tt0'],
           mode='evalexpr', expr='iif(IM0==0.0,-99.0,IM0)', outfile=sanitize_string(target) + '_' + band +
           '_final_initial_div_final.image.tt0')
    plot_image(sanitize_string(target)+'_'+band+'_final_initial_div_final.image.tt0',
               'weblog/images/'+sanitize_string(target)+'_'+band+'_final_initial_div_final.image.tt0.png',
               min=-1.0, max=1.0)
    '''
         htmlOut.writelines('Initial, Final, and  Images with scales set by Final Image<br>\n')
         htmlOut.writelines('<a href="images/'+sanitize_string(target)+'_'+band+'_initial.image.tt0.png"><img src="images/'+sanitize_string(target)+'_'+band+'_initial.image.tt0.png" ALT="pre-SC-solint image" WIDTH=400 HEIGHT=400></a>\n') 
         htmlOut.writelines('<a href="images/'+sanitize_string(target)+'_'+band+'_final.image.tt0.png"><img src="images/'+sanitize_string(target)+'_'+band+'_final.image.tt0.png" ALT="pre-SC-solint image" WIDTH=400 HEIGHT=400></a>\n')
         htmlOut.writelines('<a href="images/'+sanitize_string(target)+'_'+band+'_final_initial_div_final.image.tt0.png"><img src="images/'+sanitize_string(target)+'_'+band+'_final_initial_div_final.image.tt0.png" ALT="pre-SC-solint image" WIDTH=400 HEIGHT=400></a><br>\n')
         '''
    # SUMMARY TABLE FOR FINAL IMAGES
    htmlOut.writelines('<table cellspacing="0" cellpadding="0" border="0" bgcolor="#000000">\n')
    htmlOut.writelines('	<tr>\n')
    htmlOut.writelines('		<td>\n')
    line = '<table>\n  <tr bgcolor="#ffffff">\n    <th>Data:</th>\n    '
    for data_type in ['Initial', 'Final', 'Comparison']:
        line += '<th>'+data_type+'</th>\n    '
    line += '</tr>\n'
    htmlOut.writelines(line)
    quantities = ['Image', 'intflux', 'SNR', 'SNR_NF', 'RMS', 'RMS_NF', 'Beam']
    for key in quantities:
        if key == 'Image':
            line = '<tr bgcolor="#ffffff">\n    <td>Image: </td>\n'
        if key == 'SNR':
            line = '<tr bgcolor="#ffffff">\n    <td>SNR: </td>\n'
        if key == 'intflux':
            line = '<tr bgcolor="#ffffff">\n    <td>Integrated Flux: </td>\n'
        if key == 'RMS':
            line = '<tr bgcolor="#ffffff">\n    <td>RMS: </td>\n'
        if key == 'SNR_NF':
            line = '<tr bgcolor="#ffffff">\n    <td>SNR (near-field): </td>\n'
        if key == 'RMS_NF':
            line = '<tr bgcolor="#ffffff">\n    <td>RMS (near-field): </td>\n'
        if key == 'Beam':
            line = '<tr bgcolor="#ffffff">\n    <td>Beam: </td>\n'

        for data_type in ['orig', 'final', 'comp']:
            if data_type != 'comp':
                if key == 'Image':
                    if data_type == 'orig':
                        line += '<td><a href="images/'+sanitize_string(target)+'_'+band+'_initial.image.tt0.png"><img src="images/'+sanitize_string(
                            target)+'_'+band+'_initial.image.tt0.png" ALT="pre-SC-solint image" WIDTH=400 HEIGHT=400></a> </td>\n'
                    if data_type == 'final':
                        line += '<td><a href="images/'+sanitize_string(target)+'_'+band+'_final.image.tt0.png"><img src="images/'+sanitize_string(
                            target)+'_'+band+'_final.image.tt0.png" ALT="pre-SC-solint image" WIDTH=400 HEIGHT=400></a> </td>\n'
                if key == 'SNR':
                    line += '    <td>{:0.2f} </td>\n'.format(sclib[target][band][key+'_'+data_type])
                if key == 'intflux':
                    line += '    <td>{:0.2f} +/- {:0.2f} mJy</td>\n'.format(
                        sclib[target][band][key + '_' + data_type] * 1000.0,
                        sclib[target][band]['e_' + key + '_' + data_type] * 1000.0)
                if key == 'SNR_NF':
                    line += '    <td>{:0.2f} </td>\n'.format(sclib[target][band][key+'_'+data_type])
                if key == 'RMS':
                    line += '    <td>{:0.2f} mJy/beam </td>\n'.format(sclib[target][band][key+'_'+data_type]*1000.0)
                if key == 'RMS_NF':
                    line += '    <td>{:0.2f} mJy/beam </td>\n'.format(sclib[target][band][key+'_'+data_type]*1000.0)
                if key == 'Beam':
                    line += '    <td>{:0.2f}"x{:0.2f}" {:0.2f} deg </td>\n'.format(
                        sclib[target][band][key + '_major' + '_' + data_type],
                        sclib[target][band][key + '_minor' + '_' + data_type],
                        sclib[target][band][key + '_PA' + '_' + data_type])
            else:
                if key == 'Image':
                    line += '<td><a href="images/'+sanitize_string(target)+'_'+band+'_final_initial_div_final.image.tt0.png"><img src="images/'+sanitize_string(
                        target)+'_'+band+'_final_initial_div_final.image.tt0.png" ALT="pre-SC-solint image" WIDTH=400 HEIGHT=400></a> </td>\n'
                if key == 'intflux':
                    line += '    <td>{:0.2f} </td>\n'.format(sclib[target]
                                                             [band][key+'_final']/sclib[target][band][key+'_orig'])
                if key == 'SNR':
                    line += '    <td>{:0.2f} </td>\n'.format(sclib[target]
                                                             [band][key+'_final']/sclib[target][band][key+'_orig'])
                if key == 'SNR_NF':
                    line += '    <td>{:0.2f} </td>\n'.format(sclib[target]
                                                             [band][key+'_final']/sclib[target][band][key+'_orig'])
                if key == 'RMS':
                    line += '    <td>{:0.2f} </td>\n'.format(sclib[target]
                                                             [band][key+'_orig']/sclib[target][band][key+'_final'])
                if key == 'RMS_NF':
                    line += '    <td>{:0.2f} </td>\n'.format(sclib[target]
                                                             [band][key+'_orig']/sclib[target][band][key+'_final'])
                if key == 'Beam':
                    line += '    <td>{:0.2f}</td>\n'.format(
                        (sclib[target][band][key + '_major_final'] * sclib[target][band][key + '_minor_final']) /
                        (sclib[target][band][key + '_major_orig'] * sclib[target][band][key + '_minor_orig']))
        line += '</tr>\n    '
        htmlOut.writelines(line)
    htmlOut.writelines('</table>\n')
    htmlOut.writelines('	</td>\n')
    htmlOut.writelines('	</tr>\n')
    htmlOut.writelines('</table>\n')


def render_selfcal_solint_summary_table(htmlOut, sclib, target, band, solints, plstyle=False, weblog_dir='./'):
    #  SELFCAL SUMMARY TABLE
    # figpath: figure image relative path to the report directory (which is either weblog/ or pipeline-*/html)
    vislist = sclib[target][band]['vislist']
    solint_list = solints[band]

    if plstyle:
        htmlOut.writelines('<table table-bordered table-striped table-condensed table-custom>\n')
    else:
        htmlOut.writelines('<br>Per solint stats: <br>\n')
        htmlOut.writelines('<table cellspacing="0" cellpadding="0" border="0" bgcolor="#000000">\n')

    htmlOut.writelines('	<tr>\n')
    htmlOut.writelines('		<td>\n')
    line = '<table>\n  <tr bgcolor="#ffffff">\n    <th>Solint:</th>\n    '
    for solint in solint_list:
        line += '<th>'+solint+'</th>\n    '
    line += '</tr>\n'
    htmlOut.writelines(line)
    vis_keys = list(sclib[target][band][vislist[len(vislist)-1]].keys())
    quantities = [
        'Pass', 'intflux_final', 'intflux_improvement', 'SNR_final', 'SNR_Improvement', 'SNR_NF_final',
        'SNR_NF_Improvement', 'RMS_final', 'RMS_Improvement', 'RMS_NF_final', 'RMS_NF_Improvement', 'Beam_Ratio',
        'clean_threshold', 'Plots']
    for key in quantities:
        if key == 'Pass':
            line = '<tr bgcolor="#ffffff">\n    <td>Result: </td>\n'
        if key == 'intflux_final':
            line = '<tr bgcolor="#ffffff">\n    <td>Integrated Flux: </td>\n'
        if key == 'intflux_improvement':
            line = '<tr bgcolor="#ffffff">\n    <td>Integrated Flux Change: </td>\n'
        if key == 'SNR_final':
            line = '<tr bgcolor="#ffffff">\n    <td>Dynamic Range: </td>\n'
        if key == 'SNR_Improvement':
            line = '<tr bgcolor="#ffffff">\n    <td>DR Improvement: </td>\n'
        if key == 'SNR_NF_final':
            line = '<tr bgcolor="#ffffff">\n    <td>Dynamic Range (near-field): </td>\n'
        if key == 'SNR_NF_Improvement':
            line = '<tr bgcolor="#ffffff">\n    <td>DR Improvement (near-field): </td>\n'
        if key == 'RMS_final':
            line = '<tr bgcolor="#ffffff">\n    <td>RMS: </td>\n'
        if key == 'RMS_Improvement':
            line = '<tr bgcolor="#ffffff">\n    <td>RMS Improvement: </td>\n'
        if key == 'RMS_NF_final':
            line = '<tr bgcolor="#ffffff">\n    <td>RMS (near-field): </td>\n'
        if key == 'RMS_NF_Improvement':
            line = '<tr bgcolor="#ffffff">\n    <td>RMS Improvement (near-field): </td>\n'
        if key == 'Beam_Ratio':
            line = '<tr bgcolor="#ffffff">\n    <td>Ratio of Beam Area: </td>\n'
        if key == 'clean_threshold':
            line = '<tr bgcolor="#ffffff">\n    <td>Clean Threshold: </td>\n'
        if key == 'Plots':
            line = '<tr bgcolor="#ffffff">\n    <td>Plots: </td>\n'
        for solint in solint_list:
            if solint in vis_keys:
                vis_solint_keys = sclib[target][band][vislist[len(vislist)-1]][solint].keys()
                if key == 'Pass':
                    if sclib[target][band][vislist[len(vislist)-1]][solint]['Pass'] == False:
                        line += '    <td><font color="red">{}</font> {}</td>\n'.format(
                            'Fail', sclib[target][band]['Stop_Reason'])
                    else:
                        line += '    <td><font color="blue">{}</font></td>\n'.format('Pass')
                if key == 'intflux_final':
                    line += '    <td>{:0.2f} +/- {:0.2f} mJy</td>\n'.format(
                        sclib[target][band]
                        [vislist[len(vislist) - 1]][solint]
                        ['intflux_post'] * 1000.0,
                        sclib[target][band]
                        [vislist[len(vislist) - 1]][solint]
                        ['e_intflux_post'] * 1000.0)
                if key == 'intflux_improvement':
                    line += '    <td>{:0.2f}</td>\n'.format(
                        sclib[target][band][vislist[len(vislist) - 1]][solint]['intflux_post'] /
                        sclib[target][band][vislist[len(vislist) - 1]][solint]['intflux_pre'])
                if key == 'SNR_final':
                    line += '    <td>{:0.2f}</td>\n'.format(sclib[target]
                                                            [band][vislist[len(vislist)-1]][solint]['SNR_post'])
                if key == 'SNR_Improvement':
                    line += '    <td>{:0.2f}</td>\n'.format(
                        sclib[target][band][vislist[len(vislist) - 1]][solint]['SNR_post'] /
                        sclib[target][band][vislist[len(vislist) - 1]][solint]['SNR_pre'])
                if key == 'SNR_NF_final':
                    line += '    <td>{:0.2f}</td>\n'.format(sclib[target]
                                                            [band][vislist[len(vislist)-1]][solint]['SNR_NF_post'])
                if key == 'SNR_NF_Improvement':
                    line += '    <td>{:0.2f}</td>\n'.format(
                        sclib[target][band][vislist[len(vislist) - 1]][solint]['SNR_NF_post'] /
                        sclib[target][band][vislist[len(vislist) - 1]][solint]['SNR_NF_pre'])

                if key == 'RMS_final':
                    line += '    <td>{:0.2e} mJy/bm</td>\n'.format(sclib[target]
                                                                   [band][vislist[len(vislist)-1]][solint]['RMS_post']*1000.0)
                if key == 'RMS_Improvement':
                    line += '    <td>{:0.2e}</td>\n'.format(
                        sclib[target][band][vislist[len(vislist) - 1]][solint]['RMS_pre'] /
                        sclib[target][band][vislist[len(vislist) - 1]][solint]['RMS_post'])
                if key == 'RMS_NF_final':
                    line += '    <td>{:0.2e} mJy/bm</td>\n'.format(sclib[target]
                                                                   [band][vislist[len(vislist)-1]][solint]['RMS_NF_post']*1000.0)
                if key == 'RMS_NF_Improvement':
                    line += '    <td>{:0.2e}</td>\n'.format(
                        sclib[target][band][vislist[len(vislist) - 1]][solint]['RMS_NF_pre'] /
                        sclib[target][band][vislist[len(vislist) - 1]][solint]['RMS_NF_post'])

                if key == 'Beam_Ratio':
                    line += '    <td>{:0.2e}</td>\n'.format(
                        (sclib[target][band][vislist[len(vislist) - 1]][solint]['Beam_major_post'] *
                         sclib[target][band][vislist[len(vislist) - 1]][solint]['Beam_major_post']) /
                        (sclib[target][band][vislist[len(vislist) - 1]][solint]['Beam_major_pre'] *
                         sclib[target][band][vislist[len(vislist) - 1]][solint]['Beam_major_pre']))
                if key == 'clean_threshold':
                    if key in vis_solint_keys:
                        line += '    <td>{:0.2e} mJy/bm</td>\n'.format(
                            sclib[target][band][vislist[len(vislist)-1]][solint]['clean_threshold']*1000.0)
                    else:
                        line += '    <td>Not Available</td>\n'
                if key == 'Plots':
                    line += '    <td><a href="'+weblog_dir+target+'_'+band+'_'+solint+'.html">QA Plots</a></td>\n'

            else:
                line += '    <td> - </td>\n'
        line += '</tr>\n    '
        htmlOut.writelines(line)
    htmlOut.writelines('<tr bgcolor="#ffffff">\n    <td colspan="' + str(len(solint_list) + 1) +
                       '">Flagged solutions by antenna: </td></tr>\n')
    for vis in vislist:
        line = '<tr bgcolor="#ffffff">\n    <td>'+vis+': </td>\n'
        for solint in solint_list:
            if solint in vis_keys:
                # only evaluate last gaintable not the pre-apply table
                gaintable = sclib[target][band][vis][solint]['gaintable'][
                    len(sclib[target][band][vis][solint]['gaintable']) - 1]
                line += '<td><a href="'+weblog_dir+'/images/plot_ants_'+gaintable+'.png"><img src="'+weblog_dir + \
                    'images/plot_ants_' + gaintable+'.png" ALT="antenna positions with flagging plot" WIDTH=200 HEIGHT=200></a></td>\n'
            else:
                line += '<td>-</td>\n'
        line += '</tr>\n    '
        htmlOut.writelines(line)
        for quantity in ['Nsols', 'Flagged_Sols', 'Frac_Flagged']:
            line = '<tr bgcolor="#ffffff">\n    <td>'+quantity+'</td>\n'
            for solint in solint_list:
                if solint in vis_keys:
                    # only evaluate last gaintable not the pre-apply table
                    gaintable = sclib[target][band][vis][solint]['gaintable'][
                        len(sclib[target][band][vis][solint]['gaintable']) - 1]
                    nflagged_sols, nsols = get_sols_flagged_solns(gaintable)
                    if quantity == 'Nsols':
                        line += '<td>'+str(nsols)+'</td>\n'
                    if quantity == 'Flagged_Sols':
                        line += '<td>'+str(nflagged_sols)+'</td>\n'
                    if quantity == 'Frac_Flagged':
                        line += '<td>'+'{:0.3f}'.format(nflagged_sols/nsols)+'</td>\n'
                else:
                    line += '<td>-</td>\n'
            line += '</tr>\n    '

            htmlOut.writelines(line)
    htmlOut.writelines('</table>\n')
    htmlOut.writelines('	</td>\n')
    htmlOut.writelines('	</tr>\n')
    htmlOut.writelines('</table>\n')

    return htmlOut


def render_spw_stats_summary_table(htmlOut, sclib, target, band, plstyle=False):
    spwlist = list(sclib[target][band]['per_spw_stats'].keys())

    if plstyle:
        htmlOut.writelines('<table table-bordered table-striped table-condensed table-custom>\n')
    else:
        htmlOut.writelines('<br>Per SPW stats: <br>\n')
        htmlOut.writelines('<table cellspacing="0" cellpadding="0" border="0" bgcolor="#000000">\n')
    htmlOut.writelines('	<tr>\n')
    #htmlOut.writelines('		<td>\n')
    line = '<table>\n  <tr bgcolor="#ffffff">\n    <th></th>\n    '
    for spw in spwlist:
        line += '<th>'+spw+'</th>\n    '
    line += '</tr>\n'
    htmlOut.writelines(line)

    quantities = ['bandwidth', 'effective_bandwidth', 'SNR_orig', 'SNR_final', 'RMS_orig', 'RMS_final']
    for key in quantities:
        line = '<tr bgcolor="#ffffff">\n    <td>'+key+': </td>\n'
        for spw in spwlist:
            spwkeys = sclib[target][band]['per_spw_stats'][spw].keys()
            if 'SNR' in key and key in spwkeys:
                line += '    <td>{:0.2f}</td>\n'.format(sclib[target][band]['per_spw_stats'][spw][key])
            if 'RMS' in key and key in spwkeys:
                line += '    <td>{:0.2e} mJy/bm</td>\n'.format(sclib[target][band]['per_spw_stats'][spw][key]*1000.0)
            if 'bandwidth' in key and key in spwkeys:
                line += '    <td>{:0.4f} GHz</td>\n'.format(sclib[target][band]['per_spw_stats'][spw][key])
        line += '</tr>\n    '
        htmlOut.writelines(line)
    htmlOut.writelines('</table>\n')
    htmlOut.writelines('	</td>\n')
    htmlOut.writelines('	</tr>\n')
    htmlOut.writelines('</table>\n')
    for spw in spwlist:
        spwkeys = sclib[target][band]['per_spw_stats'][spw].keys()
        if 'delta_SNR' in spwkeys or 'delta_RMS' in spwkeys or 'delta_beamarea' in spwkeys:
            if sclib[target][band]['per_spw_stats'][spw]['delta_SNR'] < 0.0:
                htmlOut.writelines('WARNING SPW '+spw+' HAS LOWER SNR POST SELFCAL<br>\n')
            if sclib[target][band]['per_spw_stats'][spw]['delta_RMS'] > 0.0:
                htmlOut.writelines('WARNING SPW '+spw+' HAS HIGHER RMS POST SELFCAL<br>\n')
            if sclib[target][band]['per_spw_stats'][spw]['delta_beamarea'] > 0.05:
                htmlOut.writelines('WARNING SPW '+spw+' HAS A >0.05 CHANGE IN BEAM AREA POST SELFCAL<br>\n')
    return htmlOut


def render_per_solint_QA_pages(sclib, solints, bands):
  # Per Solint pages
    targets = list(sclib.keys())
    for target in targets:
        bands_obsd = list(sclib[target].keys())
        for band in bands_obsd:
            if sclib[target][band]['final_solint'] == 'None':
                final_solint_index = 0
            else:
                final_solint_index = solints[band].index(sclib[target][band]['final_solint'])

            vislist = sclib[target][band]['vislist']
            index_addition = 1
            if sclib[target][band]['final_solint'] != 'inf_ap' and sclib[target][band]['final_solint'] != 'None':
                index_addition = 2

            final_solint_to_plot = solints[band][final_solint_index+index_addition-1]
            keylist = sclib[target][band][vislist[0]].keys()
            if index_addition == 2 and final_solint_to_plot not in keylist:
                index_addition = index_addition-1

            # for i in range(final_solint_index+index_addition):
            for i in range(len(solints[band])):

                if solints[band][i] not in keylist:
                    continue
                htmlOutSolint = open('weblog/'+target+'_'+band+'_'+solints[band][i]+'.html', 'w')
                htmlOutSolint.writelines('<html>\n')
                htmlOutSolint.writelines('<title>SelfCal Weblog</title>\n')
                htmlOutSolint.writelines('<head>\n')
                htmlOutSolint.writelines('</head>\n')
                htmlOutSolint.writelines('<body>\n')
                htmlOutSolint.writelines('<a name="top"></a>\n')
                htmlOutSolint.writelines('<h2>'+target+' Plots</h2>\n')
                htmlOutSolint.writelines('<h2>'+band+'</h2>\n')
                htmlOutSolint.writelines('<h2>Targets:</h2>\n')
                keylist = sclib[target][band][vislist[0]].keys()
                solints_string = ''
                for j in range(final_solint_index+index_addition):
                    if solints[band][j] not in keylist:
                        continue
                    solints_string += '<a href="'+target+'_'+band+'_' + \
                        solints[band][j]+'.html">'+solints[band][j]+'  </a><br>\n'
                htmlOutSolint.writelines('<br>Solints: '+solints_string)

                htmlOutSolint.writelines('<h3>Solint: '+solints[band][i]+'</h3>\n')
                keylist_top = sclib[target][band].keys()
                htmlOutSolint.writelines('<a href="index.html#'+target+'_'+band+'">Back to Main Target/Band</a><br>\n')

                # must select last key for pre Jan 14th runs since they only wrote pass to the last MS dictionary entry
                passed = sclib[target][band][vislist[len(vislist)-1]][solints[band][i]]['Pass']
                '''
            if (i > final_solint_index) or ('Estimated_SNR_too_low_for_solint' not in sclib[target][band]['Stop_Reason']):
               htmlOut.writelines('<h4>Passed: <font color="red">False</font></h4>\n')
            elif 'Stop_Reason' in keylist_top:
               if (i == final_solint_index) and ('Estimated_SNR_too_low_for_solint' not in sclib[target][band]['Stop_Reason']):
                    htmlOut.writelines('<h4>Passed: <font color="red">False</font></h4>\n') 
            else:
               htmlOut.writelines('<h4>Passed: <font color="blue">True</font></h4>\n')
            '''
                if passed:
                    htmlOutSolint.writelines('<h4>Passed: <font color="blue">True</font></h4>\n')
                else:
                    htmlOutSolint.writelines('<h4>Passed: <font color="red">False</font></h4>\n')

                htmlOutSolint.writelines('Pre and Post Selfcal images with scales set to Post image<br>\n')
                plot_image(
                    sanitize_string(target) + '_' + band + '_' + solints[band][i] + '_' + str(i) + '_post.image.tt0',
                    'weblog/images/' + sanitize_string(target) + '_' + band + '_' + solints[band][i] + '_' + str(i) +
                    '_post.image.tt0.png')
                image_stats = imstat(sanitize_string(target)+'_'+band+'_'+solints[band][i]+'_'+str(i)+'_post.image.tt0')
                plot_image(
                    sanitize_string(target) + '_' + band + '_' + solints[band][i] + '_' + str(i) + '.image.tt0',
                    'weblog/images/' + sanitize_string(target) + '_' + band + '_' + solints[band][i] + '_' + str(i) +
                    '.image.tt0.png', min=image_stats['min'][0],
                    max=image_stats['max'][0])

                htmlOutSolint.writelines(
                    '<a href="images/' + sanitize_string(target) + '_' + band + '_' + solints[band][i] + '_' + str(i) +
                    '.image.tt0.png"><img src="images/' + sanitize_string(target) + '_' + band + '_' + solints[band][i] +
                    '_' + str(i) + '.image.tt0.png" ALT="pre-SC-solint image" WIDTH=400 HEIGHT=400></a>\n')
                htmlOutSolint.writelines(
                    '<a href="images/' + sanitize_string(target) + '_' + band + '_' + solints[band][i] + '_' + str(i) +
                    '_post.image.tt0.png"><img src="images/' + sanitize_string(target) + '_' + band + '_' +
                    solints[band][i] + '_' + str(i) +
                    '_post.image.tt0.png" ALT="pre-SC-solint image" WIDTH=400 HEIGHT=400></a><br>\n')
                htmlOutSolint.writelines('Post SC SNR: {:0.2f}'.format(
                    sclib[target][band][vislist[0]][solints[band][i]]['SNR_post']) +
                    '<br>Pre SC SNR: {:0.2f}'.format(
                    sclib[target][band][vislist[0]][solints[band][i]]['SNR_pre']) +
                    '<br><br>\n')
                htmlOutSolint.writelines(
                    'Post SC RMS: {:0.7f}'.format(sclib[target][band][vislist[0]][solints[band][i]]['RMS_post']) +
                    ' Jy/beam<br>Pre SC RMS: {:0.7f}'.format(
                        sclib[target][band][vislist[0]][solints[band][i]]['RMS_pre']) + ' Jy/beam<br>\n')
                htmlOutSolint.writelines(
                    'Post Beam: {:0.2f}"x{:0.2f}" {:0.2f} deg'.format(
                        sclib[target][band][vislist[0]][solints[band][i]]['Beam_major_post'],
                        sclib[target][band][vislist[0]][solints[band][i]]['Beam_minor_post'],
                        sclib[target][band][vislist[0]][solints[band][i]]['Beam_PA_post']) + '<br>\n')
                htmlOutSolint.writelines(
                    'Pre Beam: {:0.2f}"x{:0.2f}" {:0.2f} deg'.format(
                        sclib[target][band][vislist[0]][solints[band][i]]['Beam_major_pre'],
                        sclib[target][band][vislist[0]][solints[band][i]]['Beam_minor_pre'],
                        sclib[target][band][vislist[0]][solints[band][i]]['Beam_PA_pre']) + '<br><br>\n')

                if solints[band][i] == 'inf_EB':
                    htmlOutSolint.writelines('<h3>Phase vs. Frequency Plots:</h3>\n')
                else:
                    htmlOutSolint.writelines('<h3>Phase vs. Time Plots:</h3>\n')
                for vis in vislist:
                    htmlOutSolint.writelines('<h4>MS: '+vis+'</h4>\n')
                    ant_list = get_ant_list(vis)
                    gaintable = sclib[target][band][vis][solints[band][i]]['gaintable'][
                        len(sclib[target][band][vis][solints[band][i]]['gaintable']) - 1]
                    LOG.info('******************'+gaintable+'***************')
                    nflagged_sols, nsols = get_sols_flagged_solns(gaintable)
                    frac_flagged_sols = nflagged_sols/nsols
                    plot_ants_flagging_colored('weblog/images/plot_ants_'+gaintable+'.png', vis, gaintable)
                    htmlOutSolint.writelines(
                        '<a href="images/plot_ants_' + gaintable + '.png"><img src="images/plot_ants_' + gaintable +
                        '.png" ALT="antenna positions with flagging plot" WIDTH=400 HEIGHT=400></a><br>\n')
                    htmlOutSolint.writelines('N Gain solutions: {:0.0f}<br>'.format(nsols))
                    htmlOutSolint.writelines('Flagged solutions: {:0.0f}<br>'.format(nflagged_sols))
                    htmlOutSolint.writelines('Fraction Flagged Solutions: {:0.3f} <br><br>'.format(frac_flagged_sols))
                    if solints[band][i] == 'inf_EB':
                        if 'fallback' in sclib[target][band][vis][solints[band][i]].keys():
                            if sclib[target][band][vis][solints[band][i]]['fallback'] == '':
                                fallback_mode = 'None'
                            if sclib[target][band][vis][solints[band][i]]['fallback'] == 'combinespw':
                                fallback_mode = 'Combine SPW'
                            if sclib[target][band][vis][solints[band][i]]['fallback'] == 'spwmap':
                                fallback_mode = 'SPWMAP'
                            htmlOutSolint.writelines(
                                '<h4>Fallback Mode: <font color="red">'+fallback_mode+'</font></h4>\n')
                    htmlOutSolint.writelines(
                        '<h4>Spwmapping: ['+' '.join(map(str, sclib[target][band][vis][solints[band][i]]['spwmap']))+']</h4>\n')

                    for ant in ant_list:
                        sani_target = sanitize_string(target)
                        if solints[band][i] == 'inf_EB':
                            xaxis = 'frequency'
                        else:
                            xaxis = 'time'
                        if 'ap' in solints[band][i]:
                            yaxis = 'amp'
                            plotrange = [0, 0, 0, 2.0]
                        else:
                            yaxis = 'phase'
                            plotrange = [0, 0, -180, 180]
                        try:
                            plotms(gridrows=2, plotindex=0, rowindex=0, vis=gaintable, xaxis=xaxis, yaxis=yaxis,
                                   showgui=False, xselfscale=True, plotrange=plotrange, antenna=ant,
                                   customflaggedsymbol=True, title=ant + ' phase', plotfile='weblog/images/plot_' + ant +
                                   '_' + gaintable.replace('.g', '.png'),
                                   overwrite=True, clearplots=True)
                            plotms(gridrows=2, rowindex=1, plotindex=1, vis=gaintable, xaxis=xaxis, yaxis='SNR',
                                   showgui=False, xselfscale=True, antenna=ant, customflaggedsymbol=True, title=ant +
                                   ' SNR', plotfile='weblog/images/plot_' + ant + '_' + gaintable.replace('.g', '.png'),
                                   overwrite=True, clearplots=False)
                            #htmlOut.writelines('<img src="images/plot_'+ant+'_'+gaintable.replace('.g','.png')+'" ALT="gaintable antenna '+ant+'" WIDTH=200 HEIGHT=200>')
                            htmlOutSolint.writelines(
                                '<a href="images/plot_' + ant + '_' + gaintable.replace('.g', '.png') +
                                '"><img src="images/plot_' + ant + '_' + gaintable.replace('.g', '.png') +
                                '" ALT="gaintable antenna ' + ant + '" WIDTH=200 HEIGHT=200></a>\n')
                        except:
                            continue
                htmlOutSolint.writelines('</body>\n')
                htmlOutSolint.writelines('</html>\n')
                htmlOutSolint.close()


def importdata(vislist, all_targets, telescope):
    listdict = collect_listobs_per_vis(vislist)
    scantimesdict, integrationsdict, integrationtimesdict, integrationtimes, n_spws, minspw, spwsarray = fetch_scan_times(
        vislist, all_targets, listdict)
    spwslist = spwsarray.tolist()
    spwstring = ','.join(str(spw) for spw in spwslist)

    if 'VLA' in telescope:
        bands, band_properties = get_VLA_bands(vislist)

    if telescope == 'ALMA' or telescope == 'ACA':
        bands, band_properties = get_ALMA_bands(vislist, spwstring, spwsarray)

    scantimesdict = {}
    scanstartsdict = {}
    scanendsdict = {}
    integrationsdict = {}
    integrationtimesdict
    mosaic_field_dict = {}
    bands_to_remove = []

    for band in bands:
        LOG.info(band)
        scantimesdict_temp, scanstartsdict_temp, scanendsdict_temp, integrationsdict_temp, integrationtimesdict_temp,\
            integrationtimes_temp, n_spws_temp, minspw_temp, spwsarray_temp, mosaic_field_temp = fetch_scan_times_band_aware(vislist, all_targets, listdict, band_properties, band)

        scantimesdict[band] = scantimesdict_temp.copy()
        scanstartsdict[band] = scanstartsdict_temp.copy()
        scanendsdict[band] = scanendsdict_temp.copy()
        integrationsdict[band] = integrationsdict_temp.copy()
        mosaic_field_dict[band] = mosaic_field_temp.copy()
        integrationtimesdict[band] = integrationtimesdict_temp.copy()
        if n_spws_temp == -99:
            for vis in vislist:
                band_properties[vis].pop(band)
                band_properties[vis]['bands'].remove(band)
                LOG.info('Removing '+band+' bands from list due to no observations')
            bands_to_remove.append(band)
        for vis in vislist:
            for target in all_targets:
                check_target = len(integrationsdict[band][vis][target])
                if check_target == 0:
                    integrationsdict[band][vis].pop(target)
                    integrationtimesdict[band][vis].pop(target)
                    scantimesdict[band][vis].pop(target)
                    scanstartsdict[band][vis].pop(target)
                    scanendsdict[band][vis].pop(target)

                    mosaic_field_dict[band].pop(target)
    if len(bands_to_remove) > 0:
        for delband in bands_to_remove:
            bands.remove(delband)

    return listdict, bands, band_properties, scantimesdict, scanstartsdict, scanendsdict, integrationsdict, integrationtimesdict, spwslist, spwstring, spwsarray, mosaic_field_dict


def flag_spectral_lines(vislist, all_targets, spwsarray):
    LOG.info("# cont.dat file found, flagging lines identified by the pipeline.")
    for vis in vislist:
        if not os.path.exists(vis+".flagversions/flags.before_line_flags"):
            flagmanager(vis=vis, mode='save', versionname='before_line_flags',
                        comment='Flag states at start of reduction')
        else:
            flagmanager(vis=vis, mode='restore', versionname='before_line_flags')
        for target in all_targets:
            contdot_dat_flagchannels_string = flagchannels_from_contdotdat(vis, target, spwsarray)
            flagdata(vis=vis, mode='manual', spw=contdot_dat_flagchannels_string[:-2], flagbackup=False, field=target)


def split_to_selfcal_ms(vislist, band_properties, bands, spectral_average):
    for vis in vislist:
        os.system('rm -rf '+vis.replace('.ms', '.selfcal.ms')+'*')
        spwstring = ''
        chan_widths = []
        if spectral_average:
            initweights(vis=vis, wtmode='weight', dowtsp=True)  # initialize channelized weights
            for band in bands:
                desiredWidth = get_desired_width(band_properties[vis][band]['meanfreq'])
                LOG.info(f'{band} {desiredWidth}')
                widtharray, bwarray, nchanarray = get_spw_chanwidths(vis, band_properties[vis][band]['spwarray'])
                band_properties[vis][band]['chan_widths'] = get_spw_chanavg(
                    vis, widtharray, bwarray, nchanarray, desiredWidth=desiredWidth)
                LOG.info(band_properties[vis][band]['chan_widths'])
                chan_widths = chan_widths+band_properties[vis][band]['chan_widths'].astype('int').tolist()
                if spwstring == '':
                    spwstring = band_properties[vis][band]['spwstring']+''
                else:
                    spwstring = spwstring+','+band_properties[vis][band]['spwstring']
            mstransform(vis=vis, chanaverage=True, chanbin=chan_widths, spw=spwstring,
                        outputvis=vis.replace('.ms', '.selfcal.ms'), datacolumn='data', reindex=False)
            initweights(vis=vis, wtmode='delwtsp')  # remove channelized weights
        else:
            mstransform(vis=vis, outputvis=vis.replace('.ms', '.selfcal.ms'), datacolumn='data', reindex=False)


def check_mosaic(vislist, target):
    msmd.open(vis[0])
    fieldid = msmd.fieldsforname(field)
    msmd.done()
    if len(fieldid) > 1:
        mosaic = True
    else:
        mosaic = False
    return mosaic


def get_phasecenter(vis, field):
    msmd.open(vis)
    fieldid = msmd.fieldsforname(field)
    ra_phasecenter_arr = np.zeros(len(fieldid))
    dec_phasecenter_arr = np.zeros(len(fieldid))
    for i in range(len(fieldid)):
        phasecenter = msmd.phasecenter(fieldid[i])
        ra_phasecenter_arr[i] = phasecenter['m0']['value']
        dec_phasecenter_arr[i] = phasecenter['m1']['value']

    msmd.done()

    ra_phasecenter = np.median(ra_phasecenter_arr)
    dec_phasecenter = np.median(dec_phasecenter_arr)
    phasecenter_string = 'ICRS {:0.8f}rad {:0.8f}rad '.format(ra_phasecenter, dec_phasecenter)
    return phasecenter_string


def get_flagged_solns_per_spw(spwlist, gaintable):
    # Get the antenna names and offsets.
    msmd = casatools.msmetadata()
    tb = casatools.table()

    # Calculate the number of flags for each spw.
    # gaintable='"'+gaintable+'"'
    os.system('cp -r '+gaintable.replace(' ', '\ ')+' tempgaintable.g')
    gaintable = 'tempgaintable.g'
    nflags = [tb.calc('[select from '+gaintable+' where SPECTRAL_WINDOW_ID==' +
                      spwlist[i]+' giving  [ntrue(FLAG)]]')['0'].sum() for i in
              range(len(spwlist))]
    nunflagged = [tb.calc('[select from '+gaintable+' where SPECTRAL_WINDOW_ID==' +
                          spwlist[i]+' giving  [nfalse(FLAG)]]')['0'].sum() for i in
                  range(len(spwlist))]
    os.system('rm -rf tempgaintable.g')
    fracflagged = np.array(nflags)/(np.array(nflags)+np.array(nunflagged))
    # Calculate a score based on those two.
    return nflags, nunflagged, fracflagged


def analyze_inf_EB_flagging(selfcal_library, band, spwlist, gaintable, vis, target, spw_combine_test_gaintable):
    # if more than two antennas are fully flagged relative to the combinespw results, fallback to combinespw
    max_flagged_ants_combspw = 2.0
    # if only a single (or few) spw(s) has flagging, allow at most this number of antennas to be flagged before mapping
    max_flagged_ants_spwmap = 1.0
    fallback = ''
    map_index = -1
    min_spwmap_bw = 0.0
    spwmap = [False]*len(spwlist)
    nflags, nunflagged, fracflagged = get_flagged_solns_per_spw(spwlist, gaintable)
    nflags_spwcomb, nunflagged_spwcomb, fracflagged_spwcomb = get_flagged_solns_per_spw(
        spwlist[0], spw_combine_test_gaintable)
    eff_bws = np.zeros(len(spwlist))
    total_bws = np.zeros(len(spwlist))
    for i in range(len(spwlist)):
        eff_bws[i] = selfcal_library[target][band]['per_spw_stats'][spwlist[i]]['effective_bandwidth']
        total_bws[i] = selfcal_library[target][band]['per_spw_stats'][spwlist[i]]['bandwidth']
    minimum_flagged_ants_per_spw = np.min(nflags)/2.0
    # account for the fact that some antennas might be completely flagged and give
    minimum_flagged_ants_spwcomb = np.min(nflags_spwcomb)/2.0
    # the impression of a lot of flagging
    maximum_flagged_ants_per_spw = np.max(nflags)/2.0
    delta_nflags = np.array(nflags)/2.0-minimum_flagged_ants_spwcomb  # minimum_flagged_ants_per_spw

    # if there are more than 3 flagged antennas for all spws (minimum_flagged_ants_spwcomb, fallback to doing spw combine for inf_EB fitting
    # use the spw combine number of flagged ants to set the minimum otherwise could misinterpret fully flagged antennas for flagged solutions
    # captures case where no one spws has sufficient S/N, only together do they have enough
    if (minimum_flagged_ants_per_spw-minimum_flagged_ants_spwcomb) > max_flagged_ants_combspw:
        fallback = 'combinespw'

    # if certain spws have more than max_flagged_ants_spwmap flagged solutions that the least flagged spws, set those to spwmap
    for i in range(len(spwlist)):
        if np.min(delta_nflags[i]) > max_flagged_ants_spwmap:
            fallback = 'spwmap'
            spwmap[i] = True
            if total_bws[i] > min_spwmap_bw:
                min_spwmap_bw = total_bws[i]
    # also spwmap spws with similar bandwidths to the others that are getting mapped, avoid low S/N solutions
    if fallback == 'spwmap':
        for i in range(len(spwlist)):
            if total_bws[i] <= min_spwmap_bw:
                spwmap[i] = True
        if all(spwmap):
            fallback = 'combinespw'
    # want the widest bandwidth window that also has the minimum flags to use for spw mapping
    applycal_spwmap = []
    if fallback == 'spwmap':
        minflagged_index = (np.array(nflags)/2.0 == minimum_flagged_ants_per_spw).nonzero()
        max_bw_index = (eff_bws == np.max(eff_bws[minflagged_index[0]])).nonzero()
        max_bw_min_flags_index = np.intersect1d(minflagged_index[0], max_bw_index[0])
        # if len(max_bw_min_flags_index) > 1:
        # don't need the conditional since this works with array lengths of 1
        map_index = max_bw_min_flags_index[np.argmax(eff_bws[max_bw_min_flags_index])]
        # else:
        #   map_index=max_bw_min_flags_index[0]

        # make spwmap list that first maps everything to itself, need max spw to make that list
        maxspw = np.max(selfcal_library[target][band][vis]['spwsarray']+1)
        applycal_spwmap_int_list = list(np.arange(maxspw))
        for i in range(len(applycal_spwmap_int_list)):
            applycal_spwmap.append(applycal_spwmap_int_list[i])

        # replace the elements that require spwmapping (spwmap[i] == True
        for i in range(len(spwmap)):
            LOG.info(f'{i} {spwlist[i]} {spwmap[i]}')
            if spwmap[i]:
                applycal_spwmap[int(spwlist[i])] = int(spwlist[map_index])

    return fallback, map_index, spwmap, applycal_spwmap


LOG = get_selfcal_logger(__name__)
