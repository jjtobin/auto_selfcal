import numpy as np
import numpy 
import scipy.stats
import scipy.signal
import math
import os
import scipy.cluster.hierarchy as hc
import copy

import casatools
from casaplotms import plotms
from casatasks import *
from casatools import image, imager
from casatools import msmetadata as msmdtool
from casatools import table as tbtool
from casatools import ms as mstool
from PIL import Image

ms = mstool()
tb = tbtool()
msmd = msmdtool()
ia = image()
im = imager()

def tclean_wrapper(selfcal_library, imagename, band, field_str, telescope='undefined', scales=[0], smallscalebias = 0.6, mask = '',\
                   nsigma=5.0, interactive = False, robust = 0.5, gain = 0.1, niter = 50000,\
                   cycleniter = 300, uvtaper = [], savemodel = 'none',gridder='standard', sidelobethreshold=3.0,smoothfactor=1.0,noisethreshold=5.0,\
                   lownoisethreshold=1.5,parallel=False,cyclefactor=3,threshold='0.0Jy',phasecenter='',\
                   startmodel='',pblimit=0.1,pbmask=0.1,field='',datacolumn='',nfrms_multiplier=1.0, \
                   savemodel_only=False, resume=False, spw='all', image_mosaic_fields_separately=True, \
                   store_threshold=''):
    """
    Wrapper for tclean with keywords set to values desired for the Large Program imaging
    See the CASA 6.1.1 documentation for tclean to get the definitions of all the parameters
    """
    msmd.open(selfcal_library['vislist'][0])
    fieldid=msmd.fieldsforname(field)
    msmd.done()
    tb.open(selfcal_library['vislist'][0]+'/FIELD')
    try:
       ephem_column=tb.getcol('EPHEMERIS_ID')
       tb.close()
       if ephem_column[fieldid[0]] !=-1:
          phasecenter='TRACKFIELD'
    except:
       tb.close()
       phasecenter=''

    if selfcal_library['obstype']=='mosaic' and phasecenter != 'TRACKFIELD':
       phasecenter=get_phasecenter(selfcal_library['vislist'][0],selfcal_library)

    print('NF RMS Multiplier: ', nfrms_multiplier)
    # Minimize out the nfrms_multiplier at 1.
    nfrms_multiplier = max(nfrms_multiplier, 1.0)

    baselineThresholdALMA = 400.0

    if mask == '':
       if selfcal_library['usermask'] != '':
           mask = selfcal_library['usermask']
           usemask = 'user'
       else:
           usemask='auto-multithresh'
    else:
       usemask='user'

    if telescope=='ALMA':
       if selfcal_library['75thpct_uv'] > baselineThresholdALMA:
          fastnoise = True
       else:
          fastnoise = False
       sidelobethreshold=2.5
       smoothfactor=1.0
       noisethreshold=5.0*nfrms_multiplier
       lownoisethreshold=1.5*nfrms_multiplier
       cycleniter=-1
       negativethreshold = 0.0
       dogrowprune = True
       minpercentchange = 1.0
       growiterations = 75
       minbeamfrac = 0.3
       #cyclefactor=1.0
       if selfcal_library['75thpct_uv'] > 2000.0:
          sidelobethreshold=2.0

       if selfcal_library['75thpct_uv'] < 300.0:
          sidelobethreshold=2.0
          smoothfactor=1.0
          noisethreshold=4.25*nfrms_multiplier
          lownoisethreshold=1.5*nfrms_multiplier

       if selfcal_library['75thpct_uv'] < baselineThresholdALMA:
          sidelobethreshold = 2.0

    if telescope=='ACA':
       sidelobethreshold=1.25
       smoothfactor=1.0
       noisethreshold=5.0*nfrms_multiplier
       lownoisethreshold=2.0*nfrms_multiplier
       cycleniter=-1
       fastnoise=False
       negativethreshold = 0.0
       dogrowprune = True
       minpercentchange = 1.0
       growiterations = 75
       minbeamfrac = 0.1
       #cyclefactor=1.0

    elif 'VLA' in telescope:
       fastnoise=True
       sidelobethreshold=2.0
       smoothfactor=1.0
       noisethreshold=5.0*nfrms_multiplier
       lownoisethreshold=1.5*nfrms_multiplier
       if selfcal_library['obstype']!='mosaic':
           pblimit=-0.1
       cycleniter=-1
       negativethreshold = 0.0
       dogrowprune = True
       minpercentchange = 1.0
       growiterations = 75
       minbeamfrac = 0.3
       #cyclefactor=3.0
       pbmask=0.0
    wprojplanes=1
    if band=='EVLA_L' or band =='EVLA_S':
       gridder='wproject'
       wplanes=384 # normalized to S-band A-config
       #scale by 75th percentile uv distance divided by A-config value
       wplanes=wplanes * selfcal_library['75thpct_uv']/20000.0
       if band=='EVLA_L':
          wplanes=wplanes*2.0 # compensate for 1.5 GHz being 2x longer than 3 GHz


       wprojplanes=int(wplanes)
    if (band=='EVLA_L' or band =='EVLA_S') and selfcal_library['obstype']=='mosaic':
       print('WARNING DETECTED VLA L- OR S-BAND MOSAIC; WILL USE gridder="mosaic" IGNORING W-TERM')
    if selfcal_library['obstype']=='mosaic':
       gridder='mosaic'
    else:
       if gridder !='wproject':
          gridder='standard' 

    if spw == 'all':
        vlist = selfcal_library['vislist']
        spws_per_vis = selfcal_library['spws_per_vis']
        nterms = selfcal_library['nterms']
    else:
        vlist = [vis for vis in selfcal_library['vislist'] if vis in selfcal_library['spw_map'][spw]]
        spws_per_vis = [str(selfcal_library['spw_map'][spw][vis]) for vis in vlist]
        nterms = 1

    if nterms == 1:
       reffreq = ''
    else:
       reffreq = selfcal_library['reffreq']

    if "theoretical" in threshold:
        dr_mod=1.0
        if telescope =='ALMA' or telescope =='ACA':
           sensitivity=get_sensitivity(vlist,selfcal_library,field,virtual_spw=spw,
                   imsize=selfcal_library['imsize'],cellsize=selfcal_library['cellsize'])
           dr_mod=get_dr_correction(telescope,selfcal_library['SNR_dirty']*selfcal_library['RMS_dirty'],sensitivity,vlist)
           sensitivity_nomod=sensitivity.copy()
           print('DR modifier: ',dr_mod, 'SPW: ',spw)

           sensitivity=sensitivity*dr_mod   # apply DR modifier
           if (band =='Band_9' or band == 'Band_10') and spw != 'all':   # adjust for DSB noise increase
               sensitivity=4.0*sensitivity   #*4.0  might be unnecessary with DR mods

           selfcal_library['theoretical_sensitivity']=sensitivity_nomod
           for fid in selfcal_library['sub-fields']:
               selfcal_library[fid]['theoretical_sensitivity']=sensitivity_nomod
        else:
           sensitivity=0.0
           selfcal_library['theoretical_sensitivity']=-99.0
           for fid in selfcal_library['sub-fields']:
               selfcal_library[fid]['theoretical_sensitivity']=-99.0

        if threshold == "theoretical_with_drmod":
            threshold = str(4.0*sensitivity)+'Jy'
        else:
            if spw == 'all':
                sensitivity_scale_factor = 1.0
            else:
                sensitivity_agg=get_sensitivity(vlist,selfcal_library,field,virtual_spw=spw,imsize=selfcal_library['imsize'],
                        cellsize=selfcal_library['cellsize'])
                sensitivity_scale_factor=selfcal_library['RMS_NF_curr']/sensitivity_agg
            threshold = str(4.0*sensitivity_nomod*sensitivity_scale_factor)+'Jy'

    if threshold != '0.0Jy':
       nsigma=0.0

    if nsigma != 0.0:
       if nsigma*nfrms_multiplier*0.66 > nsigma:
          nsigma=nsigma*nfrms_multiplier*0.66

    if gridder=='mosaic' and startmodel!='':
       parallel=False
    if not savemodel_only:
        if not resume:
            for ext in ['.image*', '.mask', '.model*', '.pb*', '.psf*', '.residual*', '.sumwt*','.gridwt*']:
                os.system('rm -rf '+ imagename + ext)
        tclean_return = tclean(vis=vlist, 
               imagename = imagename, 
               field=field_str,
               specmode = 'mfs', 
               deconvolver = 'mtmfs',
               scales = scales, 
               gridder=gridder,
               weighting='briggs', 
               robust = robust,
               gain = gain,
               imsize = selfcal_library['imsize'],
               cell = selfcal_library['cellsize'], 
               smallscalebias = smallscalebias, #set to CASA's default of 0.6 unless manually changed
               niter = niter, #we want to end on the threshold
               interactive = interactive,
               nsigma=nsigma,    
               cycleniter = cycleniter,
               cyclefactor = selfcal_library['cyclefactor'], 
               uvtaper = uvtaper, 
               savemodel = 'none',
               mask=mask,
               usemask=usemask,
               sidelobethreshold=sidelobethreshold,
               noisethreshold=noisethreshold,
               lownoisethreshold=lownoisethreshold,
               smoothfactor=smoothfactor,
               growiterations=growiterations,
               negativethreshold=negativethreshold,
               minbeamfrac=minbeamfrac,
               dogrowprune=dogrowprune,
               minpercentchange=minpercentchange,
               fastnoise=fastnoise,
               pbmask=pbmask,
               pblimit=pblimit,
               nterms = nterms,
               reffreq = reffreq,
               uvrange=selfcal_library['uvrange'],
               threshold=threshold,
               parallel=parallel,
               phasecenter=phasecenter,
               startmodel=startmodel,
               datacolumn=datacolumn,spw=spws_per_vis,wprojplanes=wprojplanes, verbose=True)

        if store_threshold != '':
            if telescope == "ALMA" or telescope == "ACA":
                selfcal_library["clean_threshold_"+store_threshold] = float(threshold[0:-2])
            elif "VLA" in telescope and tclean_return['iterdone'] > 0:
                selfcal_library["clean_threshold_"+store_threshold] = tclean_return['summaryminor'][0][0][0]['peakRes'][-1]

        if image_mosaic_fields_separately and selfcal_library['obstype'] == 'mosaic':
            for field_id in selfcal_library['sub-fields-phasecenters']:
                if 'VLA' in telescope:
                   fov=45.0e9/selfcal_library['meanfreq']*60.0*1.5*0.5
                   #if selfcal_library['meanfreq'] < 12.0e9:
                   #   fov=fov*2.0
                if telescope=='ALMA':
                   fov=63.0*100.0e9/selfcal_library['meanfreq']*1.5*0.5*1.15
                if telescope=='ACA':
                   fov=108.0*100.0e9/selfcal_library['meanfreq']*1.5*0.5

                center = np.copy(selfcal_library['sub-fields-phasecenters'][field_id])
                if phasecenter == 'TRACKFIELD':
                    center += imhead(imagename+".image.tt0")['refval'][0:2]

                region = 'circle[[{0:f}rad, {1:f}rad], {2:f}arcsec]'.format(center[0], center[1], fov)

                for ext in [".image.tt0", ".mask", ".residual.tt0", ".psf.tt0",".pb.tt0"]:
                    target = sanitize_string(field)
                    os.system('rm -rf '+ imagename.replace(target,target+"_field_"+str(field_id)) + ext.replace("pb","mospb"))

                    if ext == ".psf.tt0":
                        os.system("cp -r "+imagename+ext+" "+imagename.replace(target,target+"_field_"+str(field_id))+ext)
                    else:
                        imsubimage(imagename+ext, outfile=imagename.replace(target,target+"_field_"+str(field_id))+\
                                ext.replace("pb","mospb.tmp"), region=region, overwrite=True)

                        if ext == ".pb.tt0":
                            immath(imagename=[imagename.replace(target,target+"_field_"+str(field_id))+ext.replace("pb","mospb.tmp")], \
                                    outfile=imagename.replace(target,target+"_field_"+str(field_id))+ext.replace("pb","mospb"), \
                                    expr="IIF(IM0 == 0, 0.1, IM0)")
                            os.system("rm -rf "+imagename.replace(target,target+"_field_"+str(field_id))+ext.replace("pb","mospb.tmp"))

                # Make an image of the primary beam for each sub-field.
                if type(selfcal_library['vislist']) == list:
                    for v in selfcal_library['vislist']:
                        # Since not every field is in every v, we need to check them all so that we don't accidentally get a v without a given field_id
                        if field_id in selfcal_library['sub-fields-fid_map'][v]:
                            fid = selfcal_library['sub-fields-fid_map'][v][field_id]
                            break

                    im.open(v)
                else:
                    fid = selfcal_library['sub-fields-fid_map'][selfcal_library['vislist']][field_id]
                    im.open(vis)

                nx, ny, nfreq, npol = imhead(imagename=imagename.replace(target,target+"_field_"+str(field_id))+".image.tt0", mode="get", \
                        hdkey="shape")

                im.selectvis(field=str(fid), spw=spws_per_vis)
                im.defineimage(nx=nx, ny=ny, cellx=selfcal_library['cellsize'], celly=selfcal_library['cellsize'], phasecenter=fid, mode="mfs")
                im.setvp(dovp=True)
                im.makeimage(type="pb", image=imagename.replace(target,target+"_field_"+str(field_id)) + ".pb.tt0")
                im.close()


     #this step is a workaround a bug in tclean that doesn't always save the model during multiscale clean. See the "Known Issues" section for CASA 5.1.1 on NRAO's website
    if savemodel=='modelcolumn' and selfcal_library['usermodel']=='':
          print("")
          print("Running tclean a second time to save the model...")
          tclean(vis= vlist, 
                 imagename = imagename, 
                 field=field_str,
                 specmode = 'mfs', 
                 deconvolver = 'mtmfs',
                 scales = scales, 
                 gridder=gridder,
                 weighting='briggs', 
                 robust = robust,
                 gain = gain,
                 imsize = selfcal_library['imsize'],
                 cell = selfcal_library['cellsize'], 
                 smallscalebias = smallscalebias, #set to CASA's default of 0.6 unless manually changed
                 niter = 0, 
                 interactive = False,
                 nsigma=0.0, 
                 cycleniter = cycleniter,
                 cyclefactor = selfcal_library['cyclefactor'], 
                 uvtaper = uvtaper, 
                 usemask='user',
                 savemodel = savemodel,
                 sidelobethreshold=sidelobethreshold,
                 noisethreshold=noisethreshold,
                 lownoisethreshold=lownoisethreshold,
                 smoothfactor=smoothfactor,
                 growiterations=growiterations,
                 negativethreshold=negativethreshold,
                 minbeamfrac=minbeamfrac,
                 dogrowprune=dogrowprune,
                 minpercentchange=minpercentchange,
                 fastnoise=fastnoise,
                 pbmask=pbmask,
                 pblimit=pblimit,
                 calcres = False,
                 calcpsf = False,
                 restoration = False,
                 nterms = nterms,
                 uvrange=selfcal_library['uvrange'],
                 reffreq = reffreq,
                 threshold=threshold,
                 parallel=False,
                 phasecenter=phasecenter,spw=spws_per_vis,wprojplanes=wprojplanes)
    
    elif savemodel=='modelcolumn' and selfcal_library['usermodel'] !='':
          print('Using user model already filled to model column, skipping model write.')
    
    if not savemodel_only:
        return tclean_return

def usermodel_wrapper(selfcal_library, imagename, band, field_str, telescope='undefined',scales=[0], smallscalebias = 0.6, mask = '',\
                   nsigma=5.0, interactive = False, robust = 0.5, gain = 0.1, niter = 50000,\
                   cycleniter = 300, uvtaper = [], savemodel = 'none',gridder='standard', sidelobethreshold=3.0,smoothfactor=1.0,noisethreshold=5.0,\
                   lownoisethreshold=1.5,parallel=False,cyclefactor=3,threshold='0.0Jy',phasecenter='',\
                   startmodel='',pblimit=0.1,pbmask=0.1,field='',datacolumn='',spw='',\
                   savemodel_only=False, resume=False):
    vlist = selfcal_library['vislist']
    if type(selfcal_library['usermodel'])==list:
       nterms=len(selfcal_library['usermodel'])
       for i, image in enumerate(selfcal_library['usermodel']):
           if 'fits' in image:
               importfits(fitsimage=image,imagename=image.replace('.fits',''))
               selfcal_library['usermodel'][i]=image.replace('.fits','')
    elif type(selfcal_library['usermodel'])==str:
       importfits(fitsimage=selfcal_library['usermodel'],imagename=usermmodel.replace('.fits',''))
       nterms=1
   
    msmd.open(vlist[0])
    fieldid=msmd.fieldsforname(field)
    msmd.done()
    tb.open(vlist[0]+'/FIELD')
    try:
       ephem_column=tb.getcol('EPHEMERIS_ID')
       tb.close()
       if ephem_column[fieldid[0]] !=-1:
          phasecenter='TRACKFIELD'
    except:
       tb.close()
       phasecenter=''

    if selfcal_library['obstype']=='mosaic' and phasecenter != 'TRACKFIELD':
       phasecenter=get_phasecenter(selfcal_library['vislist'][0],field)

    if nterms == 1:
       reffreq = ''
    else:
       reffreq = selfcal_library['reffreq']

    if mask == '':
       if selfcal_library['usermask'] != '':
           mask = selfcal_library['usermask']
           usemask = 'user'
       else:
           usemask='auto-multithresh'
    else:
       usemask='user'

    wprojplanes=1
    if band=='EVLA_L' or band =='EVLA_S':
       gridder='wproject'
       wplanes=384 # normalized to S-band A-config
       #scale by 75th percentile uv distance divided by A-config value
       wplanes=wplanes * selfcal_library['75thpct_uv']/20000.0
       if band=='EVLA_L':
          wplanes=wplanes*2.0 # compensate for 1.5 GHz being 2x longer than 3 GHz


       wprojplanes=int(wplanes)
    if (band=='EVLA_L' or band =='EVLA_S') and selfcal_library['obstype']=='mosaic':
       print('WARNING DETECTED VLA L- OR S-BAND MOSAIC; WILL USE gridder="mosaic" IGNORING W-TERM')
    if selfcal_library['obstype']=='mosaic':
       gridder='mosaic'
    else:
       if gridder !='wproject':
          gridder='standard' 

    if gridder=='mosaic' and startmodel!='':
       parallel=False
    for ext in ['.image*', '.mask', '.model*', '.pb*', '.psf*', '.residual*', '.sumwt*','.gridwt*']:
        os.system('rm -rf '+ imagename + ext)
    #regrid start model
    if not resume:
        for ext in ['.image*', '.mask', '.model*', '.pb*', '.psf*', '.residual*', '.sumwt*','.gridwt*']:
           os.system('rm -rf '+ imagename+'_usermodel_prep' + ext)
    tclean(vis= vlist, 
               imagename = imagename+'_usermodel_prep', 
               field=field_str,
               specmode = 'mfs', 
               deconvolver = 'mtmfs',
               scales = scales, 
               gridder=gridder,
               weighting='briggs', 
               robust = robust,
               gain = gain,
               imsize = selfcal_library['imsize'],
               cell = selfcal_library['cellsize'], 
               smallscalebias = smallscalebias, #set to CASA's default of 0.6 unless manually changed
               niter = 0, #we want to end on the threshold
               interactive = interactive,
               nsigma=nsigma,    
               cycleniter = cycleniter,
               cyclefactor = selfcal_library['cyclefactor'], 
               uvtaper = uvtaper, 
               mask=mask,
               usemask=usemask,
               sidelobethreshold=sidelobethreshold,
               noisethreshold=noisethreshold,
               lownoisethreshold=lownoisethreshold,
               smoothfactor=smoothfactor,
               pbmask=pbmask,
               pblimit=pblimit,
               nterms = nterms,
               reffreq = reffreq,
               uvrange=selfcal_library['uvrange'],
               threshold=threshold,
               parallel=parallel,
               phasecenter=phasecenter,
               datacolumn=datacolumn,spw=spw,wprojplanes=wprojplanes, verbose=True,startmodel=selfcal_library['usermodel'],savemodel='modelcolumn')

     #this step is a workaround a bug in tclean that doesn't always save the model during multiscale clean. See the "Known Issues" section for CASA 5.1.1 on NRAO's website
    if savemodel=='modelcolumn':
          print("")
          print("Running tclean a second time to save the model...")
          tclean(vis= vlist, 
                 imagename = imagename+'_usermodel_prep', 
                 field=field_str,
                 specmode = 'mfs', 
                 deconvolver = 'mtmfs',
                 scales = scales, 
                 gridder=gridder,
                 weighting='briggs', 
                 robust = robust,
                 gain = gain,
                 imsize = selfcal_library['imsize'],
                 cell = selfcal_library['cellsize'], 
                 smallscalebias = smallscalebias, #set to CASA's default of 0.6 unless manually changed
                 niter = 0, 
                 interactive = False,
                 nsigma=0.0, 
                 cycleniter = cycleniter,
                 cyclefactor = selfcal_library['cyclefactor'], 
                 uvtaper = uvtaper, 
                 usemask='user',
                 savemodel = savemodel,
                 sidelobethreshold=sidelobethreshold,
                 noisethreshold=noisethreshold,
                 lownoisethreshold=lownoisethreshold,
                 smoothfactor=smoothfactor,
                 pbmask=pbmask,
                 pblimit=pblimit,
                 calcres = False,
                 calcpsf = False,
                 restoration = False,
                 nterms = nterms,
                 reffreq = reffreq,
                 uvrange=selfcal_library['uvrange'],
                 threshold=threshold,
                 parallel=False,
                 phasecenter=phasecenter,spw=spw,wprojplanes=wprojplanes)
 

 

def collect_listobs_per_vis(vislist):
   listdict={}
   for vis in vislist:
      listdict[vis]=listobs(vis)
   return listdict

def fetch_scan_times_band_aware(vislist,targets,bands_for_targets,band_properties,band,telescope):
   scantimesdict={}
   scanfieldsdict={}
   scannfieldsdict={}
   scanstartsdict={}
   scanendsdict={}
   integrationsdict={}
   integrationtimesdict={}
   integrationtimes=np.array([])
   n_spws=np.array([])
   min_spws=np.array([])
   scansforspw=np.array([])
   spwslist=np.array([])
   spwslist_dict = {}
   spws_set_dict = {}
   mosaic_field={}
   scansdict={}
   for vis in vislist:
      mosaic_field[vis] = {}
      scantimesdict[vis]={}
      scanfieldsdict[vis]={}
      scannfieldsdict[vis]={}
      scanstartsdict[vis]={}
      scanendsdict[vis]={}
      integrationsdict[vis]={}
      integrationtimesdict[vis]={}
      spws_set_dict[vis] = {}
      spwslist_dict[vis]=np.array([])
      scansdict[vis]={}
      msmd.open(vis)
      for target in targets:
         scansforfield=msmd.scansforfield(target)
         for spw in band_properties[vis][band]['spwarray']:
            scansforspw_temp=msmd.scansforspw(spw)
            scansforspw=np.append(scansforspw,np.array(scansforspw_temp,dtype=int))
         scansforspw=scansforspw.astype(int)
         scansdict[vis][target]=list(set(scansforfield) & set(scansforspw))
         scansdict[vis][target].sort()
      for target in targets:
         mosaic_field[vis][target]={}
         mosaic_field[vis][target]['field_ids']=[]
         mosaic_field[vis][target]['mosaic']=False
         # ID ALMA mosaics by multiple fields with same target name
         if telescope=='ALMA':
             mosaic_field[vis][target]['field_ids']=msmd.fieldsforscans(scansdict[vis][target]).tolist()
             mosaic_field[vis][target]['field_ids']=list(set(mosaic_field[vis][target]['field_ids']))
         # ID VLA mosaics using pre-determined mosaic groupings from fetch_targets
         elif 'VLA' in telescope:
             mosaic_field[vis][target]['field_ids']=bands_for_targets['field_ids'] # need to make this have per vis information
             mosaic_field[vis][target]['field_ids']=list(set(mosaic_field[vis][target]['field_ids']))
             print('mosaic field ids', mosaic_field[vis][target]['field_ids'])
         mosaic_field[vis][target]['phasecenters'] = []
         for fid in mosaic_field[vis][target]['field_ids']:
             tb.open(vis+'/FIELD')
             mosaic_field[vis][target]['phasecenters'].append(tb.getcol("PHASE_DIR")[:,0,fid])
             tb.close()

         if len(mosaic_field[vis][target]['field_ids']) > 1:
            mosaic_field[vis][target]['mosaic']=True
         print('mosaic field', mosaic_field[vis][target])
         scantimes=np.array([])
         scanfields=np.array([])
         scannfields=np.array([])
         integrations=np.array([])
         scanstarts=np.array([])
         scanends=np.array([])

         for scan in scansdict[vis][target]:
            spws=msmd.spwsforscan(scan)
            spws_set_dict[vis][scan]=spws.copy()
            n_spws=np.append(len(spws),n_spws)
            min_spws=np.append(np.min(spws),min_spws)
            spwslist=np.append(spws,spwslist)
            spwslist_dict[vis]=np.append(spws,spwslist_dict[vis])
            integrationtime=msmd.exposuretime(scan=scan,spwid=spws[0])['value']
            integrationtimes=np.append(integrationtimes,np.array([integrationtime]))
            times=msmd.timesforscan(scan)
            scantime=np.max(times)+integrationtime-np.min(times)
            scanstarts=np.append(scanstarts,np.array([np.min(times)/86400.0]))
            scanends=np.append(scanends,np.array([(np.max(times)+integrationtime)/86400.0]))
            ints_per_scan=np.round(scantime/integrationtimes[0])
            scantimes=np.append(scantimes,np.array([scantime]))
            integrations=np.append(integrations,np.array([ints_per_scan]))
            scanfields = np.append(scanfields,np.array([','.join(msmd.fieldsforscan(scan).astype(str))]))
            scannfields = np.append(scannfields,np.array([msmd.fieldsforscan(scan).size]))

               
         scantimesdict[vis][target]=scantimes.copy()
         scanfieldsdict[vis][target]=scanfields.copy()
         scannfieldsdict[vis][target]=scannfields.copy()
         scanstartsdict[vis][target]=scanstarts.copy()
         scanendsdict[vis][target]=scanends.copy()
         #assume each band only has a single integration time
         integrationtimesdict[vis][target]=np.median(integrationtimes)
         integrationsdict[vis][target]=integrations.copy()
   # jump through some hoops to get the dictionary that has spws per scan into a dictionary of unique
   # spw sets per vis file
   for vis in vislist:
      spws_set_list=[i for i in spws_set_dict[vis].values()]
      spws_set_list=[i.tolist() for i in spws_set_list]
      unique_spws_set_list=[list(i) for i in set(tuple(i) for i in spws_set_list)]
      spws_set_list=[np.array(i) for i in unique_spws_set_list]
      spws_set_dict[vis]=np.array(spws_set_list,dtype=object)
      spwslist_dict[vis]=np.unique(spwslist_dict[vis]).astype(int)
   if len(n_spws) > 0:
      if np.mean(n_spws) != np.max(n_spws):
         print('WARNING, INCONSISTENT NUMBER OF SPWS IN SCANS/MSes (Possibly expected if Multi-band VLA data or ALMA Spectral Scan)')
      if np.max(min_spws) != np.min(min_spws):
         print('WARNING, INCONSISTENT MINIMUM SPW IN SCANS/MSes (Possibly expected if Multi-band VLA data or ALMA Spectral Scan)')
      spwslist=np.unique(spwslist).astype(int)
   else:
     return scantimesdict,scanfieldsdict,scannfieldsdict,scanstartsdict,scanendsdict,integrationsdict,integrationtimesdict, integrationtimes,-99,-99,spwslist_dict,mosaic_field
   return scantimesdict,scanfieldsdict,scannfieldsdict,scanstartsdict,scanendsdict,integrationsdict,integrationtimesdict, integrationtimes,np.max(n_spws),np.min(min_spws),spwslist_dict,spws_set_dict,mosaic_field


def fetch_spws(vislist,targets):
   scantimesdict={}
   n_spws=np.array([])
   min_spws=np.array([])
   spwslist=np.array([])
   scansdict={}
   for vis in vislist:
      scansdict[vis]={}
      msmd.open(vis)
      for target in targets:
         scansdict[vis][target]=msmd.scansforfield(target)
         scansdict[vis][target].sort()
      for target in targets:
         for scan in scansdict[vis][target]:
            spws=msmd.spwsforscan(scan)
            n_spws=np.append(len(spws),n_spws)
            min_spws=np.append(np.min(spws),min_spws)
            spwslist=np.append(spws,spwslist)
   if len(n_spws) > 1:
      if np.mean(n_spws) != np.max(n_spws):
         print('WARNING, INCONSISTENT NUMBER OF SPWS IN SCANS/MSes (Possibly expected if Multi-band VLA data or ALMA Spectral Scan)')
      if np.max(min_spws) != np.min(min_spws):
         print('WARNING, INCONSISTENT MINIMUM SPW IN SCANS/MSes (Possibly expected if Multi-band VLA data or ALMA Spectral Scan)')
   spwslist=np.unique(spwslist).astype(int)
   if len(n_spws) == 1:
      return n_spws,min_spws,spwslist
   else:
      return np.max(n_spws),np.min(min_spws),spwslist


    

#actual routine used for getting solints
def get_solints_simple(vislist,scantimesdict,scannfieldsdict,scanstartsdict,scanendsdict,integrationtimes,\
                       inf_EB_gaincal_combine,spwcombine=True,solint_decrement='fixed',solint_divider=2.0,n_solints=4.0,do_amp_selfcal=False, mosaic=False,do_scan_inf=True):
   all_integrations=np.array([])
   all_nscans_per_obs=np.array([])
   all_time_between_scans=np.array([])
   all_times_per_obs=np.array([])
   allscantimes=np.array([]) # we put all scan times from all MSes into single array
   #mix of short and long baseline data could have differing integration times and hence solints
   #could do solints per vis file, but too complex for now at least use perhaps keep scan groups different
   #per MOUS
   nscans_per_obs={}
   time_per_vis={}
   time_between_scans={}
   for vis in vislist:
      nscans_per_obs[vis]={}
      time_between_scans[vis]={}
      time_per_vis[vis]=0.0
      targets=integrationtimes[vis].keys()
      earliest_start=1.0e10
      latest_end=0.0
      for target in targets:
         nscans_per_obs[vis][target]=len(scantimesdict[vis][target])
         allscantimes=np.append(allscantimes,scantimesdict[vis][target]/scannfieldsdict[vis][target])
         for i in range(len(scanstartsdict[vis][target])):# way to get length of an EB with multiple targets without writing new functions; I could be more clever with np.where()
            if scanstartsdict[vis][target][i] < earliest_start: 
               earliest_start=scanstartsdict[vis][target][i]
            if scanendsdict[vis][target][i] > latest_end:
               latest_end=scanstartsdict[vis][target][i]
         if np.isfinite(integrationtimes[vis][target]):
            all_integrations=np.append(all_integrations,integrationtimes[vis][target])
         all_nscans_per_obs=np.append(all_nscans_per_obs,nscans_per_obs[vis][target])
         #determine time between scans
         delta_scan=np.zeros(len(scanstartsdict[vis][target])-1)
         sortedstarts=np.sort(scanstartsdict[vis][target]) #scan list isn't sorted, so sort these so they're in order and we can subtract them from each other
         sortedends=np.sort(scanstartsdict[vis][target])
         #delta_scan=(sortedends[:-1]-sortedstarts[1:])*86400.0*-1.0
         delta_scan=np.zeros(len(sortedends)-1)
         for i in range(len(sortedstarts)-1):
            delta_scan[i]=(sortedends[i]-sortedstarts[i+1])*86400.0*-1.0
         all_time_between_scans=np.append(all_time_between_scans,delta_scan)
      time_per_vis[vis]= (latest_end - earliest_start)*86400.0    # calculate length of EB
      all_times_per_obs=np.append(all_times_per_obs,np.array([time_per_vis[vis]]))
   integration_time=np.max(all_integrations) # use the longest integration time from all MS files

   max_scantime=np.median(allscantimes)
   median_scantime=np.max(allscantimes)
   min_scantime=np.min(allscantimes)
   median_scans_per_obs=np.median(all_nscans_per_obs)
   median_time_per_obs=np.median(all_times_per_obs)
   median_time_between_scans=np.median(all_time_between_scans)
   print('median scan length: ',median_scantime)
   print('median time between target scans: ',median_time_between_scans)
   print('median scans per observation: ',median_scans_per_obs)
   print('median length of observation: ',median_time_per_obs)

   solints_gt_scan=np.array([])
   gaincal_combine=[]
   
   # commented completely, no solints between inf_EB and inf
   #make solints between inf_EB and inf if more than one scan per source and scans are short
   #if median_scans_per_obs > 1 and median_scantime < 150.0:
   #   # add one solint that is meant to combine 2 short scans, otherwise go to inf_EB
   #   solint=(median_scantime*2.0+median_time_between_scans)*1.1
   #   if solint < 300.0:  # only allow solutions that are less than 5 minutes in duration
   #      solints_gt_scan=np.append(solints_gt_scan,[solint])

   #code below would make solints between inf_EB and inf by combining scans
   #sometimes worked ok, but many times selfcal would quit before solint=inf
   '''
   solint=median_time_per_obs/4.05 # divides slightly unevenly if lengths of observation are exactly equal, but better than leaving a small out of data remaining
   while solint > (median_scantime*2.0+median_time_between_scans)*1.05:      #solint should be greater than the length of time between two scans + time between to be better than inf
      solints_gt_scan=np.append(solints_gt_scan,[solint])                       # add solint to list of solints now that it is an integer number of integrations
      solint = solint/2.0  
      #print('Next solint: ',solint)                                        #divide solint by 2.0 for next solint
   '''
   print(max_scantime,integration_time)
   if solint_decrement == 'fixed':
      solint_divider=np.round(np.exp(1.0/n_solints*np.log(max_scantime/integration_time)))
   #division never less than 2.0
   if solint_divider < 2.0:
      solint_divider=2.0
   solints_lt_scan=np.array([])
   n_scans=len(allscantimes)
   solint=max_scantime/solint_divider  
   while solint > 1.90*integration_time:      #1.1*integration_time will ensure that a single int will not be returned such that solint='int' can be appended to the final list.
      ints_per_solint=solint/integration_time
      if ints_per_solint.is_integer():
         solint=solint
      else:
         remainder=ints_per_solint-float(int(ints_per_solint))     # calculate delta_T greater than an a fixed multile of integrations
         solint=solint-remainder*integration_time # add remainder to make solint a fixed number of integrations

      ints_per_solint=float(int(ints_per_solint))
      print('Checking solint = ',ints_per_solint*integration_time)
      delta=test_truncated_scans(ints_per_solint, allscantimes,integration_time) 
      solint=(ints_per_solint+delta)*integration_time
      if solint > 1.90*integration_time:
         solints_lt_scan=np.append(solints_lt_scan,[solint])                       # add solint to list of solints now that it is an integer number of integrations

      solint = solint/solint_divider  
      #print('Next solint: ',solint)                                        #divide solint by 2.0 for next solint

      

   solints_list=[]
   if len(solints_gt_scan) > 0:
      for solint in solints_gt_scan:
         solint_string='{:0.2f}s'.format(solint)
         solints_list.append(solint_string)
         if spwcombine:
            gaincal_combine.append('spw,scan')
         else:
            gaincal_combine.append('scan')



 # insert inf_EB
   solints_list.insert(0,'inf_EB')
   gaincal_combine.insert(0,inf_EB_gaincal_combine)

   # Insert scan_inf_EB if this is a mosaic.
   if mosaic and median_scans_per_obs > 1 and do_scan_inf:
       solints_list.append('scan_inf')
       if spwcombine:
           gaincal_combine.append('spw,field,scan')
       else:
           gaincal_combine.append('field,scan')

   #insert solint = inf
   if (not mosaic and (median_scans_per_obs > 2 or (median_scans_per_obs == 2 and max_scantime / min_scantime < 4))) or mosaic:                    # if only a single scan per target, redundant with inf_EB and do not include
      solints_list.append('inf')
      if spwcombine:
         gaincal_combine.append('spw')
      else:
         gaincal_combine.append('')

   for solint in solints_lt_scan:
      solint_string='{:0.2f}s'.format(solint)
      solints_list.append(solint_string)
      if spwcombine:
         gaincal_combine.append('spw')
      else:
         gaincal_combine.append('')



   #append solint = int to end
   solints_list.append('int')
   if spwcombine:
      gaincal_combine.append('spw')
   else:
      gaincal_combine.append('')
   solmode_list=['p']*len(solints_list)
   if do_amp_selfcal:
      if median_time_between_scans >150.0 or np.isnan(median_time_between_scans):
         amp_solints_list=['inf_ap']
         if spwcombine:
            amp_gaincal_combine=['spw']
         else:
            amp_gaincal_combine=['']
      else:
         amp_solints_list=['300s_ap','inf_ap']
         if spwcombine:
            amp_gaincal_combine=['scan,spw','spw']
         else:
            amp_gaincal_combine=['scan','']
      solints_list=solints_list+amp_solints_list
      gaincal_combine=gaincal_combine+amp_gaincal_combine
      solmode_list=solmode_list+['ap']*len(amp_solints_list)

      
         

   return solints_list,integration_time,gaincal_combine,solmode_list



def test_truncated_scans(ints_per_solint, allscantimes,integration_time ):
   delta_ints_per_solint=[0 , -1, 1,-2,2]
   n_truncated_scans=np.zeros(len(delta_ints_per_solint))
   n_remaining_ints=np.zeros(len(delta_ints_per_solint))
   min_index=0
   for i in range(len(delta_ints_per_solint)):
      diff_ints_per_scan=((allscantimes-((ints_per_solint+delta_ints_per_solint[i])*integration_time))/integration_time)+0.5
      diff_ints_per_scan=diff_ints_per_scan.astype(int)
      trimmed_scans=( (diff_ints_per_scan > 0.0)  & (diff_ints_per_scan < ints_per_solint+delta_ints_per_solint[i])).nonzero()
      if len(trimmed_scans[0]) >0:
         n_remaining_ints[i]=np.max(diff_ints_per_scan[trimmed_scans[0]])
      else:
         n_remaining_ints[i]=0.0
      #print((ints_per_solint+delta_ints_per_solint[i])*integration_time,ints_per_solint+delta_ints_per_solint[i],  diff_ints_per_scan)
      
      #print('Max ints remaining: ', n_remaining_ints[i])
      #print('N truncated scans: ', len(trimmed_scans[0]))
      n_truncated_scans[i]=len(trimmed_scans[0])
      # check if there are fewer truncated scans in the current trial and if
      # if one trial has more scans left off or fewer. Favor more left off, such that remainder might be able to 
      # find a solution
      # if ((i > 0) and (n_truncated_scans[i] <= n_truncated_scans[min_index]):   # if we don't care about the amount of 
      #if ((i > 0) and (n_truncated_scans[i] <= n_truncated_scans[min_index]) and (n_remaining_ints[i] > n_remaining_ints[min_index])):
      if ((i > 0) and (n_truncated_scans[i] <= n_truncated_scans[min_index]) and (n_remaining_ints[i] < n_remaining_ints[min_index])):
         min_index=i
      #print(delta_ints_per_solint[min_index])
   return delta_ints_per_solint[min_index]
   

def fetch_targets(vislist,telescope):
   targets_vis={}
   targets_band_vis={}
   vis_for_targets={}
   bands_for_targets={}
   nfields=0
   maxfieldsvis=''
   fields_superset=[]
   vis_missing_fields=[]
   vis_overflagged_fields=[]
   band_list=[]
   for vis in vislist:
      targets_vis[vis]={}
      fields=[]
      msmd.open(vis)
      fieldnames=msmd.fieldnames()
      for fieldname in fieldnames:
         scans=msmd.scansforfield(fieldname)
         if len(scans) > 0:
            fields.append(fieldname)
      msmd.close()
      fields=list(set(fields)) # convert to set to only get unique items
      targets_vis[vis]['fields']=list(set(fields))
      targets_vis[vis]['fields'].sort
      if len(targets_vis[vis]['fields']) > nfields:
         nfields=len(targets_vis[vis]['fields'])
         maxfieldvis=vis+''
         fields_superset=targets_vis[vis]['fields'].copy()

   for target in fields_superset:
      vis_for_targets[target]={}
      for vis in vislist:
         if target in targets_vis[vis]['fields']:
            bands,band_properties=get_bands([vis],[target],telescope)
            band_list=band_list+bands  
            if 'Bands' not in vis_for_targets[target].keys():
               vis_for_targets[target]['Bands']=bands.copy()
            for band in bands:
               if band not in vis_for_targets[target].keys():
                  vis_for_targets[target][band]={}
               if 'vislist' not in vis_for_targets[target][band].keys():
                  vis_for_targets[target][band]['vislist']=[]
               vis_for_targets[target][band]['vislist']+=[vis]
               vis_for_targets[target][band][vis]={}
               vis_for_targets[target][band][vis]['spwarray']=band_properties[vis][band]['spwarray'].copy()
               vis_for_targets[target][band][vis]['spwstring']=band_properties[vis][band]['spwstring']+''
               vis_for_targets[target][band]['meanfreq']=band_properties[vis][band]['meanfreq']
   flagging_dict={}
   for vis in vislist:
      flagging_dict[vis]=flagdata(vis=vis,mode='summary') 

   for target in vis_for_targets.keys():
      for band in vis_for_targets[target]['Bands']:
         for vis in vis_for_targets[target][band]['vislist']:
            vis_for_targets[target][band][vis]['flagging']=flagging_dict[vis]['field'][target].copy()
            vis_for_targets[target][band][vis]['flagging']['frac']=flagging_dict[vis]['field'][target]['flagged']/flagging_dict[vis]['field'][target]['total']
            vis_for_targets[target][band][vis]['flagging']['flagged']=flagging_dict[vis]['field'][target]['flagged']+0
            vis_for_targets[target][band][vis]['flagging']['total']=flagging_dict[vis]['field'][target]['total']+0

            if vis_for_targets[target][band][vis]['flagging']['frac'] >= 0.99:
               vis_overflagged_fields+=[vis]
               vis_for_targets[target][band]['vislist'].remove(vis)
   #Determine the mosiacs and single fields per bands incase of multi-band mosaics
   band_list=list(set(band_list))
   for band in band_list:
      bands_for_targets[band]={}
      bands_for_targets[band]['targets']=[]

      for target in vis_for_targets.keys():

         if band in vis_for_targets[target].keys():
            bands_for_targets[band]['targets'].append(target)
            if 'meanfreq' not in bands_for_targets[band].keys():
                bands_for_targets[band]['meanfreq']=vis_for_targets[target][band]['meanfreq']
      bands_for_targets[band]['targets'].sort() 
      mosaic_groups,mosaic_groups_ids,single_fields,single_fields_ids=check_targets_for_mosaic(vislist,bands_for_targets[band]['targets'],bands_for_targets[band]['meanfreq'])
      if len(mosaic_groups) > 0:
         for m,mosaic_group in enumerate(mosaic_groups):
            bands_for_targets[band][mosaic_group[0]]={}
            bands_for_targets[band][mosaic_group[0]]['fieldnames']=mosaic_group
            bands_for_targets[band][mosaic_group[0]]['field_ids']=mosaic_groups_ids[m]
            bands_for_targets[band][mosaic_group[0]]['field_str']=",".join([str(num) for num in mosaic_groups_ids[m]])
            bands_for_targets[band][mosaic_group[0]]['obstype']='mosaic'
      if len(single_fields) > 0:
         for s,single_field in enumerate(single_fields):
            bands_for_targets[band][single_field]={}
            bands_for_targets[band][single_field]['fieldnames']=single_fields
            bands_for_targets[band][single_field]['field_ids']=[int(single_fields_ids[s])]
            bands_for_targets[band][single_field]['field_str']=str(single_fields_ids[s])
            bands_for_targets[band][single_field]['obstype']='single-pointing'
   for band in band_list:
      band_targets=copy.copy(bands_for_targets[band]['targets'])
      for target in band_targets:
         if target not in bands_for_targets[band].keys():
            bands_for_targets[band]['targets'].remove(target)
            fields_superset.remove(target)
            print('Removing '+target+' as independent target since it is part of a mosaic')
   return fields_superset, targets_vis, vis_for_targets, vis_missing_fields, vis_overflagged_fields, bands_for_targets

#tolerance in units of hpbw; 1.0 means limit to an optimally sampled mosaic
# 2.0 limit means sparse, beam sampled mosaic
def create_mosaic_groups(ra_arr, dec_arr,names,ids,hpbw,overlap_tol=0.9):
    pointings=np.vstack((ra_arr, dec_arr)).T
    Z=hc.linkage(pointings,method='single',optimal_ordering=True)
    clusters=hc.fcluster(Z, t=hpbw*overlap_tol/3600.0, criterion='distance')
    unique_groups=np.unique(clusters)
    single_fields=[]
    single_fields_ids=[]
    mosaics=[]
    mosaics_ids=[]
    for group in unique_groups:
      indices=np.where(clusters==group)
      if len(indices[0]) > 1:
         mosaics.append(names[indices[0]].tolist())
         mosaics_ids.append(ids[indices[0]].tolist())
      else:
         single_fields.append(str(names[indices[0][0]]))
         single_fields_ids.append(str(ids[indices[0][0]]))
    return mosaics,mosaics_ids,single_fields,single_fields_ids     

def check_targets_for_mosaic(vislist,targets,freq):
    for vis in vislist:
        msmd.open(vis)
        fieldids=[]
        for target in targets:
            fieldids=fieldids+list(msmd.fieldsforname(target)) # convert to list from a numpy array that is returned
        ra=[]
        dec=[]
        for fid in fieldids:
            phasecenter_dict=msmd.phasecenter(fieldid=fid)
            ra.append(phasecenter_dict['m0']['value']*180.0/np.pi)
            dec.append(phasecenter_dict['m1']['value']*180.0/np.pi)

        #mosaic_fields=ascii.read('mosaic_larger.reg',names=['field','ids','ra','dec'])
        #print(mosaic_fields)
        hpbw=42.0e9/freq*60.0
        if len(targets) > 1:    mosaic_groups,mosaic_groups_ids,single_fields,single_fields_ids=create_mosaic_groups(np.array(ra),np.array(dec),np.array(targets),np.array(fieldids),hpbw,overlap_tol=0.9)
        else:
            mosaic_groups=[]
            mosaic_groups_ids=[]
            single_fields=targets
            single_fields_ids=fieldids
        print('Mosaics groupings: ',mosaic_groups)
        print('Single Fields: ',single_fields)
    return mosaic_groups,mosaic_groups_ids,single_fields,single_fields_ids
def checkmask(imagename):
   maskImage=imagename.replace('image','mask').replace('.tt0','')
   image_stats= imstat(maskImage)
   if image_stats['max'][0] == 0:
      return False
   else:
      return True

def estimate_SNR(imagename,maskname=None,verbose=True, mosaic_sub_field=False):
    MADtoRMS =  1.4826
    headerlist = imhead(imagename, mode = 'list')
    beammajor = headerlist['beammajor']['value']
    beamminor = headerlist['beamminor']['value']
    beampa = headerlist['beampa']['value']

    if mosaic_sub_field:
        os.system("rm -rf temp.image")
        immath(imagename=[imagename, imagename.replace(".image",".pb"), imagename.replace(".image",".mospb")], outfile="temp.image", \
                expr="IM0*IM1/IM2")
        image_stats= imstat(imagename = "temp.image")
        os.system("rm -rf temp.image")
    else:
        image_stats= imstat(imagename = imagename)

    if maskname is None:
       maskImage=imagename.replace('image','mask').replace('.tt0','')
    else:
       maskImage=maskname
    residualImage=imagename # change to .image JT 04-15-2024 .replace('image','residual')
    os.system('rm -rf temp.mask temp.residual')
    if os.path.exists(maskImage):
       os.system('cp -r '+maskImage+ ' temp.mask')
       maskImage='temp.mask'
    os.system('cp -r '+residualImage+ ' temp.residual')   # leave this as .residual to avoid clashing with another temp.image
    residualImage='temp.residual'
    if 'dirty' not in imagename:
       goodMask=checkmask(imagename)
    else:
       goodMask=False
    if os.path.exists(maskImage) and goodMask:
       ia.close()
       ia.done()
       ia.open(residualImage)
       #ia.calcmask(maskImage+" <0.5"+"&& mask("+residualImage+")",name='madpbmask0')
       ia.calcmask("'"+maskImage+"'"+" <0.5"+"&& mask("+residualImage+")",name='madpbmask0')
       mask0Stats = ia.statistics(robust=True,axes=[0,1])
       ia.maskhandler(op='set',name='madpbmask0')
       rms = mask0Stats['medabsdevmed'][0] * MADtoRMS
       residualMean = mask0Stats['median'][0]
    else:
       residual_stats=imstat(imagename=imagename,algorithm='chauvenet')
       rms = residual_stats['rms'][0]
    peak_intensity = image_stats['max'][0]
    SNR = peak_intensity/rms
    if verbose:
           print("#%s" % imagename)
           print("#Beam %.3f arcsec x %.3f arcsec (%.2f deg)" % (beammajor, beamminor, beampa))
           print("#Peak intensity of source: %.2f mJy/beam" % (peak_intensity*1000,))
           print("#rms: %.2e mJy/beam" % (rms*1000,))
           print("#Peak SNR: %.2f" % (SNR,))
    ia.close()
    ia.done()
    if mosaic_sub_field:
        os.system("rm -rf temp.image")
    os.system('rm -rf temp.mask temp.residual')
    return SNR,rms



def estimate_near_field_SNR(imagename,las=None,maskname=None,verbose=True, mosaic_sub_field=False, save_near_field_mask=True):
    if maskname is None:
       maskImage=imagename.replace('image','mask').replace('.tt0','')
    else:
       maskImage=maskname
    if not os.path.exists(maskImage):
       print('Does not exist')
       return np.float64(-99.0),np.float64(-99.0)
    goodMask=checkmask(maskImage)
    if not goodMask:
       print('checkmask')
       return np.float64(-99.0),np.float64(-99.0)

    MADtoRMS =  1.4826
    headerlist = imhead(imagename, mode = 'list')
    beammajor = headerlist['beammajor']['value']
    beamminor = headerlist['beamminor']['value']
    beampa = headerlist['beampa']['value']

    if mosaic_sub_field:
        immath(imagename=[imagename, imagename.replace(".image",".pb"), imagename.replace(".image",".mospb")], outfile="temp.image", \
                expr="IM0*IM1/IM2")
        image_stats= imstat(imagename = "temp.image")
        os.system("rm -rf temp.image")
    else:
        image_stats= imstat(imagename = imagename)

    residualImage=imagename   # change to .image JT 04-15-2024 .replace('image','residual')
    os.system('rm -rf temp.mask temp.residual temp.border.mask temp.smooth.ceiling.mask temp.smooth.mask temp.nearfield.mask temp.big.smooth.ceiling.mask temp.big.smooth.mask temp.nearfield.prepb.mask temp.beam.extent.image temp.delta temp.radius temp.image')
    os.system('cp -r '+maskImage+ ' temp.mask')
    os.system('cp -r '+residualImage+ ' temp.residual')   # keep as .residual to avoid clashing with another temp.image
    residualImage='temp.residual'
    maskStats=imstat(imagename='temp.mask')
    imsmooth(imagename='temp.mask',kernel='gauss',major=str(beammajor*1.0)+'arcsec',minor=str(beammajor*1.0)+'arcsec', pa='0deg',outfile='temp.smooth.mask')
    immath(imagename=['temp.smooth.mask'],expr='iif(IM0 > 0.1*max(IM0),1.0,0.0)',outfile='temp.smooth.ceiling.mask')

    # Check the extent of the beam as well.
    psfImage = maskImage.replace('mask','psf')+'.tt0'

    immath(psfImage, mode="evalexpr", expr="iif(IM0==1,IM0,0)", outfile="temp.delta")
    npix = imhead("temp.delta", mode="get", hdkey="shape")[0]
    imsmooth("temp.delta", major=str(npix/2)+"pix", minor=str(npix/2)+"pix", pa="0deg", \
            outfile="temp.radius", overwrite=True)

    bmin = imhead(imagename, mode="get", hdkey="BMIN")['value']
    bmaj = imhead(imagename, mode="get", hdkey="BMAJ")['value']
    bpa = imhead(imagename, mode="get", hdkey="BPA")['value']

    imhead(imagename="temp.radius", mode="put", hdkey="BMIN", hdvalue=str(bmin)+"arcsec")
    imhead(imagename="temp.radius", mode="put", hdkey="BMAJ", hdvalue=str(bmaj)+"arcsec")
    imhead(imagename="temp.radius", mode="put", hdkey="BPA", hdvalue=str(bpa)+"deg")

    immath(imagename=[psfImage,"temp.radius"], mode="evalexpr", expr="iif(IM0 > 0.1,1/IM1,0.0)", outfile="temp.beam.extent.image")

    centerpos = imhead(psfImage, mode="get", hdkey="maxpixpos")
    maxpos = imhead("temp.beam.extent.image", mode="get", hdkey="maxpixpos")
    center_coords = imval(psfImage, box=str(centerpos[0])+","+str(centerpos[1]))["coords"]
    max_coords = imval(psfImage, box=str(maxpos[0])+","+str(maxpos[1]))["coords"]

    beam_extent_size = ((center_coords - max_coords)**2)[0:2].sum()**0.5 * 360*60*60/(2*np.pi)

    # use the maximum of the three possibilities as the outer extent of the mask.
    print("beammajor*5 = ", beammajor*5, ", LAS = ", 5*las, ", beam_extent = ", beam_extent_size)
    outer_major = max(beammajor*5, beam_extent_size, 5*las if las is not None else 0.)

    imsmooth(imagename='temp.smooth.ceiling.mask',kernel='gauss',major=str(outer_major)+'arcsec',minor=str(outer_major)+'arcsec', pa='0deg',outfile='temp.big.smooth.mask')

    immath(imagename=['temp.big.smooth.mask'],expr='iif(IM0 > 0.01*max(IM0),1.0,0.0)',outfile='temp.big.smooth.ceiling.mask')
    immath(imagename=['temp.big.smooth.ceiling.mask','temp.smooth.ceiling.mask'],expr='((IM0-IM1)-1.0)*-1.0',outfile='temp.nearfield.prepb.mask')
    immath(imagename=[imagename,'temp.nearfield.prepb.mask'], expr='iif(MASK(IM0),IM1,1.0)',outfile='temp.nearfield.mask')

    maskImage='temp.nearfield.mask'

    mask_stats= imstat(maskImage)
    if mask_stats['min'][0] == 1:
       print('checkmask')
       SNR, rms = np.float64(-99.0), np.float64(-99.0)
    else:
       ia.close()
       ia.done()
       ia.open(residualImage)
       #ia.calcmask(maskImage+" <0.5"+"&& mask("+residualImage+")",name='madpbmask0')
       ia.calcmask("'"+maskImage+"'"+" <0.5"+"&& mask("+residualImage+")",name='madpbmask0')
       mask0Stats = ia.statistics(robust=True,axes=[0,1])
       ia.maskhandler(op='set',name='madpbmask0')
       rms = mask0Stats['medabsdevmed'][0] * MADtoRMS
       residualMean = mask0Stats['median'][0]
       peak_intensity = image_stats['max'][0]
       SNR = peak_intensity/rms
       if verbose:
              print("#%s" % imagename)
              print("#Beam %.3f arcsec x %.3f arcsec (%.2f deg)" % (beammajor, beamminor, beampa))
              print("#Peak intensity of source: %.2f mJy/beam" % (peak_intensity*1000,))
              print("#Near Field rms: %.2e mJy/beam" % (rms*1000,))
              print("#Peak Near Field SNR: %.2f" % (SNR,))
       ia.close()
       ia.done()

    if save_near_field_mask:
        os.system('cp -r '+maskImage+' '+imagename.replace('image','nearfield.mask').replace('.tt0',''))
    os.system('rm -rf temp.mask temp.residual temp.border.mask temp.smooth.ceiling.mask temp.smooth.mask temp.nearfield.mask temp.big.smooth.ceiling.mask temp.big.smooth.mask temp.nearfield.prepb.mask temp.beam.extent.image temp.delta temp.radius temp.image')
    return SNR,rms


def get_intflux(imagename,rms,maskname=None,mosaic_sub_field=False):
   headerlist = imhead(imagename, mode = 'list')
   beammajor = headerlist['beammajor']['value']
   beamminor = headerlist['beamminor']['value']
   beampa = headerlist['beampa']['value']
   cell = headerlist['cdelt2']*180.0/3.14159*3600.0
   beamarea=3.14159*beammajor*beamminor/(4.0*np.log(2.0))
   pix_per_beam=beamarea/(cell**2)
   if maskname is None:
      maskname=imagename.replace('image.tt0','mask')

   if mosaic_sub_field:
       immath(imagename=[imagename, imagename.replace(".image",".pb"), imagename.replace(".image",".mospb")], outfile="temp.image", \
               expr="IM0*IM1/IM2")
       imagestats= imstat(imagename = "temp.image", mask=maskname)
       os.system("rm -rf temp.image")
   else:
       imagestats= imstat(imagename = imagename, mask=maskname)

   if len(imagestats['flux']) > 0:
       flux=imagestats['flux'][0]
       n_beams=imagestats['npts'][0]/pix_per_beam
       e_flux=(n_beams)**0.5*rms
   else:
       flux = 0.
       e_flux = rms
   return flux,e_flux

def get_n_ants(vislist):
   #Examines number of antennas in each ms file and returns the minimum number of antennas
   msmd = casatools.msmetadata()
   tb = casatools.table()
   n_ants=50.0
   for vis in vislist:
      msmd.open(vis)
      names = msmd.antennanames(msmd.antennasforscan(msmd.scansforintent("*OBSERVE_TARGET*")[0]))
      msmd.close()
      n_ant_vis=len(names)
      if n_ant_vis < n_ants:
         n_ants=n_ant_vis
   return n_ants
    
def get_ant_list(vis):
   #Examines number of antennas in each ms file and returns the minimum number of antennas
   msmd = casatools.msmetadata()
   tb = casatools.table()
   n_ants=50.0
   msmd.open(vis)
   names = msmd.antennanames(msmd.antennasforscan(msmd.scansforintent("*OBSERVE_TARGET*")[0]))
   msmd.close()
   return names

def rank_refants(vis, caltable=None):
     # Get the antenna names and offsets.

     msmd = casatools.msmetadata()
     tb = casatools.table()

     msmd.open(vis)
     ids = msmd.antennasforscan(msmd.scansforintent("*OBSERVE_TARGET*")[0])
     names = msmd.antennanames(ids)
     offset = [msmd.antennaoffset(name) for name in names]
     msmd.close()

     # Calculate the mean longitude and latitude.

     mean_longitude = numpy.mean([offset[i]["longitude offset"]\
             ['value'] for i in range(len(names))])
     mean_latitude = numpy.mean([offset[i]["latitude offset"]\
             ['value'] for i in range(len(names))])

     # Calculate the offsets from the center.

     offsets = [numpy.sqrt((offset[i]["longitude offset"]['value'] -\
             mean_longitude)**2 + (offset[i]["latitude offset"]\
             ['value'] - mean_latitude)**2) for i in \
             range(len(names))]

     # Calculate the number of flags for each antenna.

     nflags = [tb.calc('[select from '+vis+' where ANTENNA1=='+\
             str(i)+' giving  [ntrue(FLAG)]]')['0'].sum() for i in ids]

     # Calculate the median SNR for each antenna.

     if caltable != None:
         total_snr = [tb.calc('[select from '+caltable+' where ANTENNA1=='+\
                 str(i)+' giving  [sum(SNR)]]')['0'].sum() for i in ids]

     # Calculate a score based on those two.

     score = [offsets[i] / max(offsets) + nflags[i] / max(nflags) \
             for i in range(len(names))]
     if caltable != None:
         score = [score[i] + (1 - total_snr[i] / max(total_snr)) for i in range(len(names))]

     # Print out the antenna scores.

     print("Refant list for "+vis)
     #for i in numpy.argsort(score):
     #    print(names[i], score[i])
     print(','.join(numpy.array(ids)[numpy.argsort(score)].astype(str)))
     # Return the antenna names sorted by score.

     return ','.join(numpy.array(ids)[numpy.argsort(score)].astype(str))


def get_SNR_self(selfcal_library,selfcal_plan,n_ant,inf_EB_gaincal_combine,inf_EB_gaintype):
   minsolint_spw=100
   for target in selfcal_library:
    for band in selfcal_library[target].keys():
      selfcal_plan[target][band]['solint_snr'], selfcal_plan[target][band]['solint_snr_per_spw'], selfcal_plan[target][band]['solint_snr_per_bb'] = \
              get_SNR_self_individual(selfcal_library[target][band]['vislist'], selfcal_library[target][band], n_ant, selfcal_plan[target][band]['solints'], 
              selfcal_plan[target][band]['integration_time'], inf_EB_gaincal_combine, inf_EB_gaintype)

      print('Estimated SNR per solint:')
      print(target,band)
      for solint in selfcal_plan[target][band]['solints']:
        if solint == 'inf_EB':
           print('{}: {:0.2f}'.format(solint,selfcal_plan[target][band]['solint_snr'][solint]))
           '''
           for spw in solint_snr_per_spw[target][band][solint].keys():
              print('{}: spw: {}: {:0.2f}, BW: {} GHz'.format(solint,spw,solint_snr_per_spw[target][band][solint][spw],selfcal_library[target][band]['per_spw_stats'][str(spw)]['effective_bandwidth']))
              if solint_snr_per_spw[target][band][solint][spw] < minsolint_spw:
                 minsolint_spw=solint_snr_per_spw[target][band][solint][spw]
           if minsolint_spw < 3.5 and minsolint_spw > 2.5 and inf_EB_override==False:  # if below 3.5 but above 2.5 switch to gaintype T, but leave combine=scan
              print('Switching Gaintype to T for: '+target)
              inf_EB_gaintype_dict[target][band]='T'
           elif minsolint_spw < 2.5 and inf_EB_override==False:
              print('Switching Gaincal combine to spw,scan for: '+target)
              inf_EB_gaincal_combine_dict[target][band]='scan,spw' # if below 2.5 switch to combine=spw to avoid losing spws
           '''
        else:
           print('{}: {:0.2f}'.format(solint,selfcal_plan[target][band]['solint_snr'][solint]))

      for fid in selfcal_library[target][band]['sub-fields']:
          selfcal_plan[target][band][fid] = {}
          selfcal_plan[target][band][fid]['solint_snr_per_field'], selfcal_plan[target][band][fid]['solint_snr_per_field_per_spw'], selfcal_plan[target][band][fid]['solint_snr_per_field_per_bb'] = \
                  get_SNR_self_individual(selfcal_library[target][band]['vislist'], selfcal_library[target][band][fid], n_ant, 
                  selfcal_plan[target][band]['solints'], selfcal_plan[target][band]['integration_time'], inf_EB_gaincal_combine, 
                  inf_EB_gaintype)

          print('Estimated SNR per solint:')
          print(target,band,"field "+str(fid))
          for solint in selfcal_plan[target][band]['solints']:
            if solint == 'inf_EB':
               print('{}: {:0.2f}'.format(solint,selfcal_plan[target][band][fid]['solint_snr_per_field'][solint]))
               '''
               for spw in solint_snr_per_spw[target][band][solint].keys():
                  print('{}: spw: {}: {:0.2f}, BW: {} GHz'.format(solint,spw,solint_snr_per_spw[target][band][solint][spw],selfcal_library[target][band]['per_spw_stats'][str(spw)]['effective_bandwidth']))
                  if solint_snr_per_spw[target][band][solint][spw] < minsolint_spw:
                     minsolint_spw=solint_snr_per_spw[target][band][solint][spw]
               if minsolint_spw < 3.5 and minsolint_spw > 2.5 and inf_EB_override==False:  # if below 3.5 but above 2.5 switch to gaintype T, but leave combine=scan
                  print('Switching Gaintype to T for: '+target)
                  inf_EB_gaintype_dict[target][band]='T'
               elif minsolint_spw < 2.5 and inf_EB_override==False:
                  print('Switching Gaincal combine to spw,scan for: '+target)
                  inf_EB_gaincal_combine_dict[target][band]='scan,spw' # if below 2.5 switch to combine=spw to avoid losing spws
               '''
            else:
               print('{}: {:0.2f}'.format(solint,selfcal_plan[target][band][fid]['solint_snr_per_field'][solint]))

   #return solint_snr, solint_snr_per_spw, solint_snr_per_field, solint_snr_per_field_per_spw

def get_SNR_self_individual(vislist,selfcal_library,n_ant,solints,integration_time,inf_EB_gaincal_combine,inf_EB_gaintype):
      if inf_EB_gaintype=='G':
         polscale=2.0
      else:
         polscale=1.0

      SNR = max(selfcal_library['SNR_orig'], selfcal_library['intflux_orig']/selfcal_library['e_intflux_orig'])

      solint_snr = {}
      solint_snr_per_spw = {}
      solint_snr_per_bb = {}
      for solint in solints:
         solint_snr[solint]=0.0
         solint_snr_per_spw[solint]={}       
         solint_snr_per_bb[solint]={}    
         if solint == 'inf_EB':
            SNR_self_EB=np.zeros(len(selfcal_library['vislist']))
            SNR_self_EB_spw={}
            SNR_self_EB_bb={}
            for i in range(len(selfcal_library['vislist'])):
               SNR_self_EB[i]=SNR/((n_ant)**0.5*(selfcal_library['Total_TOS']/selfcal_library[selfcal_library['vislist'][i]]['TOS'])**0.5)
               SNR_self_EB_spw[selfcal_library['vislist'][i]]={}
               SNR_self_EB_bb[selfcal_library['vislist'][i]]={}
               for spw in selfcal_library['spw_map']:
                 if selfcal_library['vislist'][i] in selfcal_library['spw_map'][spw]:
                     SNR_self_EB_spw[selfcal_library['vislist'][i]][str(spw)]=(polscale)**-0.5*SNR/((n_ant-3)**0.5*(selfcal_library['Total_TOS']/selfcal_library[selfcal_library['vislist'][i]]['TOS'])**0.5)*(selfcal_library[selfcal_library['vislist'][i]]['per_spw_stats'][selfcal_library['spw_map'][spw][selfcal_library['vislist'][i]]]['effective_bandwidth']/selfcal_library[selfcal_library['vislist'][i]]['total_effective_bandwidth'])**0.5
                 print(selfcal_library[vislist[i]]['baseband'])
               print('SNR_self_EB_spw: ',SNR_self_EB_spw)
               for baseband in selfcal_library[vislist[i]]['baseband']:
                     SNR_self_EB_bb[selfcal_library['vislist'][i]][baseband]=(polscale)**-0.5*SNR/((n_ant-3)**0.5*(selfcal_library['Total_TOS']/selfcal_library[selfcal_library['vislist'][i]]['TOS'])**0.5)*(selfcal_library[selfcal_library['vislist'][i]]['baseband'][baseband]['total_effective_bandwidth']/selfcal_library[selfcal_library['vislist'][i]]['total_effective_bandwidth'])**0.5
               print('SNR_self_EB_bb: ',SNR_self_EB_bb)
            for spw in selfcal_library['spw_map']:
               mean_SNR_spw=0.0
               total_vis = 0
               for j in range(len(selfcal_library['vislist'])):
                  if selfcal_library['vislist'][j] in selfcal_library['spw_map'][spw]:
                     mean_SNR_spw+=SNR_self_EB_spw[selfcal_library['vislist'][j]][str(spw)]
                     total_vis += 1
               mean_SNR_spw=mean_SNR_spw/total_vis
               solint_snr_per_spw[solint][str(spw)]=mean_SNR_spw
            for baseband in selfcal_library[vislist[i]]['baseband']:
               mean_SNR_bb=0.0
               for j in range(len(selfcal_library['vislist'])):
                  if baseband in SNR_self_EB_bb[selfcal_library['vislist'][j]].keys():
                     mean_SNR_bb+=SNR_self_EB_bb[selfcal_library['vislist'][j]][baseband]
               mean_SNR_bb=mean_SNR_bb/len(selfcal_library['vislist']) 
               print('mean_SNR_bb',mean_SNR_bb,baseband)
               solint_snr_per_bb[solint][baseband]=mean_SNR_bb
            solint_snr[solint]=np.mean(SNR_self_EB)
            selfcal_library['per_EB_SNR']=np.mean(SNR_self_EB)
         elif solint =='scan_inf':
               selfcal_library['per_scan_SNR']=SNR/((n_ant-3)**0.5*(selfcal_library['Total_TOS']/selfcal_library['Median_scan_time'])**0.5)
               solint_snr[solint]=selfcal_library['per_scan_SNR']
               for spw in selfcal_library['spw_map']:
                  vis = list(selfcal_library['spw_map'][spw].keys())[0]
                  true_spw = selfcal_library['spw_map'][spw][vis]
                  solint_snr_per_spw[solint][str(spw)]=SNR/((n_ant-3)**0.5*(selfcal_library['Total_TOS']/selfcal_library['Median_scan_time'])**0.5)*(selfcal_library[vis]['per_spw_stats'][true_spw]['effective_bandwidth']/selfcal_library[vis]['total_effective_bandwidth'])**0.5
               for baseband in selfcal_library[vis]['baseband']:
                  solint_snr_per_bb[solint][baseband]=SNR/((n_ant-3)**0.5*(selfcal_library['Total_TOS']/selfcal_library['Median_scan_time'])**0.5)*(selfcal_library[vis]['baseband'][baseband]['total_effective_bandwidth']/selfcal_library[vis]['total_effective_bandwidth'])**0.5
         elif solint =='inf' or solint == 'inf_ap':
               selfcal_library['per_scan_SNR']=SNR/((n_ant-3)**0.5*(selfcal_library['Total_TOS']/(selfcal_library['Median_scan_time']/selfcal_library['Median_fields_per_scan']))**0.5)
               solint_snr[solint]=selfcal_library['per_scan_SNR']
               for spw in selfcal_library['spw_map']:
                  vis = list(selfcal_library['spw_map'][spw].keys())[0]
                  true_spw = selfcal_library['spw_map'][spw][vis]
                  solint_snr_per_spw[solint][str(spw)]=SNR/((n_ant-3)**0.5*(selfcal_library['Total_TOS']/(selfcal_library['Median_scan_time']/selfcal_library['Median_fields_per_scan']))**0.5)*(selfcal_library[vis]['per_spw_stats'][true_spw]['effective_bandwidth']/selfcal_library[vis]['total_effective_bandwidth'])**0.5
               for baseband in selfcal_library[vis]['baseband']:
                  solint_snr_per_bb[solint][baseband]=SNR/((n_ant-3)**0.5*(selfcal_library['Total_TOS']/(selfcal_library['Median_scan_time']/selfcal_library['Median_fields_per_scan']))**0.5)*(selfcal_library[vis]['baseband'][baseband]['total_effective_bandwidth']/selfcal_library[vis]['total_effective_bandwidth'])**0.5
         elif solint == 'int':
               solint_snr[solint]=SNR/((n_ant-3)**0.5*(selfcal_library['Total_TOS']/integration_time)**0.5)
               for spw in selfcal_library['spw_map']:
                  vis = list(selfcal_library['spw_map'][spw].keys())[0]
                  true_spw = selfcal_library['spw_map'][spw][vis]
                  solint_snr_per_spw[solint][str(spw)]=SNR/((n_ant-3)**0.5*(selfcal_library['Total_TOS']/integration_time)**0.5)*(selfcal_library[vis]['per_spw_stats'][true_spw]['effective_bandwidth']/selfcal_library[vis]['total_effective_bandwidth'])**0.5
               for baseband in selfcal_library[vis]['baseband']:
                  solint_snr_per_bb[solint][baseband]=SNR/((n_ant-3)**0.5*(selfcal_library['Total_TOS']/integration_time)**0.5)*(selfcal_library[vis]['baseband'][baseband]['total_effective_bandwidth']/selfcal_library[vis]['total_effective_bandwidth'])**0.5
         else:
               solint_float=float(solint.replace('s','').replace('_ap',''))
               solint_snr[solint]=SNR/((n_ant-3)**0.5*(selfcal_library['Total_TOS']/solint_float)**0.5)
               for spw in selfcal_library['spw_map']:
                  vis = list(selfcal_library['spw_map'][spw].keys())[0]
                  true_spw = selfcal_library['spw_map'][spw][vis]
                  solint_snr_per_spw[solint][str(spw)]=SNR/((n_ant-3)**0.5*(selfcal_library['Total_TOS']/solint_float)**0.5)*(selfcal_library[vis]['per_spw_stats'][true_spw]['effective_bandwidth']/selfcal_library[vis]['total_effective_bandwidth'])**0.5
               for baseband in selfcal_library[vis]['baseband']:
                  solint_snr_per_bb[solint][baseband]=SNR/((n_ant-3)**0.5*(selfcal_library['Total_TOS']/solint_float)**0.5)*(selfcal_library[vis]['baseband'][baseband]['total_effective_bandwidth']/selfcal_library[vis]['total_effective_bandwidth'])**0.5
      return solint_snr,solint_snr_per_spw,solint_snr_per_bb

def get_SNR_self_update(selfcal_library,selfcal_plan,n_ant,solint_curr,solint_next,integration_time,solint_snr):
   maxspws=0
   maxspwvis=''
   for vis in selfcal_library['vislist']:
      if selfcal_library[vis]['n_spws'] >= maxspws:
         maxspws=selfcal_library[vis]['n_spws']
         maxspwvis=vis+''
   SNR = max(selfcal_library[selfcal_library['vislist'][0]][solint_curr]['SNR_post'],selfcal_library[selfcal_library['vislist'][0]][solint_curr]['intflux_post']/selfcal_library[selfcal_library['vislist'][0]][solint_curr]['e_intflux_post'])

   SNR_orig = max(selfcal_library['SNR_orig'],selfcal_library['intflux_orig']/selfcal_library['e_intflux_orig'])

   SNR_ratio = SNR / SNR_orig

   #solint_snr[solint_next]=SNR_ratio*solint_snr[solint_next]
   solint_snr[solint_next]=SNR_ratio*solint_snr[solint_next]

   for spw in selfcal_library['spw_map']:
      selfcal_plan['solint_snr_per_spw'][solint_next][str(spw)]=selfcal_plan['solint_snr_per_spw'][solint_next][str(spw)]*SNR_ratio

   for baseband in selfcal_library[vis]['baseband']:
      selfcal_plan['solint_snr_per_bb'][solint_next][baseband]=selfcal_plan['solint_snr_per_bb'][solint_next][baseband]*SNR_ratio



def get_SNR_self_update_old(selfcal_library,n_ant,solint_curr,solint_next,integration_time,solint_snr):

    SNR = max(selfcal_library[selfcal_library['vislist'][0]][solint_curr]['SNR_post'],selfcal_library[selfcal_library['vislist'][0]][solint_curr]['intflux_post']/selfcal_library[selfcal_library['vislist'][0]][solint_curr]['e_intflux_post'])

    if solint_next == 'inf' or solint_next == 'inf_ap':
       selfcal_library['per_scan_SNR']=SNR/((n_ant-3)**0.5*(selfcal_library['Total_TOS']/(selfcal_library['Median_scan_time']/selfcal_library['Median_fields_per_scan']))**0.5)
       solint_snr[solint_next]=selfcal_library['per_scan_SNR']
    elif solint_next == 'scan_inf':
       selfcal_library['per_scan_SNR']=SNR/((n_ant-3)**0.5*(selfcal_library['Total_TOS']/selfcal_library['Median_scan_time'])**0.5)
       solint_snr[solint_next]=selfcal_library['per_scan_SNR']
    elif solint_next == 'int':
       solint_snr[solint_next]=SNR/((n_ant-3)**0.5*(selfcal_library['Total_TOS']/integration_time)**0.5)
    else:
       solint_float=float(solint_next.replace('s','').replace('_ap',''))
       solint_snr[solint_next]=SNR/((n_ant-3)**0.5*(selfcal_library['Total_TOS']/solint_float)**0.5)


def get_sensitivity(vislist,selfcal_library,field='',virtual_spw='all',chan=0,cellsize='0.025arcsec',imsize=1600,robust=0.5,specmode='mfs',uvtaper=''):
   for vis in vislist:
      if virtual_spw == 'all':
          im.selectvis(vis=vis,field=selfcal_library['sub-fields-fid_map'][vis][selfcal_library['sub-fields'][0]],spw=selfcal_library[vis]['spws'])
      else:
          im.selectvis(vis=vis,field=selfcal_library['sub-fields-fid_map'][vis][selfcal_library['sub-fields'][0]],spw=selfcal_library['spw_map'][virtual_spw][vis])

   im.defineimage(mode=specmode,stokes='I',spw=[0],cellx=cellsize,celly=cellsize,nx=imsize,ny=imsize)  
   im.weight(type='briggs',robust=robust)  
   if uvtaper != '':
      if 'klambda' in uvtaper:
         uvtaper=uvtaper.replace('klambda','')
         uvtaperflt=float(uvtaper)
         bmaj=str(206.0/uvtaperflt)+'arcsec'
         bmin=bmaj
         bpa='0.0deg'
      if 'arcsec' in uvtaper:
         bmaj=uvtaper
         bmin=uvtaper
         bpa='0.0deg'
      print('uvtaper: '+bmaj+' '+bmin+' '+bpa)
      im.filter(type='gaussian', bmaj=bmaj, bmin=bmin, bpa=bpa)
   try:
       estsens=np.float64(im.apparentsens()[1])
   except:
       print('#')
       print('# Sensisitivity Calculation failed for '+vis)
       print('# Continuing to next MS') 
       print('# Data in this spw/MS may be flagged')
       print('#')
       sys.exit(0)
   print('Estimated Sensitivity: ',estsens)
   im.close()
   return estsens


def LSRKfreq_to_chan(msfile, field, spw, LSRKfreq,spwsarray,minmaxchans=False):
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
    #work around the fact that spws in DATA_DESC_ID don't match listobs

    spw=int(spw)  # work around spw begin an np.uint64
    uniquespws=np.unique(spw_col)
    matching_index=np.where(spw==spwsarray)
    alt_spw=uniquespws[matching_index[0][0]]
    alt_spw=int(alt_spw) # work around spw begin an np.uint64
    #print(spw,alt_spw,matching_index[0])
    tb.close()
    obsid = np.unique(obs_col[np.where(spw_col==alt_spw)]) 
    
    tb.open(msfile+'/SPECTRAL_WINDOW')
    chanfreqs = tb.getcol('CHAN_FREQ', startrow = spw, nrow = 1)
    tb.close()
    tb.open(msfile+'/FIELD')
    fieldnames = tb.getcol('NAME')
    tb.close()
    tb.open(msfile+'/OBSERVATION')
    obstime = np.squeeze(tb.getcol('TIME_RANGE', startrow = obsid[0], nrow = 1))[0]
    tb.close()
    nchan = len(chanfreqs)
    ms.open(msfile)
    
    lsrkfreqs = ms.cvelfreqs(spwids = [spw], fieldids = int(np.where(fieldnames==field)[0][0]), mode = 'channel', nchan = nchan, \
            obstime = str(obstime)+'s', start = 0, outframe = 'LSRK') / 1e9
    ms.close()
    #print(spw,alt_spw,field,int(np.where(fieldnames==field)[0][0]))
    #print(lsrkfreqs)

    if type(LSRKfreq)==np.ndarray:
        if minmaxchans:
            #print(nchan)
            chanwidth = lsrkfreqs[1] - lsrkfreqs[0]
            channel = ((LSRKfreq - lsrkfreqs[0])/chanwidth).astype(int)
            channel_sorted = np.sort(channel)
            channel_sorted[-1] = abs(nchan-1 - channel_sorted[-1])
            return channel_sorted/nchan
        else:
            outchans = np.zeros_like(LSRKfreq)
            for i in range(len(LSRKfreq)):
                outchans[i] = np.argmin(np.abs(lsrkfreqs - LSRKfreq[i]))
        return outchans
    else:
        if minmaxchans:
           if np.argmin(np.abs(lsrkfreqs - LSRKfreq)) == 0:
              return np.argmin(np.abs(lsrkfreqs - LSRKfreq)),"min"
           elif np.argmin(np.abs(lsrkfreqs - LSRKfreq)) == nchan-1:
              return np.argmin(np.abs(lsrkfreqs - LSRKfreq)),"max"
           else:
              return np.argmin(np.abs(lsrkfreqs - LSRKfreq)),"middle"
        else:
           return np.argmin(np.abs(lsrkfreqs - LSRKfreq))

def parse_contdotdat(contdotdat_file,target):
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
    f = open(contdotdat_file,'r')
    lines = f.readlines()
    f.close()

    while '\n' in lines:
        lines.remove('\n')

    contdotdat = {}
    desiredTarget=False
    for i, line in enumerate(lines):
        if 'ALL' in line:
           continue
        if 'Flags:' in line:
           continue
        if 'Field' in line:
            field=line.split()[-1]
            if field == target:
               desiredTarget=True
               continue
            else:
               desiredTarget=False
               continue
        if desiredTarget==True:
           if 'SpectralWindow' in line:
              #code to adapt to new cont.dat format
              splitline=line.split()
              if len(splitline)==3:
                 spw = int(splitline[-2])
                 spwname=splitline[-1]
              else:
                 spw = int(splitline[-1])
                 spwname=''
              contdotdat[spw] = []
           else:
              contdotdat[spw] += [line.split()[0].split("G")[0].split("~")]

    for spw in contdotdat:
        contdotdat[spw] = np.array(contdotdat[spw], dtype=float)

    return contdotdat

def get_spwnum_refvis(vislist,target,contdotdat,spwsarray_dict):
   # calculate a score for each visibility based on which one ends up with cont.dat freq ranges that correspond to 
   # channel limits; lowest score is chosen as the reference visibility file
   spws=list(contdotdat.keys())
   score=np.zeros(len(vislist))
   for i in range(len(vislist)):
      for spw in spws:
         if spw not in spwsarray_dict[vislist[i]]:
             score[i] += 1e8
             continue

         # The score is computed as the total distance of the top and bottom of the contdotdat range for this SPW to the
         # known edges of the SPW.
         test = LSRKfreq_to_chan(vislist[i], target, spw, np.array([contdotdat[spw][0][0],contdotdat[spw][-1][1]]), \
                 spwsarray_dict[vislist[i]], minmaxchans=True)
         score[i] += test.sum()

   # Add in some penalty for being lower after sorting the vislist, as in principle the sorted order should be the order
   # that they were observed and analyzed in?
   score += np.arange(len(vislist))[np.argsort(np.argsort(vislist))]
   print(score)
   visref=vislist[np.argmin(score)]            
   return visref

def flagchannels_from_contdotdat(vis,target,spwsarray,vislist,spwvisref,contdotdat,return_contfit_range=False):
    """
    Generates a string with the list of lines identified by the cont.dat file from the ALMA pipeline, that need to be flagged.

    Parameters
    ==========
    ms_dict: Dictionary of information about measurement set

    Returns
    =======
    String of channels to be flagged, in a format that can be passed to the spw parameter in CASA's flagdata task. 
    """

    flagchannels_string = ''
    #moved out of function to not for each MS for efficiency
    #contdotdat = parse_contdotdat('cont.dat',target)
    #spwvisref=get_spwnum_refvis(vislist,target,contdotdat,spwsarray)
    for j,spw in enumerate(contdotdat):
        msmd.open(spwvisref)
        spwname=msmd.namesforspws(spw)[0]
        msmd.close()
        msmd.open(vis)
        spws=msmd.spwsfornames(spwname)
        msmd.close()
        # must directly cast to int, otherwise the CASA tool call does not like numpy.uint64
        #loop through returned spws to see which is in the spw array rather than assuming, because assumptions be damned
        trans_spw = -1
        for check_spw in spws[spwname]:
           matching_index=np.where(check_spw == spwsarray)
           if len(matching_index[0]) == 0:
              continue
           else:
              trans_spw=check_spw
              break
        if trans_spw == -1:
           print('COULD NOT DETERMINE SPW MAPPING FOR CONT.DAT, PROCEEDING WITHOUT FLAGGING FOR '+vis)
           return ''
        #trans_spw=int(np.max(spws[spwname])) # assume higher number spw is the correct one, generally true with ALMA data structure

        flagchannels_string += '%d:' % (trans_spw)
        tb.open(vis+'/SPECTRAL_WINDOW')
        nchan = tb.getcol('CHAN_FREQ', startrow = trans_spw, nrow = 1).size
        tb.close()

        chans = np.array([])
        for k in range(contdotdat[spw].shape[0]):
            print(trans_spw, contdotdat[spw][k])

            chans = np.concatenate((LSRKfreq_to_chan(vis, target, trans_spw, contdotdat[spw][k],spwsarray),chans))

            """
            if flagchannels_string == '':
                flagchannels_string+='%d:%d~%d' % (spw, np.min([chans[0], chans[1]]), np.max([chans[0], chans[1]]))
            else:
                flagchannels_string+=', %d:%d~%d' % (spw, np.min([chans[0], chans[1]]), np.max([chans[0], chans[1]]))
            """

        chans = np.sort(chans)
        if not return_contfit_range:
           flagchannels_string += '0~%d;' % (chans[0])
           for i in range(1,chans.size-1,2):
               flagchannels_string += '%d~%d;' % (chans[i], chans[i+1])
           flagchannels_string += '%d~%d, ' % (chans[-1], nchan-1)
        else:
           for i in range(0,chans.size-1,2):
               flagchannels_string += '%d~%d;' % (chans[i], chans[i+1])
           flagchannels_string=flagchannels_string[:-1]+ ',' 
    if not return_contfit_range:
       print("# Flagchannels input string for %s in %s from cont.dat file: \'%s\'" % (target, vis, flagchannels_string))
    else:    
       flagchannels_string=flagchannels_string[:-1]
       print("# Cont range string for %s in %s from cont.dat file: \'%s\'" % (target, vis, flagchannels_string))
    return flagchannels_string

def get_fitspw_dict(vis,target,spwsarray,vislist,spwvisref,contdotdat,fitorder=1):
    """
    Generates a string with the list of lines identified by the cont.dat file from the ALMA pipeline, that need to be flagged.

    Parameters
    ==========
    ms_dict: Dictionary of information about measurement set

    Returns
    =======
    String of channels to be flagged, in a format that can be passed to the spw parameter in CASA's flagdata task. 
    """

    fitspw_dict = {}
    #moved out of function to not for each MS for efficiency
    #contdotdat = parse_contdotdat('cont.dat',target)
    #spwvisref=get_spwnum_refvis(vislist,target,contdotdat,spwsarray)
    for j,spw in enumerate(contdotdat):
        msmd.open(spwvisref)
        spwname=msmd.namesforspws(spw)[0]
        msmd.close()
        msmd.open(vis)
        spws=msmd.spwsfornames(spwname)
        msmd.close()
        # must directly cast to int, otherwise the CASA tool call does not like numpy.uint64
        #loop through returned spws to see which is in the spw array rather than assuming, because assumptions be damned
        trans_spw = -1
        for check_spw in spws[spwname]:
           matching_index=np.where(check_spw == spwsarray)
           if len(matching_index[0]) == 0:
              continue
           else:
              trans_spw=check_spw
              break
        if trans_spw==-1:
           print('COULD NOT DETERMINE SPW MAPPING FOR CONT.DAT, PROCEEDING WITHOUT FLAGGING FOR '+vis)
           return ''
        #trans_spw=int(np.max(spws[spwname])) # assume higher number spw is the correct one, generally true with ALMA data structure
        #flagchannels_string += '%d:' % (trans_spw)
        tb.open(vis+'/SPECTRAL_WINDOW')
        nchan = tb.getcol('CHAN_FREQ', startrow = trans_spw, nrow = 1).size
        tb.close()

        chans = np.array([])
        for k in range(contdotdat[spw].shape[0]):
            print(trans_spw, contdotdat[spw][k])

            chans = np.concatenate((LSRKfreq_to_chan(vis, target, trans_spw, contdotdat[spw][k],spwsarray),chans))

            """
            if flagchannels_string == '':
                flagchannels_string+='%d:%d~%d' % (spw, np.min([chans[0], chans[1]]), np.max([chans[0], chans[1]]))
            else:
                flagchannels_string+=', %d:%d~%d' % (spw, np.min([chans[0], chans[1]]), np.max([chans[0], chans[1]]))
            """
            chans = np.sort(chans)
            fitspw_dict[str(trans_spw)] = {}
            fitspw_dict[str(trans_spw)]['fitorder']=fitorder
            flagchannels_string=''
            for i in range(0,chans.size-1,2):
               flagchannels_string += '%d~%d;' % (chans[i], chans[i+1])
            fitspw_dict[str(trans_spw)]['chan']=flagchannels_string[:-1]

    if len(fitspw_dict) == 0:
        print("WARNING: No entry found in cont.dat for target "+target+", fitting all channels for continuum range.")
        fitspw_dict[','.join(spwsarray.astype(str))] = ''

    return fitspw_dict

def get_spw_chanwidths(vis,spwarray):
   widtharray=np.zeros(len(spwarray))
   bwarray=np.zeros(len(spwarray))
   nchanarray=np.zeros(len(spwarray))
   for i in range(len(spwarray)):
      tb.open(vis+'/SPECTRAL_WINDOW')
      widtharray[i]=np.abs(np.unique(tb.getcol('CHAN_WIDTH', startrow = spwarray[i], nrow = 1)))
      bwarray[i]=np.abs(np.unique(tb.getcol('TOTAL_BANDWIDTH', startrow = spwarray[i], nrow = 1)))
      nchanarray[i]=np.abs(np.unique(tb.getcol('NUM_CHAN', startrow = spwarray[i], nrow = 1)))
      tb.close()

   return widtharray,bwarray,nchanarray

def get_spw_bandwidth(vis,spwsarray_dict,target,vislist):
   spwbws={}
   spwfreqs={}
   for spw in spwsarray_dict[vis]:
      tb.open(vis+'/SPECTRAL_WINDOW')
      spwbws[spw]=np.abs(np.unique(tb.getcol('TOTAL_BANDWIDTH', startrow = spw, nrow = 1)))[0]/1.0e9 # put bandwidths into GHz
      tb.close()
      msmd.open(vis)
      spwfreqs[spw]=msmd.meanfreq(spw)
      msmd.close()
   spweffbws=spwbws.copy()
   if os.path.exists("cont.dat"):
      spweffbws=get_spw_eff_bandwidth(vis,target,vislist,spwsarray_dict)

   return spwbws,spweffbws,spwfreqs


def get_spw_eff_bandwidth(vis,target,vislist,spwsarray_dict):
   spweffbws={}
   contdotdat=parse_contdotdat('cont.dat',target)

   spwvisref=get_spwnum_refvis(vislist,target,contdotdat,spwsarray_dict)
   for key in contdotdat.keys():
      msmd.open(spwvisref)
      spwname=msmd.namesforspws(key)[0]
      msmd.close()
      msmd.open(vis)
      spws=msmd.spwsfornames(spwname)
      msmd.close()
      trans_spw=-1
      # must directly cast to int, otherwise the CASA tool call does not like numpy.uint64
      #loop through returned spws to see which is in the spw array rather than assuming, because assumptions be damned
      for check_spw in spws[spwname]:
         matching_index=np.where(check_spw == spwsarray_dict[vis])
         if len(matching_index[0]) == 0:
              continue
         else:
              trans_spw=check_spw
              break
      #trans_spw=int(np.max(spws[spwname])) # assume higher number spw is the correct one, generally true with ALMA data structure
      cumulat_bw=0.0
      for i in range(len(contdotdat[key])):
         cumulat_bw+=np.abs(contdotdat[key][i][1]-contdotdat[key][i][0])
      spweffbws[trans_spw]=cumulat_bw+0.0
   return spweffbws
   



def get_spw_chanavg(vis,widtharray,bwarray,chanarray,desiredWidth=15.625e6):
   avgarray=np.zeros(len(widtharray))
   for i in range(len(widtharray)):
      if desiredWidth > bwarray[i]:
         avgarray[i]=chanarray[i]
      else:
         nchan=bwarray[i]/desiredWidth
         nchan=np.round(nchan)
         avgarray[i]=chanarray[i]/nchan   
      if avgarray[i] < 1.0:
         avgarray[i]=1.0
   return avgarray



def get_spw_map(selfcal_library, target, band, telescope):
    # Get the list of EBs from the selfcal_library
    vislist = selfcal_library[target][band]['vislist'].copy()

    # If we are looking at VLA data, find the EB with the maximum number of SPWs so that we have the fewest "odd man out" SPWs hanging out at the end as possible.
    if "VLA" in telescope:
        maxspws=0
        maxspwvis=''
        for vis in vislist:
           if selfcal_library[target][band][vis]['n_spws'] >= maxspws:
              maxspws=selfcal_library[target][band][vis]['n_spws']
              maxspwvis=vis+''

        vislist.remove(maxspwvis)
        vislist = [maxspwvis] + vislist

    spw_map = {}
    reverse_spw_map = {}
    virtual_index = 0
    # This code is meant to be generic in order to prepare for cases where multiple EBs might have unique SPWs in them (e.g. inhomogeneous data),
    # but the criterea for which SPWs match will need to be updated for this to truly generalize.
    for vis in vislist:
        reverse_spw_map[vis] = {}
        for spw in selfcal_library[target][band][vis]['spwsarray']:
            found_match = False
            for s in spw_map:
                for v in spw_map[s].keys():
                   if vis == v:
                       continue

                   if telescope == "ALMA" or telescope == "ACA":
                       # NOTE: This assumes that matching based on SPW name is ok. Fine for now... but will need to update this for inhomogeneous data.
                       msmd.open(vis)
                       spwname=msmd.namesforspws(spw)[0]
                       msmd.close()

                       msmd.open(v)
                       sname = msmd.namesforspws(spw_map[s][v])[0]
                       msmd.close()

                       if spwname == sname:
                           found_match = True
                   elif 'VLA' in telescope:
                       msmd.open(vis)
                       bandwidth1 = msmd.bandwidths(spw)
                       chanwidth1 = msmd.chanwidths(spw)[0]
                       chanfreq1 = msmd.chanfreqs(spw)[0]
                       msmd.close()

                       msmd.open(v)
                       bandwidth2 = msmd.bandwidths(spw_map[s][v])
                       chanwidth2 = msmd.chanwidths(spw_map[s][v])[0]
                       chanfreq2 = msmd.chanfreqs(spw_map[s][v])[0]
                       msmd.close()

                       if bandwidth1 == bandwidth2 and chanwidth1 == chanwidth2 and chanfreq1 == chanfreq2:
                           found_match = True

                   if found_match:
                       spw_map[s][vis] = spw
                       reverse_spw_map[vis][spw] = s
                       break

                if found_match:
                    break

            if not found_match:
                spw_map[virtual_index] = {}
                spw_map[virtual_index][vis] = spw
                reverse_spw_map[vis][spw] = virtual_index
                virtual_index += 1

    print("spw_map:")
    print(spw_map)
    print("reverse_spw_map:")
    print(reverse_spw_map)
    return spw_map, reverse_spw_map


def largest_prime_factor(n):
    i = 2
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
    return n


def get_image_parameters(vislist,telescope,target,field_ids,band,selfcal_library,scale_fov=1.0,mosaic=False):
   cells=np.zeros(len(vislist))
   for i in range(len(vislist)):
      #im.open(vislist[i])
      im.selectvis(vis=vislist[i],spw=selfcal_library[target][band][vislist[i]]['spwsarray'])
      adviseparams= im.advise() 
      cells[i]=adviseparams[2]['value']/2.0
      im.close()
   cell=np.min(cells)
   cellsize='{:0.3f}arcsec'.format(cell)
   nterms=1
   if selfcal_library[target][band]['fracbw'] > 0.1:
      nterms=2
   reffreq = get_reffreq(vislist,field_ids,dict(zip(vislist,[selfcal_library[target][band][vis]['spwsarray'] for vis in vislist])), telescope)

   if 'VLA' in telescope:
      fov=45.0e9/selfcal_library[target][band]['meanfreq']*60.0*1.5
      if selfcal_library[target][band]['meanfreq'] < 12.0e9:
         fov=fov*2.0
   if telescope=='ALMA':
      fov=63.0*100.0e9/selfcal_library[target][band]['meanfreq']*1.5
   if telescope=='ACA':
      fov=108.0*100.0e9/selfcal_library[target][band]['meanfreq']*1.5
   fov=fov*scale_fov
   if mosaic:
       msmd.open(vislist[0])
       #get field IDs for VLA and and ALMA differently
       if telescope == 'ALMA':
          fieldid=msmd.fieldsforname(target)
       elif 'VLA' in telescope:
          fieldid=np.array([],dtype=int)
          for fid in selfcal_library[target][band]['sub-fields']:
             if fid in selfcal_library[target][band]['sub-fields-fid_map'][vislist[0]].keys():
                field_id=selfcal_library[target][band]['sub-fields-fid_map'][vislist[0]][fid]
                fieldid=np.append(fieldid,np.array([field_id]))

       ra_phasecenter_arr=np.zeros(len(fieldid))
       dec_phasecenter_arr=np.zeros(len(fieldid))
       for i in range(len(fieldid)):
          phasecenter=msmd.phasecenter(fieldid[i])
          ra_phasecenter_arr[i]=phasecenter['m0']['value']
          dec_phasecenter_arr[i]=phasecenter['m1']['value']
       msmd.done()

       mosaic_size = max(ra_phasecenter_arr.max() - ra_phasecenter_arr.min(), 
               dec_phasecenter_arr.max() - dec_phasecenter_arr.min()) * 180./np.pi * 3600.

       fov += mosaic_size

   npixels=int(np.ceil(fov/cell / 100.0)) * 100
   if npixels > 16384:
      if mosaic:
          print("WARNING: Image size = "+str(npixels)+" is excessively large. It is not being trimmed because it is needed for the mosaic, but this may not be viable for your hardware.")
      else:
          npixels=16384

   while largest_prime_factor(npixels) >= 7:
       npixels += 2

   return cellsize,npixels,nterms,reffreq


def check_image_nterms(fracbw, SNR):
   if fracbw >=0.1:
      nterms=2
   elif (SNR > 10.0) and (fracbw < 0.1):   # estimate the gain of going to nterms=2 based on nterms=1 S/N and fracbw
      #coefficients come from a empirical fit using simulated data with a spectral index of 3
      X=[fracbw,np.log10(SNR)]
      A = 2336.415
      B = 0.051
      C = -306.590
      D = 5.654
      E = 28.220
      F = -23.598
      G = -0.594
      H = -3.413 
      Z=10**(A*X[0]**3+B*X[1]**3+C*X[0]**2*X[1]+D*X[1]**2*X[0] +E*X[0]*X[1]+ F*X[0]+ G*X[1] +H)
      if Z > 0.01:
         print('SWITCHING TO NTERMS=2')
         nterms=2
      else:
         nterms=1
   else:
      nterms=1
   return nterms

def get_mean_freq(vis,spwsarray):
   tb.open(vis[0]+'/SPECTRAL_WINDOW')
   freqarray=tb.getcol('REF_FREQUENCY')
   tb.close()
   meanfreq=np.mean(freqarray[spwsarray])
   minfreq=np.min(freqarray[spwsarray])
   maxfreq=np.max(freqarray[spwsarray])
   fracbw=np.abs(maxfreq-minfreq)/meanfreq
   return meanfreq, maxfreq,minfreq,fracbw

def get_reffreq(vislist,field_ids,spwsarray,telescope):
    meanfreqs = []
    weights = []
    bandwidth_weights = []
    all_bandwidths = []
    for vis in vislist:
        for fid in field_ids[vis]:
            msmd.open(vis)
            spws = spwsarray[vis]
            data_desc_ids = [msmd.datadescids(spw)[0] for spw in spws]
            meanfreqs += [msmd.meanfreq(spw) for spw in spws]
            bandwidths = msmd.bandwidths(spws)
            all_bandwidths += bandwidths.tolist()
            msmd.close()

            weights += [tb.calc('sum([select from '+vis+' where DATA_DESC_ID=={0:d} && FIELD_ID=={1:d} giving [sum(WEIGHT)*nFalse(FLAG)]])'.\
                    format(spw, fid))[0] for spw in spws]

            nflags = np.array([tb.calc('sum([select from '+vis+' where DATA_DESC_ID=={0:d} && FIELD_ID=={1:d} giving [nTrue(FLAG)]])'.\
                    format(data_desc_id, fid))[0] for data_desc_id in data_desc_ids])
            nunflagged = np.array([tb.calc('sum([select from '+vis+' where DATA_DESC_ID=={0:d} && FIELD_ID=={1:d} giving [nFalse(FLAG)]])'.\
                    format(data_desc_id, fid))[0] for data_desc_id in data_desc_ids])
            unflagged_fraction = nunflagged / (nflags + nunflagged)

            bandwidth_weights += (unflagged_fraction*bandwidths).tolist()

    meanfreqs, weights = np.array(meanfreqs), np.array(weights)
    bandwidth_weights = np.array(bandwidth_weights)
    all_bandwidths = np.array(all_bandwidths)

    sumwt_reffreq = (meanfreqs*weights).sum() / (weights).sum()
    bandwidth_reffreq = (meanfreqs*bandwidth_weights).sum() / (bandwidth_weights).sum()

    #mean_total_bandwidth = np.sum(all_bandwidths) / len(vislist)
    bandwidth_spread = np.max(meanfreqs + 0.5*all_bandwidths) - np.min(meanfreqs - 0.5*all_bandwidths)

    print("Bandwidth reffreq = ", bandwidth_reffreq/1e9, "GHz")
    print("SumWt reffreq = ", sumwt_reffreq/1e9, "GHz")
    print("Diff = ", np.abs(bandwidth_reffreq - sumwt_reffreq)/1e9, "GHz")
    print("Bandwidth Spread = ", bandwidth_spread/1e9, "GHz")
    print("0.05*Bandwidth Spread = ", 0.05*bandwidth_spread/1e9, "GHz")
    if abs(bandwidth_reffreq - sumwt_reffreq) > 0.1*bandwidth_spread:
        print("Using sumwt reffreq")
        reffreq = sumwt_reffreq
    else:
        print("Using bandwidth reffreq")
        reffreq = bandwidth_reffreq

    return str(reffreq/1e9)+"GHz"


def get_desired_width(meanfreq):
   if meanfreq >= 50.0e9:
      desiredWidth=15.625e6
   elif (meanfreq < 50.0e9) and (meanfreq >=40.0e9):
      desiredWidth=16.0e6
   elif (meanfreq < 40.0e9) and (meanfreq >=26.0e9):
      desiredWidth=8.0e6
   elif (meanfreq < 26.0e9) and (meanfreq >=18.0e9):
      desiredWidth=8.0e6
   elif (meanfreq < 18.0e9) and (meanfreq >=8.0e9):
      desiredWidth=8.0e6
   elif (meanfreq < 8.0e9) and (meanfreq >=4.0e9):
      desiredWidth=4.0e6
   elif (meanfreq < 4.0e9) and (meanfreq >=2.0e9):
      desiredWidth=4.0e6
   elif (meanfreq < 4.0e9):
      desiredWidth=2.0e6
   return desiredWidth



def get_bands(vislist,fields,telescope):
   observed_bands={}
   for vis in vislist:
      observed_bands[vis]={}
      msmd.open(vis)
      spws_for_field=np.array([])
      for field in fields:
         spws_temp=msmd.spwsforfield(field)
         spws_for_field=np.concatenate((spws_for_field,np.array(spws_temp)))
      msmd.close()
      spws_for_field=np.unique(spws_for_field)
      spws_for_field.sort()
      spws_for_field=spws_for_field.astype('int')
      #visheader=vishead(vis,mode='list',listitems=[])
      tb.open(vis+'/SPECTRAL_WINDOW') 
      spw_names=tb.getcol('NAME')
      tb.close()
      #spw_names=visheader['spw_name'][0]
      spw_names_band=['']*len(spws_for_field)
      spw_names_band=['']*len(spws_for_field)
      spw_names_bb=['']*len(spws_for_field)
      spw_names_spw=np.zeros(len(spw_names_band)).astype('int')

      if 'VLA' in telescope:
         for i in range(len(spws_for_field)):
            spw_names_band[i]=spw_names[spws_for_field[i]].split('#')[0]
            spw_names_bb[i]=spw_names[spws_for_field[i]].split('#')[1]
            spw_names_spw[i]=spws_for_field[i]     
         all_bands=np.unique(spw_names_band)
         observed_bands[vis]['n_bands']=len(all_bands)
         observed_bands[vis]['bands']=all_bands.tolist()
         for band in all_bands:
            index=np.where(np.array(spw_names_band)==band)
            observed_bands[vis][band]={}
            # logic below removes the VLA standard pointing setups at X and C-bands
            # the code is mostly immune to this issue since we get the spws for only
            # the science targets above; however, should not ignore the possibility
            # that someone might also do pointing on what is the science target
            if (band == 'EVLA_X') and (len(index[0]) >= 2): # ignore pointing band
               observed_bands[vis][band]['spwarray']=spw_names_spw[index[0]]
               indices_to_remove=np.array([])
               for i in range(len(observed_bands[vis][band]['spwarray'])):
                   meanfreq,maxfreq,minfreq,fracbw=get_mean_freq([vis],np.array([observed_bands[vis][band]['spwarray'][i]]))
                   if (meanfreq==8.332e9) or (meanfreq==8.460e9):
                      indices_to_remove=np.append(indices_to_remove,[i])
               observed_bands[vis][band]['spwarray']=np.delete(observed_bands[vis][band]['spwarray'],indices_to_remove.astype(int))
            elif (band == 'EVLA_C') and (len(index[0]) >= 2): # ignore pointing band

               observed_bands[vis][band]['spwarray']=spw_names_spw[index[0]]
               indices_to_remove=np.array([])
               for i in range(len(observed_bands[vis][band]['spwarray'])):
                   meanfreq,maxfreq,minfreq,fracbw=get_mean_freq([vis],np.array([observed_bands[vis][band]['spwarray'][i]]))
                   if (meanfreq==4.832e9) or (meanfreq==4.960e9):
                      indices_to_remove=np.append(indices_to_remove,[i])
               observed_bands[vis][band]['spwarray']=np.delete(observed_bands[vis][band]['spwarray'],indices_to_remove.astype(int))
            else:
               observed_bands[vis][band]['spwarray']=spw_names_spw[index[0]]
            spwslist=observed_bands[vis][band]['spwarray'].tolist()
            spwstring=','.join(str(spw) for spw in spwslist)
            observed_bands[vis][band]['spwstring']=spwstring+''
            observed_bands[vis][band]['meanfreq'],observed_bands[vis][band]['maxfreq'],observed_bands[vis][band]['minfreq'],observed_bands[vis][band]['fracbw']=get_mean_freq([vis],observed_bands[vis][band]['spwarray'])
            observed_bands[vis][band]['baseband']=get_basebands(observed_bands,vis,band,observed_bands[vis][band]['spwarray'])

            msmd.open(vis)
            observed_bands[vis][band]['ncorrs']=msmd.ncorrforpol(msmd.polidfordatadesc(observed_bands[vis][band]['spwarray'][0]))
            msmd.close()
      if telescope == 'ALMA' or telescope == 'ACA':
         meanfreq, maxfreq,minfreq,fracbw=get_mean_freq(vislist,spws_for_field)
         band=get_ALMA_band_string(meanfreq)
         bands=[band]
         observed_bands[vis]['bands']=[band]
         for band in bands:
            observed_bands[vis][band]={}
            observed_bands[vis][band]['spwarray']=np.array(spws_for_field)
            observed_bands[vis][band]['spwstring']=','.join(str(spw) for spw in spws_for_field)
            observed_bands[vis][band]['meanfreq']=meanfreq
            observed_bands[vis][band]['maxfreq']=maxfreq
            observed_bands[vis][band]['minfreq']=minfreq
            observed_bands[vis][band]['fracbw']=fracbw
            observed_bands[vis][band]['baseband']=get_basebands(observed_bands,vis,band,observed_bands[vis][band]['spwarray'])

            msmd.open(vis)
            observed_bands[vis][band]['ncorrs']=msmd.ncorrforpol(msmd.polidfordatadesc(spws_for_field[0]))
            msmd.close()

   if 'VLA' in telescope:
      bands_match=True
  
      for i in range(len(vislist)):
         for j in range(i+1,len(vislist)):
            bandlist_match=(observed_bands[vislist[i]]['bands'] ==observed_bands[vislist[i+1]]['bands'])
            if not bandlist_match:
               bands_match=False
      if not bands_match:
        print('WARNING: INCONSISTENT BANDS IN THE MSFILES')

      get_max_uvdist(vislist,observed_bands[vislist[0]]['bands'].copy(),observed_bands,'VLA')
   elif telescope == 'ALMA' or telescope == 'ACA':
      get_max_uvdist(vislist,observed_bands[vislist[0]]['bands'].copy(),observed_bands,'ALMA')   
      
   

   return observed_bands[vislist[0]]['bands'].copy(),observed_bands

def get_ALMA_band_string(meanfreq):
   if (meanfreq < 950.0e9) and (meanfreq >=787.0e9):
      band='Band_10'
   elif (meanfreq < 720.0e9) and (meanfreq >=602.0e9):
      band='Band_9'
   elif (meanfreq < 500.0e9) and (meanfreq >=385.0e9):
      band='Band_8'
   elif (meanfreq < 373.0e9) and (meanfreq >=275.0e9):
      band='Band_7'
   elif (meanfreq < 275.0e9) and (meanfreq >=211.0e9):
      band='Band_6'
   elif (meanfreq < 211.0e9) and (meanfreq >=163.0e9):
      band='Band_5'
   elif (meanfreq < 163.0e9) and (meanfreq >=125.0e9):
      band='Band_4'
   elif (meanfreq < 116.0e9) and (meanfreq >=84.0e9):
      band='Band_3'
   elif (meanfreq < 84.0e9) and (meanfreq >=67.0e9):
      band='Band_2'
   elif (meanfreq < 50.0e9) and (meanfreq >=30.0e9):
      band='Band_1'
   return band

def get_basebands(observed_bands,vis,band,spwarray):
   observed_bands[vis][band]['baseband']={}
   baseband_array=np.array([])
   msmd.open(vis)
   baseband_dict=msmd.spwsforbaseband()
   for baseband in baseband_dict.keys():
      spw_overlap=np.intersect1d(baseband_dict[baseband],spwarray)
      if len(spw_overlap) > 0:
         observed_bands[vis][band]['baseband'][str(baseband)]={} 
         observed_bands[vis][band]['baseband'][str(baseband)]['spwstring']=''
         observed_bands[vis][band]['baseband'][str(baseband)]['spwarray']=np.array(spw_overlap)
         observed_bands[vis][band]['baseband'][str(baseband)]['nspws']=len(observed_bands[vis][band]['baseband'][str(baseband)]['spwarray'])
         for spw in spw_overlap:                  # loop through all spws and put them with their respective basebands
            if observed_bands[vis][band]['baseband'][str(int(baseband))]['spwstring'] == '':
               observed_bands[vis][band]['baseband'][str(int(baseband))]['spwstring']+=str(int(spw)) 
            else:
               observed_bands[vis][band]['baseband'][str(int(baseband))]['spwstring']+=','+str(int(spw))
         observed_bands[vis][band]['baseband'][str(baseband)]['spwlist']=observed_bands[vis][band]['baseband'][str(baseband)]['spwarray'].tolist()
   msmd.close()
   return observed_bands[vis][band]['baseband']



def get_telescope(vis):
   visheader=vishead(vis,mode='list',listitems=[])
   telescope=visheader['telescope'][0][0]
   if telescope == 'ALMA':
      tb.open(vis+'/ANTENNA')
      ant_diameter=np.unique(tb.getcol('DISH_DIAMETER'))[0]
      if ant_diameter==7.0:
         telescope='ACA'
   return telescope
      
def get_dr_correction(telescope,dirty_peak,theoretical_sens,vislist):
   dirty_dynamic_range=dirty_peak/theoretical_sens
   n_dr_max=2.5
   n_dr=1.0
   tlimit=2.0
   if telescope=='ALMA':
      if dirty_dynamic_range > 150.:
                    maxSciEDR = 150.0
                    new_threshold = np.max([n_dr_max * theoretical_sens, dirty_peak / maxSciEDR * tlimit])
                    n_dr=new_threshold/theoretical_sens
      else:
                    if dirty_dynamic_range > 100.:
                        n_dr = 2.5
                    elif 50. < dirty_dynamic_range <= 100.:
                        n_dr = 2.0
                    elif 20. < dirty_dynamic_range <= 50.:
                        n_dr = 1.5
                    elif dirty_dynamic_range <= 20.:
                        n_dr = 1.0
   if telescope=='ACA':
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
         n_dr=new_threshold/theoretical_sens
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
     ids = msmd.antennasforscan(msmd.scansforintent("*OBSERVE_TARGET*")[0])
     names = msmd.antennanames(ids)
     offset = [msmd.antennaoffset(id) for id in ids]
     msmd.close()
     baselines=np.array([])
     for i in range(len(offset)):
        for j in range(i+1,len(offset)):
           baseline = numpy.sqrt((offset[i]["longitude offset"]['value'] -\
             offset[j]["longitude offset"]['value'])**2 + (offset[i]["latitude offset"]\
             ['value'] - offset[j]["latitude offset"]['value'])**2)
           
           baselines=np.append(baselines,np.array([baseline]))
     return baselines



def get_max_uvdist(vislist,bands,band_properties,telescope):
   for band in bands:   
      all_baselines=np.array([])
      for vis in vislist:
         baselines=get_baseline_dist(vis)
         all_baselines=np.append(all_baselines,baselines)
      max_baseline=np.max(all_baselines)
      min_baseline=np.min(all_baselines)
      if 'VLA' in telescope:
         baseline_5=numpy.percentile(all_baselines[all_baselines > 0.05*all_baselines.max()],5.0)
      else: # ALMA
         baseline_5=numpy.percentile(all_baselines,5.0)
      baseline_75=numpy.percentile(all_baselines,75.0)
      baseline_median=numpy.percentile(all_baselines,50.0)
      for vis in vislist:
         meanlam=3.0e8/band_properties[vis][band]['meanfreq']
         max_uv_dist=max_baseline # leave maxuv in meters like the other uv entries /meanlam/1000.0
         min_uv_dist=min_baseline
         band_properties[vis][band]['maxuv']=max_uv_dist
         band_properties[vis][band]['minuv']=min_uv_dist
         band_properties[vis][band]['75thpct_uv']=baseline_75
         band_properties[vis][band]['median_uv']=baseline_median
         band_properties[vis][band]['LAS']=0.6 * (meanlam/baseline_5) * 180./np.pi * 3600.


def get_uv_range(band,band_properties,vislist):
   if (band == 'EVLA_C') or (band == 'EVLA_X') or (band == 'EVLA_S') or (band == 'EVLA_L'):
      n_vis=len(vislist)
      mean_max_uv=0.0
      for vis in vislist:
         mean_max_uv+=band_properties[vis][band]['maxuv']
      mean_max_uv=mean_max_uv/float(n_vis)
      min_uv=0.05*mean_max_uv
      uvrange='>{:0.2f}m'.format(min_uv)
   else:
      uvrange=''
   return uvrange

def sanitize_string(string):
   sani_string=string.replace('-','_').replace(' ','_').replace('+','_').replace('*','_').replace(',','_').replace(';','_').replace(':','_').replace('[','_').replace(']','_').replace('{','_').replace('}','_')

   sani_string='Target_'+sani_string
   return sani_string


def compare_beams(image1, image2):
    header_1 = imhead(image1, mode = 'list')
    beammajor_1 = header_1['beammajor']['value']
    beamminor_1 = header_1['beamminor']['value']
    beampa_1 = header_1['beampa']['value']

    header_2 = imhead(image2, mode = 'list')
    beammajor_2 = header_2['beammajor']['value']
    beamminor_2 = header_2['beamminor']['value']
    beampa_2 = header_2['beampa']['value']
    beamarea_1=beammajor_1*beamminor_1
    beamarea_2=beammajor_2*beamminor_2
    delta_beamarea=(beamarea_2-beamarea_1)/beamarea_1
    return delta_beamarea

def get_sols_flagged_solns(gaintable):
   tb.open(gaintable)
   flags=tb.getcol('FLAG').squeeze()
   nsols=flags.size
   flagged_sols=np.where(flags==True)
   nflagged_sols=flagged_sols[0].size
   return nflagged_sols, nsols

def plot_ants_flagging_colored(filename,vis,gaintable):
   names, offset_x,offset_y, offsets, nflags, nunflagged,fracflagged=get_flagged_solns_per_ant(gaintable,vis)
   import matplotlib
   matplotlib.use('Agg')
   import matplotlib.pyplot as plt
   ants_zero_flagging=np.where(fracflagged == 0.0)
   ants_lt10pct_flagging=((fracflagged <= 0.1) & (fracflagged > 0.0)).nonzero()
   ants_lt25pct_flagging=((fracflagged <= 0.25) & (fracflagged > 0.10)).nonzero()
   ants_lt50pct_flagging=((fracflagged <= 0.5) & (fracflagged > 0.25)).nonzero()
   ants_lt75pct_flagging=((fracflagged <= 0.75) & (fracflagged > 0.5)).nonzero()
   ants_gt75pct_flagging=np.where(fracflagged > 0.75)
   fig, ax = plt.subplots(1,1,figsize=(12, 12))
   ax.scatter(offset_x[ants_zero_flagging[0]],offset_y[ants_zero_flagging[0]],marker='o',color='green',label='No Flagging',s=120)
   ax.scatter(offset_x[ants_lt10pct_flagging[0]],offset_y[ants_lt10pct_flagging[0]],marker='o',color='blue',label='<10% Flagging',s=120)
   ax.scatter(offset_x[ants_lt25pct_flagging[0]],offset_y[ants_lt25pct_flagging[0]],marker='o',color='yellow',label='<25% Flagging',s=120)
   ax.scatter(offset_x[ants_lt50pct_flagging[0]],offset_y[ants_lt50pct_flagging[0]],marker='o',color='magenta',label='<50% Flagging',s=120)
   ax.scatter(offset_x[ants_lt75pct_flagging[0]],offset_y[ants_lt75pct_flagging[0]],marker='o',color='cyan',label='<75% Flagging',s=120)
   ax.scatter(offset_x[ants_gt75pct_flagging[0]],offset_y[ants_gt75pct_flagging[0]],marker='o',color='black',label='>75% Flagging',s=120)
   ax.legend(fontsize=20)
   for i in range(len(names)):
      ax.text(offset_x[i],offset_y[i],names[i])
   ax.set_xlabel('Latitude Offset (m)',fontsize=20)
   ax.set_ylabel('Longitude Offset (m)',fontsize=20)
   ax.set_title('Antenna Positions colorized by Selfcal Flagging',fontsize=20)
   plt.savefig(filename,dpi=200.0)
   plt.close()

def get_flagged_solns_per_ant_from_dict(gc_dict_list,spwlist,vis):
   msmd.open(vis)
   antids=[]
   for ant in [idant for idant in gc_dict_list[0]['solvestats']['spw'+str(spwlist[0])].keys() if idant.startswith('ant')]:
      antids.append(int(ant.replace('ant','')))
   antids.sort()
   names=msmd.antennanames(antids)
   offset = [msmd.antennaoffset(name) for name in names]
   msmd.close()
   apriori_flagged=np.zeros(len(antids))
   nflagged=np.zeros(len(antids))
   nunflagged=np.zeros(len(antids))

   # Calculate the mean longitude and latitude.
   mean_longitude = numpy.mean([offset[i]["longitude offset"]\
            ['value'] for i in range(len(names))])
   mean_latitude = numpy.mean([offset[i]["latitude offset"]\
            ['value'] for i in range(len(names))])

   # Calculate the offsets from the center.
   offsets = [numpy.sqrt((offset[i]["longitude offset"]['value'] -\
             mean_longitude)**2 + (offset[i]["latitude offset"]\
             ['value'] - mean_latitude)**2) for i in \
             range(len(names))]
   offset_y=np.array([(offset[i]["latitude offset"]['value']) for i in \
            range(len(names))])
   offset_x=np.array([(offset[i]["longitude offset"]['value']) for i in \
            range(len(names))])

   for gc_dict in gc_dict_list:
       for s, spw in enumerate(spwlist):
          for a,antenna in enumerate(antids):
             ant='ant'+str(antenna)
             for e, element in enumerate(gc_dict['solvestats']['spw'+str(spw)][ant]['above_minsnr']):
                if (gc_dict['solvestats']['spw'+str(spw)][ant]['above_minsnr'][e] < gc_dict['solvestats']['spw'+str(spw)][ant]['expected'][e] or gc_dict['solvestats']['spw'+str(spw)][ant]['above_minblperant'][e] < gc_dict['solvestats']['spw'+str(spw)][ant]['expected'][e] or gc_dict['solvestats']['spw'+str(spw)][ant]['data_unflagged'][e] < gc_dict['solvestats']['spw'+str(spw)][ant]['expected'][e]) and (gc_dict['solvestats']['spw'+str(spw)][ant]['expected'][e] > 0):
                   nflagged[a]+= (gc_dict['solvestats']['spw'+str(spw)][ant]['expected'][e] - np.min([gc_dict['solvestats']['spw'+str(spw)][ant]['above_minsnr'][e], gc_dict['solvestats']['spw'+str(spw)][ant]['above_minblperant'][e]]))
                   #nflagged[s]+=1.0
                if (gc_dict['solvestats']['spw'+str(spw)][ant]['data_unflagged'][e] < gc_dict['solvestats']['spw'+str(spw)][ant]['expected'][e]) and (gc_dict['solvestats']['spw'+str(spw)][ant]['expected'][e] > 0):
                   apriori_flagged[a]+=(gc_dict['solvestats']['spw'+str(spw)][ant]['expected'][e]- gc_dict['solvestats']['spw'+str(spw)][ant]['data_unflagged'][e])
                   #apriori_flagged[s]+=1.0
                if (gc_dict['solvestats']['spw'+str(spw)][ant]['above_minsnr'][e] > 0 and gc_dict['solvestats']['spw'+str(spw)][ant]['above_minblperant'][e] > 0 and gc_dict['solvestats']['spw'+str(spw)][ant]['data_unflagged'][e] > 0) and (gc_dict['solvestats']['spw'+str(spw)][ant]['expected'][e] > 0):
                   nunflagged[a]+=np.min([gc_dict['solvestats']['spw'+str(spw)][ant]['above_minsnr'][e],gc_dict['solvestats']['spw'+str(spw)][ant]['above_minblperant'][e]])
                   #nunflagged[s]+=1.0
   ntotal=nflagged+nunflagged
   fracflagged=nflagged/ntotal
   nflagged_non_apriori=nflagged-apriori_flagged
   ntotal_non_apriori_flagged=ntotal-apriori_flagged
   fracflagged_non_apriori=nflagged_non_apriori/ntotal_non_apriori_flagged
   fracflagged_non_apriori=np.nan_to_num(fracflagged_non_apriori)
   print('a priori flagged:',apriori_flagged)
   print('non- a priori flagged:',nflagged_non_apriori)

   return names, offset_x, offset_y, apriori_flagged, nflagged, nunflagged, ntotal, fracflagged, nflagged_non_apriori, ntotal_non_apriori_flagged, fracflagged_non_apriori


def plot_ants_flagging_colored_from_dict(filename,selfcal_library,selfcal_plan,solint,final_mode,vis):
   spwlist=selfcal_library['spwlist'].copy()
   spwlist_pass=[]
   if final_mode=='combinespw' or final_mode=='combinespwpol':
       spwlist_pass=[spwlist[0]]
   elif final_mode=='per_spw':
       spwlist_pass=spwlist.copy()
   elif final_mode=='per_bb':
       spwlist_bb=[]
       for baseband in selfcal_library['baseband'].keys():
          spwlist_bb.append(selfcal_library['baseband'][baseband]['spwlist'][0])
       spwlist_pass=spwlist_bb.copy()


   names, offset_x, offset_y, apriori_flagged, nflagged, nunflagged, ntotal, fracflagged, nflagged_non_apriori, ntotal_non_apriori_flagged, fracflagged_non_apriori=get_flagged_solns_per_ant_from_dict(selfcal_plan['solint_settings'][solint]['gaincal_return_dict'][final_mode],spwlist_pass,vis)
   fracflagged=fracflagged_non_apriori
   print(fracflagged)
   import matplotlib
   matplotlib.use('Agg')
   import matplotlib.pyplot as plt
   ants_zero_flagging=np.where(fracflagged == 0.0)
   ants_lt10pct_flagging=((fracflagged <= 0.1) & (fracflagged > 0.0)).nonzero()
   ants_lt25pct_flagging=((fracflagged <= 0.25) & (fracflagged > 0.10)).nonzero()
   ants_lt50pct_flagging=((fracflagged <= 0.5) & (fracflagged > 0.25)).nonzero()
   ants_lt75pct_flagging=((fracflagged <= 0.75) & (fracflagged > 0.5)).nonzero()
   ants_gt75pct_flagging=np.where(fracflagged > 0.75)
   fig, ax = plt.subplots(1,1,figsize=(12, 12))
   ax.scatter(offset_x[ants_zero_flagging[0]],offset_y[ants_zero_flagging[0]],marker='o',color='green',label='No Flagging',s=120)
   ax.scatter(offset_x[ants_lt10pct_flagging[0]],offset_y[ants_lt10pct_flagging[0]],marker='o',color='blue',label='<10% Flagging',s=120)
   ax.scatter(offset_x[ants_lt25pct_flagging[0]],offset_y[ants_lt25pct_flagging[0]],marker='o',color='yellow',label='<25% Flagging',s=120)
   ax.scatter(offset_x[ants_lt50pct_flagging[0]],offset_y[ants_lt50pct_flagging[0]],marker='o',color='magenta',label='<50% Flagging',s=120)
   ax.scatter(offset_x[ants_lt75pct_flagging[0]],offset_y[ants_lt75pct_flagging[0]],marker='o',color='cyan',label='<75% Flagging',s=120)
   ax.scatter(offset_x[ants_gt75pct_flagging[0]],offset_y[ants_gt75pct_flagging[0]],marker='o',color='black',label='>75% Flagging',s=120)
   ax.legend(fontsize=20)
   for i in range(len(names)):
      ax.text(offset_x[i],offset_y[i],names[i])
   ax.set_xlabel('Latitude Offset (m)',fontsize=20)
   ax.set_ylabel('Longitude Offset (m)',fontsize=20)
   ax.set_title('Antenna Positions colorized by Selfcal Flagging',fontsize=20)
   plt.savefig(filename,dpi=200.0)
   plt.close()


def plot_image(filename,outname,min_val=None,max_val=None,zoom=2):
   import matplotlib
   matplotlib.use('Agg')
   import matplotlib.pyplot as plt
   header=imhead(filename)
   tb.open(filename)
   image_data=np.rot90(tb.getcol('map').squeeze())   # rotate the image 90 degrees and get rid of degenerate axes
   tb.close()
   size=np.max(header['shape'])
   cell=header['incr'][1]*3600.0*180.0/3.14159        #get pixel size from header declination direction
   halfsize_arcsec=size/zoom/2.0*cell
   fig=plt.figure(figsize=(8.5,8))
   ax=fig.add_subplot(1,1,1)

   ll=int(size/zoom-size/(zoom*2))
   ul=int(size/zoom+size/(zoom*2))

   mask_exists=os.path.exists(filename.replace('image.tt0','mask'))
   if mask_exists: #if mask exists draw it as a contour, else don't use contours
      tb.open(filename.replace('image.tt0','mask'))
      mask_data=np.flipud(np.rot90(tb.getcol('map').squeeze()))   # rotate the image 90 degrees and get rid of degenerate axes
                                                                  #extra flip is needed for the mask data for some reason to match the casaviewer view
      tb.close()

   if min_val == None:
      img=ax.imshow(image_data[ll:ul,ll:ul],extent=[halfsize_arcsec,-halfsize_arcsec,-halfsize_arcsec,halfsize_arcsec])
      if mask_exists:
         conts=ax.contour(mask_data[ll:ul,ll:ul],levels=[0.5], colors='white', extent=[halfsize_arcsec,-halfsize_arcsec,-halfsize_arcsec,halfsize_arcsec])
   else:
      img=ax.imshow(image_data[ll:ul,ll:ul],extent=[halfsize_arcsec,-halfsize_arcsec,-halfsize_arcsec,halfsize_arcsec],vmin=min_val,vmax=max_val)
      if mask_exists:
         conts=ax.contour(mask_data[ll:ul,ll:ul],levels=[0.5], colors='white', extent=[halfsize_arcsec,-halfsize_arcsec,-halfsize_arcsec,halfsize_arcsec])

   ax.set_xlabel('Offset (arcsec)',fontsize=18)
   ax.set_ylabel('Offset (arcsec)',fontsize=18)
   ax.tick_params(axis='both', which='major', labelsize=16)
   cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
   cbar=plt.colorbar(img,cax=cax)
   cbar.ax.tick_params(labelsize=16)
   if 'final_initial_div' in filename:
      cbar.set_label('Dimensionless Ratio')
   else:
      cbar.set_label('Intensity (Jy/beam)')
   plt.savefig(outname,dpi=300.0)
   plt.close()

def get_flagged_solns_per_ant(gaintable,vis):
     # Get the antenna names and offsets.

     msmd = casatools.msmetadata()
     tb = casatools.table()

     msmd.open(vis)
     ids = msmd.antennasforscan(msmd.scansforintent("*OBSERVE_TARGET*")[0])
     names = msmd.antennanames(ids)
     offset = [msmd.antennaoffset(name) for name in names]
     msmd.close()

     # Calculate the mean longitude and latitude.

     mean_longitude = numpy.mean([offset[i]["longitude offset"]\
             ['value'] for i in range(len(names))])
     mean_latitude = numpy.mean([offset[i]["latitude offset"]\
             ['value'] for i in range(len(names))])

     # Calculate the offsets from the center.

     offsets = [numpy.sqrt((offset[i]["longitude offset"]['value'] -\
             mean_longitude)**2 + (offset[i]["latitude offset"]\
             ['value'] - mean_latitude)**2) for i in \
             range(len(names))]
     offset_y=[(offset[i]["latitude offset"]['value']) for i in \
             range(len(names))]
     offset_x=[(offset[i]["longitude offset"]['value']) for i in \
             range(len(names))]
     # Calculate the number of flags for each antenna.
     #gaintable='"'+gaintable+'"'
     os.system('cp -r '+gaintable.replace(' ','\ ')+' tempgaintable.g')
     gaintable='tempgaintable.g'
     nflags = [tb.calc('[select from '+gaintable+' where ANTENNA1=='+\
             str(i)+' giving  [ntrue(FLAG)]]')['0'].sum() for i in ids]
     nunflagged = [tb.calc('[select from '+gaintable+' where ANTENNA1=='+\
             str(i)+' giving  [nfalse(FLAG)]]')['0'].sum() for i in ids]
     os.system('rm -rf tempgaintable.g')
     fracflagged=np.array(nflags)/(np.array(nflags)+np.array(nunflagged))
     # Calculate a score based on those two.
     return names, np.array(offset_x),np.array(offset_y),offsets, nflags, nunflagged,fracflagged



def create_noise_histogram(imagename):
    MADtoRMS =  1.4826
    headerlist = imhead(imagename, mode = 'list')
    telescope=headerlist['telescope']
    beammajor = headerlist['beammajor']['value']
    beamminor = headerlist['beamminor']['value']
    beampa = headerlist['beampa']['value']
    image_stats= imstat(imagename = imagename)
    maskImage=imagename.replace('image','mask').replace('.tt0','')
    residualImage=imagename.replace('image','residual')
    os.system('rm -rf temp.mask temp.residual')
    if os.path.exists(maskImage):
       os.system('cp -r '+maskImage+ ' temp.mask')
       maskImage='temp.mask'
    os.system('cp -r '+residualImage+ ' temp.residual')
    residualImage='temp.residual'
    if os.path.exists(maskImage):
       ia.close()
       ia.done()
       ia.open(residualImage)
       #ia.calcmask(maskImage+" <0.5"+"&& mask("+residualImage+")",name='madpbmask0')
       ia.calcmask("'"+maskImage+"'"+" <0.5"+"&& mask("+residualImage+")",name='madpbmask0')
       mask0Stats = ia.statistics(robust=True,axes=[0,1])
       ia.maskhandler(op='set',name='madpbmask0')
       rms = mask0Stats['medabsdevmed'][0] * MADtoRMS
       residualMean = mask0Stats['median'][0]
       pix=np.squeeze(ia.getchunk())
       mask=np.squeeze(ia.getchunk(getmask=True))
       dimensions=mask.ndim
       if dimensions ==4:
          mask=mask[:,:,0,0]
       if dimensions == 3:
          mask=mask[:,:,0]
       unmasked=(mask == True).nonzero()
       pix_unmasked=pix[unmasked]
       N,intensity=np.histogram(pix_unmasked,bins=50)
       ia.close()
       ia.done()
    elif telescope == 'ALMA':
       ia.close()
       ia.done()
       ia.open(residualImage)
       #ia.calcmask(maskImage+" <0.5"+"&& mask("+residualImage+")",name='madpbmask0')
       ia.calcmask("mask("+residualImage+")",name='madpbmask0')
       mask0Stats = ia.statistics(robust=True,axes=[0,1])
       ia.maskhandler(op='set',name='madpbmask0')
       rms = mask0Stats['medabsdevmed'][0] * MADtoRMS
       residualMean = mask0Stats['median'][0]
       pix=np.squeeze(ia.getchunk())
       mask=np.squeeze(ia.getchunk(getmask=True))
       mask=mask[:,:,0,0]
       unmasked=(mask == True).nonzero()
       pix_unmasked=pix[unmasked]
       ia.close()
       ia.done()
    elif 'VLA' in telescope:
       residual_stats=imstat(imagename=imagename.replace('image','residual'),algorithm='chauvenet')
       rms = residual_stats['rms'][0]
       ia.open(residualImage)
       pix_unmasked=np.squeeze(ia.getchunk())
       ia.close()
       ia.done()

    N,intensity=np.histogram(pix_unmasked,bins=100)
    intensity=np.diff(intensity)+intensity[:-1]  
    ia.close()
    ia.done()
    os.system('rm -rf temp.mask temp.residual')
    return N,intensity,rms 


def create_noise_histogram_plots(N_1,N_2,intensity_1,intensity_2,rms_1,rms_2,outfile,rms_theory=0.0):
   import matplotlib
   matplotlib.use('Agg')
   import matplotlib.pyplot as plt 
   bins=50.0
   max_N_1=np.max(N_1)
   max_N_2=np.max(N_2)
   fig, ax = plt.subplots(1,1,figsize=(12, 12))
   ax.set_yscale('log')
   plt.ylim([0.0001,2.0])
   ax.step(intensity_1,N_1/np.max(N_1),label='Initial Data')
   ax.step(intensity_2,N_2/np.max(N_2),label='Final Data')
   ax.plot(intensity_1,gaussian_norm(intensity_1,0,rms_1),label='Initial Gaussian')
   ax.plot(intensity_2,gaussian_norm(intensity_2,0,rms_2),label='Final Gaussian')
   xplotrange=np.abs(ax.get_xlim()[0])+np.abs(ax.get_xlim()[1])
   if rms_theory !=0.0:
      alpha_plot=-1.0*9.0*2.0*rms_theory/xplotrange*0.75 +1.0
      if (9.0*2.0*rms_theory > xplotrange) or (alpha_plot < 0.0):
         alpha_plot=0.25
      ax.fill(intensity_2,gaussian_norm(intensity_2,0,rms_theory),color='gray',label='Theoretical Sensitivity',alpha=alpha_plot)
      #ax.plot([-1.0*rms_theory,rms_theory],[0.606,0.606],label='Theoretical Sensitivity')
   ax.legend(fontsize=20)
   ax.set_xlabel('Intensity (mJy/Beam)',fontsize=20)
   ax.set_ylabel('N',fontsize=20)
   ax.set_title('Initial vs. Final Noise (Unmasked Pixels)',fontsize=20)
   plt.savefig(outfile,dpi=200.0)
   plt.close()


def gaussian_norm(x, mean, sigma):
   gauss_dist=np.exp(-(x-mean)**2/(2*sigma**2))
   norm_gauss_dist=gauss_dist/np.max(gauss_dist)
   return norm_gauss_dist

def importdata(vislist,all_targets,bands_for_targets,telescope):
   spectral_scan=False
   listdict=collect_listobs_per_vis(vislist)

   bands, band_properties = get_bands(vislist,all_targets,telescope)

   scantimesdict={}
   scanfieldsdict={}
   scannfieldsdict={}
   scanstartsdict={}
   scanendsdict={}
   integrationsdict={}
   integrationtimesdict={}
   mosaic_field_dict={}
   bands_to_remove=[]
   spws_set_dict = {}
   nspws_sets_dict = {}

   for band in bands:
        print(band)
        scantimesdict_temp,scanfieldsdict_temp,scannfieldsdict_temp,scanstartsdict_temp,scanendsdict_temp,integrationsdict_temp,integrationtimesdict_temp,\
        integrationtimes_temp,n_spws_temp,minspw_temp,spwsarray_dict,spws_set_dict_temp,mosaic_field_temp=fetch_scan_times_band_aware(vislist,all_targets,bands_for_targets,band_properties,band,telescope)

        spwslist_dict = {}
        spwstring_dict = {}
        for vis in vislist:
             spwslist_dict[vis] = spwsarray_dict[vis].tolist()
             spwstring_dict[vis]=','.join(str(spw) for spw in spwslist_dict[vis])
        if spws_set_dict_temp[vislist[0]].ndim > 1:
           nspws_sets=spws_set_dict_temp[vislist[0]].shape[0]
        else:
           nspws_sets=1

        if telescope=='ALMA' or telescope =='ACA':
           if nspws_sets > 1 and spws_set_dict_temp[vislist[0]].ndim >1:
              spectral_scan=True


        scantimesdict[band]=scantimesdict_temp.copy()
        scanfieldsdict[band]=scanfieldsdict_temp.copy()
        scannfieldsdict[band]=scannfieldsdict_temp.copy()
        scanstartsdict[band]=scanstartsdict_temp.copy()
        scanendsdict[band]=scanendsdict_temp.copy()
        integrationsdict[band]=integrationsdict_temp.copy()
        mosaic_field_dict[band]=mosaic_field_temp.copy()
        integrationtimesdict[band]=integrationtimesdict_temp.copy()
        spws_set_dict[band] = spws_set_dict_temp.copy()
        if spws_set_dict[band][vislist[0]].ndim > 1:
           nspws_sets_dict[band]=spws_set_dict[band][vislist[0]].shape[0]
        else:
           nspws_sets_dict[band]=1
   ## Load the gain calibrator information.

   gaincalibrator_dict = {}
   for vis in vislist:
       if "targets" in vis:
           if "cont" in vis:
              vis_string = "_targets_cont"
           else: 
              vis_string = "_targets"
       else:
           if "cont" in vis:
              vis_string = "_target_cont"
           else: 
              vis_string = "_target"

       viskey = vis.replace(vis_string+".ms",vis_string+".selfcal.ms")

       original_vis = vis.replace(sanitize_string('_'.join(all_targets))+'_'+'_'.join(bands)+'_','').replace(vis_string+".ms",".ms").replace(vis_string+".selfcal.ms",".ms")


       gaincalibrator_dict[viskey] = {}
       if os.path.exists(original_vis):
           msmd.open(original_vis)
   
           for field in msmd.fieldsforintent("*CALIBRATE_PHASE*"):
               scans_for_field = msmd.scansforfield(field)
               scans_for_gaincal = msmd.scansforintent("*CALIBRATE_PHASE*")
               field_name = msmd.fieldnames()[field]
               gaincalibrator_dict[viskey][field_name] = {}
               gaincalibrator_dict[viskey][field_name]["scans"] = np.intersect1d(scans_for_field, scans_for_gaincal)
               gaincalibrator_dict[viskey][field_name]["phasecenter"] = msmd.phasecenter(field)
               gaincalibrator_dict[viskey][field_name]["intent"] = "phase"
               gaincalibrator_dict[viskey][field_name]["times"] = np.array([np.mean(msmd.timesforscan(scan)) for scan in \
                       gaincalibrator_dict[viskey][field_name]["scans"]])
   
           msmd.close()

   return listdict,bands,band_properties,scantimesdict,scanfieldsdict,scannfieldsdict,scanstartsdict,scanendsdict,integrationsdict,integrationtimesdict,spwslist_dict,spwstring_dict,spwsarray_dict,mosaic_field_dict,gaincalibrator_dict,spectral_scan,spws_set_dict

def flag_spectral_lines(vislist,all_targets,spwsarray_dict):
   print("# cont.dat file found, flagging lines identified by the pipeline.")
   contdotdat = parse_contdotdat('cont.dat',all_targets[0])
   spwvisref=get_spwnum_refvis(vislist,all_targets[0],contdotdat,spwsarray_dict)
   for vis in vislist:
      if not os.path.exists(vis+".flagversions/flags.before_line_flags"):
         flagmanager(vis=vis, mode = 'save', versionname = 'before_line_flags', comment = 'Flag states at start of reduction')
      else:
         flagmanager(vis=vis,mode='restore',versionname='before_line_flags')
      for target in all_targets:
         contdotdat = parse_contdotdat('cont.dat',target)
         if len(contdotdat) == 0:
             print("WARNING: No cont.dat entry found for target "+target+", this likely indicates that hif_findcont was mitigated. We suggest you re-run findcont without mitigation.")
             print("No flagging will be done for target "+target)
             continue
         contdot_dat_flagchannels_string = flagchannels_from_contdotdat(vis,target,spwsarray_dict[vis],vislist,spwvisref,contdotdat)
         flagdata(vis=vis, mode='manual', spw=contdot_dat_flagchannels_string[:-2], flagbackup=False, field = target)


def split_to_selfcal_ms(all_targets,vislist,band_properties,bands,spectral_average,bands_for_targets):
   for vis in vislist:
       os.system('rm -rf '+vis.replace('.ms','.selfcal.ms')+'*')
       spwstring=''
       chan_widths=[]
       if spectral_average:
          initweights(vis=vis,wtmode='weight',dowtsp=True) # initialize channelized weights
          for band in bands:
             desiredWidth=get_desired_width(band_properties[vis][band]['meanfreq'])
             print(band,desiredWidth)
             widtharray,bwarray,nchanarray=get_spw_chanwidths(vis,band_properties[vis][band]['spwarray'])
             band_properties[vis][band]['chan_widths']=get_spw_chanavg(vis,widtharray,bwarray,nchanarray,desiredWidth=desiredWidth)
             print(band_properties[vis][band]['chan_widths'])
             chan_widths=chan_widths+band_properties[vis][band]['chan_widths'].astype('int').tolist()
             if spwstring =='':
                spwstring=band_properties[vis][band]['spwstring']+''
             else:
                spwstring=spwstring+','+band_properties[vis][band]['spwstring']
          mstransform(vis=vis,field=bands_for_targets['field_str'],chanaverage=True,chanbin=chan_widths,spw=spwstring,outputvis=sanitize_string('_'.join(all_targets))+'_'+'_'.join(bands)+'_'+vis.replace('.ms','.selfcal.ms'),datacolumn='data',reindex=False)
          initweights(vis=vis,wtmode='delwtsp') # remove channelized weights
       else:
          mstransform(vis=vis,field=bands_for_targets['field_str'],outputvis=sanitize_string('_'.join(all_targets))+'_'+'_'.join(bands)+'_'+vis.replace('.ms','.selfcal.ms'),datacolumn='data',reindex=False)


def check_mosaic(vislist,target):
   msmd.open(vislist[0])
   fieldid=msmd.fieldsforname(field)
   msmd.done()
   if len(fieldid) > 1:
      mosaic=True
   else:
      mosaic=False
   return mosaic

def get_phasecenter(vis,selfcal_library):
   msmd.open(vis)
   #fieldid=msmd.fieldsforname(field) # only works for ALMA mosaics

   # should be more general
   fieldid=np.array([],dtype=int)
   for fid in selfcal_library['sub-fields']:
      if fid in selfcal_library['sub-fields-fid_map'][vis].keys():
         field_id=selfcal_library['sub-fields-fid_map'][vis][fid]
         fieldid=np.append(fieldid,np.array([field_id]))

   ra_phasecenter_arr=np.zeros(len(fieldid))
   dec_phasecenter_arr=np.zeros(len(fieldid))
   for i in range(len(fieldid)):
      phasecenter=msmd.phasecenter(fieldid[i])
      ra_phasecenter_arr[i]=phasecenter['m0']['value']
      dec_phasecenter_arr[i]=phasecenter['m1']['value']

   msmd.done()

   ra_phasecenter=np.median(ra_phasecenter_arr)
   dec_phasecenter=np.median(dec_phasecenter_arr)
   phasecenter_string='ICRS {:0.8f}rad {:0.8f}rad '.format(ra_phasecenter,dec_phasecenter)
   return phasecenter_string

def get_flagged_solns_per_spw(spwlist,gaintable,extendpol=False):
     # Get the antenna names and offsets.
     msmd = casatools.msmetadata()
     tb = casatools.table()

     # Calculate the number of flags for each spw.
     #gaintable='"'+gaintable+'"'
     os.system('cp -r '+gaintable.replace(' ','\ ')+' tempgaintable.g')
     gaintable='tempgaintable.g'
     if extendpol:
         nflags = [tb.calc('[select from '+gaintable+' where SPECTRAL_WINDOW_ID=='+\
                 spwlist[i]+' giving  [any(FLAG)]]')['0'].sum() for i in \
                 range(len(spwlist))]
         nunflagged = [tb.calc('[select from '+gaintable+' where SPECTRAL_WINDOW_ID=='+\
                 spwlist[i]+' giving  [nfalse(any(FLAG))]]')['0'].sum() for i in \
                 range(len(spwlist))]
     else:
         nflags = [tb.calc('[select from '+gaintable+' where SPECTRAL_WINDOW_ID=='+\
                 spwlist[i]+' giving  [ntrue(FLAG)]]')['0'].sum() for i in \
                 range(len(spwlist))]
         nunflagged = [tb.calc('[select from '+gaintable+' where SPECTRAL_WINDOW_ID=='+\
                 spwlist[i]+' giving  [nfalse(FLAG)]]')['0'].sum() for i in \
                 range(len(spwlist))]

     nflags = np.array(nflags)
     nunflagged = np.array(nunflagged)

     nodata = np.where(nflags + nunflagged == 0)
     nflags[nodata] = (nflags + nunflagged).max()

     os.system('rm -rf tempgaintable.g')
     fracflagged=np.array(nflags)/(np.array(nflags)+np.array(nunflagged))
     # Calculate a score based on those two.
     return nflags, nunflagged,fracflagged



def select_best_gaincal_mode(selfcal_library,selfcal_plan,vis,gaintable_prefix,solint,spectral_solution_fraction,minsnr_to_proceed,telescope):
   selected_mode='combinespw'
   spwlist=selfcal_library[vis]['spwlist'].copy()
   spwlist_str=selfcal_library[vis]['spws'].split(',')
   if telescope != 'ACA':
       # if more than two antennas are fully flagged relative to the combinespw results, fallback to combinespw or per_bb
       max_flagged_ants_per_spw=2.0
       max_flagged_ants_per_bb=1.0
       # if only a single (or few) spw(s) has flagging, allow at most this number of antennas to be flagged before mapping
       max_flagged_ants_spwmap=1.0
   else:
       # For the ACA, don't allow any flagging of antennas before trying fallbacks, because it is more damaging due to the smaller
       # number of antennas
       max_flagged_ants_per_spw=0.0
       max_flagged_ants_per_bb=0.0
       max_flagged_ants_spwmap=0.0

   fallback=''
   min_spwmap_bw=0.0
   spwmap=[0.0]*len(spwlist)
   spwmap_widest_window_in_bb=[0.0]*len(spwlist)

   selfcal_plan[vis]['solint_settings'][solint]['nflags']={}
   selfcal_plan[vis]['solint_settings'][solint]['nunflagged']={}
   selfcal_plan[vis]['solint_settings'][solint]['ntotal']={}
   selfcal_plan[vis]['solint_settings'][solint]['fracflagged']={}
   selfcal_plan[vis]['solint_settings'][solint]['nflags_non_apriori']={}
   selfcal_plan[vis]['solint_settings'][solint]['ntotal_non_apriori']={}
   selfcal_plan[vis]['solint_settings'][solint]['fracflagged_non_apriori']={}
   selfcal_plan[vis]['solint_settings'][solint]['nflags_apriori']={}
   selfcal_plan[vis]['solint_settings'][solint]['delta_nflags']={}
   selfcal_plan[vis]['solint_settings'][solint]['minimum_flagged_ants']={}
   selfcal_plan[vis]['solint_settings'][solint]['maximum_flagged_ants']={}
   selfcal_plan[vis]['solint_settings'][solint]['non_zero_fraction']={}
   selfcal_plan[vis]['solint_settings'][solint]['polscale']={}
   #for loop here to get fraction flagged, unflagged, and flag fraction per mode
   for mode in selfcal_plan[vis]['solint_settings'][solint]['modes_to_attempt']:
      gaintable=gaintable_prefix+solint+'_'+str(selfcal_plan['solints'].index(solint))+'_'+selfcal_plan[vis]['solint_settings'][solint]['solmode']+'_'+selfcal_plan[vis]['solint_settings'][solint]['filename_append'][mode]+'.g'
      print(gaintable)
      # get_gaintable_flagging_stats returns (in order):
      # apriori_flagged - flagged solutions due to data being flagged
      # nflagged - total flagged solutions 
      # nunflagged - total unflagged solutions
      # ntotal - total solutions
      # fracflagged -fraction of flagged solutions
      # nflagged_non_apriori - flagged solutions (with apriori_flagged subtracted)
      # ntotal_non_apriori_flagged - total solutions with apriori_flagged solutions subtracted
      # fracflagged_non_apriori - fraction flagged without apriori flagged solutions
      # table evaulations will use flagging stats with apriori flags omitted
      selfcal_plan[vis]['solint_settings'][solint]['polscale'][mode]=2.0
      if selfcal_plan[vis]['solint_settings'][solint]['gaincal_gaintype']=='T':
        selfcal_plan[vis]['solint_settings'][solint]['polscale'][mode]=1.0

      if 'combinespw' in mode:
         selfcal_plan[vis]['solint_settings'][solint]['nflags_apriori'][mode],selfcal_plan[vis]['solint_settings'][solint]['nflags'][mode],selfcal_plan[vis]['solint_settings'][solint]['nunflagged'][mode],selfcal_plan[vis]['solint_settings'][solint]['ntotal'][mode],selfcal_plan[vis]['solint_settings'][solint]['fracflagged'][mode],selfcal_plan[vis]['solint_settings'][solint]['nflags_non_apriori'][mode],selfcal_plan[vis]['solint_settings'][solint]['ntotal_non_apriori'][mode],selfcal_plan[vis]['solint_settings'][solint]['fracflagged_non_apriori'][mode]=get_gaintable_flagging_stats(selfcal_plan[vis]['solint_settings'][solint]['gaincal_return_dict'][mode],[spwlist[0]])

      if mode=='per_spw':
         selfcal_plan[vis]['solint_settings'][solint]['nflags_apriori'][mode],selfcal_plan[vis]['solint_settings'][solint]['nflags'][mode],selfcal_plan[vis]['solint_settings'][solint]['nunflagged'][mode],selfcal_plan[vis]['solint_settings'][solint]['ntotal'][mode],selfcal_plan[vis]['solint_settings'][solint]['fracflagged'][mode],selfcal_plan[vis]['solint_settings'][solint]['nflags_non_apriori'][mode],selfcal_plan[vis]['solint_settings'][solint]['ntotal_non_apriori'][mode],selfcal_plan[vis]['solint_settings'][solint]['fracflagged_non_apriori'][mode]=get_gaintable_flagging_stats(selfcal_plan[vis]['solint_settings'][solint]['gaincal_return_dict'][mode],spwlist)

         for s, spw in enumerate(spwlist):
            selfcal_library[vis]['per_spw_stats'][int(spw)]['nflags']=selfcal_plan[vis]['solint_settings'][solint]['nflags_non_apriori'][mode][s]
            selfcal_library[vis]['per_spw_stats'][int(spw)]['nunflagged']=selfcal_plan[vis]['solint_settings'][solint]['nunflagged'][mode][s]
            selfcal_library[vis]['per_spw_stats'][int(spw)]['fracflagged']=selfcal_plan[vis]['solint_settings'][solint]['fracflagged_non_apriori'][mode][s]

      if mode == 'per_bb':
         baseband_scale=float(len(selfcal_library[vis]['baseband'].keys()))
         nflags=0
         nunflagged=0
         fracflagged=0.0
         spwlist_bb=[]
         for baseband in selfcal_library[vis]['baseband'].keys():
            spwlist_bb.append(selfcal_library[vis]['baseband'][baseband]['spwlist'][0])
         selfcal_plan[vis]['solint_settings'][solint]['nflags_apriori'][mode],selfcal_plan[vis]['solint_settings'][solint]['nflags'][mode],selfcal_plan[vis]['solint_settings'][solint]['nunflagged'][mode],selfcal_plan[vis]['solint_settings'][solint]['ntotal'][mode],selfcal_plan[vis]['solint_settings'][solint]['fracflagged'][mode],selfcal_plan[vis]['solint_settings'][solint]['nflags_non_apriori'][mode],selfcal_plan[vis]['solint_settings'][solint]['ntotal_non_apriori'][mode],selfcal_plan[vis]['solint_settings'][solint]['fracflagged_non_apriori'][mode]=get_gaintable_flagging_stats(selfcal_plan[vis]['solint_settings'][solint]['gaincal_return_dict'][mode],spwlist_bb)
      else:
         baseband_scale=1.0
      if solint == 'inf_EB':
         n_solutions=1.0
      else:
         n_antennas=selfcal_plan[vis]['solint_settings']['inf_EB']['ntotal_non_apriori']['combinespw'][0]/selfcal_plan[vis]['solint_settings'][solint]['polscale'][mode]
         n_solutions=(selfcal_plan[vis]['solint_settings'][solint]['nflags_non_apriori']['combinespw'][0]+selfcal_plan[vis]['solint_settings'][solint]['nunflagged']['combinespw'][0])/n_antennas


      selfcal_plan[vis]['solint_settings'][solint]['minimum_flagged_ants'][mode]=np.min(selfcal_plan[vis]['solint_settings'][solint]['nflags_non_apriori'][mode])/selfcal_plan[vis]['solint_settings'][solint]['polscale'][mode]/baseband_scale/n_solutions
      selfcal_plan[vis]['solint_settings'][solint]['maximum_flagged_ants'][mode]=np.max(selfcal_plan[vis]['solint_settings'][solint]['nflags_non_apriori'][mode])/selfcal_plan[vis]['solint_settings'][solint]['polscale'][mode]/baseband_scale/n_solutions

   #calculate delta_nflags, the difference between per_spw flagging and the minimum combinespw flagging to characterize the excess flagging in per_spw solutions
   if 'per_spw' in selfcal_plan[vis]['solint_settings'][solint]['modes_to_attempt']:
      selfcal_plan[vis]['solint_settings'][solint]['delta_nflags']['per_spw']=np.array(selfcal_plan[vis]['solint_settings'][solint]['nflags_non_apriori']['per_spw'])/selfcal_plan[vis]['solint_settings'][solint]['polscale'][mode]/n_solutions-np.array(selfcal_plan[vis]['solint_settings'][solint]['minimum_flagged_ants']['combinespw'])/selfcal_plan[vis]['solint_settings'][solint]['polscale'][mode]/n_solutions

   #choose between per_spw and per_bb if both exist
   preferred_mode='combinespw'  # default, in case somehow it doesn't get chose (shouldn't happen)
   if 'per_spw' in selfcal_plan[vis]['solint_settings'][solint]['modes_to_attempt'] and 'per_bb' in selfcal_plan[vis]['solint_settings'][solint]['modes_to_attempt']:
      print('Checking flagging per_spw, per_bb: ',selfcal_plan[vis]['solint_settings'][solint]['maximum_flagged_ants']['per_spw'],selfcal_plan[vis]['solint_settings'][solint]['maximum_flagged_ants']['per_bb'])
      if ((selfcal_plan[vis]['solint_settings'][solint]['maximum_flagged_ants']['per_spw']-max_flagged_ants_per_spw)<=selfcal_plan[vis]['solint_settings'][solint]['maximum_flagged_ants']['per_bb']):
         preferred_mode='per_spw'
      else:
         preferred_mode='per_bb'
   print('intermediate report',preferred_mode)

   # decide between per_spw and combine_spw
   if 'per_spw' in selfcal_plan[vis]['solint_settings'][solint]['modes_to_attempt'] and 'per_bb' not in selfcal_plan[vis]['solint_settings'][solint]['modes_to_attempt']:
      print('Checking flagging per_spw, combinespw: ',selfcal_plan[vis]['solint_settings'][solint]['maximum_flagged_ants']['per_spw'],selfcal_plan[vis]['solint_settings'][solint]['maximum_flagged_ants']['combinespw'])
      if ((selfcal_plan[vis]['solint_settings'][solint]['maximum_flagged_ants']['per_spw']-max_flagged_ants_per_spw)<=selfcal_plan[vis]['solint_settings'][solint]['maximum_flagged_ants']['combinespw']):
         preferred_mode='per_spw'
      else:
         preferred_mode='combinespw'
   print('intermediate report',preferred_mode)

   #only try to check between per_bb and combinespw, if per_spw is not already selected
   if preferred_mode == 'per_bb' and 'per_bb' in selfcal_plan[vis]['solint_settings'][solint]['modes_to_attempt'] and 'combinespw' in selfcal_plan[vis]['solint_settings'][solint]['modes_to_attempt']:
      print('Checking flagging per_bb, combinespw: ',selfcal_plan[vis]['solint_settings'][solint]['maximum_flagged_ants']['per_bb'],selfcal_plan[vis]['solint_settings'][solint]['maximum_flagged_ants']['combinespw'])
      if ((selfcal_plan[vis]['solint_settings'][solint]['maximum_flagged_ants']['per_bb']-max_flagged_ants_per_bb)<=selfcal_plan[vis]['solint_settings'][solint]['maximum_flagged_ants']['combinespw']):   
         preferred_mode='per_bb'
      else:
         preferred_mode='combinespw'
   print('intermediate report',preferred_mode)

   #check combinespw vs. per_spw just in case
   if preferred_mode=='per_spw' and 'per_spw' in selfcal_plan[vis]['solint_settings'][solint]['modes_to_attempt'] and 'combinespw' in selfcal_plan[vis]['solint_settings'][solint]['modes_to_attempt']:
      print('Checking flagging per_spw, combinespw: ',selfcal_plan[vis]['solint_settings'][solint]['maximum_flagged_ants']['per_spw'],selfcal_plan[vis]['solint_settings'][solint]['maximum_flagged_ants']['combinespw'])
      if ((selfcal_plan[vis]['solint_settings'][solint]['maximum_flagged_ants']['per_spw']-max_flagged_ants_per_spw)<= (selfcal_plan[vis]['solint_settings'][solint]['maximum_flagged_ants']['combinespw'])):     
         preferred_mode='per_spw'
      else:
         preferred_mode='combinespw'

   print('intermediate report',preferred_mode)
   #if after checking flagging, per_spw or per_bb is selected, check to make sure the solutions are not consistent with noise 
   #don't check for zero spectral phase on inf_EB, because it might be zero
   if ((preferred_mode == 'per_spw' or preferred_mode == 'per_bb') and solint != 'inf_EB'):
      print('intermediate report checking on the per_spw solutions')
      if preferred_mode == 'per_spw':
         if 'per_bb' in selfcal_plan[vis]['solint_settings'][solint]['modes_to_attempt']:
            mode_dict={'per_spw': {'status': False}, 'per_bb': {'status':False}}
         else:
            mode_dict={'per_spw': {'status': False}}
      elif preferred_mode =='per_bb':
         mode_dict={'per_bb': {'status':False}}
      for spw_mode in mode_dict.keys():
         combinespw_table=gaintable_prefix+solint+'_'+str(selfcal_plan['solints'].index(solint))+'_'+\
                          selfcal_plan[vis]['solint_settings'][solint]['solmode']+'_'+\
                          selfcal_plan[vis]['solint_settings'][solint]['filename_append']['combinespw']+'.g'
         spectral_table=gaintable_prefix+solint+'_'+str(selfcal_plan['solints'].index(solint))+'_'+\
                          selfcal_plan[vis]['solint_settings'][solint]['solmode']+'_'+\
                          selfcal_plan[vis]['solint_settings'][solint]['filename_append'][spw_mode]+'.g'
         if selfcal_plan[vis]['solint_settings'][solint]['solmode'] == 'p':
            fraction_wspectral_phase, n_wspectral_phase,n_total_phase=\
                     evaluate_per_spw_gaintables(combinespw_table,spectral_table,vis,selfcal_library[vis],'phase')
            print('Solint: {}, Mode: {}, Fraction w/non-zero phase: {:0.3f}, N solutions w/non-zero phase: {:0.3f}, Total number of solutions: {:0.3f}'.format(solint,spw_mode,fraction_wspectral_phase, n_wspectral_phase,n_total_phase))
            selfcal_plan[vis]['solint_settings'][solint]['non_zero_fraction'][spw_mode]=fraction_wspectral_phase
         elif selfcal_plan[vis]['solint_settings'][solint]['solmode'] == 'ap':
            fraction_wspectral_phase, n_wspectral_phase,n_total_phase=\
                     evaluate_per_spw_gaintables(combinespw_table,spectral_table,vis,selfcal_library[vis],'amp')
            print('Solint: {}, Mode: {}, Fraction w/non-zero amp: {:0.3f}, N solutions w/non-zero amp: {:0.3f}, Total number of solutions: {:0.3f}'.format(solint,spw_mode,fraction_wspectral_phase, n_wspectral_phase,n_total_phase))
            selfcal_plan[vis]['solint_settings'][solint]['non_zero_fraction'][spw_mode]=fraction_wspectral_phase
         if fraction_wspectral_phase >= spectral_solution_fraction:
            mode_dict[spw_mode]['status']=True
         # we might want to consider allowing per-spw solutions if the work for inf_EB depending on experience.
         #elif solint =='inf_EB':
         #   mode_dict[spw_mode]['status']=True
      #examine the results from checking the per_spw and per_bb solutions
      # if per_spw doesn't meet the limit for solutions not consistent with 0, it will fall back to per_bb or combinespw
      if preferred_mode == 'per_spw' and mode_dict[preferred_mode]['status']==False:
         if 'per_bb' in mode_dict.keys():
            if mode_dict['per_bb']['status']:
               preferred_mode='per_bb'
            else:
               preferred_mode='combinespw'
         else:
             preferred_mode='combinespw'

   # Check whether any spws have estimated SNR < 3, in which case we should not (initially) allow 'per_spw'
   if preferred_mode == 'per_spw' and np.any([selfcal_plan['solint_snr_per_spw']['inf_EB'][str(selfcal_library['reverse_spw_map'][vis][int(spw)])] < \
           minsnr_to_proceed for spw in spwlist]):
      if 'per_bb' in selfcal_plan[vis]['solint_settings'][solint]['modes_to_attempt']:
          preferred_mode = 'per_bb'
      else:
          preferred_mode = 'combinespw'
         
   print('intermediate report',preferred_mode)
   # if certain spws have more than max_flagged_ants_spwmap flagged solutions that the least flagged spws, set those to spwmap
   # if doing amplitude selfcal, spw mapping might not be the best idea, so only do for phase-only
   applycal_spwmap=[]
   if 'per_spw' in selfcal_plan[vis]['solint_settings'][solint]['modes_to_attempt'] and (preferred_mode=='combinespw' or preferred_mode=='per_bb') and (selfcal_plan[vis]['solint_settings'][solint]['solmode'] !='ap'):
       for i in range(len(spwlist)):
          # use >= to not always map if an spw has flagged solutions for a given antenna
          if np.min(selfcal_plan[vis]['solint_settings'][solint]['delta_nflags']['per_spw'][i]) >= max_flagged_ants_spwmap or \
                selfcal_plan['solint_snr_per_spw']['inf_EB'][str(selfcal_library['reverse_spw_map'][vis][int(spwlist[i])])] < minsnr_to_proceed or \
                selfcal_plan[vis]['solint_settings'][solint]['fracflagged']['per_spw'][i] == 1.0:
             fallback='spwmap'
             spwmap[i]=1.0
             spwmap_widest_window_in_bb[i]=check_spw_widest_in_bb(selfcal_library,vis,spwlist[i])
       if np.sum(spwmap)/len(spwmap) > 0.5:  # if greater than 1/2 of spws need mapping, just assume that we should do combinespw or per_bb
          fallback=''
       if np.sum(spwmap_widest_window_in_bb) >= 1.0: # don't do spw mapping within a baseband if the spws to be mapped are the widest in the baseband
          fallback=''

       # check if narrow windows are more flagged than wide windows
       # returns string 'true', 'false', or 'identical_spw_bws'
       # mapping can proceed except on'false'
       # we really only want to bother mapping if windows are identical
       # or if there are at a minimum three unique bandwidths and the most narrow are more flagged
       if fallback == 'spwmap':
           flagging_status=check_narrow_window_flagging(selfcal_library,vis) 
           if flagging_status == 'false' or flagging_status =='identical_spw_bws':
              fallback=''

       if fallback=='spwmap':
          #make spwmap list that first maps everything to itself, need max spw to make that list
          maxspw=np.max(spwlist)+1
          applycal_spwmap_int_list=list(np.arange(maxspw))
          for i in range(len(applycal_spwmap_int_list)):
             applycal_spwmap.append(applycal_spwmap_int_list[i])
          for i, spw in enumerate(applycal_spwmap_int_list):
             if spw in spwlist:
                 index=spwlist.index(spw)
                 print(index,spwlist[index],spwmap[index]==1.0)
                 
                 if spwmap[index]==1.0:
                    mapped_spw=get_nearest_wide_bw_spw(selfcal_library,vis,spw)
                    if mapped_spw==-99:
                       fallback=''
                       break
                    applycal_spwmap[i]=int(mapped_spw)
             # never do spw mapping for spectral scans
             if fallback !='' and selfcal_library['spectral_scan']:
                fallback=''
             if fallback=='spwmap':
                preferred_mode='per_spw'

   # If all of the spws map to the same spw, we might as well do a combinespw fallback.
   if preferred_mode == 'per_spw':
       if fallback == 'spwmap' and len(np.unique(np.array(applycal_spwmap)[np.array(spwlist).astype(int)])) == 1:
           preferred_mode = 'combinespw'
           applycal_spwmap = []

   # If we end up with combinespw, check whether going to combinespw with gaintype='T' offers further improvement.
   if preferred_mode == "combinespw" and "combinespwpol" in selfcal_plan[vis]['solint_settings'][solint]['modes_to_attempt']:
      spw_combine_test_gaintable = gaintable_prefix+solint+'_'+str(selfcal_plan['solints'].index(solint))+'_'+selfcal_plan[vis]['solint_settings'][solint]['solmode']+'_'+\
              selfcal_plan[vis]['solint_settings'][solint]['filename_append']['combinespw']+'.g'
      spwpol_combine_test_gaintable = gaintable_prefix+solint+'_'+str(selfcal_plan['solints'].index(solint))+'_'+selfcal_plan[vis]['solint_settings'][solint]['solmode']+'_'+\
              selfcal_plan[vis]['solint_settings'][solint]['filename_append']['combinespwpol']+'.g'

      print(spwlist)
      print(spwlist_str)
      nflags_spwcomb,nunflagged_spwcomb,fracflagged_spwcomb=get_flagged_solns_per_spw([spwlist_str[0]],spw_combine_test_gaintable,extendpol=True)
      nflags_spwpolcomb,nunflagged_spwpolcomb,fracflagged_spwpolcomb=get_flagged_solns_per_spw([spwlist_str[0]],spwpol_combine_test_gaintable)

      if np.sqrt((nunflagged_spwcomb[0]*(nunflagged_spwcomb[0]-1)) / (nunflagged_spwpolcomb[0]*(nunflagged_spwpolcomb[0]-1))) < 0.95:
          preferred_mode='combinespwpol'

   return preferred_mode,fallback,spwmap,applycal_spwmap



def check_narrow_window_flagging(selfcal_library,vis):  # return whether the narrows bws are flagged more than the widest
    bandwidths=[]
    for spw in selfcal_library[vis]['per_spw_stats'].keys():
       bandwidths.append(selfcal_library[vis]['per_spw_stats'][spw]['bandwidth'])
    unique_bws=list(set(bandwidths))
    unique_bws.sort()
    if len(unique_bws) == 1:  # should just do baseband or combine spw if this condition is true
       return 'identical_spw_bws'
    if len(unique_bws) > 2:  
        flags=np.zeros(len(unique_bws))
        for b,bw in enumerate(unique_bws):
            flags[b]=0.0
            for spw in selfcal_library[vis]['per_spw_stats'].keys():
               if bw == selfcal_library[vis]['per_spw_stats'][spw]['bandwidth']:
                  flags[b]+=selfcal_library[vis]['per_spw_stats'][spw]['nflags']
        if flags[0] > flags[1] >= flags[-1]: 
           return 'true'
        else:
           return 'false'
    else:
       return 'false'


def check_spw_widest_in_bb(selfcal_library,vis,spw):
    mapped_spw=-99
    spwfreq=selfcal_library[vis]['per_spw_stats'][spw]['frequency']
    spwbw=selfcal_library[vis]['per_spw_stats'][spw]['bandwidth']
    spwflags=selfcal_library[vis]['per_spw_stats'][spw]['nflags']
    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx
    spw_baseband=''
    for baseband in selfcal_library[vis]['baseband'].keys():
        if spw in selfcal_library[vis]['baseband'][baseband]['spwarray'] and len(selfcal_library[vis]['baseband'][baseband]['spwarray']) > 1:
            spw_baseband=baseband

    if spw_baseband !='':
        # remove spw in question here from spw lists
        spw_index=np.where(spw ==selfcal_library[vis]['baseband'][spw_baseband]['spwarray'])
        subarray_freqs=np.delete(selfcal_library[vis]['baseband'][spw_baseband]['freq_array'],spw_index[0][0])
        subarray_spws=np.delete(selfcal_library[vis]['baseband'][spw_baseband]['spwarray'],spw_index[0][0])
        subarray_bws=np.delete(selfcal_library[vis]['baseband'][spw_baseband]['bwarray'],spw_index[0][0])
        # find spw with larger bandwidths
        gt_bw_index=np.where(subarray_bws > spwbw)
        if len(gt_bw_index[0]) == 0:  
            return 1.0
        else:
            return 0.0
    else:
        return 1.0

def get_nearest_wide_bw_spw(selfcal_library,vis,spw):
    mapped_spw=-99
    spwfreq=selfcal_library[vis]['per_spw_stats'][spw]['frequency']
    spwbw=selfcal_library[vis]['per_spw_stats'][spw]['bandwidth']
    spwflags=selfcal_library[vis]['per_spw_stats'][spw]['nflags']
    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx
    spw_baseband=''
    for baseband in selfcal_library[vis]['baseband'].keys():
        if spw in selfcal_library[vis]['baseband'][baseband]['spwarray'] and len(selfcal_library[vis]['baseband'][baseband]['spwarray']) > 1:
            spw_baseband=baseband

    if spw_baseband !='':
        # remove spw in question here from spw lists
        spw_index=np.where(spw ==selfcal_library[vis]['baseband'][spw_baseband]['spwarray'])
        subarray_freqs=np.delete(selfcal_library[vis]['baseband'][spw_baseband]['freq_array'],spw_index[0][0])
        subarray_spws=np.delete(selfcal_library[vis]['baseband'][spw_baseband]['spwarray'],spw_index[0][0])
        subarray_bws=np.delete(selfcal_library[vis]['baseband'][spw_baseband]['bwarray'],spw_index[0][0])
        # find spw with larger bandwidths
        gt_bw_index=np.where(subarray_bws >= spwbw)
        if len(gt_bw_index[0]) == 0:  # if condition is met, cannot find spws with bws equal or greater than spw in question, exit out
            return -99
        else:   # find a nearby spw with a larger bw
            subarray_freqs=subarray_freqs[gt_bw_index[0]]
            subarray_spws=subarray_spws[gt_bw_index[0]]
            subarray_bws=subarray_bws[gt_bw_index[0]]
            subarray_flags=np.zeros(len(subarray_freqs))
            subarray_unflags=np.zeros(len(subarray_freqs))
            subarray_fracflags=np.zeros(len(subarray_freqs))
            for s, sub_spw in enumerate(subarray_spws):
                subarray_flags[s]= selfcal_library[vis]['per_spw_stats'][sub_spw]['nflags']
                subarray_unflags[s]= selfcal_library[vis]['per_spw_stats'][sub_spw]['nunflagged']
                subarray_fracflags[s]= selfcal_library[vis]['per_spw_stats'][sub_spw]['fracflagged']
            subarray_frac_freqdiffs=np.abs((subarray_freqs-spwfreq)/spwfreq)
            # compute a score based on the fractional frequency difference and the flagging fraction, select minumum
            score=subarray_frac_freqdiffs+subarray_fracflags
            index_best=np.argmin(score)
  
        index=find_nearest(subarray_freqs,spwfreq)
        mapped_spw=subarray_spws[index]
    else: # if cannot find a baseband (not sure why this would ever happen) search through spws
        nspws=len(selfcal_library[vis]['per_spw_stats'].keys())
        score=np.zeros(nspws)
        subarray_spws=np.zeros(nspws)
        for s, key in enumerate(selfcal_library[vis]['per_spw_stats'].keys()):
            if key != spw:
                fracfreqdiff=np.abs((selfcal_library[vis]['per_spw_stats'][key]['frequency']-spwfreq)/spwfreq)
                score[s]=fracfreqdiff+selfcal_library[vis]['per_spw_stats'][key]['fracflagged']
                subarray_spws[s]=int(key)
        index=np.argmin(score)
        mapped_spw=subarray_spws(index)
    return mapped_spw

def get_flagging_baseline(gc_dict_list,spwlist):
   apriori_flagged=np.zeros(len(spwlist))
   for gc_dict in gc_dict_list:
       for s, spw in enumerate(spwlist):
          for ant in [idant for idant in gc_dict['solvestats']['spw'+str(spw)].keys() if idant.startswith('ant')]:
             for e, element in enumerate(gc_dict['solvestats']['spw'+str(spw)][ant]['data_unflagged']):
                if (gc_dict['solvestats']['spw'+str(spw)][ant]['data_unflagged'][e] == 0) and (gc_dict['solvestats']['spw'+str(spw)][ant]['expected'][e] == 1):
                   apriori_flagged[s]+=1.0
   print(apriori_flagged)
   return apriori_flagged

def get_gaintable_flagging_stats(gc_dict_list,spwlist):
   apriori_flagged=np.zeros(len(spwlist))
   nflagged=np.zeros(len(spwlist))
   nunflagged=np.zeros(len(spwlist))
   for gc_dict in gc_dict_list:
       for s, spw in enumerate(spwlist):
          for ant in [idant for idant in gc_dict['solvestats']['spw'+str(spw)].keys() if idant.startswith('ant')]:
             for e, element in enumerate(gc_dict['solvestats']['spw'+str(spw)][ant]['above_minsnr']):
                if (gc_dict['solvestats']['spw'+str(spw)][ant]['above_minsnr'][e] < gc_dict['solvestats']['spw'+str(spw)][ant]['expected'][e] or gc_dict['solvestats']['spw'+str(spw)][ant]['above_minblperant'][e] < gc_dict['solvestats']['spw'+str(spw)][ant]['expected'][e] or gc_dict['solvestats']['spw'+str(spw)][ant]['data_unflagged'][e] < gc_dict['solvestats']['spw'+str(spw)][ant]['expected'][e]) and (gc_dict['solvestats']['spw'+str(spw)][ant]['expected'][e] > 0):
                   nflagged[s]+= (gc_dict['solvestats']['spw'+str(spw)][ant]['expected'][e] - np.min([gc_dict['solvestats']['spw'+str(spw)][ant]['above_minsnr'][e], gc_dict['solvestats']['spw'+str(spw)][ant]['above_minblperant'][e]]))
                   #nflagged[s]+=1.0
                if (gc_dict['solvestats']['spw'+str(spw)][ant]['data_unflagged'][e] < gc_dict['solvestats']['spw'+str(spw)][ant]['expected'][e]) and (gc_dict['solvestats']['spw'+str(spw)][ant]['expected'][e] > 0):
                   apriori_flagged[s]+=(gc_dict['solvestats']['spw'+str(spw)][ant]['expected'][e]- gc_dict['solvestats']['spw'+str(spw)][ant]['data_unflagged'][e])
                   #apriori_flagged[s]+=1.0
                if (gc_dict['solvestats']['spw'+str(spw)][ant]['above_minsnr'][e] > 0 and gc_dict['solvestats']['spw'+str(spw)][ant]['above_minblperant'][e] > 0 and gc_dict['solvestats']['spw'+str(spw)][ant]['data_unflagged'][e] > 0) and (gc_dict['solvestats']['spw'+str(spw)][ant]['expected'][e] > 0):
                   nunflagged[s]+=np.min([gc_dict['solvestats']['spw'+str(spw)][ant]['above_minsnr'][e],gc_dict['solvestats']['spw'+str(spw)][ant]['above_minblperant'][e]])
                   #nunflagged[s]+=1.0
   ntotal=nflagged+nunflagged
   fracflagged=nflagged/ntotal
   nflagged_non_apriori=nflagged-apriori_flagged
   ntotal_non_apriori_flagged=ntotal-apriori_flagged
   fracflagged_non_apriori=nflagged_non_apriori/ntotal_non_apriori_flagged
   fracflagged_non_apriori=np.nan_to_num(fracflagged_non_apriori)
   print('a priori flagged:',apriori_flagged)
   print('non- a priori flagged:',nflagged_non_apriori)
   return apriori_flagged,nflagged,nunflagged,ntotal,fracflagged,nflagged_non_apriori,ntotal_non_apriori_flagged,fracflagged_non_apriori

#used only for testing
def plot_phase_err(phase_err,snr,gaintable):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1,1,figsize=(12, 12))
    ax.scatter(snr,phase_err)
    ax.set_xlabel('SNR',fontsize=20)
    ax.set_ylabel('Phase Error (deg)',fontsize=20)
    plt.savefig('phase-err-vs-snr-'+gaintable+'.png')

#used only for testing
def plot_amp_err(amp_err,snr,gaintable):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1,1,figsize=(12, 12))
    ax.scatter(snr,amp_err)
    ax.set_xlabel('SNR',fontsize=20)
    ax.set_ylabel('Amp Error',fontsize=20)
    plt.savefig('amp-err-vs-snr-'+gaintable+'.png')

#used only for testing
def plot_spectral_phase(per_spw_phase,per_spw_phase_err,freqs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1,1,figsize=(12, 12))
    for j in range(per_spw_phase.shape[1]):
        for k in range(per_spw_phase.shape[2]):
            ax.errorbar(freqs,per_spw_phase[:,j,k],yerr=per_spw_phase_err[:,j,k],fmt='o')
    plt.savefig('phase-vs-freq.png')
    for k in range(per_spw_phase.shape[2]):
        fig, ax = plt.subplots(1,1,figsize=(12, 12))
        for j in range(per_spw_phase.shape[1]):
            ax.errorbar(freqs,per_spw_phase[:,j,k],yerr=per_spw_phase_err[:,j,k],fmt='o')
        plt.savefig('phase-vs-freq-ant'+str(k)+'.png')


#used only for testing
def plot_spectral_amp(per_spw_amp,per_spw_amp_err,freqs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1,1,figsize=(12, 12))
    for j in range(per_spw_amp.shape[1]):
        for k in range(per_spw_amp.shape[2]):
            ax.errorbar(freqs,per_spw_amp[:,j,k],yerr=per_spw_amp_err[:,j,k],fmt='o')
    plt.savefig('amp-vs-freq.png')
    for k in range(per_spw_amp.shape[2]):
        fig, ax = plt.subplots(1,1,figsize=(12, 12))
        for j in range(per_spw_amp.shape[1]):
            ax.errorbar(freqs,per_spw_amp[:,j,k],yerr=per_spw_amp_err[:,j,k],fmt='o')
        plt.savefig('amp-vs-freq-ant'+str(k)+'.png')

#used only for testing
def plot_spectral_phase_per_soln(per_spw_phase,per_spw_phase_err,per_spw_flags,freqs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    chi2=np.zeros((per_spw_phase.shape[1],per_spw_phase.shape[2]))
    npts=np.zeros((per_spw_phase.shape[1],per_spw_phase.shape[2]))
    per_spw_flags=np.abs((per_spw_flags-1.0))
    for k in range(per_spw_phase.shape[2]):
        for j in range(per_spw_phase.shape[1]):
            chi2[j,k]=np.sum(per_spw_phase[:,j,k]**2/per_spw_phase_err[:,j,k]**2)
            npts[j,k]=len(per_spw_flags[:,j,k])
             
            fig, ax = plt.subplots(1,1,figsize=(12, 12))
            ax.errorbar(freqs,per_spw_phase[:,j,k],yerr=per_spw_phase_err[:,j,k],fmt='o')
            plt.savefig('phase-vs-freq-soln'+str(j)+'-ant'+str(k)+'.png')
    return chi2,npts


def chi2_spectral_phase_per_soln(per_spw_phase,per_spw_phase_err,per_spw_flags,freqs):
    chi2=np.zeros((per_spw_phase.shape[1],per_spw_phase.shape[2]))
    npts=np.zeros((per_spw_phase.shape[1],per_spw_phase.shape[2]))
    pvalue=np.zeros((per_spw_phase.shape[1],per_spw_phase.shape[2]))
    per_spw_flags=np.abs((per_spw_flags.astype(float)-1.0))
    for k in range(per_spw_phase.shape[2]):
        for j in range(per_spw_phase.shape[1]):
            chi2[j,k]=np.sum(per_spw_phase[:,j,k]**2/per_spw_phase_err[:,j,k]**2)
            npts[j,k]=np.sum(per_spw_flags[:,j,k]) 
            pvalue[j,k]=1.0-scipy.stats.chi2.cdf(chi2[j,k],npts[j,k])
    return chi2,npts,pvalue

def check_spectral_gain_gradient(combine_spw_table,per_spw_table,spw_info,gaintype):
    gaintable_dict={'combine_spw_table': {'filename': combine_spw_table},
                    'per_spw_table': {'filename': per_spw_table}}
    
    for gaintable in gaintable_dict.keys():
        print('Checking: ',gaintable_dict[gaintable]['filename'])
            
        tb.open(gaintable_dict[gaintable]['filename'])
        gaintable_dict[gaintable][gaintype]=tb.getcol('CPARAM')
        gaintable_dict[gaintable][gaintype+'_err']=tb.getcol('PARAMERR')
        gaintable_dict[gaintable]['polarizations']=gaintable_dict[gaintable][gaintype].shape[0]
        gaintable_dict[gaintable]['snr']=tb.getcol('SNR')
        gaintable_dict[gaintable]['flag']=tb.getcol('FLAG')
        gaintable_dict[gaintable]['time']=tb.getcol('TIME')
        gaintable_dict[gaintable]['antenna1']=tb.getcol('ANTENNA1')
        gaintable_dict[gaintable]['spw']=tb.getcol('SPECTRAL_WINDOW_ID')
        tb.close()
        gaintable_dict[gaintable]['nspws']=len(np.unique(gaintable_dict[gaintable]['spw']))
        gaintable_dict[gaintable]['antennas']=np.unique(gaintable_dict[gaintable]['antenna1'])
        gaintable_dict[gaintable]['nants']=len(gaintable_dict[gaintable]['antennas'])
        gaintable_dict[gaintable]['per_polarization']={}
        for i in range(gaintable_dict[gaintable]['polarizations']):
            gaintable_dict[gaintable]['per_polarization'][i]={}
            gaintable_dict[gaintable]['per_polarization'][i][gaintype]=gaintable_dict[gaintable][gaintype][i,:,:]
            gaintable_dict[gaintable]['per_polarization'][i][gaintype+'_err']=gaintable_dict[gaintable][gaintype+'_err'][i,:,:]
            gaintable_dict[gaintable]['per_polarization'][i]['snr']=gaintable_dict[gaintable]['snr'][i,:,:]
            gaintable_dict[gaintable]['per_polarization'][i]['flag']=gaintable_dict[gaintable]['flag'][i,:,:]
            gaintable_dict[gaintable]['per_polarization'][i]['time']=gaintable_dict[gaintable]['time'][:]
            gaintable_dict[gaintable]['per_polarization'][i]['antenna1']=gaintable_dict[gaintable]['antenna1'][:]
            gaintable_dict[gaintable]['per_polarization'][i]['spw']=gaintable_dict[gaintable]['spw'][:]

            gaintable_dict[gaintable]['per_polarization'][i]['nspws']=len(np.unique(gaintable_dict[gaintable]['per_polarization'][i]['spw']))
            gaintable_dict[gaintable]['per_polarization'][i]['tablesize']=len(gaintable_dict[gaintable]['per_polarization'][i]['spw'])
            gaintable_dict[gaintable]['per_polarization'][i]['antennas']=np.unique(gaintable_dict[gaintable]['per_polarization'][i]['antenna1'])
            gaintable_dict[gaintable]['per_polarization'][i]['nants']=len(gaintable_dict[gaintable]['per_polarization'][i]['antennas'])
            gaintable_dict[gaintable]['per_polarization'][i]['solns']=int(gaintable_dict[gaintable]['per_polarization'][i]['tablesize']/ \
                    gaintable_dict[gaintable]['per_polarization'][i]['nspws']/gaintable_dict[gaintable]['per_polarization'][i]['nants'])
            gaintable_dict[gaintable]['per_polarization'][i]['spw_folded']=\
                     gaintable_dict[gaintable]['per_polarization'][i]['spw'].reshape((gaintable_dict[gaintable]['per_polarization'][i]['nspws'],\
                                 gaintable_dict[gaintable]['per_polarization'][i]['solns'],gaintable_dict[gaintable]['per_polarization'][i]['nants']))
            gaintable_dict[gaintable]['per_polarization'][i]['flag_folded']=\
                     gaintable_dict[gaintable]['per_polarization'][i]['flag'].reshape((gaintable_dict[gaintable]['per_polarization'][i]['nspws'],\
                                 gaintable_dict[gaintable]['per_polarization'][i]['solns'],gaintable_dict[gaintable]['per_polarization'][i]['nants']))
            gaintable_dict[gaintable]['per_polarization'][i]['antenna1_folded']=\
                     gaintable_dict[gaintable]['per_polarization'][i]['antenna1'].reshape((gaintable_dict[gaintable]['per_polarization'][i]['nspws'],\
                                 gaintable_dict[gaintable]['per_polarization'][i]['solns'],gaintable_dict[gaintable]['per_polarization'][i]['nants']))
            gaintable_dict[gaintable]['per_polarization'][i]['time_folded']=\
                     gaintable_dict[gaintable]['per_polarization'][i]['time'].reshape((gaintable_dict[gaintable]['per_polarization'][i]['nspws'],\
                                 gaintable_dict[gaintable]['per_polarization'][i]['solns'],gaintable_dict[gaintable]['per_polarization'][i]['nants']))
            gaintable_dict[gaintable]['per_polarization'][i]['snr_folded']=\
                     gaintable_dict[gaintable]['per_polarization'][i]['snr'].reshape((gaintable_dict[gaintable]['per_polarization'][i]['nspws'],\
                                 gaintable_dict[gaintable]['per_polarization'][i]['solns'],gaintable_dict[gaintable]['per_polarization'][i]['nants']))
            gaintable_dict[gaintable]['per_polarization'][i][gaintype+'_folded']=\
                     gaintable_dict[gaintable]['per_polarization'][i][gaintype].reshape((gaintable_dict[gaintable]['per_polarization'][i]['nspws'],\
                                 gaintable_dict[gaintable]['per_polarization'][i]['solns'],gaintable_dict[gaintable]['per_polarization'][i]['nants']))
            gaintable_dict[gaintable]['per_polarization'][i][gaintype+'_err_folded']=\
                     gaintable_dict[gaintable]['per_polarization'][i][gaintype+'_err'].reshape((gaintable_dict[gaintable]['per_polarization'][i]['nspws'],\
                                 gaintable_dict[gaintable]['per_polarization'][i]['solns'],gaintable_dict[gaintable]['per_polarization'][i]['nants']))
            if i == 0:
                gaintable_dict[gaintable]['solns']=gaintable_dict[gaintable]['per_polarization'][i]['solns']
                gaintable_dict[gaintable]['spw_folded']=gaintable_dict[gaintable]['per_polarization'][i]['spw_folded'].copy()
                gaintable_dict[gaintable]['flag_folded']=gaintable_dict[gaintable]['per_polarization'][i]['flag_folded'].copy()
                gaintable_dict[gaintable]['antenna1_folded']=gaintable_dict[gaintable]['per_polarization'][i]['antenna1_folded'].copy()
                gaintable_dict[gaintable]['time_folded']=gaintable_dict[gaintable]['per_polarization'][i]['time_folded'].copy()
                gaintable_dict[gaintable]['snr_folded']=gaintable_dict[gaintable]['per_polarization'][i]['snr_folded'].copy()
                gaintable_dict[gaintable][gaintype+'_folded']=gaintable_dict[gaintable]['per_polarization'][i][gaintype+'_folded'].copy()
                gaintable_dict[gaintable][gaintype+'_err_folded']=gaintable_dict[gaintable]['per_polarization'][i][gaintype+'_err_folded'].copy()
            else:
                #exclude this one, otherwise doubles up n solutions.  
                #gaintable_dict[gaintable]['solns']+=gaintable_dict[gaintable]['per_polarization'][i]['solns']
                gaintable_dict[gaintable]['spw_folded']=np.dstack((gaintable_dict[gaintable]['spw_folded'],\
                         gaintable_dict[gaintable]['spw_folded'],gaintable_dict[gaintable]['per_polarization'][i]['spw_folded'].copy()))
                gaintable_dict[gaintable]['flag_folded']=np.dstack((gaintable_dict[gaintable]['flag_folded'],\
                         gaintable_dict[gaintable]['per_polarization'][i]['flag_folded'].copy()))
                gaintable_dict[gaintable]['antenna1_folded']=np.dstack((gaintable_dict[gaintable]['antenna1_folded'],\
                         gaintable_dict[gaintable]['per_polarization'][i]['antenna1_folded'].copy()))
                gaintable_dict[gaintable]['time_folded']=np.dstack((gaintable_dict[gaintable]['time_folded'],\
                         gaintable_dict[gaintable]['per_polarization'][i]['time_folded'].copy()))
                gaintable_dict[gaintable]['snr_folded']=np.dstack((gaintable_dict[gaintable]['snr_folded'],\
                         gaintable_dict[gaintable]['per_polarization'][i]['snr_folded'].copy()))
                gaintable_dict[gaintable][gaintype+'_folded']=np.dstack((gaintable_dict[gaintable][gaintype+'_folded'],\
                         gaintable_dict[gaintable]['per_polarization'][i][gaintype+'_folded'].copy()))
                gaintable_dict[gaintable][gaintype+'_err_folded']=np.dstack((gaintable_dict[gaintable][gaintype+'_err_folded'],\
                         gaintable_dict[gaintable]['per_polarization'][i][gaintype+'_err_folded'].copy()))
        if gaintype == 'amp':
            gaintable_dict[gaintable][gaintype+'_folded']=np.absolute(gaintable_dict[gaintable][gaintype+'_folded'])-1.0
            gaintable_dict[gaintable][gaintype+'_err_folded']=np.absolute(gaintable_dict[gaintable][gaintype+'_err_folded'])
        if gaintype == 'phase':
            gaintable_dict[gaintable]['phase_err_folded_deg']=(gaintable_dict[gaintable]['phase_err_folded']**2 *\
                        1.0/(gaintable_dict[gaintable]['phase_folded'].imag)**2 * 1.0/(1.0 + \
                        (gaintable_dict[gaintable]['phase_folded'].real/gaintable_dict[gaintable]['phase_folded'].imag)**2))**0.5 * 180.0/3.14159

            gaintable_dict[gaintable]['phase_folded_deg']= np.angle(gaintable_dict[gaintable]['phase_folded'])*180.0/3.14159
            #plot_phase_err(gaintable_dict[gaintable]['phase_err_folded_deg'],gaintable_dict[gaintable]['snr_folded'],gaintable)
            gaintable_dict[gaintable][gaintype+'_err_folded_deg']=\
                         (gaintable_dict[gaintable][gaintype+'_err_folded']**2 * \
                          1.0/(gaintable_dict[gaintable][gaintype+'_folded'].imag)**2 *\
                          1.0/(1.0 + (gaintable_dict[gaintable][gaintype+'_folded'].real / \
                         gaintable_dict[gaintable][gaintype+'_folded'].imag)**2))**0.5 * 180.0/3.14159

            gaintable_dict[gaintable][gaintype+'_folded_deg']=\
                         np.angle(gaintable_dict[gaintable][gaintype+'_folded'])*180.0/3.14159
            #plot_phase_err(gaintable_dict[gaintable][gaintype+'_err_folded_deg'],gaintable_dict[gaintable]['snr_folded'],gaintable)
      
    #for gaintable in gaintable_dict.keys():
    #    for i in range(gaintable_dict[gaintable]['nspws']):
    #      for j in range(gaintable_dict[gaintable]['solns']):
	#          for k in range(gaintable_dict[gaintable]['nants']):
	#             print(i,j,k,gaintable_dict[gaintable]['time_folded'][i,j,k])

    spw_freqs=np.zeros(gaintable_dict['per_spw_table']['nspws'])
    for i,spw in enumerate(np.unique(gaintable_dict['per_spw_table']['spw'])):
        spw_freqs[i]=spw_info[spw]['frequency']

    if gaintype == 'phase':
        phase_per_spw_gradient=gaintable_dict['per_spw_table'][gaintype+'_folded'].copy()
        phase_per_spw_gradient_err=gaintable_dict['per_spw_table'][gaintype+'_folded'].copy()

        for i in range(gaintable_dict[gaintable]['nspws']):
            phase_per_spw_gradient[i,:,:]=(gaintable_dict['per_spw_table'][gaintype+'_folded_deg'][i,:,:]-gaintable_dict['combine_spw_table'][gaintype+'_folded_deg'][0,:,:])
            phase_per_spw_gradient_err[i,:,:]=(gaintable_dict['per_spw_table'][gaintype+'_err_folded_deg'][i,:,:]**2+gaintable_dict['combine_spw_table'][gaintype+'_err_folded_deg'][0,:,:]**2)**0.5
        for i in range(gaintable_dict[gaintable]['nspws']):
            for j in range(gaintable_dict[gaintable]['solns']):
                for k in range(gaintable_dict[gaintable]['nants']*gaintable_dict[gaintable]['polarizations']):
                    while abs(phase_per_spw_gradient[i,j,k]) >180.0:
                        if phase_per_spw_gradient[i,j,k] > 180.0:
                            phase_per_spw_gradient[i,j,k]=phase_per_spw_gradient[i,j,k]-360.0
                        elif phase_per_spw_gradient[i,j,k] < -180.0:
                            phase_per_spw_gradient[i,j,k]=phase_per_spw_gradient[i,j,k]+360.0
        print(phase_per_spw_gradient.shape,phase_per_spw_gradient_err.shape)
        return phase_per_spw_gradient,phase_per_spw_gradient_err,gaintable_dict['combine_spw_table'][gaintype+'_folded_deg'],gaintable_dict['per_spw_table'][gaintype+'_folded_deg'],gaintable_dict['combine_spw_table'][gaintype+'_err_folded_deg'],gaintable_dict['per_spw_table'][gaintype+'_err_folded_deg'],gaintable_dict['combine_spw_table']['flag_folded'],gaintable_dict['per_spw_table']['flag_folded'],spw_freqs

    if gaintype == 'amp':
        gaintable_dict[gaintable]['amp_err_folded']=np.absolute(gaintable_dict[gaintable]['amp_err_folded'])

        #plot_amp_err(gaintable_dict[gaintable]['amp_err_folded'],gaintable_dict[gaintable]['snr_folded'],gaintable)
      
        return gaintable_dict['per_spw_table']['amp_folded'],gaintable_dict['per_spw_table']['amp_err_folded'],gaintable_dict['per_spw_table']['flag_folded'],spw_freqs



def evaluate_per_spw_gaintables(combine_spw_table,per_spw_table,vis,selfcal_library,gaintype):
    try:
        if gaintype=='phase':
            phase_gradient,phase_gradient_err,phase_combinespw,phase_per_spw,\
                           phase_err_combinespw,phase_err_per_spw,combinespw_flags,\
                           per_spw_flags,freqs=\
                           check_spectral_gain_gradient(combine_spw_table,per_spw_table,selfcal_library['per_spw_stats'],gaintype)
            #plot_spectral_phase(phase_gradient,phase_gradient_err,freqs)
            #chi2,npts=plot_spectral_phase_per_soln(phase_gradient,phase_gradient_err,per_spw_flags,freqs)
            chi2,npts,pvalue=chi2_spectral_phase_per_soln(phase_gradient,phase_gradient_err,per_spw_flags,freqs)


        elif gaintype=='amp':
            amp_gradient,amp_gradient_err,per_spw_flags,freqs=check_spectral_gain_gradient(combine_spw_table,per_spw_table,selfcal_library['per_spw_stats'],gaintype)
            #plot_spectral_amp(amp_gradient,amp_gradient_err,freqs)
            #chi2,npts=plot_spectral_phase_per_soln(phase_gradient,phase_gradient_err,per_spw_flags,freqs)
            chi2,npts,pvalue=chi2_spectral_phase_per_soln(amp_gradient,amp_gradient_err,per_spw_flags,freqs)

        sigma_5=0.000000573303
        sigma_4=0.000063342484
        sigma_3=0.002699796063
        pvalue_unfolded=pvalue.reshape(pvalue.size)
        inconsistent_withzero=(pvalue_unfolded < sigma_3).nonzero()
        frac_inconsistent_withzero=float(np.size(inconsistent_withzero)/np.sum(np.isfinite(pvalue_unfolded)))
        print('Nsolutions inconsistent with 0 spectral amp or phase: {}/{}; {:0.3f}'.format(np.size(inconsistent_withzero),np.sum(np.isfinite(pvalue_unfolded)),frac_inconsistent_withzero))
        return frac_inconsistent_withzero,np.size(inconsistent_withzero),np.sum(np.isfinite(pvalue_unfolded))
    except:
        return -99.9,-99.9,-99.9





def unflag_failed_antennas(vis, caltable, gaincal_return, flagged_fraction=0.25, only_long_baselines=False, solnorm=True, calonly_max_flagged=0., spwmap=[], 
        fb_to_prev_solint=False, solints=[], iteration=0, plot=False, plot_directory="./"):
    tb.open(caltable, nomodify=plot) # Because we only modify if we aren't plotting, i.e. in the selfcal loop itself plot=False
    antennas = tb.getcol("ANTENNA1")
    flags = tb.getcol("FLAG")
    cals = tb.getcol("CPARAM")
    snr = tb.getcol("SNR")

    if len(spwmap) > 0:
        spws = tb.getcol("SPECTRAL_WINDOW_ID")
        good_spws = np.repeat(False, spws.size)
        good_spw_ids = np.unique(spwmap)
        for spw in good_spw_ids:
            good_spws = np.logical_or(good_spws, spws == spw)
    else:
        good_spw_ids = np.unique(np.concatenate([gcdict['selectvis']['spw'] for gcdict in gaincal_return]))
        good_spws = np.repeat(True, antennas.size)

    msmd.open(vis)
    good_antenna_ids = msmd.antennasforscan(msmd.scansforintent("*OBSERVE_TARGET*")[0])
    good_antennas = np.repeat(False, antennas.size)
    for ant in np.unique(antennas):
        if ant in good_antenna_ids:
            good_antennas[antennas == ant] = True

    good_spws = np.logical_and(good_spws, good_antennas)
 
    antennas = antennas[good_spws]
    flags = flags[:,:,good_spws]
    cals = cals[:,:,good_spws]
    snr = snr[:,:,good_spws]

    # Get the percentage of flagged solutions for each antenna.
    unique_antennas = np.unique(antennas)
    nants = unique_antennas.size
    """
    ordered_flags = flags.reshape(flags.shape[0:2] + (flags.shape[2]//nants, nants))
    percentage_flagged = (ordered_flags.sum(axis=2) / ordered_flags.shape[2]).mean(axis=0).mean(axis=0)
    """

    nflagged = np.array([[np.sum([gcdict['solvestats']['spw'+str(spw)]['ant'+str(ant)]["data_unflagged"].sum() - 
            gcdict['solvestats']['spw'+str(spw)]['ant'+str(ant)]["above_minsnr"].sum() for gcdict in gaincal_return]) for ant in good_antenna_ids] 
            for spw in good_spw_ids])
    nsolutions = np.array([[np.sum([gcdict['solvestats']['spw'+str(spw)]['ant'+str(ant)]["data_unflagged"].sum() for gcdict in gaincal_return]) 
            for ant in good_antenna_ids] for spw in good_spw_ids])

    percentage_flagged = nflagged.sum(axis=0) / np.clip(nsolutions.sum(axis=0), 1., np.inf)

 
    # Load in the positions of the antennas and calculate their offsets from the geometric center.
    msmd.open(vis)
    offsets = [msmd.antennaoffset(a) for a in antennas]
    unique_offsets = [msmd.antennaoffset(a) for a in unique_antennas]
    msmd.close()
 
    mean_longitude = np.mean([offsets[i]["longitude offset"]['value'] for i in range(nants)])
    mean_latitude = np.mean([offsets[i]["latitude offset"]['value'] for i in range(nants)])
    offsets = np.array([np.sqrt((offsets[i]["longitude offset"]['value'] - \
            mean_longitude)**2 + (offsets[i]["latitude offset"]['value'] - mean_latitude)**2) for i in range(len(antennas))])
    unique_offsets = np.array([np.sqrt((unique_offsets[i]["longitude offset"]['value'] - \
            mean_longitude)**2 + (unique_offsets[i]["latitude offset"]['value'] - mean_latitude)**2) for i in range(len(unique_antennas))])

    flagged_offsets = np.array([])
    offsets = np.array([])
    for i, ant in enumerate(unique_antennas):
        offsets = np.concatenate((offsets, np.repeat(unique_offsets[i], np.array([[gcdict['solvestats'][f'spw{spw}'][f'ant{ant}']['data_unflagged'] 
                for spw in good_spw_ids] for gcdict in gaincal_return]).sum())))
        flagged_offsets = np.concatenate((flagged_offsets, np.repeat(unique_offsets[i], np.array([[gcdict['solvestats'][f'spw{spw}'][f'ant{ant}']['data_unflagged'] - 
                gcdict['solvestats'][f'spw{spw}'][f'ant{ant}']['above_minsnr'] for spw in good_spw_ids] for gcdict in gaincal_return]).sum())))
          
    # Get a smoothed number of antennas flagged as a function of offset.
    test_r = np.linspace(0., offsets.max(), 1000)
    neff = (nants)**(-1./(1+4))
    kernal2 = scipy.stats.gaussian_kde(offsets, bw_method=neff)

    divisor = 1
    multiplier = cals.shape[0]
    if len(np.unique(flagged_offsets)) == 1:
        flagged_offsets = np.concatenate((flagged_offsets, flagged_offsets*1.05))
        divisor = 2
    elif len(flagged_offsets) == 0:
        tb.close()
        print("Not unflagging any antennas because there are no flags! The beam size probably changed because of calwt=True.")
        return
    kernel = scipy.stats.gaussian_kde(flagged_offsets,
            bw_method=kernal2.factor*offsets.std()/flagged_offsets.std())
    normalized = kernel(test_r) * len(flagged_offsets) / divisor / np.trapz(kernel(test_r), test_r)
    normalized2 = kernal2(test_r) * antennas.size * multiplier / np.trapz(kernal2(test_r), test_r)
    fraction_flagged_antennas = normalized / normalized2

    # Calculate the derivatives to see where flagged fraction is sharply changing.

    derivative = np.gradient(fraction_flagged_antennas, test_r)
    second_derivative = np.gradient(derivative, test_r)

    # Check which minima include enough antennas to explain the beam ratio.

    maxima = scipy.signal.argrelextrema(second_derivative, np.greater)[0]
    # We only want positive accelerations and positive velocities, i.e. flagging increasing. That said, if you happen to have the
    # case of a significantly flagged short baseline antenna and a lot of minimally flagged long baseline antennas, the velocity
    # might be negative because you have a shallow gap at the intersection of the two. So we need to do a check, and if there's no
    # peaks that satisfy this condition, ignore the velocity criterion.
    positive_velocity_maxima = maxima[np.logical_and(second_derivative[maxima] > 0, derivative[maxima] > 0)]
    maxima = maxima[second_derivative[maxima] > 0]
    # If we have enough peaks (i.e. the whole thing isn't flagged, then take only the peaks outside the inner 5%.
    if len(maxima) > 1:
        maxima = maxima[test_r[maxima] > test_r.max()*0.1]
    # Pick the shortest baseline "significant" maximum.
    if len(positive_velocity_maxima) > 0:
        good = second_derivative[maxima] / second_derivative[positive_velocity_maxima].max() > 0.5
    elif len(maxima) > 0:
        good = second_derivative[maxima] / second_derivative[maxima].max() > 0.5
    else:
        good = []

    if len(maxima) == 0 or np.all(good == False):
        maxima = np.array([0])
        good = np.array([True])

    m = maxima[good].min()
    # If thats not the shortest baseline maximum, we can go one lower as long as the velocity doesn't go below 0.
    if m != maxima.min():
        index = np.where(maxima == m)[0][0]
        m_test = maxima[index-1]
        if np.all(derivative[m_test:m]/derivative.max() > -0.05):
            m = m_test

    offset_limit = test_r[m]
    max_velocity = derivative[m]
    flagged_fraction = fraction_flagged_antennas[m]

    if only_long_baselines:
        ok_to_flag_antennas = unique_antennas[unique_offsets > offset_limit]
    else:
        ok_to_flag_antennas = unique_antennas

    # Make a plot of all of this info

    if plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        #from matplotlib import rc
        from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker, VPacker

        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()

        ax1.plot(unique_offsets, percentage_flagged, "o")

        ax1.plot(test_r, fraction_flagged_antennas, "k-")
        ax2.plot(test_r, derivative / derivative.max(), "g-")
        if len(positive_velocity_maxima) > 0:
            ax2.plot(test_r, second_derivative / second_derivative[positive_velocity_maxima].max(), "r-")
        else:
            ax2.plot(test_r, second_derivative / second_derivative[maxima].max(), "r-")

        for m in maxima[::-1]:
            if second_derivative[m] < 0:
                continue

            if test_r[m] == offset_limit:
                ax1.axvline(test_r[m], linestyle="--")
                ax1.axhline(fraction_flagged_antennas[m], linestyle="--")
            else:
                ax1.axvline(test_r[m])

        #rc('text',usetex=True)
        #rc('text.latex', preamble=r'\usepackage{color}')
        ax1.set_xlabel("Baseline (m)")
        ax1.set_ylabel("Flagged Fraction")
        #ax2.set_ylabel("Normalized Smoothed Flagged Fraction \n Velocity / Acceleration")
        ybox1 = TextArea("Normalized Smoothed Flagged Fraction ", \
                textprops=dict(color="k", rotation=90, ha='left',va='bottom'))
        ybox2 = TextArea("Velocity ", \
                textprops=dict(color="g", rotation=90, ha='left',va='bottom'))
        ybox3 = TextArea("Acceleration ", \
                textprops=dict(color="r", rotation=90, ha='left',va='bottom'))
        ybox4 = TextArea("/ ", \
                textprops=dict(color="k", rotation=90, ha='left',va='bottom'))
        ybox5 = TextArea("\n", \
                textprops=dict(color="k", rotation=90, ha='left',va='bottom'))

        ybox = VPacker(children=[ybox1], align="bottom", pad=0, sep=5)
        ybox6 = VPacker(children=[ybox3, ybox4, ybox2], align="bottom", pad=0, sep=5)

        anchored_ybox = AnchoredOffsetbox(loc=8, child=ybox, pad=0., frameon=False, \
                bbox_to_anchor=(1.15, 0.15), bbox_transform=ax2.transAxes, borderpad=0.)
        anchored_ybox2 = AnchoredOffsetbox(loc=8, child=ybox6, pad=0., frameon=False, \
                bbox_to_anchor=(1.2, 0.26), bbox_transform=ax2.transAxes, borderpad=0.)

        ax2.add_artist(anchored_ybox)
        ax2.add_artist(anchored_ybox2)

        fig.tight_layout()
        fig.savefig(plot_directory+"/"+caltable.replace(".g",".pass.png"))
        plt.close(fig)

        tb.close()

        return

    # Now combine the cluster of antennas with high flagging fraction with the antennas that actually have enough
    # flagging to warrant passing through to get the list of pass through antennas.
    bad_antennas = unique_antennas[percentage_flagged >= flagged_fraction]

    pass_through_antennas = np.intersect1d(ok_to_flag_antennas, bad_antennas)

    # For the antennas we just identified, we just pass them through without doing anything. I.e. we set flags to False and the caltable value to 1.0+0j.
    for a in pass_through_antennas:
        indices = np.where(antennas == a)

        flagged_fraction_double_snr = (snr[:,:,indices] < 10).sum() / snr[:,:,indices].size
        if flagged_fraction_double_snr < calonly_max_flagged: 
            flags[:,:,indices] = False
        else:
            flags[:,:,indices] = False
            cals[:,:,indices] = 1.0+0j

    if solnorm:
        scale = np.mean(np.abs(cals[flags == False])**2)**0.5
        print("Normalizing the amplitudes by a factor of ", scale)
        cals = cals / scale

    modified_flags = tb.getcol("FLAG")
    modified_cals = tb.getcol("CPARAM")

    modified_flags[:,:,good_spws] = flags
    modified_cals[:,:,good_spws] = cals

    tb.putcol("FLAG", modified_flags)
    tb.putcol("CPARAM", modified_cals)
    tb.flush()

    tb.close()

    # Check whether earlier solints have acceptable solutions, and if so use, those instead.

    if fb_to_prev_solint:
        if "ap" in solints[iteration]:
            for i in range(len(solints)):
                if "ap" in solints[i]:
                    min_iter = i
                    break
        else:
            min_iter = 1

        for i, solint in enumerate(solints[min_iter:iteration][::-1]):
            print("Testing solint ", solint)
            print("Opening gaintable ", caltable.replace(solints[iteration]+"_"+str(iteration), solint+"_"+str(iteration-i-1)))
            tb.open(caltable.replace(solints[iteration]+"_"+str(iteration), solint+"_"+str(iteration-i-1)))
            antennas = tb.getcol("ANTENNA1")
            flags = tb.getcol("FLAG")
            cals = tb.getcol("CPARAM")
            snr = tb.getcol("SNR")
            tb.close()

            new_pass_through_antennas = []
            print(list(pass_through_antennas))
            for ant in pass_through_antennas:
                good = antennas == ant
                if np.all(cals[:,:,good].real == 1) and np.all(cals[:,:,good].imag == 0) and np.all(flags[:,:,good] == False):
                    new_pass_through_antennas.append(ant)
                    print("Skipping ant ",ant," because it was passed through in solint = ", solint)
                else:
                    tb.open(caltable, nomodify=False)
                    bad_rows = np.where(tb.getcol("ANTENNA1") == ant)[0]
                    tb.removerows(rownrs=bad_rows)
                    tb.flush()
                    tb.close()

                    tb.open(caltable.replace(solints[iteration]+"_"+str(iteration), solint+"_"+str(iteration-i-1)))
                    good_rows = np.where(tb.getcol("ANTENNA1") == ant)[0]
                    print("Copying these rows into ", caltable, ":")
                    print(good_rows)
                    for row in good_rows:
                        tb.copyrows(outtable=caltable, startrowin=row, nrow=1)
                    tb.close()

            pass_through_antennas = new_pass_through_antennas

        tb.open(caltable)
        rownumbers = tb.rownumbers()
        subt = tb.query("OBSERVATION_ID==0", sortlist="TIME,ANTENNA1")
        tb.close()

        subt.copyrows(outtable=caltable)
        tb.open(caltable, nomodify=False)
        tb.removerows(rownrs=rownumbers)
        tb.flush()
        tb.close()
        subt.close()



def triage_calibrators(vis, target, potential_calibrators, max_distance=10.0, max_time=600.):
    gaincalibrator_dict = {}

    if os.path.exists(vis.replace("_target.selfcal.ms",".ms")):
        msmd.open(vis.replace("_target.selfcal.ms",".ms"))

        for field in msmd.fieldsforintent("*CALIBRATE_PHASE*"):
            scans_for_field = msmd.scansforfield(field)
            scans_for_gaincal = msmd.scansforintent("*CALIBRATE_PHASE*")
            field_name = msmd.fieldnames()[field]
            gaincalibrator_dict[field_name] = {}
            gaincalibrator_dict[field_name]["scans"] = np.intersect1d(scans_for_field, scans_for_gaincal)
            gaincalibrator_dict[field_name]["phasecenter"] = msmd.phasecenter(field)
            gaincalibrator_dict[field_name]["intent"] = "phase"
            gaincalibrator_dict[field_name]["times"] = np.array([np.mean(msmd.timesforscan(scan)) for scan in \
                    gaincalibrator_dict[field_name]["scans"]])

        gaincal_info_found = len(gaincalibrator_dict) > 0

        msmd.close()
    else:
        gaincal_info_found = False

    all_targets = potential_calibrators + [target]

    msmd.open(vis)
    targets_ids = [msmd.fieldsforname(field)[0] for field in all_targets]
    for i, field in enumerate(targets_ids):
        scans_for_field = msmd.scansforfield(field)
        scans_for_science = msmd.scansforintent("*OBSERVE_TARGET*")
        field_name = all_targets[i]
        gaincalibrator_dict[field_name] = {}
        gaincalibrator_dict[field_name]["scans"] = np.intersect1d(scans_for_field, scans_for_science)
        gaincalibrator_dict[field_name]["phasecenter"] = msmd.phasecenter(field)
        gaincalibrator_dict[field_name]["intent"] = "target" if field_name == target else "science"
        gaincalibrator_dict[field_name]["times"] = np.array([np.mean(msmd.timesforscan(scan)) for scan in gaincalibrator_dict[field_name]["scans"]])

    msmd.close()

    fields = []
    scans = []
    distances = []
    intents = []
    times = []
    #import matplotlib.pyplot as plt
    for t in gaincalibrator_dict.keys():
        dRA = (gaincalibrator_dict[t]["phasecenter"]["m0"]["value"] - gaincalibrator_dict[target]["phasecenter"]["m0"]["value"]) * 360/(2*np.pi)
        dDec = (gaincalibrator_dict[t]["phasecenter"]["m1"]["value"] - gaincalibrator_dict[target]["phasecenter"]["m1"]["value"]) * 360/(2*np.pi)
        #plt.plot(dRA, dDec, "ko")
        #plt.annotate(t, (dRA, dDec))
        d = (dRA**2 + dDec**2)**0.5

        scans += [gaincalibrator_dict[t]["scans"]]
        distances += [np.repeat(d,gaincalibrator_dict[t]["scans"].size)]
        intents += [np.repeat(gaincalibrator_dict[t]["intent"],gaincalibrator_dict[t]["scans"].size)]
        fields += [np.repeat(t,gaincalibrator_dict[t]["scans"].size)]
        times += [gaincalibrator_dict[t]["times"]]

    times = np.concatenate(times)
    order = np.argsort(times)
    times = times[order]

    scans = np.concatenate(scans)[order]
    distances = np.concatenate(distances)[order]
    intents = np.concatenate(intents)[order]
    fields = np.concatenate(fields)[order]
    good = np.repeat(False, scans.size)
    case = np.repeat(0, scans.size)

    if gaincal_info_found:
        is_gaincalibrator = intents == "phase"
        gaincal_interval = np.median(times[is_gaincalibrator][1:] - times[is_gaincalibrator][0:-1])
        print(times[is_gaincalibrator] - times[is_gaincalibrator][0])
    else:
        gaincal_interval = np.inf
    print("gaincal_interval = ", gaincal_interval)

    prev_target = -1
    prev_calibrator = -2
    for i in range(scans.size):
        if gaincal_info_found:
            next_calibrator = np.where(intents[i:] == "phase")[0][0] + i
        else:
            next_calibrator = np.inf

        if "target" in intents[i:]:
            next_target = np.where(intents[i:] == "target")[0][0] + i
        else:
            next_target = scans.size

        if next_calibrator == i:
            prev_calibrator = i
        elif next_target == i:
            prev_target = i

        """
        if distances[i] < distances[current_calibrator] and ((current_calibrator < next_calibrator and next_calibrator > next_target) \
                or (current_calibrator == next_calibrator)):
            good[i] = True
        """

        #print(prev_target, prev_calibrator, next_target, next_calibrator)
        next_target_time = times[next_target] if next_target < times.size else np.inf
        prev_target_time = times[prev_target] if prev_target > 0 else 0

        next_calibrator_distance = distances[next_calibrator] if next_calibrator < distances.size else np.inf
        prev_calibrator_distance = distances[prev_calibrator] if prev_calibrator >= 0 else np.inf

        if prev_target < prev_calibrator < next_target < next_calibrator:
            good[i] = distances[i] < min(prev_calibrator_distance,max_distance) and \
                    (abs(times[i] - next_target_time) < min(gaincal_interval,max_time) or \
                    abs(times[i] - prev_target_time) < min(gaincal_interval,max_time))
            case[i] = 1
        elif prev_calibrator < prev_target < next_calibrator < next_target:
            good[i] = distances[i] < min(next_calibrator_distance,max_distance) and \
                    (abs(times[i] - prev_target_time) < min(gaincal_interval,max_time) or \
                    abs(times[i] - next_target_time) < min(gaincal_interval,max_time))
            case[i] = 2
        elif prev_target < prev_calibrator < next_calibrator < next_target:
            good[i] = (distances[i] < min(next_calibrator_distance,max_distance) and \
                    abs(times[i] - next_target_time) < min(gaincal_interval,max_time)) or \
                    (distances[i] < min(prev_calibrator_distance,max_distance) and \
                    abs(times[i] - prev_target_time) < min(gaincal_interval,max_time))
            case[i] = 3
        elif prev_calibrator < prev_target < next_target < next_calibrator:
            good[i] = (distances[i] < min(next_calibrator_distance,max_distance) and \
                    abs(times[i] - next_target_time) < min(gaincal_interval,max_time)) or \
                    (distances[i] < min(prev_calibrator_distance,max_distance) and \
                    abs(times[i] - prev_target_time) < min(gaincal_interval,max_time))
            case[i] = 4
        

        next_calibrator_scan = scans[next_calibrator] if gaincal_info_found else scans.size
        prev_calibrator_scan = scans[prev_calibrator] if gaincal_info_found else -1
        if next_target < scans.size and prev_target > 0:
            print("{0:3d}   {1:5.2f}   {2:7s}   {3:20s}   {4:4.0f}   {5:5s}   {6:1d}   {7:3d}   {8:3d}   {9:3d}   {10:3d}".format(\
                    scans[i], distances[i], intents[i], fields[i], times[i]-times[0], str(good[i]), case[i], prev_calibrator_scan, \
                    next_calibrator_scan, scans[prev_target], scans[next_target]))
        elif prev_target < 0:
            print("{0:3d}   {1:5.2f}   {2:7s}   {3:20s}   {4:4.0f}   {5:5s}   {6:1d}   {7:3d}   {8:3d}   {9:3d}   {10:3d}".format(\
                    scans[i], distances[i], intents[i], fields[i], times[i]-times[0], str(good[i]), case[i], prev_calibrator_scan, \
                    next_calibrator_scan, -1, scans[next_target]))
        else:
            print("{0:3d}   {1:5.2f}   {2:7s}   {3:20s}   {4:4.0f}   {5:5s}   {6:1d}   {7:3d}   {8:3d}   {9:3d}   {10:3d}".format(\
                    scans[i], distances[i], intents[i], fields[i], times[i]-times[0], str(good[i]), case[i], prev_calibrator_scan, \
                    next_calibrator_scan, scans[prev_target], scans.max()+1))

    #good = intents == "science"
    print(scans[good].astype(str))
    print(np.unique(fields[good]))

    return ",".join(np.unique(fields[good])), ",".join(scans[good].astype(str))


def zero_out_gc_return_dict(gaincal_return):
   gaincal_return['solvestats']['data_unflagged']=np.zeros(gaincal_return['solvestats']['data_unflagged'].size)
   gaincal_return['solvestats']['above_minsnr']=np.zeros(gaincal_return['solvestats']['above_minsnr'].size)
   for key1 in gaincal_return['solvestats'].keys():
      if 'spw' in key1:
         gaincal_return['solvestats'][key1]['data_unflagged']=np.zeros(gaincal_return['solvestats'][key1]['data_unflagged'].size)
         gaincal_return['solvestats'][key1]['above_minsnr']=np.zeros(gaincal_return['solvestats'][key1]['above_minsnr'].size)
         for key2 in gaincal_return['solvestats'][key1].keys():
            if 'ant' in key2 and key2 != 'above_minblperant':
               gaincal_return['solvestats'][key1][key2]['data_unflagged']=np.zeros(gaincal_return['solvestats'][key1][key2]['data_unflagged'].size)
               gaincal_return['solvestats'][key1][key2]['above_minsnr']=np.zeros(gaincal_return['solvestats'][key1][key2]['above_minsnr'].size)
          
   return gaincal_return

def make_distance_time_phaseerr_plots(vislist):

    time_collection = []
    distance_collection = []
    rms_collection = []

    for vis in vislist:
        msmd.open(vis.replace("_target.selfcal.ms",".ms"))

        gaincalibrator_dict = {}
        for field in msmd.fieldsforintent("*CALIBRATE_PHASE*"):
            scans_for_field = msmd.scansforfield(field)
            scans_for_gaincal = msmd.scansforintent("*CALIBRATE_PHASE*")
            field_name = msmd.fieldnames()[field]
            gaincalibrator_dict[field_name] = {}
            gaincalibrator_dict[field_name]["scans"] = np.intersect1d(scans_for_field, scans_for_gaincal)
            gaincalibrator_dict[field_name]["phasecenter"] = msmd.phasecenter(field)
            gaincalibrator_dict[field_name]["intent"] = "phase"
            gaincalibrator_dict[field_name]["times"] = np.array([np.mean(msmd.timesforscan(scan)) for scan in gaincalibrator_dict[field_name]["scans"]])

        #print(gaincalibrator_dict.keys())
        gaincalibrator = list(gaincalibrator_dict.keys())[0]

        msmd.close()

        msmd.open(vis)
        all_targets = [msmd.fieldnames()[i] for i in msmd.fieldsforintent("*OBSERVE_TARGET*")]
        msmd.close()
        #print(all_targets)

        for target in all_targets:
            #print(target)
            if not os.path.exists(sanitize_string(target)+'_'+vis+'_Band_7_inf_1_p.g'):
                continue

            msmd.open(vis)
            targets_ids = [msmd.fieldsforname(field)[0] for field in [target]]
            for i, field in enumerate(targets_ids):
                scans_for_field = msmd.scansforfield(field)
                scans_for_science = msmd.scansforintent("*OBSERVE_TARGET*")
                field_name = [target][i]
                gaincalibrator_dict[field_name] = {}
                gaincalibrator_dict[field_name]["scans"] = np.intersect1d(scans_for_field, scans_for_science)
                gaincalibrator_dict[field_name]["phasecenter"] = msmd.phasecenter(field)
                gaincalibrator_dict[field_name]["intent"] = "target" if field_name == target else "science"
                gaincalibrator_dict[field_name]["times"] = np.array([np.mean(msmd.timesforscan(scan)) for scan in gaincalibrator_dict[field_name]["scans"]])

            msmd.close()

            fields = []
            scans = []
            distances = []
            intents = []
            times = []
            import matplotlib.pyplot as plt
            for t in gaincalibrator_dict.keys():
                dRA = (gaincalibrator_dict[t]["phasecenter"]["m0"]["value"] - gaincalibrator_dict[gaincalibrator]["phasecenter"]["m0"]["value"]) * 360/(2*np.pi)
                dDec = (gaincalibrator_dict[t]["phasecenter"]["m1"]["value"] - gaincalibrator_dict[gaincalibrator]["phasecenter"]["m1"]["value"]) * 360/(2*np.pi)
                #plt.plot(dRA, dDec, "ko")
                #plt.annotate(t, (dRA, dDec))
                d = (dRA**2 + dDec**2)**0.5

                scans += [gaincalibrator_dict[t]["scans"]]
                distances += [np.repeat(d,gaincalibrator_dict[t]["scans"].size)]
                intents += [np.repeat(gaincalibrator_dict[t]["intent"],gaincalibrator_dict[t]["scans"].size)]
                fields += [np.repeat(t,gaincalibrator_dict[t]["scans"].size)]
                times += [gaincalibrator_dict[t]["times"]]

            times = np.concatenate(times)
            order = np.argsort(times)
            times = times[order]

            scans = np.concatenate(scans)[order]
            distances = np.concatenate(distances)[order]
            intents = np.concatenate(intents)[order]
            fields = np.concatenate(fields)[order]

            nearest_calibrator_scan = np.repeat(0, scans.size)
            time_to_calibrator = np.repeat(0.0, scans.size)

            for i in range(scans.size):
                nearest_calibrator = np.where(np.abs(times - times[i]) == np.abs(times[intents == "phase"] - times[i]).min())[0][0]
                nearest_calibrator_scan[i] = scans[nearest_calibrator]

                time_to_calibrator[i] = np.abs(times[i] - times[nearest_calibrator])

                #print("{0:3d}   {1:5.2f}   {2:7s}   {3:20s}   {4:4.0f}   {5:3d}   {6:4.0f}".format(\
                #        scans[i], distances[i], intents[i], fields[i], times[i]-times[0], nearest_calibrator_scan[i], \
                #        time_to_calibrator[i]))


            tb.open(sanitize_string(target)+'_'+vis+'_Band_7_inf_1_p.g')
            cals = tb.getcol("CPARAM").flatten()
            flags = tb.getcol("FLAG").flatten()
            scan_numbers = tb.getcol("SCAN_NUMBER")
            #print(scan_numbers.shape)
            tb.close()

            cals = cals[flags == False]
            scan_numbers = scan_numbers[flags == False]

            phase = np.angle(cals) * 180./np.pi

            #plt.hist(phase.flatten(), 20)

            for scan in np.unique(scans[intents == "target"]):
                #time_arr = np.repeat(time_to_calibrator[scans == scan][0], (scan_numbers == scan).sum())
                time_collection += [time_to_calibrator[scans == scan][0]]
                distance_collection += [distances[scans == scan][0]] 
                rms_collection += [np.std(phase[scan_numbers == scan])]
                #plt.plot(time_arr, phase[scan_numbers == scan], "o")

            del gaincalibrator_dict[target]

    plt.scatter(distance_collection, time_collection, c=rms_collection)

def plotcals(source):
    import glob
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    vislist = glob.glob("*.selfcal.ms")

    runs = glob.glob("inf_fb*full_tclean*")
    print(runs)
    runs = np.array(runs)[np.array([3,0,2])]
    #runs = np.concatenate((runs, ["original"]))

    nants = 0
    for vis in vislist:
        msmd.open(vis)
        if len(msmd.antennanames(msmd.antennasforscan(msmd.scansforintent("*OBSERVE_TARGET*")[0]))) > nants:
            nants = len(msmd.antennanames(msmd.antennasforscan(msmd.scansforintent("*OBSERVE_TARGET*")[0])))
    print(nants)

    for ant in range(nants):
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15,5))

        for run in runs:
            for i, ax in enumerate(axes.flatten()):
                if run == "original":
                    tb.open(run+"/Target_"+source+"_"+vislist[i]+"_Band_7_inf_1_p.g")
                else:
                    tb.open(run+"/Target_"+source+"_"+vislist[i]+"_Band_7_inf_fb_6_p.g")

                times = tb.getcol("TIME")
                cals = tb.getcol("CPARAM")
                flags = tb.getcol("FLAG")
                antennas = tb.getcol("ANTENNA1")
                tb.close()

                phase = np.angle(cals) * 180./np.pi

                good = np.logical_and(antennas == ant,  flags[0,0,:] == False)

                ax.plot(times[good], phase[0,0,good], "o")

        fig.tight_layout()

        plt.savefig("Target_"+source+"ANT"+str(ant)+"_phase.png")

        plt.clf()
        plt.close(fig)


def get_min_SNR_spw(snr_per_spw):
   minsnr=1000000000000000.0
   for spw in snr_per_spw.keys():
      if snr_per_spw[spw] < minsnr: minsnr=snr_per_spw[spw]
   return minsnr
      
def remove_modes(selfcal_plan,vis,start_index):  # remove the per_spw and/or per_bb modes for solints following current solint
    preferred_mode=selfcal_plan[vis]['solint_settings'][selfcal_plan['solints'][start_index]]['final_mode']
    for j in range(start_index+1,len(selfcal_plan['solints'])):
       if 'ap' in selfcal_plan['solints'][j] and 'ap' not in selfcal_plan['solints'][start_index]: # exempt over ap solints since they go back to a longer solint
          continue
       if preferred_mode == 'per_bb' or preferred_mode == 'combinespw':
          if 'per_spw' in selfcal_plan[vis]['solint_settings'][selfcal_plan['solints'][j]]['modes_to_attempt']:
             selfcal_plan[vis]['solint_settings'][selfcal_plan['solints'][j]]['modes_to_attempt'].remove('per_spw')
       if preferred_mode == 'combinespw':
          if 'per_bb' in selfcal_plan[vis]['solint_settings'][selfcal_plan['solints'][j]]['modes_to_attempt']:
             selfcal_plan[vis]['solint_settings'][selfcal_plan['solints'][j]]['modes_to_attempt'].remove('per_bb')


def is_iterable(obj):
    try:
        iter(obj)
        return True
    except TypeError:
        return False




