import numpy as np
import numpy 
import scipy.stats
import scipy.signal
import math
import os

import casatools
from casaplotms import plotms
from casatasks import *
from casatools import image, imager
from casatools import msmetadata as msmdtool
from casatools import table as tbtool
from casatools import ms as mstool
from casaviewer import imview
from PIL import Image

ms = mstool()
tb = tbtool()
msmd = msmdtool()
ia = image()
im = imager()

def tclean_wrapper(vis, imagename, band_properties,band,telescope='undefined',scales=[0], smallscalebias = 0.6, mask = '',\
                   nsigma=5.0, imsize = None, cellsize = None, interactive = False, robust = 0.5, gain = 0.1, niter = 50000,\
                   cycleniter = 300, uvtaper = [], savemodel = 'none',gridder='standard', sidelobethreshold=3.0,smoothfactor=1.0,noisethreshold=5.0,\
                   lownoisethreshold=1.5,parallel=False,nterms=1,reffreq='',cyclefactor=3,uvrange='',threshold='0.0Jy',phasecenter='',\
                   startmodel='',pblimit=0.1,pbmask=0.1,field='',datacolumn='',spw='',obstype='single-point', nfrms_multiplier=1.0, \
                   savemodel_only=False, resume=False, image_mosaic_fields_separately=False, mosaic_field_phasecenters={}, mosaic_field_fid_map={},usermodel=''):
    """
    Wrapper for tclean with keywords set to values desired for the Large Program imaging
    See the CASA 6.1.1 documentation for tclean to get the definitions of all the parameters
    """
    msmd.open(vis[0])
    fieldid=msmd.fieldsforname(field)
    msmd.done()
    tb.open(vis[0]+'/FIELD')
    try:
       ephem_column=tb.getcol('EPHEMERIS_ID')
       tb.close()
       if ephem_column[fieldid[0]] !=-1:
          phasecenter='TRACKFIELD'
    except:
       tb.close()
       phasecenter=''

    if obstype=='mosaic' and phasecenter != 'TRACKFIELD':
       phasecenter=get_phasecenter(vis[0],field)

    print('NF RMS Multiplier: ', nfrms_multiplier)
    # Minimize out the nfrms_multiplier at 1.
    nfrms_multiplier = max(nfrms_multiplier, 1.0)

    if nterms == 1:
       reffreq = ''

    baselineThresholdALMA = 400.0

    if mask == '':
       usemask='auto-multithresh'
    else:
       usemask='user'
    if threshold != '0.0Jy':
       nsigma=0.0

    if nsigma != 0.0:
       if nsigma*nfrms_multiplier*0.66 > nsigma:
          nsigma=nsigma*nfrms_multiplier*0.66
       
    

    if telescope=='ALMA':
       if band_properties[vis[0]][band]['75thpct_uv'] > baselineThresholdALMA:
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
       print(band_properties)
       if band_properties[vis[0]][band]['75thpct_uv'] > 2000.0:   # not in ALMA heuristics, but we've been using it
          sidelobethreshold=2.0

       if band_properties[vis[0]][band]['75thpct_uv'] < 300.0:
          sidelobethreshold=2.0
          smoothfactor=1.0
          noisethreshold=4.25*nfrms_multiplier
          lownoisethreshold=1.5*nfrms_multiplier

       if band_properties[vis[0]][band]['75thpct_uv'] < baselineThresholdALMA:
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
       minbeamfrac = 0.3
       #cyclefactor=1.0

    elif 'VLA' in telescope:
       fastnoise=True
       sidelobethreshold=2.0
       smoothfactor=1.0
       noisethreshold=5.0*nfrms_multiplier
       lownoisethreshold=1.5*nfrms_multiplier
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
       wplanes=wplanes * band_properties[vis[0]][band]['75thpct_uv']/20000.0
       if band=='EVLA_L':
          wplanes=wplanes*2.0 # compensate for 1.5 GHz being 2x longer than 3 GHz


       wprojplanes=int(wplanes)
    if (band=='EVLA_L' or band =='EVLA_S') and obstype=='mosaic':
       print('WARNING DETECTED VLA L- OR S-BAND MOSAIC; WILL USE gridder="mosaic" IGNORING W-TERM')
    if obstype=='mosaic':
       gridder='mosaic'
    else:
       if gridder !='wproject':
          gridder='standard' 

    if gridder=='mosaic' and startmodel!='':
       parallel=False
    if not savemodel_only:
        if not resume:
            for ext in ['.image*', '.mask', '.model*', '.pb*', '.psf*', '.residual*', '.sumwt*','.gridwt*']:
                os.system('rm -rf '+ imagename + ext)
        tclean_return = tclean(vis= vis, 
               imagename = imagename, 
               field=field,
               specmode = 'mfs', 
               deconvolver = 'mtmfs',
               scales = scales, 
               gridder=gridder,
               weighting='briggs', 
               robust = robust,
               gain = gain,
               imsize = imsize,
               cell = cellsize, 
               smallscalebias = smallscalebias, #set to CASA's default of 0.6 unless manually changed
               niter = niter, #we want to end on the threshold
               interactive = interactive,
               nsigma=nsigma,    
               cycleniter = cycleniter,
               cyclefactor = cyclefactor, 
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
               uvrange=uvrange,
               threshold=threshold,
               parallel=parallel,
               phasecenter=phasecenter,
               startmodel=startmodel,
               datacolumn=datacolumn,spw=spw,wprojplanes=wprojplanes, verbose=True)

        if image_mosaic_fields_separately:
            for field_id in mosaic_field_phasecenters:
                if 'VLA' in telescope:
                   fov=45.0e9/band_properties[vis[0]][band]['meanfreq']*60.0*1.5*0.5
                   if band_properties[vis[0]][band]['meanfreq'] < 12.0e9:
                      fov=fov*2.0
                if telescope=='ALMA':
                   fov=63.0*100.0e9/band_properties[vis[0]][band]['meanfreq']*1.5*0.5*1.15
                if telescope=='ACA':
                   fov=108.0*100.0e9/band_properties[vis[0]][band]['meanfreq']*1.5*0.5

                center = np.copy(mosaic_field_phasecenters[field_id])
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
                if type(vis) == list:
                    for v in vis:
                        # Since not every field is in every v, we need to check them all so that we don't accidentally get a v without a given field_id
                        if field_id in mosaic_field_fid_map[v]:
                            fid = mosaic_field_fid_map[v][field_id]
                            break

                    im.open(v)
                else:
                    fid = mosaic_field_fid_map[vis][field_id]
                    im.open(vis)

                nx, ny, nfreq, npol = imhead(imagename=imagename.replace(target,target+"_field_"+str(field_id))+".image.tt0", mode="get", \
                        hdkey="shape")

                im.selectvis(field=str(fid), spw=spw)
                im.defineimage(nx=nx, ny=ny, cellx=cellsize, celly=cellsize, phasecenter=fid, mode="mfs", spw=spw)
                im.setvp(dovp=True)
                im.makeimage(type="pb", image=imagename.replace(target,target+"_field_"+str(field_id)) + ".pb.tt0")
                im.close()


     #this step is a workaround a bug in tclean that doesn't always save the model during multiscale clean. See the "Known Issues" section for CASA 5.1.1 on NRAO's website
    if savemodel=='modelcolumn' and usermodel=='':
          print("")
          print("Running tclean a second time to save the model...")
          tclean(vis= vis, 
                 imagename = imagename, 
                 field=field,
                 specmode = 'mfs', 
                 deconvolver = 'mtmfs',
                 scales = scales, 
                 gridder=gridder,
                 weighting='briggs', 
                 robust = robust,
                 gain = gain,
                 imsize = imsize,
                 cell = cellsize, 
                 smallscalebias = smallscalebias, #set to CASA's default of 0.6 unless manually changed
                 niter = 0, 
                 interactive = False,
                 nsigma=0.0, 
                 cycleniter = cycleniter,
                 cyclefactor = cyclefactor, 
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
                 reffreq = reffreq,
                 uvrange=uvrange,
                 threshold=threshold,
                 parallel=False,
                 phasecenter=phasecenter,spw=spw,wprojplanes=wprojplanes)
    elif usermodel !='':
          print('Using user model already filled to model column, skipping model write.')
    
    if not savemodel_only:
        return tclean_return

def usermodel_wrapper(vis, imagename, band_properties,band,telescope='undefined',scales=[0], smallscalebias = 0.6, mask = '',\
                   nsigma=5.0, imsize = None, cellsize = None, interactive = False, robust = 0.5, gain = 0.1, niter = 50000,\
                   cycleniter = 300, uvtaper = [], savemodel = 'none',gridder='standard', sidelobethreshold=3.0,smoothfactor=1.0,noisethreshold=5.0,\
                   lownoisethreshold=1.5,parallel=False,nterms=1,reffreq='',cyclefactor=3,uvrange='',threshold='0.0Jy',phasecenter='',\
                   startmodel='',pblimit=0.1,pbmask=0.1,field='',datacolumn='',spw='',obstype='single-point',\
                   savemodel_only=False, resume=False, image_mosaic_fields_separately=False, mosaic_field_phasecenters={}, mosaic_field_fid_map={},usermodel=''):
    
    if type(usermodel)==list:
       nterms=len(usermodel)
       for i, image in enumerate(usermodel):
           if 'fits' in image:
               importfits(fitsimage=image,imagename=image.replace('.fits',''))
               usermodel[i]=image.replace('.fits','')
    elif type(usermodel)==str:
       importfits(fitsimage=usermodel,imagename=usermmodel.replace('.fits',''))
       nterms=1
   
    msmd.open(vis[0])
    fieldid=msmd.fieldsforname(field)
    msmd.done()
    tb.open(vis[0]+'/FIELD')
    try:
       ephem_column=tb.getcol('EPHEMERIS_ID')
       tb.close()
       if ephem_column[fieldid[0]] !=-1:
          phasecenter='TRACKFIELD'
    except:
       tb.close()
       phasecenter=''

    if obstype=='mosaic' and phasecenter != 'TRACKFIELD':
       phasecenter=get_phasecenter(vis[0],field)

    if nterms == 1:
       reffreq = ''

    if mask == '':
       usemask='auto-multithresh'
    else:
       usemask='user'
    wprojplanes=1
    if band=='EVLA_L' or band =='EVLA_S':
       gridder='wproject'
       wplanes=384 # normalized to S-band A-config
       #scale by 75th percentile uv distance divided by A-config value
       wplanes=wplanes * band_properties[vis[0]][band]['75thpct_uv']/20000.0
       if band=='EVLA_L':
          wplanes=wplanes*2.0 # compensate for 1.5 GHz being 2x longer than 3 GHz


       wprojplanes=int(wplanes)
    if (band=='EVLA_L' or band =='EVLA_S') and obstype=='mosaic':
       print('WARNING DETECTED VLA L- OR S-BAND MOSAIC; WILL USE gridder="mosaic" IGNORING W-TERM')
    if obstype=='mosaic':
       gridder='mosaic'
    else:
       if gridder !='wproject':
          gridder='standard' 

    if gridder=='mosaic' and startmodel!='':
       parallel=False
    for ext in ['.image*', '.mask', '.model*', '.pb*', '.psf*', '.residual*', '.sumwt*','.gridwt*']:
        os.system('rm -rf '+ imagename + ext)
    #regrid start model
    tclean(vis= vis, 
               imagename = imagename+'_usermodel_prep', 
               field=field,
               specmode = 'mfs', 
               deconvolver = 'mtmfs',
               scales = scales, 
               gridder=gridder,
               weighting='briggs', 
               robust = robust,
               gain = gain,
               imsize = imsize,
               cell = cellsize, 
               smallscalebias = smallscalebias, #set to CASA's default of 0.6 unless manually changed
               niter = 0, #we want to end on the threshold
               interactive = interactive,
               nsigma=nsigma,    
               cycleniter = cycleniter,
               cyclefactor = cyclefactor, 
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
               uvrange=uvrange,
               threshold=threshold,
               parallel=parallel,
               phasecenter=phasecenter,
               datacolumn=datacolumn,spw=spw,wprojplanes=wprojplanes, verbose=True,startmodel=usermodel,savemodel='modelcolumn')

     #this step is a workaround a bug in tclean that doesn't always save the model during multiscale clean. See the "Known Issues" section for CASA 5.1.1 on NRAO's website
    if savemodel=='modelcolumn':
          print("")
          print("Running tclean a second time to save the model...")
          tclean(vis= vis, 
                 imagename = imagename+'_usermodel_prep', 
                 field=field,
                 specmode = 'mfs', 
                 deconvolver = 'mtmfs',
                 scales = scales, 
                 gridder=gridder,
                 weighting='briggs', 
                 robust = robust,
                 gain = gain,
                 imsize = imsize,
                 cell = cellsize, 
                 smallscalebias = smallscalebias, #set to CASA's default of 0.6 unless manually changed
                 niter = 0, 
                 interactive = False,
                 nsigma=0.0, 
                 cycleniter = cycleniter,
                 cyclefactor = cyclefactor, 
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
                 uvrange=uvrange,
                 threshold=threshold,
                 parallel=False,
                 phasecenter=phasecenter,spw=spw,wprojplanes=wprojplanes)
 

def collect_listobs_per_vis(vislist):
   listdict={}
   for vis in vislist:
      listdict[vis]=listobs(vis)
   return listdict

def fetch_scan_times(vislist,targets):
   scantimesdict={}
   integrationsdict={}
   integrationtimesdict={}
   integrationtimes=np.array([])
   n_spws=np.array([])
   min_spws=np.array([])
   spwslist_dict = {}
   spws_set_dict = {}
   scansdict={}
   for vis in vislist:
      scantimesdict[vis]={}
      integrationsdict[vis]={}
      integrationtimesdict[vis]={}
      scansdict[vis]={}
      spws_set_dict[vis]={}
      spwslist_dict[vis]=np.array([])
      msmd.open(vis)
      for target in targets:
         scansdict[vis][target]=msmd.scansforfield(target)

      for target in targets:
         scantimes=np.array([])
         integrations=np.array([])
         for scan in scansdict[vis][target]:
            spws_set_dict[vis][scan]=np.array([])
            spws=msmd.spwsforscan(scan)
            #print(scan, spws)
            spws_set_dict[vis][scan]=spws.copy()
            n_spws=np.append(len(spws),n_spws)
            min_spws=np.append(np.min(spws),min_spws)
            spwslist_dict[vis]=np.append(spws,spwslist_dict[vis])
            integrationtime=msmd.exposuretime(scan=scan,spwid=spws[0])['value']
            integrationtimes=np.append(integrationtimes,np.array([integrationtime]))
            times=msmd.timesforscan(scan)
            scantime=np.max(times)+integrationtime-np.min(times)
            ints_per_scan=np.round(scantime/integrationtimes[0])
            scantimes=np.append(scantimes,np.array([scantime]))
            integrations=np.append(integrations,np.array([ints_per_scan]))



         scantimesdict[vis][target]=scantimes.copy()
         #assume each band only has a single integration time
         integrationtimesdict[vis][target]=np.median(integrationtimes)
         integrationsdict[vis][target]=integrations.copy()
      msmd.close()
   if np.mean(n_spws) != np.max(n_spws):
      print('WARNING, INCONSISTENT NUMBER OF SPWS IN SCANS/MSes (Possibly expected if Multi-band VLA data or ALMA Spectral Scan)')
   if np.max(min_spws) != np.min(min_spws):
      print('WARNING, INCONSISTENT MINIMUM SPW IN SCANS/MSes (Possibly expected if Multi-band VLA data or ALMA Spectral Scan)')

   for vis in vislist:
      spwslist_dict[vis]=np.unique(spwslist_dict[vis]).astype(int)
   # jump through some hoops to get the dictionary that has spws per scan into a dictionary of unique
   # spw sets per vis file
   for vis in vislist:
      spws_set_list=[i for i in spws_set_dict[vis].values()]
      spws_set_list=[i.tolist() for i in spws_set_list]
      unique_spws_set_list=[list(i) for i in set(tuple(i) for i in spws_set_list)]
      spws_set_list=[np.array(i) for i in unique_spws_set_list]
      spws_set_dict[vis]=np.array(spws_set_list,dtype=object)

   return scantimesdict,integrationsdict,integrationtimesdict, integrationtimes,np.max(n_spws),np.min(min_spws),spwslist_dict,spws_set_dict

def fetch_scan_times_band_aware(vislist,targets,band_properties,band):
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

         mosaic_field[vis][target]['field_ids']=msmd.fieldsforscans(scansdict[vis][target]).tolist()
         mosaic_field[vis][target]['field_ids']=list(set(mosaic_field[vis][target]['field_ids']))

         mosaic_field[vis][target]['phasecenters'] = []
         for fid in mosaic_field[vis][target]['field_ids']:
             tb.open(vis+'/FIELD')
             mosaic_field[vis][target]['phasecenters'].append(tb.getcol("PHASE_DIR")[:,0,fid])
             tb.close()

         if len(mosaic_field[vis][target]['field_ids']) > 1:
            mosaic_field[vis][target]['mosaic']=True
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
   if len(n_spws) > 0:
      if np.mean(n_spws) != np.max(n_spws):
         print('WARNING, INCONSISTENT NUMBER OF SPWS IN SCANS/MSes (Possibly expected if Multi-band VLA data or ALMA Spectral Scan)')
      if np.max(min_spws) != np.min(min_spws):
         print('WARNING, INCONSISTENT MINIMUM SPW IN SCANS/MSes (Possibly expected if Multi-band VLA data or ALMA Spectral Scan)')
      spwslist=np.unique(spwslist).astype(int)
   else:
     return scantimesdict,scanfieldsdict,scannfieldsdict,scanstartsdict,scanendsdict,integrationsdict,integrationtimesdict, integrationtimes,-99,-99,spwslist,mosaic_field
   return scantimesdict,scanfieldsdict,scannfieldsdict,scanstartsdict,scanendsdict,integrationsdict,integrationtimesdict, integrationtimes,np.max(n_spws),np.min(min_spws),spwslist,spws_set_dict,mosaic_field


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
                       inf_EB_gaincal_combine,spwcombine=True,solint_decrement='fixed',solint_divider=2.0,n_solints=4.0,do_amp_selfcal=False, mosaic=False):
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
   if mosaic and median_scans_per_obs > 1:
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
   
def fetch_targets(vis):
      fields=[]
      msmd.open(vis)
      fieldnames=msmd.fieldnames()
      for fieldname in fieldnames:
         scans=msmd.scansforfield(fieldname)
         if len(scans) > 0:
            fields.append(fieldname)
      msmd.close()
      fields=list(set(fields)) # convert to set to only get unique items
      return fields

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
      names = msmd.antennanames()
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
   names = msmd.antennanames()
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


def get_SNR_self(all_targets,bands,vislist,selfcal_library,n_ant,solints,integration_time,inf_EB_gaincal_combine,inf_EB_gaintype):
   solint_snr={}
   solint_snr_per_field={}
   solint_snr_per_spw={}
   solint_snr_per_field_per_spw={}
   for target in all_targets:
    solint_snr[target]={}
    solint_snr_per_field[target]={}
    solint_snr_per_spw[target]={}
    solint_snr_per_field_per_spw[target]={}
    for band in selfcal_library[target].keys():
      if target in solints[band].keys():
         solint_snr[target][band], solint_snr_per_spw[target][band] = get_SNR_self_individual(vislist, selfcal_library[target][band], n_ant, \
              solints[band][target], integration_time, inf_EB_gaincal_combine, inf_EB_gaintype)

         solint_snr_per_field[target][band] = {}
         solint_snr_per_field_per_spw[target][band] = {}
         for fid in selfcal_library[target][band]['sub-fields']:
             solint_snr_per_field[target][band][fid], solint_snr_per_field_per_spw[target][band][fid] = get_SNR_self_individual(vislist, \
                     selfcal_library[target][band][fid], n_ant, solints[band][target], integration_time, inf_EB_gaincal_combine, inf_EB_gaintype)

   return solint_snr, solint_snr_per_spw, solint_snr_per_field, solint_snr_per_field_per_spw

def get_SNR_self_individual(vislist,selfcal_library,n_ant,solints,integration_time,inf_EB_gaincal_combine,inf_EB_gaintype):
      if inf_EB_gaintype=='G':
         polscale=2.0
      else:
         polscale=1.0

      SNR = max(selfcal_library['SNR_orig'], selfcal_library['intflux_orig']/selfcal_library['e_intflux_orig'])

      solint_snr = {}
      solint_snr_per_spw = {}
      for solint in solints:
         solint_snr[solint]=0.0
         solint_snr_per_spw[solint]={}       
         if solint == 'inf_EB':
            SNR_self_EB=np.zeros(len(selfcal_library['vislist']))
            SNR_self_EB_spw={}
            for i in range(len(selfcal_library['vislist'])):
               SNR_self_EB[i]=SNR/((n_ant)**0.5*(selfcal_library['Total_TOS']/selfcal_library[selfcal_library['vislist'][i]]['TOS'])**0.5)
               SNR_self_EB_spw[selfcal_library['vislist'][i]]={}
               for spw in selfcal_library['spw_map']:
                 if selfcal_library['vislist'][i] in selfcal_library['spw_map'][spw]:
                     SNR_self_EB_spw[selfcal_library['vislist'][i]][str(spw)]=(polscale)**-0.5*SNR/((n_ant-3)**0.5*(selfcal_library['Total_TOS']/selfcal_library[selfcal_library['vislist'][i]]['TOS'])**0.5)*(selfcal_library[selfcal_library['vislist'][i]]['per_spw_stats'][selfcal_library['spw_map'][spw][selfcal_library['vislist'][i]]]['effective_bandwidth']/selfcal_library[selfcal_library['vislist'][i]]['total_effective_bandwidth'])**0.5
            for spw in selfcal_library['spw_map']:
               mean_SNR=0.0
               total_vis = 0
               for j in range(len(selfcal_library['vislist'])):
                  if selfcal_library['vislist'][j] in selfcal_library['spw_map'][spw]:
                     mean_SNR+=SNR_self_EB_spw[selfcal_library['vislist'][j]][str(spw)]
                     total_vis += 1
               mean_SNR=mean_SNR/total_vis
               solint_snr_per_spw[solint][str(spw)]=mean_SNR
            solint_snr[solint]=np.mean(SNR_self_EB)
            selfcal_library['per_EB_SNR']=np.mean(SNR_self_EB)
         elif solint =='scan_inf':
               selfcal_library['per_scan_SNR']=SNR/((n_ant-3)**0.5*(selfcal_library['Total_TOS']/selfcal_library['Median_scan_time'])**0.5)
               solint_snr[solint]=selfcal_library['per_scan_SNR']
               for spw in selfcal_library['spw_map']:
                  vis = list(selfcal_library['spw_map'][spw].keys())[0]
                  true_spw = selfcal_library['spw_map'][spw][vis]
                  solint_snr_per_spw[solint][str(spw)]=SNR/((n_ant-3)**0.5*(selfcal_library['Total_TOS']/selfcal_library['Median_scan_time'])**0.5)*(selfcal_library[vis]['per_spw_stats'][true_spw]['effective_bandwidth']/selfcal_library[vis]['total_effective_bandwidth'])**0.5
         elif solint =='inf' or solint == 'inf_ap':
               selfcal_library['per_scan_SNR']=SNR/((n_ant-3)**0.5*(selfcal_library['Total_TOS']/(selfcal_library['Median_scan_time']/selfcal_library['Median_fields_per_scan']))**0.5)
               solint_snr[solint]=selfcal_library['per_scan_SNR']
               for spw in selfcal_library['spw_map']:
                  vis = list(selfcal_library['spw_map'][spw].keys())[0]
                  true_spw = selfcal_library['spw_map'][spw][vis]
                  solint_snr_per_spw[solint][str(spw)]=SNR/((n_ant-3)**0.5*(selfcal_library['Total_TOS']/(selfcal_library['Median_scan_time']/selfcal_library['Median_fields_per_scan']))**0.5)*(selfcal_library[vis]['per_spw_stats'][true_spw]['effective_bandwidth']/selfcal_library[vis]['total_effective_bandwidth'])**0.5
         elif solint == 'int':
               solint_snr[solint]=SNR/((n_ant-3)**0.5*(selfcal_library['Total_TOS']/integration_time)**0.5)
               for spw in selfcal_library['spw_map']:
                  vis = list(selfcal_library['spw_map'][spw].keys())[0]
                  true_spw = selfcal_library['spw_map'][spw][vis]
                  solint_snr_per_spw[solint][str(spw)]=SNR/((n_ant-3)**0.5*(selfcal_library['Total_TOS']/integration_time)**0.5)*(selfcal_library[vis]['per_spw_stats'][true_spw]['effective_bandwidth']/selfcal_library[vis]['total_effective_bandwidth'])**0.5
         else:
               solint_float=float(solint.replace('s','').replace('_ap',''))
               solint_snr[solint]=SNR/((n_ant-3)**0.5*(selfcal_library['Total_TOS']/solint_float)**0.5)
               for spw in selfcal_library['spw_map']:
                  vis = list(selfcal_library['spw_map'][spw].keys())[0]
                  true_spw = selfcal_library['spw_map'][spw][vis]
                  solint_snr_per_spw[solint][str(spw)]=SNR/((n_ant-3)**0.5*(selfcal_library['Total_TOS']/solint_float)**0.5)*(selfcal_library[vis]['per_spw_stats'][true_spw]['effective_bandwidth']/selfcal_library[vis]['total_effective_bandwidth'])**0.5
      return solint_snr,solint_snr_per_spw

def get_SNR_self_update(all_targets,band,vislist,selfcal_library,n_ant,solint_curr,solint_next,integration_time,solint_snr):
   for target in all_targets:

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
              spw = int(line.split()[-1])
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
   for spw in spwsarray_dict[vis]:
      tb.open(vis+'/SPECTRAL_WINDOW')
      spwbws[spw]=np.abs(np.unique(tb.getcol('TOTAL_BANDWIDTH', startrow = spw, nrow = 1)))[0]/1.0e9 # put bandwidths into GHz
      tb.close()
   spweffbws=spwbws.copy()
   if os.path.exists("cont.dat"):
      spweffbws=get_spw_eff_bandwidth(vis,target,vislist,spwsarray_dict)

   return spwbws,spweffbws


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


def get_image_parameters(vislist,telescope,target,field_ids,band,band_properties,scale_fov=1.0,mosaic=False):
   cells=np.zeros(len(vislist))
   for i in range(len(vislist)):
      #im.open(vislist[i])
      im.selectvis(vis=vislist[i],spw=band_properties[vislist[i]][band]['spwarray'])
      adviseparams= im.advise() 
      cells[i]=adviseparams[2]['value']/2.0
      im.close()
   cell=np.min(cells)
   cellsize='{:0.3f}arcsec'.format(cell)
   nterms=1
   if band_properties[vislist[0]][band]['fracbw'] > 0.1:
      nterms=2
   reffreq = get_reffreq(vislist,field_ids,dict(zip(vislist,[band_properties[vis][band]['spwarray'] for vis in vislist])), telescope)

   if 'VLA' in telescope:
      fov=45.0e9/band_properties[vislist[0]][band]['meanfreq']*60.0*1.5
      if band_properties[vislist[0]][band]['meanfreq'] < 12.0e9:
         fov=fov*2.0
   if telescope=='ALMA':
      fov=63.0*100.0e9/band_properties[vislist[0]][band]['meanfreq']*1.5
   if telescope=='ACA':
      fov=108.0*100.0e9/band_properties[vislist[0]][band]['meanfreq']*1.5
   fov=fov*scale_fov
   if mosaic:
       msmd.open(vislist[0])
       fieldid=msmd.fieldsforname(target)
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

def get_mean_freq(vislist,spwsarray):
   tb.open(vislist[0]+'/SPECTRAL_WINDOW')
   freqarray=tb.getcol('REF_FREQUENCY')
   tb.close()
   meanfreq=np.mean(freqarray[spwsarray[vislist[0]]])
   minfreq=np.min(freqarray[spwsarray[vislist[0]]])
   maxfreq=np.max(freqarray[spwsarray[vislist[0]]])
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


def get_ALMA_bands(vislist,spwstring,spwarray):
   meanfreq, maxfreq,minfreq,fracbw=get_mean_freq(vislist,spwarray)
   observed_bands={}
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
   bands=[band]
   for vis in vislist:
      observed_bands[vis]={}
      observed_bands[vis]['bands']=[band]
      for band in bands:
         observed_bands[vis][band]={}
         observed_bands[vis][band]['spwarray']=spwarray[vis]
         observed_bands[vis][band]['spwstring']=spwstring[vis]+''
         observed_bands[vis][band]['meanfreq']=meanfreq
         observed_bands[vis][band]['maxfreq']=maxfreq
         observed_bands[vis][band]['minfreq']=minfreq
         observed_bands[vis][band]['fracbw']=fracbw

         msmd.open(vis)
         observed_bands[vis][band]['ncorrs']=msmd.ncorrforpol(msmd.polidfordatadesc(spwarray[vis][0]))
         msmd.close()
   get_max_uvdist(vislist,observed_bands[vislist[0]]['bands'].copy(),observed_bands,'ALMA')
   return bands,observed_bands


def get_VLA_bands(vislist,fields):
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
                meanfreq,maxfreq,minfreq,fracbw=get_mean_freq([vis],{vis: np.array([observed_bands[vis][band]['spwarray'][i]])})
                if (meanfreq==8.332e9) or (meanfreq==8.460e9):
                   indices_to_remove=np.append(indices_to_remove,[i])
            observed_bands[vis][band]['spwarray']=np.delete(observed_bands[vis][band]['spwarray'],indices_to_remove.astype(int))
         elif (band == 'EVLA_C') and (len(index[0]) >= 2): # ignore pointing band

            observed_bands[vis][band]['spwarray']=spw_names_spw[index[0]]
            indices_to_remove=np.array([])
            for i in range(len(observed_bands[vis][band]['spwarray'])):
                meanfreq,maxfreq,minfreq,fracbw=get_mean_freq([vis],{vis: np.array([observed_bands[vis][band]['spwarray'][i]])})
                if (meanfreq==4.832e9) or (meanfreq==4.960e9):
                   indices_to_remove=np.append(indices_to_remove,[i])
            observed_bands[vis][band]['spwarray']=np.delete(observed_bands[vis][band]['spwarray'],indices_to_remove.astype(int))
         else:
            observed_bands[vis][band]['spwarray']=spw_names_spw[index[0]]
         spwslist=observed_bands[vis][band]['spwarray'].tolist()
         spwstring=','.join(str(spw) for spw in spwslist)
         observed_bands[vis][band]['spwstring']=spwstring+''
         observed_bands[vis][band]['meanfreq'],observed_bands[vis][band]['maxfreq'],observed_bands[vis][band]['minfreq'],observed_bands[vis][band]['fracbw']=get_mean_freq([vis],{vis: [observed_bands[vis][band]['spwarray']]})

         msmd.open(vis)
         observed_bands[vis][band]['ncorrs']=msmd.ncorrforpol(msmd.polidfordatadesc(observed_bands[vis][band]['spwarray'][0]))
         msmd.close()
   bands_match=True
   for i in range(len(vislist)):
      for j in range(i+1,len(vislist)):
         bandlist_match=(observed_bands[vislist[i]]['bands'] ==observed_bands[vislist[i+1]]['bands'])
         if not bandlist_match:
            bands_match=False
   if not bands_match:
     print('WARNING: INCONSISTENT BANDS IN THE MSFILES')
   get_max_uvdist(vislist,observed_bands[vislist[0]]['bands'].copy(),observed_bands,'VLA')
   return observed_bands[vislist[0]]['bands'].copy(),observed_bands


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
     names = msmd.antennanames()
     offset = [msmd.antennaoffset(name) for name in names]
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

def plot_image(filename,outname,min=None,max=None,zoom=2):
   header=imhead(filename)
   size=np.max(header['shape'])
   if os.path.exists(filename.replace('image.tt0','mask')): #if mask exists draw it as a contour, else don't use contours
      if min == None:
         imview(raster={'file': filename, 'scaling': -1, 'colorwedge': True},\
               contour={'file': filename.replace('image.tt0','mask'), 'levels': [1] },\
             zoom={'blc': [int(size/2-size/(zoom*2)),int(size/2-size/(zoom*2))],\
                   'trc': [int(size/2+size/(zoom*2)),int(size/2+size/(zoom*2))]},\
             out={'file': outname, 'orient': 'landscape'})
      else:
         imview(raster={'file': filename, 'scaling': -1, 'range': [min,max], 'colorwedge': True},\
               contour={'file': filename.replace('image.tt0','mask'), 'levels': [1] },\
             zoom={'blc': [int(size/2-size/(zoom*2)),int(size/2-size/(zoom*2))],\
                   'trc': [int(size/2+size/(zoom*2)),int(size/2+size/(zoom*2))]},\
             out={'file': outname, 'orient': 'landscape'})
   else:
      if min == None:
         imview(raster={'file': filename, 'scaling': -1, 'colorwedge': True},\
             zoom={'blc': [int(size/2-size/(zoom*2)),int(size/2-size/(zoom*2))],\
                   'trc': [int(size/2+size/(zoom*2)),int(size/2+size/(zoom*2))]},\
             out={'file': outname, 'orient': 'landscape'})
      else:
         imview(raster={'file': filename, 'scaling': -1, 'range': [min,max], 'colorwedge': True},\
             zoom={'blc': [int(size/2-size/(zoom*2)),int(size/2-size/(zoom*2))],\
                   'trc': [int(size/2+size/(zoom*2)),int(size/2+size/(zoom*2))]},\
             out={'file': outname, 'orient': 'landscape'})
   #make image square since imview makes it a strange dimension
   im = Image.open(outname)
   width, height = im.size
   if height > width:
      remainder=height-width
      trim_amount=int(remainder/2.0)
      im1=im.crop((0,trim_amount,width-1,height-trim_amount-1))
   else:
      remainder=width-height
      trim_amount=int(remainder/2.0)
      im1=im.crop((trim_amount,0,width-trim_amount-1,height-1))
   im1.save(outname)

def get_flagged_solns_per_ant(gaintable,vis):
     # Get the antenna names and offsets.

     msmd = casatools.msmetadata()
     tb = casatools.table()

     msmd.open(vis)
     names = msmd.antennanames()
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
             str(i)+' giving  [ntrue(FLAG)]]')['0'].sum() for i in \
             range(len(names))]
     nunflagged = [tb.calc('[select from '+gaintable+' where ANTENNA1=='+\
             str(i)+' giving  [nfalse(FLAG)]]')['0'].sum() for i in \
             range(len(names))]
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

def generate_weblog(sclib,solints,bands,directory='weblog'):
   from datetime import datetime
   os.system('rm -rf '+directory)
   os.system('mkdir '+directory)
   os.system('mkdir '+directory+'/images')
   htmlOut=open(directory+'/index.html','w')
   htmlOut.writelines('<html>\n')
   htmlOut.writelines('<title>SelfCal Weblog</title>\n')
   htmlOut.writelines('<head>\n')
   htmlOut.writelines('</head>\n')
   htmlOut.writelines('<body>\n')
   htmlOut.writelines('<a name="top"></a>\n')
   htmlOut.writelines('<h1>SelfCal Weblog</h1>\n')
   htmlOut.writelines('<h4>Date Executed:'+datetime.today().strftime('%Y-%m-%d')+'</h4>\n')
   htmlOut.writelines('<h2>Targets:</h2>\n')
   targets=list(sclib.keys())
   for target in targets:
      htmlOut.writelines('<a href="#'+target+'">'+target+'</a><br>\n')
   htmlOut.writelines('<h2>Bands:</h2>\n')
   bands_string=', '.join([str(elem) for elem in bands])
   htmlOut.writelines(''+bands_string+'\n')
   htmlOut.writelines('<h2>Solints to Attempt:</h2>\n')
   for band in bands:
      for target in solints[band].keys():     
         solints_string=', '.join([str(elem) for elem in solints[band][target]])
         htmlOut.writelines('<br>'+band+': '+solints_string)

   for target in targets:
      htmlOut.writelines('<a name="'+target+'"></a>\n')
      htmlOut.writelines('<h2>'+target+' Summary</h2>\n')
      htmlOut.writelines('<a href="#top">Back to Top</a><br>\n')
      bands_obsd=list(sclib[target].keys())

      for band in bands_obsd:
         htmlOut.writelines('<h2>'+band+'</h2>\n')
         htmlOut.writelines('<a name="'+target+'_'+band+'"></a>\n')
         htmlOut.writelines('Selfcal Success?: '+str(sclib[target][band]['SC_success'])+'<br>\n')
         keylist=sclib[target][band].keys()
         if 'Stop_Reason' not in keylist:
            htmlOut.writelines('Stop Reason: Estimated Selfcal S/N too low for solint<br><br>\n')
            if sclib[target][band]['SC_success']==False:
               render_summary_table(htmlOut,sclib,target,band,directory=directory)
               continue
         else:   
            htmlOut.writelines('Stop Reason: '+str(sclib[target][band]['Stop_Reason'])+'<br><br>\n')
            print(target,band,sclib[target][band]['Stop_Reason'])
            if (('Estimated_SNR_too_low_for_solint' in sclib[target][band]['Stop_Reason']) or ('Selfcal_Not_Attempted' in sclib[target][band]['Stop_Reason'])) and sclib[target][band]['final_solint']=='None':
               render_summary_table(htmlOut,sclib,target,band,directory=directory)
               continue
         htmlOut.writelines('Final Successful solint: '+str(sclib[target][band]['final_solint'])+'<br><br>\n')
         if sclib[target][band]['obstype'] == 'mosaic':
             htmlOut.writelines('<a href="'+target+'_field-by-field/index.html">Field-by-Field Summary</a><br><br>\n')
         if 'Found_contdotdat' in sclib[target][band]:
             htmlOut.write('<font color="red">WARNING:</font> No cont.dat entry found for target '+target+', this likely indicates that hif_findcont was mitigated. We suggest you re-run findcont without mitigation.<br><br>')
         # Summary table for before/after SC
         render_summary_table(htmlOut,sclib,target,band,directory=directory)

         #Noise Summary plot
         N_initial,intensity_initial,rms_inital=create_noise_histogram(sanitize_string(target)+'_'+band+'_initial.image.tt0')
         N_final,intensity_final,rms_final=create_noise_histogram(sanitize_string(target)+'_'+band+'_final.image.tt0')
         if 'theoretical_sensitivity' in keylist:
            rms_theory=sclib[target][band]['theoretical_sensitivity']
            if rms_theory != -99.0:
               rms_theory=sclib[target][band]['theoretical_sensitivity']
            else:
               rms_theory=0.0
         else:
            rms_theory=0.0
         create_noise_histogram_plots(N_initial,N_final,intensity_initial,intensity_final,rms_inital,rms_final,\
                                      directory+'/images/'+sanitize_string(target)+'_'+band+'_noise_plot.png',rms_theory)
         htmlOut.writelines('<br>Initial vs. Final Noise Characterization<br>')
         htmlOut.writelines('<a href="images/'+sanitize_string(target)+'_'+band+'_noise_plot.png"><img src="images/'+sanitize_string(target)+'_'+band+'_noise_plot.png" ALT="Noise Characteristics" WIDTH=300 HEIGHT=300></a><br>\n')
         
         # Solint summary table
         render_selfcal_solint_summary_table(htmlOut,sclib,target,band,solints)

         # PER SPW STATS TABLE
         if 'per_spw_stats' in sclib[target][band].keys():
            render_spw_stats_summary_table(htmlOut,sclib,target,band)

   # Close main weblog file
   htmlOut.writelines('</body>\n')
   htmlOut.writelines('</html>\n')
   htmlOut.close()
   
   # Pages for each solint
   render_per_solint_QA_pages(sclib,solints,bands,directory=directory)
 

def render_summary_table(htmlOut,sclib,target,band,directory='weblog'):
         plot_image(sanitize_string(target)+'_'+band+'_final.image.tt0',\
                      directory+'/images/'+sanitize_string(target)+'_'+band+'_final.image.tt0.png', zoom=2 if directory=="weblog" else 1)
         image_stats=imstat(sanitize_string(target)+'_'+band+'_final.image.tt0')
         
         plot_image(sanitize_string(target)+'_'+band+'_initial.image.tt0',\
                      directory+'/images/'+sanitize_string(target)+'_'+band+'_initial.image.tt0.png',min=image_stats['min'][0],max=image_stats['max'][0], zoom=2 if directory=="weblog" else 1) 
         os.system('rm -rf '+sanitize_string(target)+'_'+band+'_final_initial_div_final.image.tt0 '+sanitize_string(target)+'_'+band+'_final_initial_div_final.temp.image.tt0')

         ### Hacky way to suppress stuff outside mask in ratio images.
         immath(imagename=[sanitize_string(target)+'_'+band+'_final.image.tt0',sanitize_string(target)+'_'+band+'_initial.image.tt0',sanitize_string(target)+'_'+band+'_final.mask'],\
                mode='evalexpr',expr='((IM0-IM1)/IM0)*IM2',outfile=sanitize_string(target)+'_'+band+'_final_initial_div_final.temp.image.tt0')
         immath(imagename=[sanitize_string(target)+'_'+band+'_final_initial_div_final.temp.image.tt0'],\
                mode='evalexpr',expr='iif(IM0==0.0,-99.0,IM0)',outfile=sanitize_string(target)+'_'+band+'_final_initial_div_final.image.tt0')
         plot_image(sanitize_string(target)+'_'+band+'_final_initial_div_final.image.tt0',\
                      directory+'/images/'+sanitize_string(target)+'_'+band+'_final_initial_div_final.image.tt0.png',\
                       min=-1.0,max=1.0, zoom=2 if directory=="weblog" else 1) 
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
         line='<table>\n  <tr bgcolor="#ffffff">\n    <th>Data:</th>\n    '
         for data_type in ['Initial', 'Final', 'Comparison']:
            line+='<th>'+data_type+'</th>\n    '
         line+='</tr>\n'
         htmlOut.writelines(line)
         quantities=['Image','intflux','SNR','SNR_NF','RMS','RMS_NF','Beam']
         for key in quantities:
            if key =='Image':
               line='<tr bgcolor="#ffffff">\n    <td>Image: </td>\n'
            if key =='SNR':
               line='<tr bgcolor="#ffffff">\n    <td>SNR: </td>\n'
            if key =='intflux':
               line='<tr bgcolor="#ffffff">\n    <td>Integrated Flux: </td>\n'
            if key =='RMS':
               line='<tr bgcolor="#ffffff">\n    <td>RMS: </td>\n'
            if key =='SNR_NF':
               line='<tr bgcolor="#ffffff">\n    <td>SNR (near-field): </td>\n'
            if key =='RMS_NF':
               line='<tr bgcolor="#ffffff">\n    <td>RMS (near-field): </td>\n'
            if key =='Beam':
               line='<tr bgcolor="#ffffff">\n    <td>Beam: </td>\n'

            for data_type in ['orig', 'final', 'comp']:
               if data_type !='comp':
                  if key =='Image':
                     if data_type=='orig':
                        line+='<td><a href="images/'+sanitize_string(target)+'_'+band+'_initial.image.tt0.png"><img src="images/'+sanitize_string(target)+'_'+band+'_initial.image.tt0.png" ALT="pre-SC-solint image" WIDTH=400 HEIGHT=400></a> </td>\n'
                     if data_type=='final':
                        line+='<td><a href="images/'+sanitize_string(target)+'_'+band+'_final.image.tt0.png"><img src="images/'+sanitize_string(target)+'_'+band+'_final.image.tt0.png" ALT="pre-SC-solint image" WIDTH=400 HEIGHT=400></a> </td>\n'
                  if key =='SNR':
                     line+='    <td>{:0.3f} </td>\n'.format(sclib[target][band][key+'_'+data_type])
                  if key =='intflux':
                     line+='    <td>{:0.3f} +/- {:0.3f} mJy</td>\n'.format(sclib[target][band][key+'_'+data_type]*1000.0,sclib[target][band]['e_'+key+'_'+data_type]*1000.0)
                  if key =='SNR_NF':
                     line+='    <td>{:0.3f} </td>\n'.format(sclib[target][band][key+'_'+data_type])
                  if key =='RMS':
                     line+='    <td>{:0.3f} mJy/beam </td>\n'.format(sclib[target][band][key+'_'+data_type]*1000.0)
                  if key =='RMS_NF':
                     line+='    <td>{:0.3f} mJy/beam </td>\n'.format(sclib[target][band][key+'_'+data_type]*1000.0)
                  if key=='Beam':
                     line+='    <td>{:0.3f}"x{:0.3f}" {:0.3f} deg </td>\n'.format(sclib[target][band][key+'_major'+'_'+data_type],sclib[target][band][key+'_minor'+'_'+data_type],sclib[target][band][key+'_PA'+'_'+data_type])
               else:
                  if key =='Image':
                        line+='<td><a href="images/'+sanitize_string(target)+'_'+band+'_final_initial_div_final.image.tt0.png"><img src="images/'+sanitize_string(target)+'_'+band+'_final_initial_div_final.image.tt0.png" ALT="pre-SC-solint image" WIDTH=400 HEIGHT=400></a> </td>\n'
                  if key =='intflux':
                     if sclib[target][band][key+'_orig'] == 0:
                         line+='    <td>{:0.3f} </td>\n'.format(1.0)
                     else:
                         line+='    <td>{:0.3f} </td>\n'.format(sclib[target][band][key+'_final']/sclib[target][band][key+'_orig'])
                  if key =='SNR':
                     line+='    <td>{:0.3f} </td>\n'.format(sclib[target][band][key+'_final']/sclib[target][band][key+'_orig'])
                  if key =='SNR_NF':
                     line+='    <td>{:0.3f} </td>\n'.format(sclib[target][band][key+'_final']/sclib[target][band][key+'_orig'])
                  if key =='RMS':
                     line+='    <td>{:0.3f} </td>\n'.format(sclib[target][band][key+'_orig']/sclib[target][band][key+'_final'])
                  if key =='RMS_NF':
                     line+='    <td>{:0.3f} </td>\n'.format(sclib[target][band][key+'_orig']/sclib[target][band][key+'_final'])
                  if key=='Beam':
                     line+='    <td>{:0.3f}</td>\n'.format((sclib[target][band][key+'_major_final']*sclib[target][band][key+'_minor_final'])/(sclib[target][band][key+'_major_orig']*sclib[target][band][key+'_minor_orig']))
            line+='</tr>\n    '
            htmlOut.writelines(line)
         htmlOut.writelines('</table>\n')
         htmlOut.writelines('	</td>\n')
         htmlOut.writelines('	</tr>\n')
         htmlOut.writelines('</table>\n')

def render_selfcal_solint_summary_table(htmlOut,sclib,target,band,solints):
         #  SELFCAL SUMMARY TABLE   
         vislist=sclib[target][band]['vislist']
         solint_list=solints[band][target]
         htmlOut.writelines('<br>Per solint stats: <br>\n')
         htmlOut.writelines('<table cellspacing="0" cellpadding="0" border="0" bgcolor="#000000">\n')
         htmlOut.writelines('	<tr>\n')
         htmlOut.writelines('		<td>\n')
         line='<table>\n  <tr bgcolor="#ffffff">\n    <th>Solint:</th>\n    '
         for solint in solint_list:
            line+='<th>'+solint+'</th>\n    '
         line+='</tr>\n'
         htmlOut.writelines(line)
         vis_keys=list(sclib[target][band][vislist[len(vislist)-1]].keys())
         quantities=['Pass','intflux_final','intflux_improvement','SNR_final','SNR_Improvement','SNR_NF_final','SNR_NF_Improvement','RMS_final','RMS_Improvement','RMS_NF_final','RMS_NF_Improvement','Beam_Ratio','clean_threshold','Plots']
         for key in quantities:
            if key =='Pass':
               line='<tr bgcolor="#ffffff">\n    <td>Result: </td>\n'
            if key =='intflux_final':
               line='<tr bgcolor="#ffffff">\n    <td>Integrated Flux: </td>\n'
            if key =='intflux_improvement':
               line='<tr bgcolor="#ffffff">\n    <td>Integrated Flux Change: </td>\n'
            if key =='SNR_final':
               line='<tr bgcolor="#ffffff">\n    <td>Dynamic Range: </td>\n'
            if key =='SNR_Improvement':
               line='<tr bgcolor="#ffffff">\n    <td>DR Improvement: </td>\n'
            if key =='SNR_NF_final':
               line='<tr bgcolor="#ffffff">\n    <td>Dynamic Range (near-field): </td>\n'
            if key =='SNR_NF_Improvement':
               line='<tr bgcolor="#ffffff">\n    <td>DR Improvement (near-field): </td>\n'
            if key =='RMS_final':
               line='<tr bgcolor="#ffffff">\n    <td>RMS: </td>\n'
            if key =='RMS_Improvement':
               line='<tr bgcolor="#ffffff">\n    <td>RMS Improvement: </td>\n'
            if key =='RMS_NF_final':
               line='<tr bgcolor="#ffffff">\n    <td>RMS (near-field): </td>\n'
            if key =='RMS_NF_Improvement':
               line='<tr bgcolor="#ffffff">\n    <td>RMS Improvement (near-field): </td>\n'
            if key =='Beam_Ratio':
               line='<tr bgcolor="#ffffff">\n    <td>Ratio of Beam Area: </td>\n'
            if key =='clean_threshold':
               line='<tr bgcolor="#ffffff">\n    <td>Clean Threshold: </td>\n'
            if key =='Plots':
               line='<tr bgcolor="#ffffff">\n    <td>Plots: </td>\n'
            for solint in solint_list:
               if solint in vis_keys:
                  vis_solint_keys=sclib[target][band][vislist[len(vislist)-1]][solint].keys()
                  if key != 'Pass' and sclib[target][band][vislist[len(vislist)-1]][solint]['Pass'] == 'None':
                      line+='    <td> - </td>\n'
                      continue
                  if key=='Pass':
                     if sclib[target][band][vislist[len(vislist)-1]][solint]['Pass'] == False:
                        line+='    <td><font color="red">{}</font> {}</td>\n'.format('Fail',sclib[target][band][vislist[len(vislist)-1]][solint]['Fail_Reason'])
                     elif sclib[target][band][vislist[len(vislist)-1]][solint]['Pass'] == 'None':
                        line+='    <td><font color="green">{}</font> {}</td>\n'.format('Not attempted',sclib[target][band][vislist[len(vislist)-1]][solint]['Fail_Reason'])
                     else:
                        line+='    <td><font color="blue">{}</font></td>\n'.format('Pass')
                  if key=='intflux_final':
                     line+='    <td>{:0.3f} +/- {:0.3f} mJy</td>\n'.format(sclib[target][band][vislist[len(vislist)-1]][solint]['intflux_post']*1000.0,sclib[target][band][vislist[len(vislist)-1]][solint]['e_intflux_post']*1000.0)
                  if key=='intflux_improvement':
                     if sclib[target][band][vislist[len(vislist)-1]][solint]['intflux_pre'] == 0:
                        line+='    <td>{:0.3f}</td>\n'.format(1.0)
                     else:
                        line+='    <td>{:0.3f}</td>\n'.format(sclib[target][band][vislist[len(vislist)-1]][solint]['intflux_post']/sclib[target][band][vislist[len(vislist)-1]][solint]['intflux_pre'])                      
                  if key=='SNR_final':
                     line+='    <td>{:0.3f}</td>\n'.format(sclib[target][band][vislist[len(vislist)-1]][solint]['SNR_post'])
                  if key=='SNR_Improvement':
                     line+='    <td>{:0.3f}</td>\n'.format(sclib[target][band][vislist[len(vislist)-1]][solint]['SNR_post']/sclib[target][band][vislist[len(vislist)-1]][solint]['SNR_pre'])
                  if key=='SNR_NF_final':
                     line+='    <td>{:0.3f}</td>\n'.format(sclib[target][band][vislist[len(vislist)-1]][solint]['SNR_NF_post'])
                  if key=='SNR_NF_Improvement':
                     line+='    <td>{:0.3f}</td>\n'.format(sclib[target][band][vislist[len(vislist)-1]][solint]['SNR_NF_post']/sclib[target][band][vislist[len(vislist)-1]][solint]['SNR_NF_pre'])

                  if key=='RMS_final':
                     line+='    <td>{:0.3e} mJy/bm</td>\n'.format(sclib[target][band][vislist[len(vislist)-1]][solint]['RMS_post']*1000.0)
                  if key=='RMS_Improvement':
                     line+='    <td>{:0.3e}</td>\n'.format(sclib[target][band][vislist[len(vislist)-1]][solint]['RMS_pre']/sclib[target][band][vislist[len(vislist)-1]][solint]['RMS_post'])
                  if key=='RMS_NF_final':
                     line+='    <td>{:0.3e} mJy/bm</td>\n'.format(sclib[target][band][vislist[len(vislist)-1]][solint]['RMS_NF_post']*1000.0)
                  if key=='RMS_NF_Improvement':
                     line+='    <td>{:0.3e}</td>\n'.format(sclib[target][band][vislist[len(vislist)-1]][solint]['RMS_NF_pre']/sclib[target][band][vislist[len(vislist)-1]][solint]['RMS_NF_post'])

                  if key=='Beam_Ratio':
                     line+='    <td>{:0.3e}</td>\n'.format((sclib[target][band][vislist[len(vislist)-1]][solint]['Beam_major_post']*sclib[target][band][vislist[len(vislist)-1]][solint]['Beam_minor_post'])/(sclib[target][band]['Beam_major_orig']*sclib[target][band]['Beam_minor_orig']))
                  if key =='clean_threshold':
                     if key in vis_solint_keys:
                        line+='    <td>{:0.3e} mJy/bm</td>\n'.format(sclib[target][band][vislist[len(vislist)-1]][solint]['clean_threshold']*1000.0)
                     else:
                        line+='    <td>Not Available</td>\n'
                  if key =='Plots':
                     line+='    <td><a href="'+target+'_'+band+'_'+solint+'.html">QA Plots</a></td>\n'

               else:
                  line+='    <td> - </td>\n'
            line+='</tr>\n    '
            htmlOut.writelines(line)
         htmlOut.writelines('<tr bgcolor="#ffffff">\n    <td colspan="'+str(len(solint_list)+1)+'">Flagged solutions by antenna: </td></tr>\n')
         for vis in vislist:
            line='<tr bgcolor="#ffffff">\n    <td>'+vis+': </td>\n'
            for solint in solint_list:
               if solint in vis_keys and sclib[target][band][vis][solint]['Pass'] != 'None' and 'gaintable' in sclib[target][band][vis][solint]:
                  # only evaluate last gaintable not the pre-apply table
                  gaintable=sclib[target][band][vis][solint]['gaintable'][len(sclib[target][band][vis][solint]['gaintable'])-1]
                  line+='<td><a href="images/plot_ants_'+gaintable+'.png"><img src="images/plot_ants_'+gaintable+'.png" ALT="antenna positions with flagging plot" WIDTH=200 HEIGHT=200></a></td>\n'
               else:
                   line+='<td>-</td>\n'
            line+='</tr>\n    '
            htmlOut.writelines(line)
            for quantity in ['Nsols','Flagged_Sols','Frac_Flagged']:
               line='<tr bgcolor="#ffffff">\n    <td>'+quantity+'</td>\n'
               for solint in solint_list:
                  if solint in vis_keys and sclib[target][band][vis][solint]['Pass'] != 'None' and 'gaintable' in sclib[target][band][vis][solint]:
                     # only evaluate last gaintable not the pre-apply table
                     gaintable=sclib[target][band][vis][solint]['gaintable'][len(sclib[target][band][vis][solint]['gaintable'])-1]
                     nflagged_sols, nsols=get_sols_flagged_solns(gaintable)
                     if quantity =='Nsols':
                        line+='<td>'+str(nsols)+'</td>\n'
                     if quantity =='Flagged_Sols':
                        line+='<td>'+str(nflagged_sols)+'</td>\n'
                     if quantity =='Frac_Flagged':
                        line+='<td>'+'{:0.3f}'.format(nflagged_sols/nsols)+'</td>\n'
                  else:
                     line+='<td>-</td>\n'
               line+='</tr>\n    '

               htmlOut.writelines(line)
         htmlOut.writelines('</table>\n')
         htmlOut.writelines('	</td>\n')
         htmlOut.writelines('	</tr>\n')
         htmlOut.writelines('</table>\n')

def render_spw_stats_summary_table(htmlOut,sclib,target,band):
   spwlist=list(sclib[target][band]['spw_map'].keys())
   htmlOut.writelines('<br>Per SPW stats: <br>\n')
   htmlOut.writelines('<table cellspacing="0" cellpadding="0" border="0" bgcolor="#000000">\n')
   htmlOut.writelines('	<tr>\n')
   htmlOut.writelines('		<td>\n')
   line='<table>\n  <tr bgcolor="#ffffff">\n    <th>Virtual SPW ID:</th>\n    '
   for spw in spwlist:
      line+='<th>'+str(spw)+'</th>\n    '
   line+='</tr>\n'
   line+='<tr bgcolor="#ffffff">\n    <td colspan="{0:d}" style="text-align: center">Virtual SPW to real SPW mapping</td>\n</tr>\n'.format(len(spwlist)+1)
   htmlOut.writelines(line)

   quantities=sclib[target][band]['vislist'] + ['bandwidth','effective_bandwidth','SNR_orig','SNR_final','RMS_orig','RMS_final']
   for key in quantities:
      if key == 'bandwidth':
         line = '<tr bgcolor="#ffffff">\n    <td colspan="{0:d}" style="text-align: center">Metadata and Statistics</td>\n</tr>\n'.format(len(spwlist)+1)
      else:
         line = ''
      line+='<tr bgcolor="#ffffff">\n    <td>'+key+': </td>\n'
      for spw in spwlist:
         if spw in sclib[target][band]['per_spw_stats']:
             spwkeys=sclib[target][band]['per_spw_stats'][spw].keys()
             if 'SNR' in key and key in spwkeys:
                line+='    <td>{:0.3f}</td>\n'.format(sclib[target][band]['per_spw_stats'][spw][key])
             if 'RMS' in key and key in spwkeys:
                line+='    <td>{:0.3e} mJy/bm</td>\n'.format(sclib[target][band]['per_spw_stats'][spw][key]*1000.0)
         vis = list(sclib[target][band]['spw_map'][spw].keys())[0]
         if 'bandwidth' in key and key in sclib[target][band][vis]['per_spw_stats'][sclib[target][band]['spw_map'][spw][vis]].keys():
            line+='    <td>{:0.4f} GHz</td>\n'.format(sclib[target][band][vis]['per_spw_stats'][sclib[target][band]['spw_map'][spw][vis]][key])
         if key in sclib[target][band]['vislist'] and key in sclib[target][band]['spw_map'][spw]:
            line+='    <td>{:0d}</td>\n'.format(sclib[target][band]['spw_map'][spw][key])
         elif key in sclib[target][band]['vislist'] and key not in sclib[target][band]['spw_map'][spw]:
            line+='    <td>-</td>\n'
      line+='</tr>\n    '
      htmlOut.writelines(line)
   htmlOut.writelines('</table>\n')
   htmlOut.writelines('	</td>\n')
   htmlOut.writelines('	</tr>\n')
   htmlOut.writelines('</table>\n')
   for spw in spwlist:
      if spw in sclib[target][band]['per_spw_stats']:
          spwkeys=sclib[target][band]['per_spw_stats'][spw].keys()
          if 'delta_SNR' in spwkeys or 'delta_RMS' in spwkeys or 'delta_beamarea' in spwkeys:
             if sclib[target][band]['per_spw_stats'][spw]['delta_SNR'] < 0.0:
                htmlOut.writelines('WARNING SPW '+str(spw)+' HAS LOWER SNR POST SELFCAL<br>\n')
             if sclib[target][band]['per_spw_stats'][spw]['delta_RMS'] > 0.0:
                htmlOut.writelines('WARNING SPW '+str(spw)+' HAS HIGHER RMS POST SELFCAL<br>\n')
             if sclib[target][band]['per_spw_stats'][spw]['delta_beamarea'] > 0.05:
                htmlOut.writelines('WARNING SPW '+str(spw)+' HAS A >0.05 CHANGE IN BEAM AREA POST SELFCAL<br>\n')

def render_per_solint_QA_pages(sclib,solints,bands,directory='weblog'):
  ## Per Solint pages
   targets=list(sclib.keys())
   for target in targets:
      bands_obsd=list(sclib[target].keys())
      for band in bands_obsd:
         if sclib[target][band]['final_solint'] == 'None':
            final_solint_index=0
         else:
            final_solint_index=solints[band][target].index(sclib[target][band]['final_solint']) 

         vislist=sclib[target][band]['vislist']
         index_addition=1
         if sclib[target][band]['final_solint'] != 'inf_ap' and sclib[target][band]['final_solint'] != 'None':
            index_addition=2
         # if it's a dataset where inf_EB == inf, make sure to take out the assumption that there would be an 'inf' solution
         if 'inf' not in solints[band][target]:
            index_addition=index_addition-1
         #add an additional check to make sure the final index will be in the array
         if final_solint_index+index_addition-1 > (len(solints[band][target])-1):
            index_addition=(final_solint_index+index_addition-1) - (len(solints[band][target])-1)
         

         final_solint_to_plot=solints[band][target][final_solint_index+index_addition-1]
         keylist=sclib[target][band][vislist[0]].keys()
         if index_addition == 2 and final_solint_to_plot not in keylist:
           index_addition=index_addition-1


         
         #for i in range(final_solint_index+index_addition):
         for i in range(len(solints[band][target])):

            if solints[band][target][i] not in keylist or sclib[target][band][vislist[len(vislist)-1]][solints[band][target][i]]['Pass'] == 'None':
               continue
            htmlOutSolint=open(directory+'/'+target+'_'+band+'_'+solints[band][target][i]+'.html','w')
            htmlOutSolint.writelines('<html>\n')
            htmlOutSolint.writelines('<title>SelfCal Weblog</title>\n')
            htmlOutSolint.writelines('<head>\n')
            htmlOutSolint.writelines('</head>\n')
            htmlOutSolint.writelines('<body>\n')
            htmlOutSolint.writelines('<a name="top"></a>\n')
            htmlOutSolint.writelines('<h2>'+target+' Plots</h2>\n')
            htmlOutSolint.writelines('<h2>'+band+'</h2>\n')
            htmlOutSolint.writelines('<h2>Targets:</h2>\n')
            keylist=sclib[target][band][vislist[0]].keys()
            solints_string=''
            for j in range(final_solint_index+index_addition):
               if solints[band][target][j] not in keylist:
                  continue
               solints_string+='<a href="'+target+'_'+band+'_'+solints[band][target][j]+'.html">'+solints[band][target][j]+'  </a><br>\n'
            htmlOutSolint.writelines('<br>Solints: '+solints_string)

            htmlOutSolint.writelines('<h3>Solint: '+solints[band][target][i]+'</h3>\n')       
            keylist_top=sclib[target][band].keys()
            htmlOutSolint.writelines('<a href="index.html#'+target+'_'+band+'">Back to Main Target/Band</a><br>\n')


            #must select last key for pre Jan 14th runs since they only wrote pass to the last MS dictionary entry
            passed=sclib[target][band][vislist[len(vislist)-1]][solints[band][target][i]]['Pass']
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
            plot_image(sanitize_string(target)+'_'+band+'_'+solints[band][target][i]+'_'+str(i)+'_post.image.tt0',\
                      directory+'/images/'+sanitize_string(target)+'_'+band+'_'+solints[band][target][i]+'_'+str(i)+'_post.image.tt0.png', \
                      zoom=2 if directory=="weblog" else 1) 
            image_stats=imstat(sanitize_string(target)+'_'+band+'_'+solints[band][target][i]+'_'+str(i)+'_post.image.tt0')
            plot_image(sanitize_string(target)+'_'+band+'_'+solints[band][target][i]+'_'+str(i)+'.image.tt0',\
                      directory+'/images/'+sanitize_string(target)+'_'+band+'_'+solints[band][target][i]+'_'+str(i)+'.image.tt0.png',min=image_stats['min'][0],max=image_stats['max'][0], \
                      zoom=2 if directory=="weblog" else 1) 

            htmlOutSolint.writelines('<a href="images/'+sanitize_string(target)+'_'+band+'_'+solints[band][target][i]+'_'+str(i)+'.image.tt0.png"><img src="images/'+sanitize_string(target)+'_'+band+'_'+solints[band][target][i]+'_'+str(i)+'.image.tt0.png" ALT="pre-SC-solint image" WIDTH=400 HEIGHT=400></a>\n')
            htmlOutSolint.writelines('<a href="images/'+sanitize_string(target)+'_'+band+'_'+solints[band][target][i]+'_'+str(i)+'_post.image.tt0.png"><img src="images/'+sanitize_string(target)+'_'+band+'_'+solints[band][target][i]+'_'+str(i)+'_post.image.tt0.png" ALT="pre-SC-solint image" WIDTH=400 HEIGHT=400></a><br>\n')
            htmlOutSolint.writelines('Post SC SNR: {:0.3f}'.format(sclib[target][band][vislist[0]][solints[band][target][i]]['SNR_post'])+'<br>Pre SC SNR: {:0.3f}'.format(sclib[target][band][vislist[0]][solints[band][target][i]]['SNR_pre'])+'<br><br>\n')
            htmlOutSolint.writelines('Post SC RMS: {:0.7f}'.format(sclib[target][band][vislist[0]][solints[band][target][i]]['RMS_post'])+' Jy/beam<br>Pre SC RMS: {:0.7f}'.format(sclib[target][band][vislist[0]][solints[band][target][i]]['RMS_pre'])+' Jy/beam<br>\n')
            htmlOutSolint.writelines('Post Beam: {:0.3f}"x{:0.3f}" {:0.3f} deg'.format(sclib[target][band][vislist[0]][solints[band][target][i]]['Beam_major_post'],sclib[target][band][vislist[0]][solints[band][target][i]]['Beam_minor_post'],sclib[target][band][vislist[0]][solints[band][target][i]]['Beam_PA_post'])+'<br>\n')
            htmlOutSolint.writelines('Pre Beam: {:0.3f}"x{:0.3f}" {:0.3f} deg'.format(sclib[target][band][vislist[0]][solints[band][target][i]]['Beam_major_pre'],sclib[target][band][vislist[0]][solints[band][target][i]]['Beam_minor_pre'],sclib[target][band][vislist[0]][solints[band][target][i]]['Beam_PA_pre'])+'<br><br>\n')


            if solints[band][target][i] =='inf_EB':
               htmlOutSolint.writelines('<h3>Phase vs. Frequency Plots:</h3>\n')
            else:
               htmlOutSolint.writelines('<h3>Phase vs. Time Plots:</h3>\n')
            for vis in vislist:
               htmlOutSolint.writelines('<h4>MS: '+vis+'</h4>\n')
               if 'gaintable' not in sclib[target][band][vis][solints[band][target][i]]:
                    htmlOutSolint.writelines('No gaintable available <br><br>')
                    continue
               ant_list=get_ant_list(vis)
               gaintable=sclib[target][band][vis][solints[band][target][i]]['gaintable'][len(sclib[target][band][vis][solints[band][target][i]]['gaintable'])-1]
               print('******************'+gaintable+'***************')
               nflagged_sols, nsols=get_sols_flagged_solns(gaintable)
               frac_flagged_sols=nflagged_sols/nsols
               plot_ants_flagging_colored(directory+'/images/plot_ants_'+gaintable+'.png',vis,gaintable)
               htmlOutSolint.writelines('<a href="images/plot_ants_'+gaintable+'.png"><img src="images/plot_ants_'+gaintable+'.png" ALT="antenna positions with flagging plot" WIDTH=400 HEIGHT=400></a>')
               if 'unflagged_lbs' in sclib[target][band][vis][solints[band][target][i]]:
                   unflag_failed_antennas(vis, gaintable.replace('.g','.pre-pass.g'), sclib[target][band][vis][solints[band][target][i]]['gaincal_return'], flagged_fraction=0.25, \
                           spwmap=sclib[target][band][vis][solints[band][target][i]]['unflag_spwmap'], \
                           plot=True, plot_directory=directory+'/images/')
                   htmlOutSolint.writelines('\n<a href="images/'+gaintable.replace('.g.','.pre-pass.pass')+'.png"><img src="images/'+gaintable.replace('.g','.pre-pass.pass')+'.png" ALT="Long baseline unflagging thresholds" HEIGHT=400></a><br>\n')
               else:
                   htmlOutSolint.writelines('<br>\n')
               htmlOutSolint.writelines('N Gain solutions: {:0.0f}<br>'.format(nsols))
               htmlOutSolint.writelines('Flagged solutions: {:0.0f}<br>'.format(nflagged_sols))
               htmlOutSolint.writelines('Fraction Flagged Solutions: {:0.3f} <br><br>'.format(frac_flagged_sols))
               if solints[band][target][i] =='inf_EB':
                  if 'fallback' in sclib[target][band][vis][solints[band][target][i]].keys():
                     if sclib[target][band][vis][solints[band][target][i]]['fallback'] == '':
                        fallback_mode='None'
                     if sclib[target][band][vis][solints[band][target][i]]['fallback'] == 'combinespw':
                        fallback_mode='Combine SPW'
                     if sclib[target][band][vis][solints[band][target][i]]['fallback'] == 'spwmap':
                        fallback_mode='SPWMAP'
                     if sclib[target][band][vis][solints[band][target][i]]['fallback'] == 'combinespwpol':
                        fallback_mode='Combine SPW & Pol'
                     htmlOutSolint.writelines('<h4>Fallback Mode: <font color="red">'+fallback_mode+'</font></h4>\n')
               htmlOutSolint.writelines('<h4>Spwmapping: ['+' '.join(map(str,sclib[target][band][vis][solints[band][target][i]]['spwmap']))+']</h4>\n')


               for ant in ant_list:
                  sani_target=sanitize_string(target)
                  if solints[band][target][i] =='inf_EB':
                     xaxis='frequency'
                  else:
                     xaxis='time'
                  if 'ap' in solints[band][target][i]:
                     yaxis='amp'
                     plotrange=[0,0,0,2.0]
                  else:
                     yaxis='phase'
                     plotrange=[0,0,-180,180]
                  try:
                     plotms(gridrows=2,plotindex=0,rowindex=0,vis=gaintable,xaxis=xaxis, yaxis=yaxis,showgui=False,\
                         xselfscale=True,plotrange=plotrange, antenna=ant,customflaggedsymbol=True,title=ant+' phase',\
                         plotfile=directory+'/images/plot_'+ant+'_'+gaintable.replace('.g','.png'),overwrite=True, clearplots=True)
                     plotms(gridrows=2,rowindex=1,plotindex=1,vis=gaintable,xaxis=xaxis, yaxis='SNR',showgui=False,\
                         xselfscale=True, antenna=ant,customflaggedsymbol=True,title=ant+' SNR',\
                         plotfile=directory+'/images/plot_'+ant+'_'+gaintable.replace('.g','.png'),overwrite=True, clearplots=False)
                     #htmlOut.writelines('<img src="images/plot_'+ant+'_'+gaintable.replace('.g','.png')+'" ALT="gaintable antenna '+ant+'" WIDTH=200 HEIGHT=200>')
                     htmlOutSolint.writelines('<a href="images/plot_'+ant+'_'+gaintable.replace('.g','.png')+'"><img src="images/plot_'+ant+'_'+gaintable.replace('.g','.png')+'" ALT="gaintable antenna '+ant+'" WIDTH=200 HEIGHT=200></a>\n')
                  except:
                     continue
            htmlOutSolint.writelines('</body>\n')
            htmlOutSolint.writelines('</html>\n')
            htmlOutSolint.close()

def importdata(vislist,all_targets,telescope):
   spectral_scan=False
   listdict=collect_listobs_per_vis(vislist)
   scantimesdict,integrationsdict,integrationtimesdict,integrationtimes,n_spws,minspw,spwsarray_dict,spws_set=fetch_scan_times(vislist,all_targets)
   spwslist_dict = {}
   spwstring_dict = {}
   for vis in vislist:
        spwslist_dict[vis] = spwsarray_dict[vis].tolist()
        spwstring_dict[vis]=','.join(str(spw) for spw in spwslist_dict[vis])
   if spws_set[vislist[0]].ndim > 1:
      nspws_sets=spws_set[vislist[0]].shape[0]
   else:
      nspws_sets=1
   if 'VLA' in telescope:
      bands,band_properties=get_VLA_bands(vislist,all_targets)
  
   if telescope=='ALMA' or telescope =='ACA':
      bands,band_properties=get_ALMA_bands(vislist,spwstring_dict,spwsarray_dict)
      if nspws_sets > 1 and spws_set[vislist[0]].ndim >1:
         spectral_scan=True

   scantimesdict={}
   scanfieldsdict={}
   scannfieldsdict={}
   scanstartsdict={}
   scanendsdict={}
   integrationsdict={}
   integrationtimesdict
   mosaic_field_dict={}
   bands_to_remove=[]
   spws_set_dict = {}
   nspws_sets_dict = {}

   for band in bands:
        print(band)
        scantimesdict_temp,scanfieldsdict_temp,scannfieldsdict_temp,scanstartsdict_temp,scanendsdict_temp,integrationsdict_temp,integrationtimesdict_temp,\
        integrationtimes_temp,n_spws_temp,minspw_temp,spwsarray_temp,spws_set_dict_temp,mosaic_field_temp=fetch_scan_times_band_aware(vislist,all_targets,band_properties,band)

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
        if n_spws_temp == -99:
           for vis in vislist:
              band_properties[vis].pop(band)
              band_properties[vis]['bands'].remove(band)
              print('Removing '+band+' bands from list due to no observations')
           bands_to_remove.append(band)
        loopcount=0
        for vis in vislist:
           for target in all_targets:
              check_target=len(integrationsdict[band][vis][target])
              if check_target == 0:
                 integrationsdict[band][vis].pop(target)
                 integrationtimesdict[band][vis].pop(target)
                 scantimesdict[band][vis].pop(target)
                 scanfieldsdict[band][vis].pop(target)
                 scannfieldsdict[band][vis].pop(target)
                 scanstartsdict[band][vis].pop(target)
                 scanendsdict[band][vis].pop(target) 
                 #handle case of multiMS mosaic data; assumes mosaic info is the same for MSes
                 if loopcount == 0:
                    mosaic_field_dict[band][vis].pop(target)
           loopcount+=1        
   if len(bands_to_remove) > 0:
      for delband in bands_to_remove:
         bands.remove(delband)
   
   ## Load the gain calibrator information.

   gaincalibrator_dict = {}
   for vis in vislist:
       if "targets" in vis:
           vis_string = "_targets"
       else:
           vis_string = "_target"

       viskey = vis.replace(vis_string+".ms",vis_string+".selfcal.ms")

       gaincalibrator_dict[viskey] = {}
       if os.path.exists(vis.replace(vis_string+".ms",".ms").replace(vis_string+".selfcal.ms",".ms")):
           msmd.open(vis.replace(vis_string+".ms",".ms").replace(vis_string+".selfcal.ms",".ms"))
   
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


def split_to_selfcal_ms(vislist,band_properties,bands,spectral_average):
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
          mstransform(vis=vis,chanaverage=True,chanbin=chan_widths,spw=spwstring,outputvis=vis.replace('.ms','.selfcal.ms'),datacolumn='data',reindex=False)
          initweights(vis=vis,wtmode='delwtsp') # remove channelized weights
       else:
          mstransform(vis=vis,outputvis=vis.replace('.ms','.selfcal.ms'),datacolumn='data',reindex=False)


def check_mosaic(vislist,target):
   msmd.open(vis[0])
   fieldid=msmd.fieldsforname(field)
   msmd.done()
   if len(fieldid) > 1:
      mosaic=True
   else:
      mosaic=False
   return mosaic

def get_phasecenter(vis,field):
   msmd.open(vis)
   fieldid=msmd.fieldsforname(field)
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


def analyze_inf_EB_flagging(selfcal_library,band,spwlist,gaintable,vis,target,spw_combine_test_gaintable,spectral_scan,telescope, solint_snr_per_spw, minsnr_to_proceed,spwpol_combine_test_gaintable=None):
   if telescope != 'ACA':
       # if more than two antennas are fully flagged relative to the combinespw results, fallback to combinespw
       max_flagged_ants_combspw=2.0
       # if only a single (or few) spw(s) has flagging, allow at most this number of antennas to be flagged before mapping
       max_flagged_ants_spwmap=1.0
   else:
       # For the ACA, don't allow any flagging of antennas before trying fallbacks, because it is more damaging due to the smaller
       # number of antennas
       max_flagged_ants_combspw=0.0
       max_flagged_ants_spwmap=0.0

   fallback=''
   map_index=-1
   min_spwmap_bw=0.0
   spwmap=[False]*len(spwlist)
   nflags,nunflagged,fracflagged=get_flagged_solns_per_spw(spwlist,gaintable)
   nflags_spwcomb,nunflagged_spwcomb,fracflagged_spwcomb=get_flagged_solns_per_spw([spwlist[0]],spw_combine_test_gaintable)
   eff_bws=np.zeros(len(spwlist))
   total_bws=np.zeros(len(spwlist))
   keylist=list(selfcal_library[target][band][vis]['per_spw_stats'].keys())
   for i in range(len(spwlist)):
      eff_bws[i]=selfcal_library[target][band][vis]['per_spw_stats'][keylist[i]]['effective_bandwidth']
      total_bws[i]=selfcal_library[target][band][vis]['per_spw_stats'][keylist[i]]['bandwidth']
   minimum_flagged_ants_per_spw=np.min(nflags)/2.0
   minimum_flagged_ants_spwcomb=np.min(nflags_spwcomb)/2.0 # account for the fact that some antennas might be completely flagged and give 
                                                           # the impression of a lot of flagging
   maximum_flagged_ants_per_spw=np.max(nflags)/2.0
   delta_nflags=np.array(nflags)/2.0-minimum_flagged_ants_spwcomb #minimum_flagged_ants_per_spw

   # if there are more than 3 flagged antennas for all spws (minimum_flagged_ants_spwcomb, fallback to doing spw combine for inf_EB fitting
   # use the spw combine number of flagged ants to set the minimum otherwise could misinterpret fully flagged antennas for flagged solutions
   # captures case where no one spws has sufficient S/N, only together do they have enough
   if (minimum_flagged_ants_per_spw-minimum_flagged_ants_spwcomb) > max_flagged_ants_combspw:
      fallback='combinespw'
   
   #if certain spws have more than max_flagged_ants_spwmap flagged solutions that the least flagged spws, set those to spwmap
   for i in range(len(spwlist)):
      if np.min(delta_nflags[i]) > max_flagged_ants_spwmap or \
            solint_snr_per_spw[target][band]['inf_EB'][str(selfcal_library[target][band]['reverse_spw_map'][vis][int(spwlist[i])])] < minsnr_to_proceed:
         fallback='spwmap'
         spwmap[i]=True
         if total_bws[i] > min_spwmap_bw:
            min_spwmap_bw=total_bws[i]
   #also spwmap spws with similar bandwidths to the others that are getting mapped, avoid low S/N solutions
   if fallback=='spwmap':
      for i in range(len(spwlist)):
         if total_bws[i] <= min_spwmap_bw:
            spwmap[i]=True
      if all(spwmap):
         fallback='combinespw'
   #want the widest bandwidth window that also has the minimum flags to use for spw mapping
   applycal_spwmap=[]
   if fallback=='spwmap':
      minflagged_index=(np.array(nflags)/2.0 == minimum_flagged_ants_per_spw).nonzero()
      max_bw_index = (eff_bws == np.max(eff_bws[minflagged_index[0]])).nonzero()
      max_bw_min_flags_index=np.intersect1d( minflagged_index[0],max_bw_index[0])
      #if len(max_bw_min_flags_index) > 1:
      #don't need the conditional since this works with array lengths of 1
      map_index=max_bw_min_flags_index[np.argmax(eff_bws[max_bw_min_flags_index])]   
      #else:
      #   map_index=max_bw_min_flags_index[0]
      
      #make spwmap list that first maps everything to itself, need max spw to make that list
      maxspw=np.max(selfcal_library[target][band][vis]['spwsarray']+1)
      applycal_spwmap_int_list=list(np.arange(maxspw))
      for i in range(len(applycal_spwmap_int_list)):
         applycal_spwmap.append(applycal_spwmap_int_list[i])

      #replace the elements that require spwmapping (spwmap[i] == True
      for i in range(len(spwmap)):
         print(i,spwlist[i],spwmap[i])
         if spwmap[i]:
            applycal_spwmap[int(spwlist[i])]=int(spwlist[map_index])
      # always fallback to combinespw for spectral scans
      if fallback !='' and spectral_scan:
         fallback='combinespw'

   # If all of the spws map to the same spw, we might as well do a combinespw fallback.
   if len(np.unique(applycal_spwmap)) == 1:
       fallback = 'combinespw'
       applycal_spwmap = []

   if fallback == "combinespw":
      # If we end up with combinespw, check whether going to combinespw with gaintype='T' offers further improvement.
      nflags_spwcomb,nunflagged_spwcomb,fracflagged_spwcomb=get_flagged_solns_per_spw([spwlist[0]],spw_combine_test_gaintable,extendpol=True)
      nflags_spwpolcomb,nunflagged_spwpolcomb,fracflagged_spwpolcomb=get_flagged_solns_per_spw([spwlist[0]],spwpol_combine_test_gaintable)

      if np.sqrt((nunflagged_spwcomb[0]*(nunflagged_spwcomb[0]-1)) / (nunflagged_spwpolcomb[0]*(nunflagged_spwpolcomb[0]-1))) < 0.95:
          fallback='combinespwpol'

   return fallback,map_index,spwmap,applycal_spwmap



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
 
    # Get a smoothed number of antennas flagged as a function of offset.
    test_r = np.linspace(0., offsets.max(), 1000)
    neff = (nants)**(-1./(1+4))
    kernal2 = scipy.stats.gaussian_kde(offsets, bw_method=neff)

    flagged_offsets = offsets[np.any(flags, axis=(0,1))]
    if len(np.unique(flagged_offsets)) == 1:
        flagged_offsets = np.concatenate((flagged_offsets, flagged_offsets*1.05))
    elif len(flagged_offsets) == 0:
        tb.close()
        print("Not unflagging any antennas because there are no flags! The beam size probably changed because of calwt=True.")
        return
    kernel = scipy.stats.gaussian_kde(flagged_offsets,
            bw_method=kernal2.factor*offsets.std()/flagged_offsets.std())
    normalized = kernel(test_r) * len(flagged_offsets) / np.trapz(kernel(test_r), test_r)
    normalized2 = kernal2(test_r) * antennas.size / np.trapz(kernal2(test_r), test_r)
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

            # Estimated change ine the size of the beam.
            beam_change = np.percentile(offsets, 80) / np.percentile(offsets[np.logical_or(flags.any(axis=0).any(axis=0) == False, \
                    offsets > test_r[m])], 80)

            #if beam_change < 1.05:
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
