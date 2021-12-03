#future improvements
# get_sensitivity to properly weight the estimated sensitivity by the relative fraction of time on source
# heuristics for switching between calonly and calflag
# heuristics to switch from combine=spw to combine=''
# switch heirarchy of selfcal_library such that solint is at a higher level than vis. makes storage of some parameters awkward since they live
#    in the per vis level instead of per solint
# clean final image with appropriate RMS noise level derived empirically, like self-cal loops, rather than theoretical

import numpy as np
from scipy import stats
import glob
execfile('selfcal_helpers.py',globals())


###################################################################################################
######################## All code until line ~170 is just jumping through hoops ###################
######################## to get at metadata pipeline should have in the context ###################
#################### And it will do flagging of lines and/or spectral averaging ###################
######################## Some of this code is not elegant nor efficient ###########################
###################################################################################################

##
## Get list of MS files in directory
##
vislist=glob.glob('*_target.ms')

##
## save starting flags or restore to the starting flags
##
for vis in vislist:
   if os.path.exists(vis+".flagversions/flags.starting_flags"):
       flagmanager(vis=vis, mode = 'restore', versionname = 'starting_flags', comment = 'Flag states at start of reduction')
   else:
      flagmanager(vis=vis,mode='save',versionname='starting_flags')

## 
## Find targets, assumes all targets are in all ms files for simplicity and only science targets, will fail otherwise
##
all_targets=fetch_targets(vislist[0])

##
## Global environment variables for control of selfcal
##
spectral_average=False
parallel=True
gaincal_minsnr=2.0
minsnr_to_proceed=3.0
delta_beam_thresh=0.05
n_ants=get_n_ants(vislist)
telescope=get_telescope(vislist[0])
apply_cal_mode_default='calflag'


##
## Import inital MS files to get relevant meta data
##
listdict=collect_listobs_per_vis(vislist)

scantimesdict,integrationsdict,integrationtimesdict,integrationtimes,n_spws,minspw,spwsarray=fetch_scan_times(vislist,all_targets,listdict)
spwslist=spwsarray.tolist()
spwstring=','.join(str(spw) for spw in spwslist)

if 'VLA' in telescope:
  bands,band_properties=get_VLA_bands(vislist)

if telescope=='ALMA' or telescope =='ACA':
  bands,band_properties=get_ALMA_bands(vislist,spwstring,spwsarray)


scantimesdict={}
integrationsdict={}
integrationtimesdict={}
bands_to_remove=[]
for band in bands:
     print(band)
     scantimesdict_temp,integrationsdict_temp,integrationtimesdict_temp,\
     integrationtimes_temp,n_spws_temp,minspw_temp,spwsarray_temp=fetch_scan_times_band_aware(vislist,all_targets,listdict,band_properties,band)
     scantimesdict[band]=scantimesdict_temp.copy()
     integrationsdict[band]=integrationsdict_temp.copy()
     integrationtimesdict[band]=integrationtimesdict_temp.copy()
     if n_spws_temp == -99:
        for vis in vislist:
           band_properties[vis].pop(band)
           band_properties[vis]['bands'].remove(band)
           bands_to_remove.append(band)
           print('Removing '+band+' bands from list due to no observations')

if len(bands_to_remove) > 0:
   for delband in bands_to_remove:
      bands.remove(delband)

##
## flag spectral lines in MS(es) if there is a cont.dat file present
##
if os.path.exists("cont.dat"):
   print("# cont.dat file found, flagging lines identified by the pipeline.")
   for vis in vislist:
      if not os.path.exists(vis+".flagversions/flags.before_line_flags"):
         flagmanager(vis=vis, mode = 'save', versionname = 'before_line_flags', comment = 'Flag states at start of reduction')
      else:
         flagmanager(vis=vis,mode='restore',versionname='before_line_flags')
      for target in all_targets:
         contdot_dat_flagchannels_string = flagchannels_from_contdotdat(vis,target,spwsarray)
         flagdata(vis=vis, mode='manual', spw=contdot_dat_flagchannels_string[:-2], flagbackup=False, field = target)

##
## spectrally average ALMA or VLA data with telescope/frequency specific averaging properties
##
for vis in vislist:
    os.system('rm -rf '+vis.replace('.ms','.selfcal.ms')+'*')
    spwstring=''
    chan_widths=[]
    if spectral_average:
       for band in bands:
          desiredWidth=get_desired_width(band_properties[vis][band]['meanfreq'])
          print(band,desiredWidth)
          band_properties[vis][band]['chan_widths']=get_spw_chanavg(vis,get_spw_chanwidths(vis,band_properties[vis][band]['spwarray']),desiredWidth=desiredWidth)
          print(band_properties[vis][band]['chan_widths'])
          chan_widths=chan_widths+band_properties[vis][band]['chan_widths'].astype('int').tolist()
          if spwstring =='':
             spwstring=band_properties[vis][band]['spwstring']+''
          else:
             spwstring=spwstring+','+band_properties[vis][band]['spwstring']
       split(vis=vis,width=chan_widths,spw=spwstring,outputvis=vis.replace('.ms','.selfcal.ms'),datacolumn='data')
    else:
       os.system('cp -r '+vis+' '+vis.replace('.ms','.selfcal.ms'))
       if os.path.exists(vis+".flagversions/flags.before_line_flags"):
          flagmanager(vis=vis,mode='restore',versionname='before_line_flags')     


##
## Reimport MS(es) to self calibrate since frequency averaging and splitting may have changed it
##
vislist=glob.glob('*selfcal.ms')
listdict=collect_listobs_per_vis(vislist)
scantimesdict,integrationsdict,integrationtimesdict,integrationtimes,n_spws,minspw,spwsarray=fetch_scan_times(vislist,all_targets,listdict)
spwslist=spwsarray.tolist()
spwstring=','.join(str(spw) for spw in spwslist)

if 'VLA' in telescope:
  bands,band_properties=get_VLA_bands(vislist)

if telescope=='ALMA' or telescope =='ACA':
  bands,band_properties=get_ALMA_bands(vislist,spwstring,spwsarray)

scantimesdict={}
integrationsdict={}
integrationtimesdict={}
for band in bands:
     print(band)
     scantimesdict_temp,integrationsdict_temp,integrationtimesdict_temp,\
     integrationtimes_temp,n_spws_temp,minspw_temp,spwsarray_temp=fetch_scan_times_band_aware(vislist,all_targets,listdict,band_properties,band)

     scantimesdict[band]=scantimesdict_temp.copy()
     integrationsdict[band]=integrationsdict_temp.copy()
     integrationtimesdict[band]=integrationtimesdict_temp.copy()
     if n_spws_temp == -99:
        for vis in vislist:
           band_properties[vis].pop(band)
           band_properties[vis]['bands'].remove(band)
           print('Removing '+band+' bands from list due to no observations')

if len(bands_to_remove) > 0:
   for delband in bands_to_remove:
      bands.remove(delband)

##
## set image parameters based on the visibility data properties and frequency
##
cellsize={}
imsize={}
nterms={}
applycal_interp={}

for band in bands:
   cellsize[band],imsize[band],nterms[band]=get_image_parameters(vislist,telescope,band,band_properties)
   if band_properties[vislist[0]][band]['meanfreq'] >12.0e9:
      applycal_interp[band]='linearPD'
   else:
      applycal_interp[band]='linear'




###################################################################################################
################################# End Metadata gathering for Selfcal ##############################
###################################################################################################



###################################################################################################
############################# Start Actual important stuff for selfcal ############################
###################################################################################################

##
## begin setting up a selfcal_library with all relevant metadata to keep track of during selfcal
## 
selfcal_library={}

for target in all_targets:
   selfcal_library[target]={}
   for band in bands:
      selfcal_library[target][band]={}
      for vis in vislist:
         selfcal_library[target][band][vis]={}
##
## finds solints, starting with inf, ending with int, and tries to align
## solints with number of integrations
## solints reduce by factor of 2 in each self-cal interation
## e.g., inf, max_scan_time/2.0, prev_solint/2.0, ..., int
## starting solints will have solint the length of the entire EB to correct bulk offsets
##
solints={}
gaincal_combine={}
applycal_mode={}
for band in bands:
   solints[band],integration_time=get_solints_simple(vislist,scantimesdict[band],integrationtimesdict[band])
   print(band,solints[band])
   gaincal_combine[band]=['spw']*len(solints[band])
   solints[band].insert(0,'inf_EB')
   gaincal_combine[band].insert(0,'spw,scan')
   applycal_mode[band]=[apply_cal_mode_default]*len(solints[band])



##
## puts stuff in right place from other MS metadata to perform proper data selections
## in tclean, gaincal, and applycal
## Also gets relevant times on source to estimate SNR per EB/scan
##
for target in all_targets:
 for band in bands:
   print(target, band)
   selfcal_library[target][band]['SC_success']=False
   selfcal_library[target][band]['final_solint']='None'
   selfcal_library[target][band]['Total_TOS']=0.0
   selfcal_library[target][band]['spws']=[]
   selfcal_library[target][band]['spws_per_vis']=[]
   selfcal_library[target][band]['vislist']=vislist.copy()
   allscantimes=np.array([])
   for vis in vislist:
      selfcal_library[target][band][vis]['gaintable']=[]
      selfcal_library[target][band][vis]['TOS']=np.sum(scantimesdict[band][vis][target])
      selfcal_library[target][band][vis]['Median_scan_time']=np.median(scantimesdict[band][vis][target])
      allscantimes=np.append(allscantimes,scantimesdict[band][vis][target])
      selfcal_library[target][band][vis]['flags']=[]
      selfcal_library[target][band][vis]['refant'] = rank_refants(vis)
      n_spws,minspw,spwsarray=fetch_spws([vis],[target],listdict)
      spwslist=spwsarray.tolist()
      spwstring=','.join(str(spw) for spw in spwslist)
      selfcal_library[target][band][vis]['spws']=band_properties[vis][band]['spwstring']
      selfcal_library[target][band][vis]['spwsarray']=band_properties[vis][band]['spwarray']
      selfcal_library[target][band][vis]['spwlist']=band_properties[vis][band]['spwarray'].tolist()
      selfcal_library[target][band][vis]['n_spws']=len(selfcal_library[target][band][vis]['spwsarray'])
      selfcal_library[target][band][vis]['minspw']=int(np.min(selfcal_library[target][band][vis]['spwsarray']))
      selfcal_library[target][band][vis]['spwmap']=[selfcal_library[target][band][vis]['minspw']]*(np.max(selfcal_library[target][band][vis]['spwsarray'])+1)
      selfcal_library[target][band]['Total_TOS']=selfcal_library[target][band][vis]['TOS']+selfcal_library[target][band]['Total_TOS']
      selfcal_library[target][band]['spws_per_vis'].append(band_properties[vis][band]['spwstring'])
   selfcal_library[target][band]['Median_scan_time']=np.median(allscantimes)
   selfcal_library[target][band]['uvrange']=get_uv_range(band,band_properties,vislist)
   print(selfcal_library[target][band]['uvrange'])

##
## 
## 
for target in all_targets:
   for band in bands:
      if selfcal_library[target][band]['Total_TOS'] == 0.0:
         selfcal_library[target].pop(band)


##
## create initial images for each target to evaluate SNR and beam
## replicates what a preceding hif_makeimages would do
## Enables before/after comparison and thresholds to be calculated
## based on the achieved S/N in the real data
##
for target in all_targets:
 sani_target=sanitize_string(target)
 for band in selfcal_library[target].keys():
   #make images using the appropriate tclean heuristics for each telescope
   if not os.path.exists(sani_target+'_'+band+'_dirty.image.tt0'):
      tclean_wrapper(vis=vislist, imagename=sani_target+'_'+band+'_dirty',
                     telescope=telescope,nsigma=3.0, scales=[0],
                     threshold='0.0Jy',niter=0,
                     savemodel='none',parallel=parallel,cellsize=cellsize[band],imsize=imsize[band],nterms=nterms[band],
                     field=target,spw=selfcal_library[target][band]['spws_per_vis'],uvrange=selfcal_library[target][band]['uvrange'])
   dirty_SNR,dirty_RMS=estimate_SNR(sani_target+'_'+band+'_dirty.image.tt0')
   dr_mod=1.0
   if telescope =='ALMA' or telescope =='ACA':
      sensitivity=get_sensitivity(vislist,selfcal_library[target][band][vis]['spws'],spw=selfcal_library[target][band][vis]['spwsarray'],imsize=imsize[band],cellsize=cellsize[band])
      dr_mod=get_dr_correction(telescope,dirty_SNR*dirty_RMS,sensitivity,vislist)
      print('DR modifier: ',dr_mod)
   if not os.path.exists(sani_target+'_'+band+'_initial.image.tt0'):
      if telescope=='ALMA' or telescope =='ACA':
         sensitivity=sensitivity*dr_mod   # apply DR modifier
         if band =='Band_9' or band == 'Band_10':   # adjust for DSB noise increase
            sensitivity=sensitivity   #*4.0  might be unnecessary with DR mods
      else:
         sensitivity=0.0
      tclean_wrapper(vis=vislist, imagename=sani_target+'_'+band+'_initial',
                     telescope=telescope,nsigma=3.0, scales=[0],
                     threshold=str(sensitivity*3.0)+'Jy',
                     savemodel='none',parallel=parallel,cellsize=cellsize[band],imsize=imsize[band],nterms=nterms[band],
                     field=target,spw=selfcal_library[target][band]['spws_per_vis'],uvrange=selfcal_library[target][band]['uvrange'])
   initial_SNR,initial_RMS=estimate_SNR(sani_target+'_'+band+'_initial.image.tt0')
   header=imhead(imagename=sani_target+'_'+band+'_initial.image.tt0')
   selfcal_library[target][band]['SNR_orig']=initial_SNR
   selfcal_library[target][band]['RMS_orig']=initial_RMS
   selfcal_library[target][band]['RMS_curr']=initial_RMS
   selfcal_library[target][band]['SNR_dirty']=dirty_SNR
   selfcal_library[target][band]['RMS_dirty']=dirty_RMS
   selfcal_library[target][band]['Beam_major_orig']=header['restoringbeam']['major']['value']
   selfcal_library[target][band]['Beam_minor_orig']=header['restoringbeam']['minor']['value']
   selfcal_library[target][band]['Beam_PA_orig']=header['restoringbeam']['positionangle']['value'] 

##
## estimate per scan/EB S/N using time on source and median scan times
##
solint_snr=get_SNR_self(all_targets,bands,vislist,selfcal_library,n_ants,solints,integration_time)
for target in all_targets:
 for band in solint_snr[target].keys():
   print('Estimated SNR per solint:')
   print(target,band)
   for solint in solints[band]:
     print('{}: {:0.2f}'.format(solint,solint_snr[target][band][solint]))

##
## Set clean selfcal thresholds
### Open question about determining the starting and progression of clean threshold for
### each iteration
### Peak S/N > 100; SNR/15 for first, successivly reduce to 3.0 sigma through each iteration?
### Peak S/N < 100; SNR/10.0 
##
## Switch to a sensitivity for low frequency that is based on the residuals of the initial image for the
# first couple rounds and then switch to straight nsigma? Determine based on fraction of pixels that the # initial mask covers to judge very extended sources?

for target in all_targets:
  for band in selfcal_library[target].keys():
   dividing_factor=15.0
   if band_properties[selfcal_library[target][band]['vislist'][0]][band]['meanfreq'] <8.0e9:
      dividing_factor=40.0
   else:
      dividing_factor=15.0
   nsigma_init=np.max([selfcal_library[target][band]['SNR_orig']/dividing_factor,5.0]) # restricts initial nsigma to be at least 5
   #selfcal_library[target]['nsigma']=np.linspace(nsigma_init,3.0,len(solints))
   #logspace to reduce in nsigma more quickly
   selfcal_library[target][band]['nsigma']=10**np.linspace(np.log10(nsigma_init),np.log10(3.0),len(solints[band]))
   if telescope=='ALMA' or telescope =='ACA': #or ('VLA' in telescope) 
      sensitivity=get_sensitivity(vislist,selfcal_library[target][band][vis]['spws'],spw=selfcal_library[target][band][vis]['spwsarray'],imsize=imsize[band],cellsize=cellsize[band])
      if band =='Band_9' or band == 'Band_10':   # adjust for DSB noise increase
         sensitivity=sensitivity*4.0 
      if ('VLA' in telescope):
         sensitivity=sensitivity*0.0 # empirical correction, VLA estimates for sensitivity have tended to be a factor of ~3 low
   else:
      sensitivity=0.0
   selfcal_library[target][band]['thresholds']=selfcal_library[target][band]['nsigma']*sensitivity

##
## Save self-cal library
##
import pickle
with open('selfcal_library.pickle', 'wb') as handle:
    pickle.dump(selfcal_library, handle, protocol=pickle.HIGHEST_PROTOCOL)



##
## Begin Self-cal loops
##
for target in all_targets:
 sani_target=sanitize_string(target)
 for band in selfcal_library[target].keys():
   vislist=selfcal_library[target][band]['vislist'].copy()
   print('Starting selfcal procedure on: '+target+' '+band)
   for iteration in range(len(solints[band])):
      if solint_snr[target][band][solints[band][iteration]] < minsnr_to_proceed:
         print('*********** estimated solint '+solint_snr[target][band][solints[band][iteration]]+' SNR too low, measured: '+str(solint_snr[target][band][solints[band][iteration]])+', Min SNR Required: '+str(minsnr_to_proceed)+' **************')
         break
      else:
         solint=solints[band][iteration]
         if iteration == 0:
            print('Starting with solint: '+solint)
         else:
            print('Continuing with solint: '+solint)
         os.system('rm -rf '+target+'_'+band+'_'+solint+'_'+str(iteration)+'*')
         ##
         ## make images using the appropriate tclean heuristics for each telescope
         ## set threshold based on RMS of initial image and lower if value becomes lower
         ## during selfcal by resetting 'RMS_curr' after the post-applycal evaluation
         ##
         tclean_wrapper(vis=vislist, imagename=sani_target+'_'+band+'_'+solint+'_'+str(iteration),
                     telescope=telescope,nsigma=selfcal_library[target][band]['nsigma'][iteration], scales=[0],
                     threshold=str(selfcal_library[target][band]['nsigma'][iteration]*selfcal_library[target][band]['RMS_curr'])+'Jy',
                     savemodel='modelcolumn',parallel=parallel,cellsize=cellsize[band],imsize=imsize[band],nterms=nterms[band],
                     field=target,spw=selfcal_library[target][band]['spws_per_vis'],uvrange=selfcal_library[target][band]['uvrange'])
         print('Pre selfcal assessemnt: '+target)
         SNR,RMS=estimate_SNR(sani_target+'_'+band+'_'+solint+'_'+str(iteration)+'.image.tt0')
         header=imhead(imagename=sani_target+'_'+band+'_'+solint+'_'+str(iteration)+'.image.tt0')

         if iteration == 0:
            gaintables={}
            spwmaps={}
         for vis in vislist:
            gaintables[vis]=[]
            spwmaps[vis]=[]
            ##
            ## Solve gain solutions per MS, target, solint, and band
            ##
            os.system('rm -rf '+target+'_'+vis+'_'+band+'_'+solint+'_'+str(iteration)+'.g')
            gaincal(vis=vis,\
                    caltable=target+'_'+vis+'_'+band+'_'+solint+'_'+str(iteration)+'.g',\
                    gaintype='T', spw=selfcal_library[target][band][vis]['spws'],refant=selfcal_library[target][band][vis]['refant'], calmode='p', solint=solint.replace('_EB',''),\
                    minsnr=gaincal_minsnr, minblperant=4,combine=gaincal_combine[band][iteration],field=target,gaintable='',spwmap='',uvrange=selfcal_library[target][band]['uvrange']) 
                    # for simplicity don't do incremental solutions,gaintable=gaintables[vis],spwmap=spwmaps[vis])
         for vis in vislist:
            if os.path.exists(vis+".flagversions/flags.selfcal_"+target+'_'+band+'_'+solint):
               flagmanager(vis=vis,mode='delete',versionname='selfcal_'+target+'_'+band+'_'+solint)
            flagmanager(vis=vis,mode='save',versionname='selfcal_'+target+'_'+band+'_'+solint)
            ##
            ## Apply gain solutions per MS, target, solint, and band
            ##
            applycal(vis=vis,\
                     gaintable=gaintables[vis]+[target+'_'+vis+'_'+band+'_'+solint+'_'+str(iteration)+'.g'],\
                     interp=applycal_interp[band], calwt=True,spwmap=[selfcal_library[target][band][vis]['spwmap']],\
                     applymode=applycal_mode[band][iteration],field=target,spw=selfcal_library[target][band][vis]['spws'])
         for vis in vislist:
            ##
            ## record self cal results/details for this solint
            ##
            selfcal_library[target][band][vis][solint]={}
            selfcal_library[target][band][vis][solint]['SNR_pre']=SNR.copy()
            selfcal_library[target][band][vis][solint]['RMS_pre']=RMS.copy()
            selfcal_library[target][band][vis][solint]['Beam_major_pre']=header['restoringbeam']['major']['value']
            selfcal_library[target][band][vis][solint]['Beam_minor_pre']=header['restoringbeam']['minor']['value']
            selfcal_library[target][band][vis][solint]['Beam_PA_pre']=header['restoringbeam']['positionangle']['value'] 
            selfcal_library[target][band][vis][solint]['gaintable']=target+'_'+vis+'_'+band+'_'+solint+'_'+str(iteration)+'.g'
            selfcal_library[target][band][vis][solint]['iteration']=iteration+0
            selfcal_library[target][band][vis][solint]['flags']='selfcal_'+target+'_'+band+'_'+solint
            selfcal_library[target][band][vis][solint]['spwmap']=selfcal_library[target][band][vis]['spwmap']
            selfcal_library[target][band][vis][solint]['applycal_mode']=applycal_mode[band][iteration]+''
            selfcal_library[target][band][vis][solint]['gaincal_combine']=gaincal_combine[band][iteration]+''
         ##
         ## Create post self-cal image using the model as a startmodel to evaluate how much selfcal helped
         ##
         if nterms[band]==1:
            startmodel=[sani_target+'_'+band+'_'+solint+'_'+str(iteration)+'.model.tt0']
         elif nterms[band]==2:
            startmodel=[sani_target+'_'+band+'_'+solint+'_'+str(iteration)+'.model.tt0',sani_target+'_'+band+'_'+solint+'_'+str(iteration)+'.model.tt1']
         tclean_wrapper(vis=vislist, imagename=sani_target+'_'+band+'_'+solint+'_'+str(iteration)+'_post',
                  telescope=telescope,scales=[0], nsigma=0.0,\
                  savemodel='none',parallel=parallel,cellsize=cellsize[band],imsize=imsize[band],nterms=nterms[band],\
                  niter=0,startmodel=startmodel,field=target,spw=selfcal_library[target][band]['spws_per_vis'],uvrange=selfcal_library[target][band]['uvrange'])
         print('Post selfcal assessemnt: '+target)
         #copy mask for use in post-selfcal SNR measurement
         os.system('cp -r '+sani_target+'_'+band+'_'+solint+'_'+str(iteration)+'.mask '+sani_target+'_'+band+'_'+solint+'_'+str(iteration)+'_post.mask')
         post_SNR,post_RMS=estimate_SNR(sani_target+'_'+band+'_'+solint+'_'+str(iteration)+'_post.image.tt0')
         for vis in vislist:
            selfcal_library[target][band][vis][solint]['SNR_post']=post_SNR.copy()
            selfcal_library[target][band][vis][solint]['RMS_post']=post_RMS.copy()
            ## Update RMS value if necessary
            if selfcal_library[target][band][vis][solint]['RMS_post'] < selfcal_library[target][band]['RMS_curr']:
               selfcal_library[target][band]['RMS_curr']=selfcal_library[target][band][vis][solint]['RMS_post'].copy()
            header=imhead(imagename=sani_target+'_'+band+'_'+solint+'_'+str(iteration)+'_post.image.tt0')
            selfcal_library[target][band][vis][solint]['Beam_major_post']=header['restoringbeam']['major']['value']
            selfcal_library[target][band][vis][solint]['Beam_minor_post']=header['restoringbeam']['minor']['value']
            selfcal_library[target][band][vis][solint]['Beam_PA_post']=header['restoringbeam']['positionangle']['value'] 


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
         if ((post_SNR >= SNR) and (delta_beamarea < delta_beam_thresh)) or ((solint =='inf_EB') and ((post_SNR-SNR)/SNR > -0.02)): 
            selfcal_library[target][band]['SC_success']=True
            for vis in vislist:
               selfcal_library[target][band][vis]['gaintable']=selfcal_library[target][band][vis][solint]['gaintable']
               selfcal_library[target][band][vis]['spwmap']=selfcal_library[target][band][vis][solint]['spwmap'].copy()
               selfcal_library[target][band][vis]['flags']=selfcal_library[target][band][vis][solint]['flags']
               selfcal_library[target][band][vis]['applycal_mode']=selfcal_library[target][band][vis][solint]['applycal_mode']
               selfcal_library[target][band][vis]['gaincal_combine']=selfcal_library[target][band][vis][solint]['applycal_mode']
               selfcal_library[target][band][vis][solint]['Pass']=True

            selfcal_library[target][band]['final_solint']=solint
            selfcal_library[target][band]['iteration']=iteration
            if (iteration < len(solints[band])-1) and (selfcal_library[target][band][vis][solint]['SNR_post'] > selfcal_library[target][band]['SNR_orig']): #(iteration == 0) and 
               print('Updating solint = '+solints[band][iteration+1]+' SNR')
               print('Was: ',solint_snr[target][band][solints[band][iteration+1]])
               get_SNR_self_update([target],band,vislist,selfcal_library,n_ants,solint,solints[band][iteration+1],integration_time,solint_snr)
               print('Now: ',solint_snr[target][band][solints[band][iteration+1]])
               
            if iteration < (len(solints[band])-1):
               print('****************Selfcal passed, shortening solint*************')
            else:
               print('****************Selfcal passed for Minimum solint*************')
         ## 
         ## if S/N worsens, and/or beam area increases reject current solutions and reapply previous (or revert to origional data)
         ##  
         else: 
            selfcal_library[target][band][vis][solint]['Pass']=False
            reason=''
            if (post_SNR <= SNR):
               reason=reason+' S/N decrease;'
            if (delta_beamarea > delta_beam_thresh):
               reason=reason+' Beam change beyond '+str(delta_beam_thresh)
            print('****************Selfcal failed*************')
            print('REASON: '+reason)
            if iteration > 0: # reapply only the previous gain tables, to get rid of solutions from this selfcal round
               print('****************Reapplying previous solint solutions*************')
               for vis in vislist:
                  print('****************Applying '+selfcal_library[target][band][vis]['gaintable']+' to '+target+' '+band+'*************')
                  flagmanager(vis=vis,mode='restore',versionname=selfcal_library[target][band][vis]['flags'])
                  applycal(vis=vis,\
                          gaintable=selfcal_library[target][band][vis]['gaintable'],\
                          interp=applycal_interp[band],\
                          calwt=True,spwmap=[selfcal_library[target][band][vis]['spwmap']],\
                          applymode=applycal_mode[band][selfcal_library[target][band]['iteration']],field=target,spw=selfcal_library[target][band][vis]['spws'])    
            else:            
               print('****************Removing all calibrations for '+target+' '+band+'**************')
               for vis in vislist:
                  flagmanager(vis=vis,mode='restore',versionname=selfcal_library[target][band][vis][solints[band][0]]['flags'])
                  clearcal(vis=vis,field=target,spw=selfcal_library[target][band][vis]['spws'])
                  selfcal_library[target][band]['SNR_post']=selfcal_library[target][band]['SNR_orig'].copy()
                  selfcal_library[target][band]['RMS_post']=selfcal_library[target][band]['RMS_orig'].copy()
            print('****************Aborting further self-calibration attempts for '+target+' '+band+'**************')
            break # breakout of loops of successive solints since solutions are getting worse


##
## If we want to try amplitude selfcal, should we do it as a function out of the main loop or a separate loop?
## Mechanics are likely to be a bit more simple since I expect we'd only try a single solint=inf solution
##

##
## Make a final image per target to assess overall improvement
##
for target in all_targets:
 for band in selfcal_library[target].keys():
   vislist=selfcal_library[target][band]['vislist'].copy()
   ## omit DR modifiers here since we should have increased DR significantly
   if telescope=='ALMA' or telescope =='ACA':
      sensitivity=get_sensitivity(vislist,selfcal_library[target][band][vis]['spws'],spw=selfcal_library[target][band][vis]['spwsarray'],imsize=imsize[band],cellsize=cellsize[band])
      dr_mod=1.0
      if not selfcal_library[target][band]['SC_success']: # fetch the DR modifier if selfcal failed on source
         dr_mod=get_dr_correction(telescope,selfcal_library[target][band]['SNR_dirty']*selfcal_library[target][band]['RMS_dirty'],sensitivity,vislist)
         print('DR modifier: ',dr_mod)
         sensitivity=sensitivity*dr_mod 
      if ((band =='Band_9') or (band == 'Band_10')) and dr_mod != 1.0:   # adjust for DSB noise increase
         sensitivity=sensitivity*4.0 
   else:
      sensitivity=0.0
   tclean_wrapper(vis=vislist,imagename=sani_target+'_'+band+'_final',\
               telescope=telescope,nsigma=3.0, threshold=str(sensitivity*3.0)+'Jy',scales=[0],\
               savemodel='none',parallel=parallel,cellsize=cellsize[band],imsize=imsize[band],
               nterms=nterms[band],field=target,datacolumn='corrected',spw=selfcal_library[target][band]['spws_per_vis'],uvrange=selfcal_library[target][band]['uvrange'])
   final_SNR,final_RMS=estimate_SNR(sani_target+'_'+band+'_final.image.tt0')
   selfcal_library[target][band]['SNR_final']=final_SNR
   selfcal_library[target][band]['RMS_final']=final_RMS


##
## Print final results
##
for target in all_targets:
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

##
## Save final library results
##
import pickle
with open('selfcal_library.pickle', 'wb') as handle:
    pickle.dump(selfcal_library, handle, protocol=pickle.HIGHEST_PROTOCOL)


