#future improvements
# get_sensitivity to properly weight the estimated sensitivity by the relative fraction of time on source
# heuristics for switching between calonly and calflag
# heuristics to switch from combine=spw to combine=''
# telescope to argument to tclean wrapper and set defaults inside

import numpy as np
from scipy import stats
import glob
execfile('selfcal_helpers.py',globals())
vislist=glob.glob('*_target.ms')

for vis in vislist:
   if os.path.exists(vis+".flagversions/flags.starting_flags"):
       flagmanager(vis=vis, mode = 'restore', versionname = 'starting_flags', comment = 'Flag states at start of reduction')
   else:
      flagmanager(vis=vis,mode='save',versionname='starting_flags')
#assume all targets are in all ms files for simplicity and only science targets
all_targets=fetch_targets(vislist[0])

spectral_average=False
parallel=True
# C7 and above
longbaseline=False
gaincal_minsnr=2.0
minsnr_to_proceed=3.0
delta_beam_thresh=0.05
n_ants=get_n_ants(vislist)
visheader=vishead(vislist[0],mode='list',listitems=[])
telescope=visheader['telescope'][0][0]
apply_cal_mode_default='calflag'

#Find scan times, integration times and numbers of integrations
listdict,scantimesdict,integrationsdict,integrationtimesdict,integrationtimes,n_spws,minspw,spwsarray=fetch_scan_times(vislist,all_targets)
spwslist=spwsarray.tolist()
spwstring=','.join(str(spw) for spw in spwslist)

#flag based on the cont.dat
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


if 'VLA' in telescope:
  bands,band_properties=get_VLA_bands(vislist)
  listdict={}
  scantimesdict={}
  integrationsdict={}
  integrationtimesdict={}
  for band in bands:
     listdict_temp,scantimesdict_temp,integrationsdict_temp,integrationtimesdict_temp,\
     integrationtimes_temp,n_spws_temp,minspw_temp,spwsarray_temp=fetch_scan_times_band_aware(vislist,all_targets,band_properties,band)
     listdict[band]=listdict_temp.copy
     scantimesdict[band]=scantimesdict_temp.copy()
     integrationsdict[band]=integrationsdict_temp.copy()
     integrationtimesdict[band]=integrationtimesdict_temp.copy()
if telescope=='ALMA':
  meanfreq=get_mean_freq(vislist,spwsarray)
  bands,band_properties=get_ALMA_bands(vislist,meanfreq,spwstring,spwsarray)
  listdict={}
  scantimesdict={}
  integrationsdict={}
  integrationtimesdict={}
  for band in bands:
     listdict_temp,scantimesdict_temp,integrationsdict_temp,integrationtimesdict_temp,\
     integrationtimes_temp,n_spws_temp,minspw_temp,spwsarray_temp=fetch_scan_times_band_aware(vislist,all_targets,band_properties,band)
     listdict[band]=listdict_temp.copy
     scantimesdict[band]=scantimesdict_temp.copy()
     integrationsdict[band]=integrationsdict_temp.copy()
     integrationtimesdict[band]=integrationtimesdict_temp.copy()



#spectrally average all to have a minimum channel width of 15.625 MHz and restore flags to original MS
for vis in vislist:
    os.system('rm -rf '+vis.replace('.ms','.selfcal.ms')+'*')
    spwstring=''
    chan_widths=[]
    if spectral_average:
       for band in band_properties[vis]['bands']:
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

vislist=glob.glob('*selfcal.ms')
listdict,scantimesdict,integrationsdict,integrationtimesdict,integrationtimes,n_spws,minspw,spwsarray=fetch_scan_times(vislist,all_targets)
spwslist=spwsarray.tolist()
spwstring=','.join(str(spw) for spw in spwslist)

if 'VLA' in telescope:
  bands,band_properties=get_VLA_bands(vislist)
  listdict={}
  scantimesdict={}
  integrationsdict={}
  integrationtimesdict={}
  for band in bands:
     listdict_temp,scantimesdict_temp,integrationsdict_temp,integrationtimesdict_temp,\
     integrationtimes_temp,n_spws_temp,minspw_temp,spwsarray_temp=fetch_scan_times_band_aware(vislist,all_targets,band_properties,band)
     listdict[band]=listdict_temp.copy
     scantimesdict[band]=scantimesdict_temp.copy()
     integrationsdict[band]=integrationsdict_temp.copy()
     integrationtimesdict[band]=integrationtimesdict_temp.copy()
if telescope=='ALMA':
  bands,band_properties=get_ALMA_bands(vislist,meanfreq,spwstring,spwsarray)
  listdict={}
  scantimesdict={}
  integrationsdict={}
  integrationtimesdict={}
  for band in bands:
     listdict_temp,scantimesdict_temp,integrationsdict_temp,integrationtimesdict_temp,\
     integrationtimes_temp,n_spws_temp,minspw_temp,spwsarray_temp=fetch_scan_times_band_aware(vislist,all_targets,band_properties,band)
     listdict[band]=listdict_temp.copy
     scantimesdict[band]=scantimesdict_temp.copy()
     integrationsdict[band]=integrationsdict_temp.copy()
     integrationtimesdict[band]=integrationtimesdict_temp.copy()

#set image parameters based on data properties
cellsize={}
imsize={}
nterms={}
meanfreq={}
applycal_interp={}

for band in bands:
   cellsize[band],imsize[band],nterms[band],meanfreq[band]=get_image_parameters(vislist,telescope,spwsarray)
   if meanfreq[band] >12.0e9:
      applycal_interp[band]='linearPD'
   else:
      applycal_interp[band]='linear'

'''
#Redundant if splitting to new MS
for vis in vislist:
   if not os.path.exists(vis+".flagversions/flags.starting_flags"):
       flagmanager(vis=vis, mode = 'save', versionname = 'starting_flags', comment = 'Flag states at start of reduction')
   else:
      clearcal(vis=vis)
      flagmanager(vis=vis,mode='restore',versionname='starting_flags')
'''


  
selfcal_library={}

for target in all_targets:
   selfcal_library[target]={}
   for band in bands:
      selfcal_library[target][band]={}
      for vis in vislist:
         selfcal_library[target][band][vis]={}

#finds solints, starting with inf, ending with int, and tries to align
#solints with number of integrations
#solints reduce by factor of 2 in each self-cal interation
#e.g., inf, max_scan_time/2.0, prev_solint/2.0, ..., int
solints={}
gaincal_combine={}
applycal_mode={}
for band in bands:
   solints[band]=get_solints_simple(vislist,scantimesdict[band],integrationtimesdict[band])
   print(band,solints[band])
   gaincal_combine[band]=['spw']*len(solints[band])
   solints[band].insert(0,'inf_EB')
   gaincal_combine[band].insert(0,'spw,scan')
   applycal_mode[band]=[apply_cal_mode_default]*len(solints[band])
#insert another solint, but with combine='scan,spw' for long baseline data to align average phases of each EB


#don't do calflag first two iterations to avoid flagging too much data initially (tends to be the long baselines)
'''
if longbaseline:
   applycal_mode=['calonly']*2+['calflag']*(len(solints)-2) 
else:
   applycal_mode=['calflag']*len(solints)
'''

for target in all_targets:
 for band in bands:
   print(target)
   selfcal_library[target][band]['SC_success']=False
   selfcal_library[target][band]['final_solint']='None'
   selfcal_library[target][band]['Total_TOS']=0.0
   selfcal_library[target][band]['spws']=[]
   allscantimes=np.array([])
   for vis in vislist:
      selfcal_library[target][band][vis]['gaintable']=[]
      selfcal_library[target][band][vis]['TOS']=np.sum(scantimesdict[band][vis][target])
      selfcal_library[target][band][vis]['Median_scan_time']=np.median(scantimesdict[band][vis][target])
      allscantimes=np.append(allscantimes,scantimesdict[band][vis][target])
      selfcal_library[target][band][vis]['flags']=[]
      selfcal_library[target][band][vis]['refant'] = rank_refants(vis)
      dm0,n_spws,minspw,spwsarray=fetch_spws([vis],[target])
      spwslist=spwsarray.tolist()
      spwstring=','.join(str(spw) for spw in spwslist)
      selfcal_library[target][band][vis]['spws']=band_properties[vis][band]['spwstring']
      selfcal_library[target][band][vis]['spwsarray']=band_properties[vis][band]['spwarray']
      selfcal_library[target][band][vis]['spwlist']=band_properties[vis][band]['spwarray'].tolist()
      selfcal_library[target][band][vis]['n_spws']=len(selfcal_library[target][band][vis]['spwsarray'])
      selfcal_library[target][band][vis]['minspw']=int(np.min(selfcal_library[target][band][vis]['spwsarray']))
      selfcal_library[target][band][vis]['spwmap']=[selfcal_library[target][band][vis]['minspw']]*(np.max(selfcal_library[target][band][vis]['spwsarray'])+1)
      selfcal_library[target][band]['Total_TOS']=selfcal_library[target][band][vis]['TOS']+selfcal_library[target][band]['Total_TOS']
      selfcal_library[target][band]['spws_per_vis']=selfcal_library[target][band]['spws'].append([band_properties[vis][band]['spwstring']])
   selfcal_library[target][band]['Median_scan_time']=np.median(allscantimes)



#create initial images for each target to evaluate SNR and beam
#replicates what a preceding hif_makeimages would do
for target in all_targets:
 for band in bands:
   #make images using the appropriate tclean heuristics for each telescope
   if not os.path.exists(target+'_'+band+'_initial.image.tt0'):
      if telescope=='ALMA':
         sensitivity=get_sensitivity(vislist,selfcal_library[target][band][vis]['spws'],spw=selfcal_library[target][band][vis]['spwsarray'],imsize=imsize[band],cellsize=cellsize[band])
      else:
         sensitivity=0.0
      tclean_wrapper(vis=vislist, imagename=target+'_'+band+'_initial',
                     telescope=telescope,nsigma=3.0, scales=[0],
                     threshold=str(sensitivity*3.0)+'Jy',
                     savemodel='none',parallel=parallel,cellsize=cellsize[band],imsize=imsize[band],nterms=nterms[band],
                     field=target,spw=selfcal_library[target][band][vis]['spws'])
   initial_SNR,initial_RMS=estimate_SNR(target+'_'+band+'_initial.image.tt0')
   header=imhead(imagename=target+'_'+band+'_initial.image.tt0')
   selfcal_library[target][band]['SNR_orig']=initial_SNR
   selfcal_library[target][band]['RMS_orig']=initial_RMS
   selfcal_library[target][band]['Beam_major_orig']=header['restoringbeam']['major']['value']
   selfcal_library[target][band]['Beam_minor_orig']=header['restoringbeam']['minor']['value']
   selfcal_library[target][band]['Beam_PA_orig']=header['restoringbeam']['positionangle']['value'] 

   #estimate per scan S/N using total time on source divided by average scan time
get_SNR_self(all_targets,bands,vislist,selfcal_library,n_ants)
for band in bands:
 for target in all_targets:
   print(target,band)
   print('Per Scan SNR: ',selfcal_library[target][band]['per_scan_SNR'])
   print('Per EB SNR: ',selfcal_library[target][band]['per_EB_SNR'])
   nsigma_init=np.max([selfcal_library[target][band]['SNR_orig']/15.0,5.0]) # restricts initial nsigma to be at least 5
   #selfcal_library[target]['nsigma']=np.linspace(nsigma_init,3.0,len(solints))
   #logspace to reduce in nsigma more quickly
   selfcal_library[target][band]['nsigma']=10**np.linspace(np.log10(nsigma_init),np.log10(3.0),len(solints[band]))
   if telescope=='ALMA':
      sensitivity=get_sensitivity(vislist,selfcal_library[target][band][vis]['spws'],spw=selfcal_library[target][band][vis]['spwsarray'],imsize=imsize[band],cellsize=cellsize[band])
   else:
      sensitivity=0.0
   selfcal_library[target][band]['thresholds']=selfcal_library[target][band]['nsigma']*sensitivity


import pickle
with open('selfcal_library.pickle', 'wb') as handle:
    pickle.dump(selfcal_library, handle, protocol=pickle.HIGHEST_PROTOCOL)

for target in all_targets:
 for band in bands:
   print('Starting selfcal procedure on: '+target+' '+band)
   for iteration in range(len(solints[band])):
      if iteration==0:
         SNR_key='per_EB_SNR'
      else:
         SNR_key='per_scan_SNR'

      if selfcal_library[target][band][SNR_key] < minsnr_to_proceed:
         print('*********** '+SNR_key+' too low, measured: '+str(selfcal_library[target][band][SNR_key])+', Min SNR Required: '+str(minsnr_to_proceed)+' **************')
         break
      else:
         solint=solints[band][iteration]
         if iteration == 0:
            print('Starting with solint: '+solint)
         else:
            print('Continuing with solint: '+solint)
         os.system('rm -rf '+target+'_'+band+'_'+solint+'_'+str(iteration)+'*')
         ### Open question about determining the starting and progression of clean threshold for
         ### each iteration
         ### Peak S/N > 100; SNR/10 for first, successivly reduce to 3.0 sigma through each iteration?
         ### Peak S/N < 100; SNR/10.0 

         #make images using the appropriate tclean heuristics for each telescope
         tclean_wrapper(vis=vislist, imagename=target+'_'+band+'_'+solint+'_'+str(iteration),
                     telescope=telescope,nsigma=selfcal_library[target][band]['nsigma'][iteration], scales=[0],
                     threshold=str(selfcal_library[target][band]['thresholds'][iteration])+'Jy',
                     savemodel='modelcolumn',parallel=parallel,cellsize=cellsize[band],imsize=imsize[band],nterms=nterms[band],
                     field=target,spw=selfcal_library[target][band][vis]['spws'])
         print('Pre selfcal assessemnt: '+target)
         SNR,RMS=estimate_SNR(target+'_'+band+'_'+solint+'_'+str(iteration)+'.image.tt0')
         header=imhead(imagename=target+'_'+band+'_'+solint+'_'+str(iteration)+'.image.tt0')
         b_maj_pre=header['restoringbeam']['major']['value']
         b_min_pre=header['restoringbeam']['minor']['value']
         b_pa_pre=header['restoringbeam']['positionangle']['value'] 

         if iteration == 0:
            gaintables={}
            spwmaps={}
         for vis in vislist:
            gaintables[vis]=[]
            spwmaps[vis]=[]
            os.system('rm -rf '+target+'_'+vis+'_'+band+'_'+solint+'_'+str(iteration)+'.g')
            gaincal(vis=vis,\
                    caltable=target+'_'+vis+'_'+band+'_'+solint+'_'+str(iteration)+'.g',\
                    gaintype='T', spw=selfcal_library[target][band][vis]['spws'],refant=selfcal_library[target][band][vis]['refant'], calmode='p', solint=solint.replace('_EB',''),\
                    minsnr=gaincal_minsnr, minblperant=4,combine=gaincal_combine[band][iteration],field=target,gaintable='',spwmap='') 
                    # for simplicity don't do incremental solutions,gaintable=gaintables[vis],spwmap=spwmaps[vis])
         for vis in vislist:
            if os.path.exists(vis+".flagversions/flags.selfcal_"+target+'_'+band+'_'+solint):
               flagmanager(vis=vis,mode='delete',versionname='selfcal_'+target+'_'+band+'_'+solint)
            flagmanager(vis=vis,mode='save',versionname='selfcal_'+target+'_'+band+'_'+solint)
            applycal(vis=vis,\
                     gaintable=gaintables[vis]+[target+'_'+vis+'_'+band+'_'+solint+'_'+str(iteration)+'.g'],\
                     interp=applycal_interp[band], calwt=True,spwmap=[selfcal_library[target][band][vis]['spwmap']],\
                     applymode=applycal_mode[band][iteration],field=target,spw=selfcal_library[target][band][vis]['spws'])
         for vis in vislist:
            selfcal_library[target][band][vis][solint]={}
            selfcal_library[target][band][vis][solint]['SNR_pre']=SNR.copy()
            selfcal_library[target][band][vis][solint]['RMS_pre']=RMS.copy()
            selfcal_library[target][band][vis][solint]['gaintable']=target+'_'+vis+'_'+band+'_'+solint+'_'+str(iteration)+'.g'
            selfcal_library[target][band][vis][solint]['iteration']=iteration+0
            selfcal_library[target][band][vis][solint]['flags']='selfcal_'+target+'_'+band+'_'+solint
            selfcal_library[target][band][vis][solint]['spwmap']=selfcal_library[target][band][vis]['spwmap']
            selfcal_library[target][band][vis][solint]['applycal_mode']=applycal_mode[band][iteration]+''
            selfcal_library[target][band][vis][solint]['gaincal_combine']=gaincal_combine[band][iteration]+''
         if nterms[band]==1:
            startmodel=[target+'_'+band+'_'+solint+'_'+str(iteration)+'.model.tt0']
         elif nterms[band]==2:
            startmodel=[target+'_'+band+'_'+solint+'_'+str(iteration)+'.model.tt0',target+'_'+band+'_'+solint+'_'+str(iteration)+'.model.tt1']
         tclean_wrapper(vis=vislist, imagename=target+'_'+band+'_'+solint+'_'+str(iteration)+'_post',
                  telescope=telescope,scales=[0], nsigma=0.0,\
                  savemodel='none',parallel=parallel,cellsize=cellsize[band],imsize=imsize[band],nterms=nterms[band],\
                  niter=0,startmodel=startmodel,field=target,spw=selfcal_library[target][band][vis]['spws'])
         print('Post selfcal assessemnt: '+target)
         post_SNR,post_RMS=estimate_SNR(target+'_'+band+'_'+solint+'_'+str(iteration)+'_post.image.tt0')
         header=imhead(imagename=target+'_'+band+'_'+solint+'_'+str(iteration)+'_post.image.tt0')
         b_maj_post=header['restoringbeam']['major']['value']
         b_min_post=header['restoringbeam']['minor']['value']
         b_pa_post=header['restoringbeam']['positionangle']['value'] 

         #compare relative to original image to ensure we are not incrementally changing the beam in each iteration
         beamarea_pre=selfcal_library[target][band]['Beam_major_orig']*selfcal_library[target][band]['Beam_minor_orig']
         beamarea_post=b_maj_post*b_min_post
         '''
         frac_delta_b_maj=np.abs((b_maj_post-selfcal_library[target]['Beam_major_orig'])/selfcal_library[target]['Beam_major_orig'])
         frac_delta_b_min=np.abs((b_min_post-selfcal_library[target]['Beam_minor_orig'])/selfcal_library[target]['Beam_minor_orig'])
         delta_b_pa=np.abs((b_pa_post-selfcal_library[target]['Beam_PA_orig']))
         '''
         delta_beamarea=(beamarea_post-beamarea_pre)/beamarea_pre
         if (post_SNR >= SNR) and (delta_beamarea < delta_beam_thresh): 
            #if S/N improvement, and beamarea is changing by < delta_beam_thresh, accept solutions to main calibration dictionary
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
            selfcal_library[target][band]['SNR_post']=post_SNR.copy()
            selfcal_library[target][band]['RMS_post']=post_RMS.copy()

            if iteration < (len(solints[band])-1):
               print('****************Selfcal passed, shortening solint*************')
            else:
               print('****************Selfcal passed for Minimum solint*************')
         else: #if S/N worsens, and/or beam area increases reject solutions and backout calibration
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
                  print('****************Applying '+selfcal_library[target][band][vis]['gaintable']+' to '+target+'*************')
                  flagmanager(vis=vis,mode='restore',versionname=selfcal_library[target][band][vis]['flags'])
                  applycal(vis=vis,\
                          gaintable=selfcal_library[target][band][vis]['gaintable'],\
                          interp=applycal_interp[band],\
                          calwt=True,spwmap=[selfcal_library[target][band][vis]['spwmap']],\
                          applymode=applycal_mode[band][selfcal_library[target][band]['iteration']],field=target,spw=selfcal_library[target][band][vis]['spws'])    
            else:            
               print('****************Removing all calibrations for '+target+'**************')
               clearcal(vis=vis,field=target)
               selfcal_library[target][band]['SNR_post']=selfcal_library[target][band]['SNR_orig'].copy()
               selfcal_library[target][band]['RMS_post']=selfcal_library[target][band]['RMS_orig'].copy()
            print('****************Aborting further self-calibration attempts for '+target+'**************')
            break # breakout of loops of successive solints since solutions are getting worse

for target in all_targets:
 for band in bands:
   #make images using the appropriate tclean heuristics for each telescope
   if telescope=='ALMA':
      sensitivity=get_sensitivity(vislist,selfcal_library[target][band][vis]['spws'],spw=selfcal_library[target][band][vis]['spwsarray'],imsize=imsize[band],cellsize=cellsize[band])
   else:
      sensitivity=0.0
   tclean_wrapper(vis=vislist,imagename=target+'_'+band+'_final',\
               telescope=telescope,nsigma=3.0, threshold=str(sensitivity*3.0)+'Jy',scales=[0],\
               savemodel='none',parallel=parallel,cellsize=cellsize[band],imsize=imsize[band],
               nterms=nterms[band],field=target,datacolumn='corrected',spw=selfcal_library[target][band][vis]['spws'])
   final_SNR,final_RMS=estimate_SNR(target+'_'+band+'_final.image.tt0')
   selfcal_library[target][band]['SNR_final']=final_SNR
   selfcal_library[target][band]['RMS_final']=final_RMS


for target in all_targets:
 for band in bands:
   print(target+' '+band+' Summary')
   print('At least 1 successful selfcal iteration?: ', selfcal_library[target][band]['SC_success'])
   if selfcal_library[target][band]['SC_success']:
      print('Final solint: ',selfcal_library[target][band]['final_solint'])
      print('Original SNR: ',selfcal_library[target][band]['SNR_orig'])
      print('Final SNR: ',selfcal_library[target][band]['SNR_final'])
      print('Original RMS: ',selfcal_library[target][band]['RMS_orig'])
      print('Final RMS: ',selfcal_library[target][band]['RMS_final'])
      for vis in vislist:
         print('Final gaintables: '+selfcal_library[target][band][vis]['gaintable'])
         print('Final spwmap: ',selfcal_library[target][band][vis]['spwmap'])
   else:
      print('Selfcal failed on '+target+'. No solutions applied.')

import pickle
with open('selfcal_library.pickle', 'wb') as handle:
    pickle.dump(selfcal_library, handle, protocol=pickle.HIGHEST_PROTOCOL)


