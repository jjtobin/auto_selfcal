#future improvements
# heuristics for switching between calonly and calflag
# heuristics to switch from combine=spw to combine=''
# switch heirarchy of selfcal_library such that solint is at a higher level than vis. makes storage of some parameters awkward since they live
#    in the per vis level instead of per solint

import numpy as np
from scipy import stats
import glob
import sys
#execfile('selfcal_helpers.py',globals())
sys.path.append("./")
from selfcal_helpers import *
from run_selfcal import run_selfcal
from casampi.MPIEnvironment import MPIEnvironment 
parallel=MPIEnvironment.is_mpi_enabled

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
if len(vislist) == 0:
   vislist=glob.glob('*_targets.ms')   # adaptation for PL2022 output
   if len(vislist)==0:
      vislist=glob.glob('*_cont.ms')   # adaptation for PL2022 output
   elif len(vislist)==0:
      sys.exit('No Measurement sets found in current working directory, exiting')

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
spectral_average=True
do_amp_selfcal=True
inf_EB_gaincal_combine='scan'
inf_EB_gaintype='G'
inf_EB_override=False
gaincal_minsnr=2.0
gaincal_unflag_minsnr=5.0
minsnr_to_proceed=3.0
delta_beam_thresh=0.05
n_ants=get_n_ants(vislist)
telescope=get_telescope(vislist[0])
apply_cal_mode_default='calflag'
unflag_only_lbants = False
unflag_only_lbants_onlyap = False
calonly_max_flagged = 0.0
second_iter_solmode = ""
unflag_fb_to_prev_solint = False
rerank_refants=False
allow_gain_interpolation=False
rel_thresh_scaling='log10'  #can set to linear, log10, or loge (natural log)
dividing_factor=-99.0  # number that the peak SNR is divided by to determine first clean threshold -99.0 uses default
                       # default is 40 for <8ghz and 15.0 for all other frequencies
check_all_spws=False   # generate per-spw images to check phase transfer did not go poorly for narrow windows
apply_to_target_ms=False # apply final selfcal solutions back to the input _target.ms files

if 'VLA' in telescope:
   check_all_spws=False
   #inf_EB_gaincal_combine='spw,scan'
##
## Import inital MS files to get relevant meta data
##
listdict,bands,band_properties,scantimesdict,scanfieldsdict,scannfieldsdict,scanstartsdict,scanendsdict,integrationsdict,\
integrationtimesdict,spwslist,spwstring,spwsarray,mosaic_field,gaincalibrator_dict=importdata(vislist,all_targets,telescope)

##
## flag spectral lines in MS(es) if there is a cont.dat file present
##
if os.path.exists("cont.dat"):
   flag_spectral_lines(vislist,all_targets,spwsarray)


##
## spectrally average ALMA or VLA data with telescope/frequency specific averaging properties
##
split_to_selfcal_ms(vislist,band_properties,bands,spectral_average)

##
## put flagging back at original state for originally input ms for when they are used next time
##
for vis in vislist:
    if os.path.exists(vis+".flagversions/flags.before_line_flags"):
       flagmanager(vis=vis,mode='restore',versionname='before_line_flags')     


##
## Reimport MS(es) to self calibrate since frequency averaging and splitting may have changed it
##
spwslist_orig=spwslist.copy()
vislist_orig=vislist.copy()
spwstring_orig=spwstring+''
spwsarray_orig =spwsarray.copy()

vislist=glob.glob('*selfcal.ms')
listdict,bands,band_properties,scantimesdict,scanfieldsdict,scannfieldsdict,scanstartsdict,scanendsdict,integrationsdict,\
integrationtimesdict,spwslist,spwstring,spwsarray,mosaic_field,gaincalibrator_dict=importdata(vislist,all_targets,telescope)

##
## Save/restore starting flags
##

for vis in vislist:
   if os.path.exists(vis+'.flagversions/flags.selfcal_starting_flags'):
      flagmanager(vis=vis,mode='restore',versionname='selfcal_starting_flags')
   else:
      flagmanager(vis=vis,mode='save',versionname='selfcal_starting_flags')


##
## set image parameters based on the visibility data properties and frequency
##
cellsize={}
imsize={}
nterms={}
applycal_interp={}

for target in all_targets:
    cellsize[target], imsize[target], nterms[target], applycal_interp[target] = {}, {}, {}, {}
    for band in bands:
       cellsize[target][band],imsize[target][band],nterms[target][band] = \
               get_image_parameters(vislist,telescope,target,band, \
               band_properties,mosaic=mosaic_field[band][all_targets[0]]['mosaic'])

       if band_properties[vislist[0]][band]['meanfreq'] >12.0e9:
          applycal_interp[target][band]='linearPD'
       else:
          applycal_interp[target][band]='linear'





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
      if target in scantimesdict[band][vislist[0]].keys():
         selfcal_library[target][band]={}
      else:
         continue
      for vis in vislist:
         selfcal_library[target][band][vis]={}

      for fid in mosaic_field[band][target]["field_ids"]:
          selfcal_library[target][band][fid] = {}

          for vis in vislist:
              selfcal_library[target][band][fid][vis] = {}

import json
print(json.dumps(selfcal_library, indent=4))
##
## finds solints, starting with inf, ending with int, and tries to align
## solints with number of integrations
## solints reduce by factor of 2 in each self-cal interation
## e.g., inf, max_scan_time/2.0, prev_solint/2.0, ..., int
## starting solints will have solint the length of the entire EB to correct bulk offsets
##
solints={}
gaincal_combine={}
solmode={}
applycal_mode={}
for band in bands:
   solints[band],integration_time,gaincal_combine[band],solmode[band]=get_solints_simple(vislist,scantimesdict[band],scannfieldsdict[band],scanstartsdict[band],scanendsdict[band],integrationtimesdict[band],inf_EB_gaincal_combine,do_amp_selfcal=do_amp_selfcal,mosaic=mosaic_field[band][all_targets[0]]['mosaic'])
   print(band,solints[band])
   applycal_mode[band]=[apply_cal_mode_default]*len(solints[band])



##
## puts stuff in right place from other MS metadata to perform proper data selections
## in tclean, gaincal, and applycal
## Also gets relevant times on source to estimate SNR per EB/scan
##
for target in all_targets:
 for band in selfcal_library[target].keys():
   print(target, band)
   selfcal_library[target][band]['SC_success']=False
   selfcal_library[target][band]['final_solint']='None'
   selfcal_library[target][band]['Total_TOS']=0.0
   selfcal_library[target][band]['spws']=[]
   selfcal_library[target][band]['spws_per_vis']=[]
   selfcal_library[target][band]['nterms']=nterms[target][band]
   selfcal_library[target][band]['vislist']=vislist.copy()
   if mosaic_field[band][target]['mosaic']:
      selfcal_library[target][band]['obstype']='mosaic'
   else:
      selfcal_library[target][band]['obstype']='single-point'
   selfcal_library[target][band]['sub-fields'] = mosaic_field[band][target]['field_ids']
   selfcal_library[target][band]['sub-fields-to-selfcal'] = mosaic_field[band][target]['field_ids']
   allscantimes=np.array([])
   allscannfields=np.array([])
   for vis in vislist:
      selfcal_library[target][band][vis]['gaintable']=[]
      selfcal_library[target][band][vis]['TOS']=np.sum(scantimesdict[band][vis][target])
      selfcal_library[target][band][vis]['Median_scan_time']=np.median(scantimesdict[band][vis][target])
      selfcal_library[target][band][vis]['Median_fields_per_scan']=np.median(scannfieldsdict[band][vis][target])
      allscantimes=np.append(allscantimes,scantimesdict[band][vis][target])
      allscannfields=np.append(allscannfields,scannfieldsdict[band][vis][target])
      selfcal_library[target][band][vis]['refant'] = rank_refants(vis)
      n_spws,minspw,spwsarray=fetch_spws([vis],[target])
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
   selfcal_library[target][band]['Median_fields_per_scan']=np.median(allscannfields)
   selfcal_library[target][band]['uvrange']=get_uv_range(band,band_properties,vislist)
   selfcal_library[target][band]['75thpct_uv']=band_properties[vislist[0]][band]['75thpct_uv']
   selfcal_library[target][band]['LAS']=band_properties[vislist[0]][band]['LAS']
   selfcal_library[target][band]['fracbw']=band_properties[vislist[0]][band]['fracbw']
   print(selfcal_library[target][band]['uvrange'])

   for fid in mosaic_field[band][target]['field_ids']:
       selfcal_library[target][band][fid]['SC_success']=False
       selfcal_library[target][band][fid]['final_solint']='None'
       selfcal_library[target][band][fid]['Total_TOS']=0.0
       selfcal_library[target][band][fid]['spws']=[]
       selfcal_library[target][band][fid]['spws_per_vis']=[]
       selfcal_library[target][band][fid]['nterms']=nterms[target][band]
       selfcal_library[target][band][fid]['vislist']=vislist.copy()
       selfcal_library[target][band][fid]['obstype'] = 'single-point'
       allscantimes=np.array([])
       allscannfields=np.array([])
       for vis in vislist:
          good = np.array([str(fid) in scan_fields for scan_fields in scanfieldsdict[band][vis][target]])
          selfcal_library[target][band][fid][vis]['gaintable']=[]
          selfcal_library[target][band][fid][vis]['TOS']=np.sum(scantimesdict[band][vis][target][good]/scannfieldsdict[band][vis][target][good])
          selfcal_library[target][band][fid][vis]['Median_scan_time']=np.median(scantimesdict[band][vis][target][good]/scannfieldsdict[band][vis][target][good])
          selfcal_library[target][band][fid][vis]['Median_fields_per_scan']=1
          allscantimes=np.append(allscantimes,scantimesdict[band][vis][target][good]/scannfieldsdict[band][vis][target][good])
          allscannfields=np.append(allscannfields,[1])
          selfcal_library[target][band][fid][vis]['refant'] = selfcal_library[target][band][vis]['refant']
          n_spws,minspw,spwsarray=fetch_spws([vis],[target])
          spwslist=spwsarray.tolist()
          spwstring=','.join(str(spw) for spw in spwslist)
          selfcal_library[target][band][fid][vis]['spws']=band_properties[vis][band]['spwstring']
          selfcal_library[target][band][fid][vis]['spwsarray']=band_properties[vis][band]['spwarray']
          selfcal_library[target][band][fid][vis]['spwlist']=band_properties[vis][band]['spwarray'].tolist()
          selfcal_library[target][band][fid][vis]['n_spws']=len(selfcal_library[target][band][fid][vis]['spwsarray'])
          selfcal_library[target][band][fid][vis]['minspw']=int(np.min(selfcal_library[target][band][fid][vis]['spwsarray']))
          selfcal_library[target][band][fid][vis]['spwmap']=[selfcal_library[target][band][fid][vis]['minspw']]*(np.max(selfcal_library[target][band][fid][vis]['spwsarray'])+1)
          selfcal_library[target][band][fid]['Total_TOS']=selfcal_library[target][band][fid][vis]['TOS']+selfcal_library[target][band][fid]['Total_TOS']
          selfcal_library[target][band][fid]['spws_per_vis'].append(band_properties[vis][band]['spwstring'])
       selfcal_library[target][band][fid]['Median_scan_time']=np.median(allscantimes)
       selfcal_library[target][band][fid]['Median_fields_per_scan']=np.median(allscannfields)
       selfcal_library[target][band][fid]['uvrange']=get_uv_range(band,band_properties,vislist)
       selfcal_library[target][band][fid]['75thpct_uv']=band_properties[vislist[0]][band]['75thpct_uv']
       selfcal_library[target][band][fid]['LAS']=band_properties[vislist[0]][band]['LAS']

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

##
## 
## 
for target in all_targets:
 for band in selfcal_library[target].keys():
      if selfcal_library[target][band]['Total_TOS'] == 0.0:
         selfcal_library[target].pop(band)


print(json.dumps(selfcal_library, indent=4, cls=NpEncoder))
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
      # Because tclean doesn't deal in NF masks, the automask from the initial image is likely to contain a lot of noise unless
      # we can get an estimate of the NF modifier for the auto-masking thresholds. To do this, we need to create a very basic mask
      # with the dirty image. So we just use one iteration with a tiny gain so that nothing is really subtracted off.
      tclean_wrapper(vislist,sani_target+'_'+band+'_dirty',
                     band_properties,band,telescope=telescope,nsigma=4.0, scales=[0],
                     threshold='0.0Jy',niter=1, gain=0.00001,
                     savemodel='none',parallel=parallel,cellsize=cellsize[target][band],imsize=imsize[target][band],nterms=nterms[target][band],
                     field=target,spw=selfcal_library[target][band]['spws_per_vis'],uvrange=selfcal_library[target][band]['uvrange'],obstype=selfcal_library[target][band]['obstype'], image_mosaic_fields_separately=selfcal_library[target][band]['obstype']=='mosaic')
   dirty_SNR,dirty_RMS=estimate_SNR(sani_target+'_'+band+'_dirty.image.tt0')
   if telescope!='ACA':
      dirty_NF_SNR,dirty_NF_RMS=estimate_near_field_SNR(sani_target+'_'+band+'_dirty.image.tt0', las=selfcal_library[target][band]['LAS'])
   else:
      dirty_NF_SNR,dirty_NF_RMS=dirty_SNR,dirty_RMS

   mosaic_dirty_SNR, mosaic_dirty_RMS, mosaic_dirty_NF_SNR, mosaic_dirty_NF_RMS = {}, {}, {}, {}
   for fid in selfcal_library[target][band]['sub-fields']:
       if selfcal_library[target][band]['obstype'] == 'mosaic':
           imagename = sani_target+'_field_'+str(fid)+'_'+band+'_dirty.image.tt0'
       else:
           imagename = sani_target+'_'+band+'_dirty.image.tt0'

       mosaic_dirty_SNR[fid], mosaic_dirty_RMS[fid] = estimate_SNR(imagename, mosaic_sub_field=selfcal_library[target][band]["obstype"]=="mosaic")
       if telescope !='ACA':
          mosaic_dirty_NF_SNR[fid],mosaic_dirty_NF_RMS[fid]=estimate_near_field_SNR(imagename, las=selfcal_library[target][band]['LAS'], \
                  mosaic_sub_field=selfcal_library[target][band]["obstype"]=="mosaic")
       else:
          mosaic_dirty_NF_SNR[fid],mosaic_dirty_NF_RMS[fid]=mosaic_dirty_SNR[fid],mosaic_dirty_RMS[fid]

   if telescope == "VLA" or (selfcal_library[target][band]['obstype'] == 'mosaic' and \
           selfcal_library[target][band]['Median_scan_time'] / selfcal_library[target][band]['Median_fields_per_scan'] < 60.) \
           or selfcal_library[target][band]['75thpct_uv'] > 2000.0:
       selfcal_library[target][band]['cyclefactor'] = 3.0
   else:
       selfcal_library[target][band]['cyclefactor'] = 1.0

   dr_mod=1.0
   if telescope =='ALMA' or telescope =='ACA':
      sensitivity=get_sensitivity(vislist,selfcal_library[target][band],target,selfcal_library[target][band][vis]['spws'],spw=selfcal_library[target][band][vis]['spwsarray'],imsize=imsize[target][band],cellsize=cellsize[target][band])
      dr_mod=get_dr_correction(telescope,dirty_SNR*dirty_RMS,sensitivity,vislist)
      sensitivity_nomod=sensitivity.copy()
      print('DR modifier: ',dr_mod)
   if not os.path.exists(sani_target+'_'+band+'_initial.image.tt0'):
      if telescope=='ALMA' or telescope =='ACA':
         sensitivity=sensitivity*dr_mod   # apply DR modifier
         if band =='Band_9' or band == 'Band_10':   # adjust for DSB noise increase
            sensitivity=sensitivity   #*4.0  might be unnecessary with DR mods
      else:
         sensitivity=0.0
      tclean_wrapper(vislist,sani_target+'_'+band+'_initial',
                     band_properties,band,telescope=telescope,nsigma=4.0, scales=[0],
                     threshold=str(sensitivity*4.0)+'Jy',
                     savemodel='none',parallel=parallel,cellsize=cellsize[target][band],imsize=imsize[target][band],nterms=nterms[target][band],
                     field=target,spw=selfcal_library[target][band]['spws_per_vis'],uvrange=selfcal_library[target][band]['uvrange'],obstype=selfcal_library[target][band]['obstype'], nfrms_multiplier=dirty_NF_RMS/dirty_RMS, image_mosaic_fields_separately=selfcal_library[target][band]['obstype']=='mosaic', cyclefactor=selfcal_library[target][band]['cyclefactor'])
   initial_SNR,initial_RMS=estimate_SNR(sani_target+'_'+band+'_initial.image.tt0')
   if telescope!='ACA':
      initial_NF_SNR,initial_NF_RMS=estimate_near_field_SNR(sani_target+'_'+band+'_initial.image.tt0', las=selfcal_library[target][band]['LAS'])
   else:
      initial_NF_SNR,initial_NF_RMS=initial_SNR,initial_RMS

   mosaic_initial_SNR, mosaic_initial_RMS, mosaic_initial_NF_SNR, mosaic_initial_NF_RMS = {}, {}, {}, {}
   for fid in selfcal_library[target][band]['sub-fields']:
       if selfcal_library[target][band]['obstype'] == 'mosaic':
           imagename = sani_target+'_field_'+str(fid)+'_'+band+'_initial.image.tt0'
       else:
           imagename = sani_target+'_'+band+'_initial.image.tt0'

       mosaic_initial_SNR[fid], mosaic_initial_RMS[fid] = estimate_SNR(imagename, mosaic_sub_field=selfcal_library[target][band]["obstype"]=="mosaic")
       if telescope !='ACA':
          mosaic_initial_NF_SNR[fid],mosaic_initial_NF_RMS[fid]=estimate_near_field_SNR(imagename, las=selfcal_library[target][band]['LAS'], \
                  mosaic_sub_field=selfcal_library[target][band]["obstype"]=="mosaic")
       else:
          mosaic_initial_NF_SNR[fid],mosaic_initial_NF_RMS[fid]=mosaic_initial_SNR[fid],mosaic_initial_RMS[fid]

   header=imhead(imagename=sani_target+'_'+band+'_initial.image.tt0')
   if telescope =='ALMA' or telescope == 'ACA':
      selfcal_library[target][band]['theoretical_sensitivity']=sensitivity_nomod
   if 'VLA' in telescope:
      selfcal_library[target][band]['theoretical_sensitivity']=-99.0
   selfcal_library[target][band]['SNR_orig']=initial_SNR
   if selfcal_library[target][band]['nterms'] == 1:  # updated nterms if needed based on S/N and fracbw
      selfcal_library[target][band]['nterms']=check_image_nterms(selfcal_library[target][band]['fracbw'],selfcal_library[target][band]['SNR_orig'])
   selfcal_library[target][band]['RMS_orig']=initial_RMS
   selfcal_library[target][band]['SNR_NF_orig']=initial_NF_SNR
   selfcal_library[target][band]['RMS_NF_orig']=initial_NF_RMS
   selfcal_library[target][band]['RMS_curr']=initial_RMS
   selfcal_library[target][band]['RMS_NF_curr']=initial_NF_RMS
   selfcal_library[target][band]['SNR_dirty']=dirty_SNR
   selfcal_library[target][band]['RMS_dirty']=dirty_RMS
   selfcal_library[target][band]['Beam_major_orig']=header['restoringbeam']['major']['value']
   selfcal_library[target][band]['Beam_minor_orig']=header['restoringbeam']['minor']['value']
   selfcal_library[target][band]['Beam_PA_orig']=header['restoringbeam']['positionangle']['value'] 
   goodMask=checkmask(imagename=sani_target+'_'+band+'_initial.image.tt0')
   if goodMask:
      selfcal_library[target][band]['intflux_orig'],selfcal_library[target][band]['e_intflux_orig']=get_intflux(sani_target+'_'+band+'_initial.image.tt0',initial_RMS)
   else:
      selfcal_library[target][band]['intflux_orig'],selfcal_library[target][band]['e_intflux_orig']=-99.0,-99.0

   for fid in selfcal_library[target][band]['sub-fields']:
       if selfcal_library[target][band]['obstype'] == 'mosaic':
           imagename = sani_target+'_field_'+str(fid)+'_'+band+'_initial.image.tt0'
       else:
           imagename = sani_target+'_'+band+'_initial.image.tt0'

       header=imhead(imagename=imagename)
       if telescope =='ALMA' or telescope == 'ACA':
          selfcal_library[target][band][fid]['theoretical_sensitivity']=sensitivity_nomod
       if 'VLA' in telescope:
          selfcal_library[target][band][fid]['theoretical_sensitivity']=-99.0
       selfcal_library[target][band][fid]['SNR_orig']=mosaic_initial_SNR[fid]
       if selfcal_library[target][band][fid]['SNR_orig'] > 500.0:
          selfcal_library[target][band][fid]['nterms']=2
       selfcal_library[target][band][fid]['RMS_orig']=mosaic_initial_RMS[fid]
       selfcal_library[target][band][fid]['SNR_NF_orig']=mosaic_initial_NF_SNR[fid]
       selfcal_library[target][band][fid]['RMS_NF_orig']=mosaic_initial_NF_RMS[fid]
       selfcal_library[target][band][fid]['RMS_curr']=mosaic_initial_RMS[fid]
       selfcal_library[target][band][fid]['RMS_NF_curr']=mosaic_initial_NF_RMS[fid]
       selfcal_library[target][band][fid]['SNR_dirty']=mosaic_dirty_SNR[fid]
       selfcal_library[target][band][fid]['RMS_dirty']=mosaic_dirty_RMS[fid]
       selfcal_library[target][band][fid]['Beam_major_orig']=header['restoringbeam']['major']['value']
       selfcal_library[target][band][fid]['Beam_minor_orig']=header['restoringbeam']['minor']['value']
       selfcal_library[target][band][fid]['Beam_PA_orig']=header['restoringbeam']['positionangle']['value'] 
       goodMask=checkmask(imagename=imagename)
       if goodMask:
          selfcal_library[target][band][fid]['intflux_orig'],selfcal_library[target][band][fid]['e_intflux_orig']=get_intflux(imagename,\
                  mosaic_initial_RMS[fid], mosaic_sub_field=selfcal_library[target][band]["obstype"]=="mosaic")
       else:
          selfcal_library[target][band][fid]['intflux_orig'],selfcal_library[target][band][fid]['e_intflux_orig']=-99.0,-99.0



print(json.dumps(selfcal_library, indent=4, cls=NpEncoder))
####MAKE DIRTY PER SPW IMAGES TO PROPERLY ASSESS DR MODIFIERS
##
## Make a initial image per spw images to assess overall improvement
##   

for target in all_targets:
   for band in selfcal_library[target].keys():
      selfcal_library[target][band]['per_spw_stats']={}
      vislist=selfcal_library[target][band]['vislist'].copy()
      #code to work around some VLA data not having the same number of spws due to missing BlBPs
      #selects spwlist from the visibilities with the greates number of spws
      maxspws=0
      maxspwvis=''
      for vis in vislist:
         if selfcal_library[target][band][vis]['n_spws'] >= maxspws:
            maxspws=selfcal_library[target][band][vis]['n_spws']
            maxspwvis=vis+''
         selfcal_library[target][band][vis]['spwlist']=selfcal_library[target][band][vis]['spws'].split(',')
      spwlist=selfcal_library[target][band][maxspwvis]['spwlist']
       
      spw_bandwidths,spw_effective_bandwidths=get_spw_bandwidth(vis,selfcal_library[target][band][maxspwvis]['spwsarray'],target)

      selfcal_library[target][band]['total_bandwidth']=0.0
      selfcal_library[target][band]['total_effective_bandwidth']=0.0
      if len(spw_effective_bandwidths.keys()) != len(spw_bandwidths.keys()):
         print('cont.dat does not contain all spws; falling back to total bandwidth')
         for spw in spw_bandwidths.keys():
            if spw not in spw_effective_bandwidths.keys():
               spw_effective_bandwidths[spw]=spw_bandwidths[spw]

      for spw in spwlist:
         keylist=selfcal_library[target][band]['per_spw_stats'].keys()
         if spw not in keylist:
            selfcal_library[target][band]['per_spw_stats'][spw]={}

         selfcal_library[target][band]['per_spw_stats'][spw]['effective_bandwidth']=spw_effective_bandwidths[spw]
         selfcal_library[target][band]['per_spw_stats'][spw]['bandwidth']=spw_bandwidths[spw]
         selfcal_library[target][band]['total_bandwidth']+=spw_bandwidths[spw]
         selfcal_library[target][band]['total_effective_bandwidth']+=spw_effective_bandwidths[spw]

      for fid in selfcal_library[target][band]['sub-fields']:
          selfcal_library[target][band][fid]['per_spw_stats']={}
          vislist=selfcal_library[target][band][fid]['vislist'].copy()
          for vis in vislist:
              selfcal_library[target][band][fid][vis]['spwlist']=selfcal_library[target][band][fid][vis]['spws'].split(',')
          spwlist=selfcal_library[target][band][fid][maxspwvis]['spwlist']
          spw_bandwidths,spw_effective_bandwidths=get_spw_bandwidth(vis,selfcal_library[target][band][fid][maxspwvis]['spwsarray'],target)
          selfcal_library[target][band][fid]['total_bandwidth']=0.0
          selfcal_library[target][band][fid]['total_effective_bandwidth']=0.0
          if len(spw_effective_bandwidths.keys()) != len(spw_bandwidths.keys()):
             print('cont.dat does not contain all spws; falling back to total bandwidth')
             for spw in spw_bandwidths.keys():
                if spw not in spw_effective_bandwidths.keys():
                   spw_effective_bandwidths[spw]=spw_bandwidths[spw]
          for spw in spwlist:
             keylist=selfcal_library[target][band][fid]['per_spw_stats'].keys()
             if spw not in keylist:
                selfcal_library[target][band][fid]['per_spw_stats'][spw]={}
             selfcal_library[target][band][fid]['per_spw_stats'][spw]['effective_bandwidth']=spw_effective_bandwidths[spw]
             selfcal_library[target][band][fid]['per_spw_stats'][spw]['bandwidth']=spw_bandwidths[spw]
             selfcal_library[target][band][fid]['total_bandwidth']+=spw_bandwidths[spw]
             selfcal_library[target][band][fid]['total_effective_bandwidth']+=spw_effective_bandwidths[spw]

if check_all_spws:
   for target in all_targets:
      sani_target=sanitize_string(target)
      for band in selfcal_library[target].keys():
         vislist=selfcal_library[target][band]['vislist'].copy()
         #potential place where diff spws for different VLA EBs could cause problems
         spwlist=selfcal_library[target][band][vis]['spws'].split(',')
         for spw in spwlist:
            keylist=selfcal_library[target][band]['per_spw_stats'].keys()
            if spw not in keylist:
               selfcal_library[target][band]['per_spw_stats'][spw]={}
            if not os.path.exists(sani_target+'_'+band+'_'+spw+'_dirty.image.tt0'):
               spws_per_vis=[spw]*len(vislist)
               tclean_wrapper(vislist,sani_target+'_'+band+'_'+spw+'_dirty',
                     band_properties,band,telescope=telescope,nsigma=4.0, scales=[0],
                     threshold='0.0Jy',niter=0,
                     savemodel='none',parallel=parallel,cellsize=cellsize[target][band],imsize=imsize[target][band],nterms=1,
                     field=target,spw=spws_per_vis,
                     uvrange=selfcal_library[target][band]['uvrange'],obstype=selfcal_library[target][band]['obstype'])
            dirty_SNR,dirty_RMS=estimate_SNR(sani_target+'_'+band+'_'+spw+'_dirty.image.tt0')
            if telescope!='ACA':
               dirty_per_spw_NF_SNR,dirty_per_spw_NF_RMS=estimate_near_field_SNR(sani_target+'_'+band+'_'+spw+'_dirty.image.tt0', las=selfcal_library[target][band]['LAS'])
            else:
               dirty_per_spw_NF_SNR,dirty_per_spw_NF_RMS=per_spw_SNR,per_spw_RMS
            if not os.path.exists(sani_target+'_'+band+'_'+spw+'_initial.image.tt0'):
               if telescope=='ALMA' or telescope =='ACA':
                  sensitivity=get_sensitivity(vislist,selfcal_library[target][band],target,spw,spw=np.array([int(spw)]),imsize=imsize[target][band],cellsize=cellsize[target][band])
                  dr_mod=1.0
                  dr_mod=get_dr_correction(telescope,dirty_SNR*dirty_RMS,sensitivity,vislist)
                  print('DR modifier: ',dr_mod,'SPW: ',spw)
                  sensitivity=sensitivity*dr_mod 
                  if ((band =='Band_9') or (band == 'Band_10')) and dr_mod != 1.0:   # adjust for DSB noise increase
                     sensitivity=sensitivity*4.0 
               else:
                  sensitivity=0.0
               spws_per_vis=[spw]*len(vislist)  #assumes all spw ids are identical in each MS file

               tclean_wrapper(vislist,sani_target+'_'+band+'_'+spw+'_initial',\
                          band_properties,band,telescope=telescope,nsigma=4.0, threshold=str(sensitivity*4.0)+'Jy',scales=[0],\
                          savemodel='none',parallel=parallel,cellsize=cellsize[target][band],imsize=imsize[target][band],\
                          nterms=1,field=target,datacolumn='corrected',\
                          spw=spws_per_vis,uvrange=selfcal_library[target][band]['uvrange'],obstype=selfcal_library[target][band]['obstype'], \
                          nfrms_multiplier=dirty_per_spw_NF_RMS/dirty_RMS, cyclefactor=selfcal_library[target][band]['cyclefactor'])

            per_spw_SNR,per_spw_RMS=estimate_SNR(sani_target+'_'+band+'_'+spw+'_initial.image.tt0')
            if telescope!='ACA':
               initial_per_spw_NF_SNR,initial_per_spw_NF_RMS=estimate_near_field_SNR(sani_target+'_'+band+'_'+spw+'_initial.image.tt0', las=selfcal_library[target][band]['LAS'])
            else:
               initial_per_spw_NF_SNR,initial_per_spw_NF_RMS=per_spw_SNR,per_spw_RMS
            selfcal_library[target][band]['per_spw_stats'][spw]['SNR_orig']=per_spw_SNR
            selfcal_library[target][band]['per_spw_stats'][spw]['RMS_orig']=per_spw_RMS
            selfcal_library[target][band]['per_spw_stats'][spw]['SNR_NF_orig']=initial_per_spw_NF_SNR
            selfcal_library[target][band]['per_spw_stats'][spw]['RMS_NF_orig']=initial_per_spw_NF_RMS
            goodMask=checkmask(sani_target+'_'+band+'_'+spw+'_initial.image.tt0')
            if goodMask:
               selfcal_library[target][band]['per_spw_stats'][spw]['intflux_orig'],selfcal_library[target][band]['per_spw_stats'][spw]['e_intflux_orig']=get_intflux(sani_target+'_'+band+'_'+spw+'_initial.image.tt0',per_spw_RMS)
            else:
               selfcal_library[target][band]['per_spw_stats'][spw]['intflux_orig'],selfcal_library[target][band]['per_spw_stats'][spw]['e_intflux_orig']=-99.0,-99.0               





##
## estimate per scan/EB S/N using time on source and median scan times
##
inf_EB_gaincal_combine_dict={} #'scan'
inf_EB_gaintype_dict={} #'G'
inf_EB_fallback_mode_dict={} #'scan'

solint_snr,solint_snr_per_spw,solint_snr_per_field,solint_snr_per_field_per_spw=get_SNR_self(all_targets,bands,vislist,selfcal_library,n_ants,solints,integration_time,inf_EB_gaincal_combine,inf_EB_gaintype)
minsolint_spw=100.0
for target in all_targets:
 inf_EB_gaincal_combine_dict[target]={} #'scan'
 inf_EB_fallback_mode_dict[target]={} #'scan'
 inf_EB_gaintype_dict[target]={} #'G'
 for band in solint_snr[target].keys():
   inf_EB_gaincal_combine_dict[target][band]={}
   inf_EB_gaintype_dict[target][band]={}
   inf_EB_fallback_mode_dict[target][band]={}
   for vis in vislist:
    inf_EB_gaincal_combine_dict[target][band][vis]=inf_EB_gaincal_combine #'scan'
    if selfcal_library[target][band]['obstype']=='mosaic':
       inf_EB_gaincal_combine_dict[target][band][vis]+=',field'   
    inf_EB_gaintype_dict[target][band][vis]=inf_EB_gaintype #G
    inf_EB_fallback_mode_dict[target][band][vis]='' #'scan'
    print('Estimated SNR per solint:')
    print(target,band)
    for solint in solints[band]:
      if solint == 'inf_EB':
         print('{}: {:0.2f}'.format(solint,solint_snr[target][band][solint]))
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
         print('{}: {:0.2f}'.format(solint,solint_snr[target][band][solint]))

    for fid in selfcal_library[target][band]['sub-fields']:
        print('Estimated SNR per solint:')
        print(target,band,"field "+str(fid))
        for solint in solints[band]:
          if solint == 'inf_EB':
             print('{}: {:0.2f}'.format(solint,solint_snr_per_field[target][band][fid][solint]))
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
             print('{}: {:0.2f}'.format(solint,solint_snr_per_field[target][band][fid][solint]))

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
   if band_properties[selfcal_library[target][band]['vislist'][0]][band]['meanfreq'] <8.0e9 and (dividing_factor ==-99.0):
      dividing_factor=40.0
   elif (dividing_factor ==-99.0):
      dividing_factor=15.0
   nsigma_init=np.max([selfcal_library[target][band]['SNR_NF_orig']/dividing_factor,5.0]) # restricts initial nsigma to be at least 5

   n_ap_solints=sum(1 for solint in solints[band] if 'ap' in solint)  # count number of amplitude selfcal solints, repeat final clean depth of phase-only for amplitude selfcal
   if rel_thresh_scaling == 'loge':
      selfcal_library[target][band]['nsigma']=np.append(np.exp(np.linspace(np.log(nsigma_init),np.log(3.0),len(solints[band])-n_ap_solints)),np.array([np.exp(np.log(3.0))]*n_ap_solints))
   elif rel_thresh_scaling == 'linear':
      selfcal_library[target][band]['nsigma']=np.append(np.linspace(nsigma_init,3.0,len(solints[band])-n_ap_solints),np.array([3.0]*n_ap_solints))
   else: #implicitly making log10 the default
      selfcal_library[target][band]['nsigma']=np.append(10**np.linspace(np.log10(nsigma_init),np.log10(3.0),len(solints[band])-n_ap_solints),np.array([10**(np.log10(3.0))]*n_ap_solints))

   if telescope=='ALMA' or telescope =='ACA': #or ('VLA' in telescope) 
      sensitivity=get_sensitivity(vislist,selfcal_library[target][band],target,selfcal_library[target][band][vis]['spws'],spw=selfcal_library[target][band][vis]['spwsarray'],imsize=imsize[target][band],cellsize=cellsize[target][band])
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
 for band in selfcal_library[target].keys():
   run_selfcal(selfcal_library, target, band, solints, solint_snr, solint_snr_per_field, applycal_mode, solmode, band_properties, telescope, n_ants, cellsize[target], imsize[target], \
           inf_EB_gaintype_dict, inf_EB_gaincal_combine_dict, inf_EB_fallback_mode_dict, gaincal_combine, applycal_interp[target], integration_time, \
           gaincal_minsnr=gaincal_minsnr, gaincal_unflag_minsnr=gaincal_unflag_minsnr, minsnr_to_proceed=minsnr_to_proceed, delta_beam_thresh=delta_beam_thresh, do_amp_selfcal=do_amp_selfcal, \
           inf_EB_gaincal_combine=inf_EB_gaincal_combine, inf_EB_gaintype=inf_EB_gaintype, unflag_only_lbants=unflag_only_lbants, \
           unflag_only_lbants_onlyap=unflag_only_lbants_onlyap, calonly_max_flagged=calonly_max_flagged, \
           second_iter_solmode=second_iter_solmode, unflag_fb_to_prev_solint=unflag_fb_to_prev_solint, rerank_refants=rerank_refants, \
           gaincalibrator_dict=gaincalibrator_dict, allow_gain_interpolation=allow_gain_interpolation)

print(json.dumps(selfcal_library, indent=4, cls=NpEncoder))


##
## Save the flags following the main iteration of self-calibration since we will need to revert to the beginning for the fallback mode.
##
# PS: I don't need this anymore?
for vis in vislist:
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
            elif 'inf' in solints[band]:
                fallback_fields[band].append(target)
        else:
            fallback_fields[band].append(target)

    # Update the relevant lists if we are going to do a fallback mode.
    if len(fallback_fields[band]) > 0:
        solints[band] += ["inf_EB_fb","inf_fb1","inf_fb2","inf_fb3"]
        solmode[band] += ["p","p","p","p"]
        gaincal_combine[band] += [gaincal_combine[band][0], gaincal_combine[band][1], gaincal_combine[band][1], gaincal_combine[band][1]]
        applycal_mode[band] += [applycal_mode[band][0], applycal_mode[band][1], applycal_mode[band][1], applycal_mode[band][1]]
        calibrators[band] = [inf_EB_fields[band], inf_fields[band], inf_fields[band], inf_fields[band]]
        for target in all_targets:
            selfcal_library[target][band]["nsigma"] = np.concatenate((selfcal_library[target][band]["nsigma"],[selfcal_library[target][band]["nsigma"][0], \
                    selfcal_library[target][band]["nsigma"][1], selfcal_library[target][band]["nsigma"][1], selfcal_library[target][band]["nsigma"][1]]))

print(inf_EB_fields)
print(inf_fields)
print(fallback_fields)

##
## Reset the inf_EB informational dictionaries.
##

for target in all_targets:
 for band in solint_snr[target].keys():
   # If the target had a successful inf_EB solution, no need to reset.
   if target in inf_EB_fields[band]:
       continue

   for vis in vislist:
    inf_EB_gaincal_combine_dict[target][band][vis]=inf_EB_gaincal_combine #'scan'
    if selfcal_library[target][band]['obstype']=='mosaic':
       inf_EB_gaincal_combine_dict[target][band][vis]+=',field'   
    inf_EB_gaintype_dict[target][band][vis]=inf_EB_gaintype #G
    inf_EB_fallback_mode_dict[target][band][vis]='' #'scan'


calculate_inf_EB_fb_anyways = True
preapply_targets_own_inf_EB = False

## The below sets the calibrations back to what they were prior to starting the fallback mode. It should not be needed
## for the final version of the codue, but is used for testing.


for target in all_targets:
 sani_target=sanitize_string(target)
 for band in selfcal_library[target].keys():
   if target not in fallback_fields[band]:
       continue
   if 'gaintable_final' in selfcal_library[target][band][vislist[0]]:
      print('****************Reapplying previous solint solutions*************')
      for vis in vislist:
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
      for vis in vislist:
         flagmanager(vis=vis,mode='delete',versionname='fb_selfcal_starting_flags_'+sani_target)
         clearcal(vis=vis,field=target,spw=selfcal_library[target][band][vis]['spws'])
## END
            

##
## Begin fallback self-cal loops
##
for target in all_targets:
 for band in selfcal_library[target].keys():
   if target not in fallback_fields[band]:
       continue

   run_selfcal(selfcal_library, target, band, solints, solint_snr, solint_snr_per_field, applycal_mode, solmode, band_properties, telescope, n_ants, cellsize[target], imsize[target], \
           inf_EB_gaintype_dict, inf_EB_gaincal_combine_dict, inf_EB_fallback_mode_dict, gaincal_combine, applycal_interp[target], integration_time, \
           gaincal_minsnr=gaincal_minsnr, gaincal_unflag_minsnr=gaincal_unflag_minsnr, minsnr_to_proceed=minsnr_to_proceed, delta_beam_thresh=delta_beam_thresh, do_amp_selfcal=do_amp_selfcal, \
           inf_EB_gaincal_combine=inf_EB_gaincal_combine, inf_EB_gaintype=inf_EB_gaintype, unflag_only_lbants=unflag_only_lbants, \
           unflag_only_lbants_onlyap=unflag_only_lbants_onlyap, calonly_max_flagged=calonly_max_flagged, \
           second_iter_solmode=second_iter_solmode, unflag_fb_to_prev_solint=unflag_fb_to_prev_solint, rerank_refants=rerank_refants, \
           mode="cocal", calibrators=calibrators, calculate_inf_EB_fb_anyways=calculate_inf_EB_fb_anyways, \
           preapply_targets_own_inf_EB=preapply_targets_own_inf_EB, gaincalibrator_dict=gaincalibrator_dict)

print(json.dumps(selfcal_library, indent=4, cls=NpEncoder))

##
## If we want to try amplitude selfcal, should we do it as a function out of the main loop or a separate loop?
## Mechanics are likely to be a bit more simple since I expect we'd only try a single solint=inf solution
##

##
## Make a final image per target to assess overall improvement
##
for target in all_targets:
 sani_target=sanitize_string(target)
 for band in selfcal_library[target].keys():
   vislist=selfcal_library[target][band]['vislist'].copy()
   nfsnr_modifier = selfcal_library[target][band]['RMS_NF_curr'] / selfcal_library[target][band]['RMS_curr']
   tclean_wrapper(vislist,sani_target+'_'+band+'_final',\
               band_properties,band,telescope=telescope,nsigma=3.0, threshold=str(selfcal_library[target][band]['RMS_NF_curr']*3.0)+'Jy',scales=[0],\
               savemodel='none',parallel=parallel,cellsize=cellsize[target][band],imsize=imsize[target][band],
               nterms=selfcal_library[target][band]['nterms'],field=target,datacolumn='corrected',spw=selfcal_library[target][band]['spws_per_vis'],uvrange=selfcal_library[target][band]['uvrange'],obstype=selfcal_library[target][band]['obstype'], \
               nfrms_multiplier=nfsnr_modifier, image_mosaic_fields_separately=selfcal_library[target][band]['obstype']=='mosaic',\
               cyclefactor=selfcal_library[target][band]['cyclefactor'])
   final_SNR,final_RMS=estimate_SNR(sani_target+'_'+band+'_final.image.tt0')
   if telescope !='ACA':
      final_NF_SNR,final_NF_RMS=estimate_near_field_SNR(sani_target+'_'+band+'_final.image.tt0', las=selfcal_library[target][band]['LAS'])
   else:
      final_NF_SNR,final_NF_RMS=final_SNR,final_RMS

   mosaic_final_SNR, mosaic_final_RMS, mosaic_final_NF_SNR, mosaic_final_NF_RMS = {}, {}, {}, {}
   for fid in selfcal_library[target][band]['sub-fields']:
       if selfcal_library[target][band]['obstype'] == 'mosaic':
           imagename = sani_target+'_field_'+str(fid)+'_'+band+'_final.image.tt0'
       else:
           imagename = sani_target+'_'+band+'_final.image.tt0'

       mosaic_final_SNR[fid], mosaic_final_RMS[fid] = estimate_SNR(imagename, mosaic_sub_field=selfcal_library[target][band]["obstype"]=="mosaic")
       if telescope !='ACA':
          mosaic_final_NF_SNR[fid],mosaic_final_NF_RMS[fid]=estimate_near_field_SNR(imagename, las=selfcal_library[target][band]['LAS'], \
                  mosaic_sub_field=selfcal_library[target][band]["obstype"]=="mosaic")
       else:
          mosaic_final_NF_SNR[fid],mosaic_final_NF_RMS[fid]=mosaic_final_SNR[fid],mosaic_final_RMS[fid]

   selfcal_library[target][band]['SNR_final']=final_SNR
   selfcal_library[target][band]['RMS_final']=final_RMS
   selfcal_library[target][band]['SNR_NF_final']=final_NF_SNR
   selfcal_library[target][band]['RMS_NF_final']=final_NF_RMS
   header=imhead(imagename=sani_target+'_'+band+'_final.image.tt0')
   selfcal_library[target][band]['Beam_major_final']=header['restoringbeam']['major']['value']
   selfcal_library[target][band]['Beam_minor_final']=header['restoringbeam']['minor']['value']
   selfcal_library[target][band]['Beam_PA_final']=header['restoringbeam']['positionangle']['value'] 
   #recalc inital stats using final mask
   final_SNR,final_RMS=estimate_SNR(sani_target+'_'+band+'_initial.image.tt0',maskname=sani_target+'_'+band+'_final.mask')
   if telescope!='ACA':
      final_NF_SNR,final_NF_RMS=estimate_near_field_SNR(sani_target+'_'+band+'_initial.image.tt0',maskname=sani_target+'_'+band+'_final.mask', las=selfcal_library[target][band]['LAS'])
   else:
      final_NF_SNR,final_NF_RMS=final_SNR,final_RMS
   selfcal_library[target][band]['SNR_orig']=final_SNR
   selfcal_library[target][band]['RMS_orig']=final_RMS
   selfcal_library[target][band]['SNR_NF_orig']=final_NF_SNR
   selfcal_library[target][band]['RMS_NF_orig']=final_NF_RMS
   goodMask=checkmask(imagename=sani_target+'_'+band+'_final.image.tt0')
   if goodMask:
      selfcal_library[target][band]['intflux_final'],selfcal_library[target][band]['e_intflux_final']=get_intflux(sani_target+'_'+band+'_final.image.tt0',final_RMS)
      selfcal_library[target][band]['intflux_orig'],selfcal_library[target][band]['e_intflux_orig']=get_intflux(sani_target+'_'+band+'_initial.image.tt0',selfcal_library[target][band]['RMS_orig'],maskname=sani_target+'_'+band+'_final.mask')
   else:
      selfcal_library[target][band]['intflux_final'],selfcal_library[target][band]['e_intflux_final']=-99.0,-99.0

   for fid in selfcal_library[target][band]['sub-fields']:
       if selfcal_library[target][band]['obstype'] == 'mosaic':
           imagename = sani_target+'_field_'+str(fid)+'_'+band
       else:
           imagename = sani_target+'_'+band

       selfcal_library[target][band][fid]['SNR_final']=mosaic_final_SNR[fid]
       selfcal_library[target][band][fid]['RMS_final']=mosaic_final_RMS[fid]
       selfcal_library[target][band][fid]['SNR_NF_final']=mosaic_final_NF_SNR[fid]
       selfcal_library[target][band][fid]['RMS_NF_final']=mosaic_final_NF_RMS[fid]
       header=imhead(imagename=imagename+'_final.image.tt0')
       selfcal_library[target][band][fid]['Beam_major_final']=header['restoringbeam']['major']['value']
       selfcal_library[target][band][fid]['Beam_minor_final']=header['restoringbeam']['minor']['value']
       selfcal_library[target][band][fid]['Beam_PA_final']=header['restoringbeam']['positionangle']['value'] 
       #recalc inital stats using final mask
       mosaic_initial_final_SNR,mosaic_initial_final_RMS=estimate_SNR(imagename+'_initial.image.tt0',maskname=imagename+'_final.mask', \
               mosaic_sub_field=selfcal_library[target][band]["obstype"]=="mosaic")
       if telescope!='ACA':
          mosaic_initial_final_NF_SNR,mosaic_initial_final_NF_RMS=estimate_near_field_SNR(imagename+'_initial.image.tt0', \
                  maskname=imagename+'_final.mask', las=selfcal_library[target][band]['LAS'], \
                  mosaic_sub_field=selfcal_library[target][band]["obstype"]=="mosaic")
       else:
          mosaic_initial_final_NF_SNR,mosaic_initial_final_NF_RMS=mosaic_initial_final_SNR,mosaic_initial_final_RMS
       selfcal_library[target][band][fid]['SNR_orig']=mosaic_initial_final_SNR
       selfcal_library[target][band][fid]['RMS_orig']=mosaic_initial_final_RMS
       selfcal_library[target][band][fid]['SNR_NF_orig']=mosaic_initial_final_NF_SNR
       selfcal_library[target][band][fid]['RMS_NF_orig']=mosaic_initial_final_NF_RMS

       if selfcal_library[target][band]['obstype'] == 'mosaic':
           imagename = sani_target+'_field_'+str(fid)+'_'+band
       else:
           imagename = sani_target+'_'+band

       goodMask=checkmask(imagename=imagename+'_final.image.tt0')
       if goodMask:
          selfcal_library[target][band][fid]['intflux_final'],selfcal_library[target][band][fid]['e_intflux_final']=\
                  get_intflux(imagename+'_final.image.tt0', mosaic_final_RMS[fid], mosaic_sub_field=selfcal_library[target][band]["obstype"]=="mosaic")
          selfcal_library[target][band][fid]['intflux_orig'],selfcal_library[target][band][fid]['e_intflux_orig']=\
                  get_intflux(imagename+'_initial.image.tt0', selfcal_library[target][band][fid]['RMS_orig'], \
                  maskname=imagename+'_final.mask', mosaic_sub_field=selfcal_library[target][band]["obstype"]=="mosaic")
       else:
          selfcal_library[target][band][fid]['intflux_final'],selfcal_library[target][band][fid]['e_intflux_final']=-99.0,-99.0





print(json.dumps(selfcal_library, indent=4, cls=NpEncoder))

##
## Make a final image per spw images to assess overall improvement
##
if check_all_spws:
   for target in all_targets:
      sani_target=sanitize_string(target)
      for band in selfcal_library[target].keys():
         vislist=selfcal_library[target][band]['vislist'].copy()

         spwlist=selfcal_library[target][band][vis]['spws'].split(',')
         print('Generating final per-SPW images for '+target+' in '+band)
         for spw in spwlist:
   ## omit DR modifiers here since we should have increased DR significantly
            if not os.path.exists(sani_target+'_'+band+'_'+spw+'_final.image.tt0'):
               if telescope=='ALMA' or telescope =='ACA':
                  sensitivity=get_sensitivity(vislist,selfcal_library[target][band],target,spw,spw=np.array([int(spw)]),imsize=imsize[target][band],cellsize=cellsize[target][band])
                  dr_mod=1.0
                  if not selfcal_library[target][band]['SC_success']: # fetch the DR modifier if selfcal failed on source
                     dr_mod=get_dr_correction(telescope,selfcal_library[target][band]['SNR_dirty']*selfcal_library[target][band]['RMS_dirty'],sensitivity,vislist)
                  print('DR modifier: ',dr_mod, 'SPW: ',spw)
                  sensitivity=sensitivity*dr_mod 
                  if ((band =='Band_9') or (band == 'Band_10')) and dr_mod != 1.0:   # adjust for DSB noise increase
                     sensitivity=sensitivity*4.0 
               else:
                  sensitivity=0.0
               spws_per_vis=[spw]*len(vislist)  #assumes all spw ids are identical in each MS file
               sensitivity_agg=get_sensitivity(vislist,selfcal_library[target][band],target,selfcal_library[target][band][vis]['spws'],spw=selfcal_library[target][band][vis]['spwsarray'],imsize=imsize[band],cellsize=cellsize[band])
               sensitivity_scale_factor=selfcal_library[target][band]['RMS_NF_curr']/sensitivity_agg

               nfsnr_modifier = selfcal_library[target][band]['RMS_NF_curr'] / selfcal_library[target][band]['RMS_curr']
               tclean_wrapper(vislist,sani_target+'_'+band+'_'+spw+'_final',\
                          band_properties,band,telescope=telescope,nsigma=4.0, threshold=str(sensitivity*sensitivity_scale_factor*4.0)+'Jy',scales=[0],\
                          savemodel='none',parallel=parallel,cellsize=cellsize[target][band],imsize=imsize[target][band],\
                          nterms=1,field=target,datacolumn='corrected',\
                          spw=spws_per_vis,uvrange=selfcal_library[target][band]['uvrange'],obstype=selfcal_library[target][band]['obstype'],
                          nfrms_multiplier=nfsnr_modifier, cyclefactor=selfcal_library[target][band]['cyclefactor'])
            final_per_spw_SNR,final_per_spw_RMS=estimate_SNR(sani_target+'_'+band+'_'+spw+'_final.image.tt0')
            if telescope !='ACA':
               final_per_spw_NF_SNR,final_per_spw_NF_RMS=estimate_near_field_SNR(sani_target+'_'+band+'_'+spw+'_final.image.tt0', las=selfcal_library[target][band]['LAS'])
            else:
               final_per_spw_NF_SNR,final_per_spw_NF_RMS=final_per_spw_SNR,final_per_spw_RMS

            selfcal_library[target][band]['per_spw_stats'][spw]['SNR_final']=final_per_spw_SNR
            selfcal_library[target][band]['per_spw_stats'][spw]['RMS_final']=final_per_spw_RMS
            selfcal_library[target][band]['per_spw_stats'][spw]['SNR_NF_final']=final_per_spw_NF_SNR
            selfcal_library[target][band]['per_spw_stats'][spw]['RMS_NF_final']=final_per_spw_NF_RMS
            #reccalc initial stats with final mask
            final_per_spw_SNR,final_per_spw_RMS=estimate_SNR(sani_target+'_'+band+'_'+spw+'_initial.image.tt0',maskname=sani_target+'_'+band+'_'+spw+'_final.mask')
            if telescope !='ACA':
               final_per_spw_NF_SNR,final_per_spw_NF_RMS=estimate_near_field_SNR(sani_target+'_'+band+'_'+spw+'_initial.image.tt0',maskname=sani_target+'_'+band+'_'+spw+'_final.mask', las=selfcal_library[target][band]['LAS'])
            else:
               final_per_spw_NF_SNR,final_per_spw_NF_RMS=final_per_spw_SNR,final_per_spw_RMS
            selfcal_library[target][band]['per_spw_stats'][spw]['SNR_orig']=final_per_spw_SNR
            selfcal_library[target][band]['per_spw_stats'][spw]['RMS_orig']=final_per_spw_RMS
            selfcal_library[target][band]['per_spw_stats'][spw]['SNR_NF_orig']=final_per_spw_NF_SNR
            selfcal_library[target][band]['per_spw_stats'][spw]['RMS_NF_orig']=final_per_spw_NF_RMS



            goodMask=checkmask(sani_target+'_'+band+'_'+spw+'_final.image.tt0')
            if goodMask:
               selfcal_library[target][band]['per_spw_stats'][spw]['intflux_final'],selfcal_library[target][band]['per_spw_stats'][spw]['e_intflux_final']=get_intflux(sani_target+'_'+band+'_'+spw+'_final.image.tt0',final_per_spw_RMS)
            else:
               selfcal_library[target][band]['per_spw_stats'][spw]['intflux_final'],selfcal_library[target][band]['per_spw_stats'][spw]['e_intflux_final']=-99.0,-99.0               








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

   for fid in selfcal_library[target][band]['sub-fields']:
       print(target+' '+band+' field '+str(fid)+' Summary')
       print('At least 1 successful selfcal iteration?: ', selfcal_library[target][band][fid]['SC_success'])
       print('Final solint: ',selfcal_library[target][band][fid]['final_solint'])
       print('Original SNR: ',selfcal_library[target][band][fid]['SNR_orig'])
       print('Final SNR: ',selfcal_library[target][band][fid]['SNR_final'])
       print('Original RMS: ',selfcal_library[target][band][fid]['RMS_orig'])
       print('Final RMS: ',selfcal_library[target][band][fid]['RMS_final'])


applyCalOut=open('applycal_to_orig_MSes.py','w')
#apply selfcal solutions back to original ms files
if apply_to_target_ms:
   for vis in vislist_orig:
      clearcal(vis=vis)
for target in all_targets:
   for band in selfcal_library[target].keys():
      if selfcal_library[target][band]['SC_success']:
         for vis in vislist: 
            solint=selfcal_library[target][band]['final_solint']
            iteration=selfcal_library[target][band][vis][solint]['iteration']    
            line='applycal(vis="'+vis.replace('.selfcal','')+'",gaintable='+str(selfcal_library[target][band][vis]['gaintable_final'])+',interp='+str(selfcal_library[target][band][vis]['applycal_interpolate_final'])+', calwt=False,spwmap='+str(selfcal_library[target][band][vis]['spwmap_final'])+', applymode="'+selfcal_library[target][band][vis]['applycal_mode_final']+'",field="'+target+'",spw="'+spwstring_orig+'")\n'
            applyCalOut.writelines(line)
            if apply_to_target_ms:
               if os.path.exists(vis.replace('.selfcal','')+".flagversions/flags.starting_flags"):
                  flagmanager(vis=vis.replace('.selfcal',''), mode = 'restore', versionname = 'starting_flags', comment = 'Flag states at start of reduction')
               else:
                  flagmanager(vis=vis.replace('.selfcal',''),mode='save',versionname='before_final_applycal')
               applycal(vis=vis.replace('.selfcal',''),\
                    gaintable=selfcal_library[target][band][vis]['gaintable_final'],\
                    interp=selfcal_library[target][band][vis]['applycal_interpolate_final'], calwt=False,spwmap=[selfcal_library[target][band][vis]['spwmap_final']],\
                    applymode=selfcal_library[target][band][vis]['applycal_mode_final'],field=target,spw=spwstring_orig)

applyCalOut.close()


if os.path.exists("cont.dat"):
   uvcontsubOut=open('uvcontsub_orig_MSes.py','w')
   line='import os\n'
   uvcontsubOut.writelines(line)
   for target in all_targets:
      sani_target=sanitize_string(target)
      for band in selfcal_library[target].keys():
         for vis in vislist:      
            contdot_dat_flagchannels_string = flagchannels_from_contdotdat(vis.replace('.selfcal',''),target,spwsarray)[:-2]
            line='uvcontsub(vis="'+vis.replace('.selfcal','')+'",field="'+target+'", spw="'+spwstring_orig+'",fitspw="'+contdot_dat_flagchannels_string+'",excludechans=True, combine="spw")\n'
            uvcontsubOut.writelines(line)
            line='os.system("mv '+vis.replace('.selfcal','')+'.contsub '+sani_target+'_'+vis+'.contsub")\n'
            uvcontsubOut.writelines(line)
   uvcontsubOut.close()



#
# Perform a check on the per-spw images to ensure they didn't lose quality in self-calibration
#
if check_all_spws:
   for target in all_targets:
      sani_target=sanitize_string(target)
      for band in selfcal_library[target].keys():
         vislist=selfcal_library[target][band]['vislist'].copy()

         spwlist=selfcal_library[target][band][vis]['spws'].split(',')
         for spw in spwlist:
            delta_beamarea=compare_beams(sani_target+'_'+band+'_'+spw+'_initial.image.tt0',\
                                         sani_target+'_'+band+'_'+spw+'_final.image.tt0')
            delta_SNR=selfcal_library[target][band]['per_spw_stats'][spw]['SNR_final']-\
                      selfcal_library[target][band]['per_spw_stats'][spw]['SNR_orig']
            delta_RMS=selfcal_library[target][band]['per_spw_stats'][spw]['RMS_final']-\
                      selfcal_library[target][band]['per_spw_stats'][spw]['RMS_orig']
            selfcal_library[target][band]['per_spw_stats'][spw]['delta_SNR']=delta_SNR
            selfcal_library[target][band]['per_spw_stats'][spw]['delta_RMS']=delta_RMS
            selfcal_library[target][band]['per_spw_stats'][spw]['delta_beamarea']=delta_beamarea
            print(sani_target+'_'+band+'_'+spw,\
                  'Pre SNR: {:0.2f}, Post SNR: {:0.2f} Pre RMS: {:0.3f}, Post RMS: {:0.3f}'.format(selfcal_library[target][band]['per_spw_stats'][spw]['SNR_orig'],\
                   selfcal_library[target][band]['per_spw_stats'][spw]['SNR_final'],selfcal_library[target][band]['per_spw_stats'][spw]['RMS_orig']*1000.0,selfcal_library[target][band]['per_spw_stats'][spw]['RMS_final']*1000.0))
            if delta_SNR < 0.0:
               print('WARNING SPW '+spw+' HAS LOWER SNR POST SELFCAL')
            if delta_RMS > 0.0:
               print('WARNING SPW '+spw+' HAS HIGHER RMS POST SELFCAL')
            if delta_beamarea > 0.05:
               print('WARNING SPW '+spw+' HAS A >0.05 CHANGE IN BEAM AREA POST SELFCAL')


##
## Save final library results
##
import pickle
with open('selfcal_library.pickle', 'wb') as handle:
    pickle.dump(selfcal_library, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('solints.pickle', 'wb') as handle:
    pickle.dump(solints, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('bands.pickle', 'wb') as handle:
    pickle.dump(bands, handle, protocol=pickle.HIGHEST_PROTOCOL)

generate_weblog(selfcal_library,solints,bands,directory='weblog')

# For simplicity, instead of redoing all of the weblog code, create a new selfcal_library dictionary where all of the sub-fields exist at the
# same level as the main field so that they all get their own entry in the weblog, in addition to the entry for the main field.
for target in all_targets:
    new_selfcal_library = {}
    for band in selfcal_library[target].keys():
        if selfcal_library[target][band]['obstype'] == 'mosaic':
            for fid in selfcal_library[target][band]['sub-fields']:
                if target+'_field_'+str(fid) not in new_selfcal_library:
                    new_selfcal_library[target+'_field_'+str(fid)] = {}
                new_selfcal_library[target+'_field_'+str(fid)][band] = selfcal_library[target][band][fid]

    if len(new_selfcal_library) > 0:
        generate_weblog(new_selfcal_library,solints,bands,directory='weblog/'+target+'_field-by-field')


