import numpy as np
import glob
import sys
#execfile('selfcal_helpers.py',globals())
sys.path.append("./")
from selfcal_helpers import *

def prepare_selfcal(vislist, 
        spectral_average=True, 
        sort_targets_and_EBs=False,
        scale_fov=1.0,
        inf_EB_gaincal_combine='scan',
        inf_EB_gaintype='G',
        apply_cal_mode_default='calflag',
        do_amp_selfcal=True):

    n_ants=get_n_ants(vislist)
    telescope=get_telescope(vislist[0])

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
    if sort_targets_and_EBs:
        all_targets.sort()
        vislist.sort()

    ##
    ## Import inital MS files to get relevant meta data
    ##
    listdict,bands,band_properties,scantimesdict,scanfieldsdict,scannfieldsdict,scanstartsdict,scanendsdict,integrationsdict,\
    integrationtimesdict,spwslist_dict,spwstring_dict,spwsarray_dict,mosaic_field,gaincalibrator_dict,spectral_scan,spws_set=importdata(vislist,all_targets,telescope)

    ##
    ## flag spectral lines in MS(es) if there is a cont.dat file present
    ##
    if os.path.exists("cont.dat"):
       flag_spectral_lines(vislist,all_targets,spwsarray_dict)


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
    band_properties_orig = band_properties.copy()
    spwslist_dict_orig=spwslist_dict.copy()
    vislist_orig=vislist.copy()
    spwstring_dict_orig=spwstring_dict.copy()
    spwsarray_dict_orig =spwsarray_dict.copy()

    vislist=[vis.replace(".ms",".selfcal.ms") for vis in vislist]

    listdict,bands,band_properties,scantimesdict,scanfieldsdict,scannfieldsdict,scanstartsdict,scanendsdict,integrationsdict,\
    integrationtimesdict,spwslist_dict,spwstring_dict,spwsarray_dict,mosaic_field,gaincalibrator_dict,spectral_scan,spws_set=importdata(vislist,all_targets,telescope)

    ##
    ## Save/restore starting flags
    ##

    for vis in vislist:
       if os.path.exists(vis+'.flagversions/flags.selfcal_starting_flags'):
          flagmanager(vis=vis,mode='restore',versionname='selfcal_starting_flags')
       else:
          flagmanager(vis=vis,mode='save',versionname='selfcal_starting_flags')


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

          if mosaic_field[band][vislist[0]][target]['mosaic']:
             selfcal_library[target][band]['obstype']='mosaic'
          else:
             selfcal_library[target][band]['obstype']='single-point'

          # Make sure the fields get mapped properly, in case the order in which they are observed changes from EB to EB.

          selfcal_library[target][band]['sub-fields-fid_map'] = {}
          all_phasecenters = []
          for vis in vislist:
              selfcal_library[target][band]['sub-fields-fid_map'][vis] = {}
              for i in range(len(mosaic_field[band][vis][target]['field_ids'])):
                  found = False
                  for j in range(len(all_phasecenters)):
                      distance = ((all_phasecenters[j]["m0"]["value"] - mosaic_field[band][vis][target]['phasecenters'][i]["m0"]["value"])**2 + \
                              (all_phasecenters[j]["m1"]["value"] - mosaic_field[band][vis][target]['phasecenters'][i]["m1"]["value"])**2)**0.5

                      if distance < 4.84814e-6:
                          selfcal_library[target][band]['sub-fields-fid_map'][vis][j] = mosaic_field[band][vis][target]['field_ids'][i]
                          found = True
                          break

                  if not found:
                      all_phasecenters.append(mosaic_field[band][vis][target]['phasecenters'][i])
                      selfcal_library[target][band]['sub-fields-fid_map'][vis][len(all_phasecenters)-1] = mosaic_field[band][vis][target]['field_ids'][i]

          selfcal_library[target][band]['sub-fields'] = list(range(len(all_phasecenters)))
          selfcal_library[target][band]['sub-fields-to-selfcal'] = list(range(len(all_phasecenters)))
          selfcal_library[target][band]['sub-fields-phasecenters'] = dict(zip(selfcal_library[target][band]['sub-fields'], all_phasecenters))

          # Now we can start to create a sub-field selfcal_library entry for each sub-field.

          for fid in selfcal_library[target][band]['sub-fields']:
              selfcal_library[target][band][fid] = {}

              for vis in vislist:
                  if not fid in selfcal_library[target][band]['sub-fields-fid_map'][vis]:
                      continue

                  selfcal_library[target][band][fid][vis] = {}

    import json
    print(json.dumps(selfcal_library, indent=4))

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
       selfcal_library[target][band]['vislist']=vislist.copy()
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
          #n_spws,minspw,spwsarray=fetch_spws([vis],[target])
          #spwslist=spwsarray.tolist()
          #spwstring=','.join(str(spw) for spw in spwslist)
          selfcal_library[target][band][vis]['spws']=band_properties[vis][band]['spwstring']
          selfcal_library[target][band][vis]['spwsarray']=band_properties[vis][band]['spwarray']
          selfcal_library[target][band][vis]['spwlist']=band_properties[vis][band]['spwarray'].tolist()
          selfcal_library[target][band][vis]['spws_orig']=band_properties_orig[vis.replace('.selfcal','')][band]['spwstring']
          selfcal_library[target][band][vis]['spwsarray_orig']=band_properties_orig[vis.replace('.selfcal','')][band]['spwarray']
          selfcal_library[target][band][vis]['spwlist_orig']=band_properties_orig[vis.replace('.selfcal','')][band]['spwarray'].tolist()
          selfcal_library[target][band][vis]['n_spws']=len(selfcal_library[target][band][vis]['spwsarray'])
          selfcal_library[target][band][vis]['minspw']=int(np.min(selfcal_library[target][band][vis]['spwsarray']))
          selfcal_library[target][band][vis]['baseband']=band_properties[vis][band]['baseband']
          if spectral_scan:
             spwmap=np.zeros(np.max(spws_set[band][vis])+1,dtype='int')
             spwmap.fill(np.min(spws_set[band][vis]))
             for i in range(spws_set[band][vis].shape[0]):
                indices=np.arange(np.min(spws_set[band][vis][i]),np.max(spws_set[band][vis][i])+1)
                spwmap[indices]=np.min(spws_set[band][vis][i])
             selfcal_library[target][band][vis]['spwmap']=spwmap.tolist()
          else:
             selfcal_library[target][band][vis]['spwmap']=[selfcal_library[target][band][vis]['minspw']]*(np.max(selfcal_library[target][band][vis]['spwsarray'])+1)
             baseband_spwmap=[]
             for baseband in selfcal_library[target][band][vis]['baseband']:
                if len(baseband_spwmap) == 0:
                   baseband_spwmap=[selfcal_library[target][band][vis]['minspw']]*(selfcal_library[target][band][vis]['baseband'][baseband]['spwlist'][0]+1)
                else:   
                   baseband_spwmap=baseband_spwmap+([selfcal_library[target][band][vis]['baseband'][baseband]['spwlist'][0]]*len(selfcal_library[target][band][vis]['baseband'][baseband]['spwlist']))
             selfcal_library[target][band][vis]['baseband_spwmap']=baseband_spwmap.copy()
          selfcal_library[target][band]['Total_TOS']=selfcal_library[target][band][vis]['TOS']+selfcal_library[target][band]['Total_TOS']
          selfcal_library[target][band]['spws_per_vis'].append(band_properties[vis][band]['spwstring'])
       selfcal_library[target][band]['Median_scan_time']=np.median(allscantimes)
       selfcal_library[target][band]['Median_fields_per_scan']=np.median(allscannfields)
       selfcal_library[target][band]['uvrange']=get_uv_range(band,band_properties,vislist)
       selfcal_library[target][band]['75thpct_uv']=band_properties[vislist[0]][band]['75thpct_uv']
       selfcal_library[target][band]['LAS']=band_properties[vislist[0]][band]['LAS']
       selfcal_library[target][band]['fracbw']=band_properties[vislist[0]][band]['fracbw']
       selfcal_library[target][band]['meanfreq']=band_properties[vislist[0]][band]['meanfreq']
       selfcal_library[target][band]['spectral_scan'] = spectral_scan
       selfcal_library[target][band]['spws_set'] = spws_set[band]
       print(selfcal_library[target][band]['uvrange'])

       for fid in selfcal_library[target][band]['sub-fields']:
           selfcal_library[target][band][fid]['SC_success']=False
           selfcal_library[target][band][fid]['final_solint']='None'
           selfcal_library[target][band][fid]['Total_TOS']=0.0
           selfcal_library[target][band][fid]['spws']=[]
           selfcal_library[target][band][fid]['spws_per_vis']=[]
           selfcal_library[target][band][fid]['vislist']=[vis for vis in vislist if fid in selfcal_library[target][band]['sub-fields-fid_map'][vis]]
           selfcal_library[target][band][fid]['obstype'] = 'single-point'
           allscantimes=np.array([])
           allscannfields=np.array([])
           for vis in selfcal_library[target][band][fid]['vislist']:
              good = np.array([str(selfcal_library[target][band]['sub-fields-fid_map'][vis][fid]) in scan_fields for scan_fields in scanfieldsdict[band][vis][target]])
              selfcal_library[target][band][fid][vis]['gaintable']=[]
              selfcal_library[target][band][fid][vis]['TOS']=np.sum(scantimesdict[band][vis][target][good]/scannfieldsdict[band][vis][target][good])
              selfcal_library[target][band][fid][vis]['Median_scan_time']=np.median(scantimesdict[band][vis][target][good]/scannfieldsdict[band][vis][target][good])
              selfcal_library[target][band][fid][vis]['Median_fields_per_scan']=1
              allscantimes=np.append(allscantimes,scantimesdict[band][vis][target][good]/scannfieldsdict[band][vis][target][good])
              allscannfields=np.append(allscannfields,[1])
              selfcal_library[target][band][fid][vis]['refant'] = selfcal_library[target][band][vis]['refant']
              #n_spws,minspw,spwsarray=fetch_spws([vis],[target])
              #spwslist=spwsarray.tolist()
              #spwstring=','.join(str(spw) for spw in spwslist)
              selfcal_library[target][band][fid][vis]['spws']=band_properties[vis][band]['spwstring']
              selfcal_library[target][band][fid][vis]['spwsarray']=band_properties[vis][band]['spwarray']
              selfcal_library[target][band][fid][vis]['spwlist']=band_properties[vis][band]['spwarray'].tolist()
              selfcal_library[target][band][fid][vis]['spws_orig']=band_properties_orig[vis.replace('.selfcal','')][band]['spwstring']
              selfcal_library[target][band][fid][vis]['spwsarray_orig']=band_properties_orig[vis.replace('.selfcal','')][band]['spwarray']
              selfcal_library[target][band][fid][vis]['spwlist_orig']=band_properties_orig[vis.replace('.selfcal','')][band]['spwarray'].tolist()
              selfcal_library[target][band][fid][vis]['n_spws']=len(selfcal_library[target][band][fid][vis]['spwsarray'])
              selfcal_library[target][band][fid][vis]['minspw']=int(np.min(selfcal_library[target][band][fid][vis]['spwsarray']))
              selfcal_library[target][band][fid][vis]['baseband']=band_properties[vis][band]['baseband']
              if spectral_scan:
                 spwmap=np.zeros(np.max(spws_set[band][vis])+1,dtype='int')
                 spwmap.fill(np.min(spws_set[band][vis]))
                 for i in range(spws_set[band][vis].shape[0]):
                    indices=np.arange(np.min(spws_set[band][vis][i]),np.max(spws_set[band][vis][i])+1)
                    spwmap[indices]=np.min(spws_set[band][vis][i])
                 selfcal_library[target][band][fid][vis]['spwmap']=spwmap.tolist()
              else:
                 selfcal_library[target][band][fid][vis]['spwmap']=[selfcal_library[target][band][fid][vis]['minspw']]*(np.max(selfcal_library[target][band][fid][vis]['spwsarray'])+1)
                 baseband_spwmap=[]
                 for baseband in selfcal_library[target][band][fid][vis]['baseband']:
                    if len(baseband_spwmap) == 0:
                       baseband_spwmap=[selfcal_library[target][band][fid][vis]['minspw']]*(selfcal_library[target][band][vis]['baseband'][baseband]['spwlist'][0]+1)
                    else:   
                       baseband_spwmap=baseband_spwmap+([selfcal_library[target][band][fid][vis]['baseband'][baseband]['spwlist'][0]]*len(selfcal_library[target][band][fid][vis]['baseband'][baseband]['spwlist']))
                 selfcal_library[target][band][fid][vis]['baseband_spwmap']=baseband_spwmap.copy()

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
    ## Get the per-spw stats
    ##   

    for target in all_targets:
       for band in selfcal_library[target].keys():
          selfcal_library[target][band]['per_spw_stats']={}
          vislist=selfcal_library[target][band]['vislist'].copy()

          selfcal_library[target][band]['spw_map'] = get_spw_map(selfcal_library, 
                  target, band, telescope)

          #code to work around some VLA data not having the same number of spws due to missing BlBPs
          #selects spwlist from the visibilities with the greates number of spws
          #PS: We now track spws on an EB by EB basis soI have removed much of the maxspwvis code.
          spw_bandwidths_dict={}
          spw_effective_bandwidths_dict={}
          spw_freqs_dict={}
          for vis in selfcal_library[target][band]['vislist']:
             selfcal_library[target][band][vis]['per_spw_stats'] = {}
              
             spw_bandwidths_dict[vis],spw_effective_bandwidths_dict[vis],spw_freqs_dict[vis]=get_spw_bandwidth(vis,spwsarray_dict,target,vislist)

             selfcal_library[target][band][vis]['total_bandwidth']=0.0
             selfcal_library[target][band][vis]['total_effective_bandwidth']=0.0
             if len(spw_effective_bandwidths_dict[vis].keys()) != len(spw_bandwidths_dict[vis].keys()):
                print('cont.dat does not contain all spws; falling back to total bandwidth')
                for spw in spw_bandwidths_dict[vis].keys():
                   if spw not in spw_effective_bandwidths_dict[vis].keys():
                      spw_effective_bandwidths_dict[vis][spw]=spw_bandwidths_dict[vis][spw]

             for spw in selfcal_library[target][band][vis]['spwlist']:
                keylist=selfcal_library[target][band][vis]['per_spw_stats'].keys()
                if spw not in keylist:
                   selfcal_library[target][band][vis]['per_spw_stats'][spw]={}

                selfcal_library[target][band][vis]['per_spw_stats'][spw]['effective_bandwidth']=spw_effective_bandwidths_dict[vis][spw]
                selfcal_library[target][band][vis]['per_spw_stats'][spw]['bandwidth']=spw_bandwidths_dict[vis][spw]
                selfcal_library[target][band][vis]['per_spw_stats'][spw]['frequency']=spw_freqs_dict[vis][spw]
                selfcal_library[target][band][vis]['total_bandwidth']+=spw_bandwidths_dict[vis][spw]
                selfcal_library[target][band][vis]['total_effective_bandwidth']+=spw_effective_bandwidths_dict[vis][spw]
             for baseband in selfcal_library[target][band][vis]['baseband'].keys():
                selfcal_library[target][band][vis]['baseband'][baseband]['total_bandwidth']=0.0
                selfcal_library[target][band][vis]['baseband'][baseband]['total_effective_bandwidth']=0.0
                selfcal_library[target][band][vis]['baseband'][baseband]['bwarray']=np.zeros(len(selfcal_library[target][band][vis]['baseband'][baseband]['spwarray']))
                selfcal_library[target][band][vis]['baseband'][baseband]['eff_bwarray']=np.zeros(len(selfcal_library[target][band][vis]['baseband'][baseband]['spwarray']))
                selfcal_library[target][band][vis]['baseband'][baseband]['freq_array']=np.zeros(len(selfcal_library[target][band][vis]['baseband'][baseband]['spwarray']))
                for s, spw in enumerate(selfcal_library[target][band][vis]['baseband'][baseband]['spwarray']):
                   selfcal_library[target][band][vis]['baseband'][baseband]['bwarray'][s]=spw_bandwidths_dict[vis][spw]
                   selfcal_library[target][band][vis]['baseband'][baseband]['eff_bwarray'][s]=spw_effective_bandwidths_dict[vis][spw]
                   selfcal_library[target][band][vis]['baseband'][baseband]['freq_array'][s]=spw_freqs_dict[vis][spw]
                   selfcal_library[target][band][vis]['baseband'][baseband]['total_bandwidth']+=spw_bandwidths_dict[vis][spw]
                   selfcal_library[target][band][vis]['baseband'][baseband]['total_effective_bandwidth']+=spw_bandwidths_dict[vis][spw]
          for fid in selfcal_library[target][band]['sub-fields']:
              selfcal_library[target][band][fid]['per_spw_stats']={}
              selfcal_library[target][band][fid]['spw_map'] = selfcal_library[target][band]['spw_map']
              for vis in selfcal_library[target][band][fid]['vislist']:
                  selfcal_library[target][band][fid][vis]['per_spw_stats'] = {}

                  spw_bandwidths_dict[vis],spw_effective_bandwidths_dict[vis],spw_freqs_dict[vis]=get_spw_bandwidth(vis,spwsarray_dict,target,vislist)

                  selfcal_library[target][band][fid][vis]['total_bandwidth']=0.0
                  selfcal_library[target][band][fid][vis]['total_effective_bandwidth']=0.0
                  if len(spw_effective_bandwidths_dict[vis].keys()) != len(spw_bandwidths_dict[vis].keys()):
                     print('cont.dat does not contain all spws; falling back to total bandwidth')
                     for spw in spw_bandwidths_dict[vis].keys():
                        if spw not in spw_effective_bandwidths[vis].keys():
                           spw_effective_bandwidths_dict[vis][spw]=spw_bandwidths[spw]
                  for spw in selfcal_library[target][band][fid][vis]['spwlist']:
                     keylist=selfcal_library[target][band][fid][vis]['per_spw_stats'].keys()
                     if spw not in keylist:
                        selfcal_library[target][band][fid][vis]['per_spw_stats'][spw]={}
                     selfcal_library[target][band][fid][vis]['per_spw_stats'][spw]['effective_bandwidth']=spw_effective_bandwidths_dict[vis][spw]
                     selfcal_library[target][band][fid][vis]['per_spw_stats'][spw]['bandwidth']=spw_bandwidths_dict[vis][spw]
                     selfcal_library[target][band][fid][vis]['per_spw_stats'][spw]['frequency']=spw_freqs_dict[vis][spw]
                     selfcal_library[target][band][fid][vis]['total_bandwidth']+=spw_bandwidths_dict[vis][spw]
                     selfcal_library[target][band][fid][vis]['total_effective_bandwidth']+=spw_effective_bandwidths_dict[vis][spw]

                  for baseband in selfcal_library[target][band][fid][vis]['baseband'].keys():
                     selfcal_library[target][band][fid][vis]['baseband'][baseband]['total_bandwidth']=0.0
                     selfcal_library[target][band][fid][vis]['baseband'][baseband]['total_effective_bandwidth']=0.0
                     selfcal_library[target][band][fid][vis]['baseband'][baseband]['freq_array']=np.zeros(len(selfcal_library[target][band][fid][vis]['baseband'][baseband]['spwarray']))
                     for s, spw in enumerate(selfcal_library[target][band][fid][vis]['baseband'][baseband]['spwarray']):
                        selfcal_library[target][band][vis]['baseband'][baseband]['bwarray'][s]=spw_bandwidths_dict[vis][spw]
                        selfcal_library[target][band][vis]['baseband'][baseband]['eff_bwarray'][s]=spw_effective_bandwidths_dict[vis][spw]
                        selfcal_library[target][band][vis]['baseband'][baseband]['freq_array'][s]=spw_freqs_dict[vis][spw]
                        selfcal_library[target][band][fid][vis]['baseband'][baseband]['total_bandwidth']+=spw_bandwidths_dict[vis][spw]
                        selfcal_library[target][band][fid][vis]['baseband'][baseband]['total_effective_bandwidth']+=spw_bandwidths_dict[vis][spw]



    ##
    ## set image parameters based on the visibility data properties and frequency
    ##
    #cellsize={}
    #imsize={}
    #nterms={}
    #applycal_interp={}

    for target in selfcal_library:
        #cellsize[target], imsize[target], nterms[target], applycal_interp[target] = {}, {}, {}, {}
        for band in selfcal_library[target]:
           selfcal_library[target][band]['cellsize'],selfcal_library[target][band]['imsize'],selfcal_library[target][band]['nterms'] = \
                   get_image_parameters(selfcal_library[target][band]['vislist'],telescope,target,band, \
                   selfcal_library,scale_fov=scale_fov,mosaic=selfcal_library[target][band]['obstype']=='mosaic')

           if selfcal_library[target][band]['meanfreq'] >12.0e9:
              selfcal_library[target][band]['applycal_interp']='linearPD'
           else:
              selfcal_library[target][band]['applycal_interp']='linear'

           for fid in selfcal_library[target][band]['sub-fields']:
               selfcal_library[target][band][fid]['nterms']=selfcal_library[target][band]['nterms']

           if "VLA" in telescope or (selfcal_library[target][band]['obstype'] == 'mosaic' and \
                   selfcal_library[target][band]['Median_scan_time'] / selfcal_library[target][band]['Median_fields_per_scan'] < 60.) \
                   or selfcal_library[target][band]['75thpct_uv'] > 2000.0:
               selfcal_library[target][band]['cyclefactor'] = 3.0
           else:
               selfcal_library[target][band]['cyclefactor'] = 1.0

    ##
    ## finds solints, starting with inf, ending with int, and tries to align
    ## solints with number of integrations
    ## solints reduce by factor of 2 in each self-cal interation
    ## e.g., inf, max_scan_time/2.0, prev_solint/2.0, ..., int
    ## starting solints will have solint the length of the entire EB to correct bulk offsets
    ##
    selfcal_plan = {}
    for target in all_targets:
       selfcal_plan[target] = {}

       for band in bands:
          selfcal_plan[target][band] = {}
          if band in selfcal_library[target]:
             selfcal_plan[target][band]['solints'],selfcal_plan[target][band]['integration_time'],selfcal_plan[target][band]['gaincal_combine'], \
                    selfcal_plan[target][band]['solmode']=get_solints_simple(selfcal_library[target][band]['vislist'],scantimesdict[band],
                    scannfieldsdict[band],scanstartsdict[band],scanendsdict[band],integrationtimesdict[band],\
                    inf_EB_gaincal_combine,do_amp_selfcal=do_amp_selfcal,mosaic=selfcal_library[target][band]['obstype'] == 'mosaic')
             print(band,target,selfcal_plan[target][band]['solints'])
             selfcal_plan[target][band]['applycal_mode']=[apply_cal_mode_default]*len(selfcal_plan[target][band]['solints'])

    ##
    ## estimate per scan/EB S/N using time on source and median scan times
    ##

    for target in selfcal_plan:
     for band in selfcal_plan[target]:
       for vis in selfcal_library[target][band]['vislist']:
        selfcal_plan[target][band][vis] = {}
        selfcal_plan[target][band][vis]['inf_EB_gaincal_combine']=inf_EB_gaincal_combine #'scan'
        if selfcal_library[target][band]['obstype']=='mosaic':
           selfcal_plan[target][band][vis]['inf_EB_gaincal_combine']+=',field'   
        selfcal_plan[target][band][vis]['inf_EB_gaintype']=inf_EB_gaintype #G
        selfcal_plan[target][band][vis]['inf_EB_fallback_mode']='' #'scan'

    return selfcal_library, selfcal_plan, gaincalibrator_dict

def plan_selfcal_per_solint(selfcal_library, selfcal_plan):
   # there are some extra keys in this dictionary that stem from how my thinking was orginally and how it evolved
   # the current philosophy is that for each solint it will specify how to apply each gain table, and if it should 
   # be pre-applied for gaincal solves. Then in gaincal wrapper, the parameters for preapplying all tables and for applying
   # them are accumulated together for the execution of gaincal and applycal.
   #
   # The accumulated parameters for gaincal and applycal are then stored in the selfcal_library for each solution interval
   #
   #
   # 
   for target in selfcal_library.keys():
      for band in selfcal_library[target].keys():
         for vis in selfcal_library[target][band]['vislist']:
             maxspws_per_bb=0
             if selfcal_library[target][band]['meanfreq'] > 12.0e9:
                applycal_interp='linearPD'
             else:
                applycal_interp='linear'
             for baseband in selfcal_library[target][band][vis]['baseband'].keys():
                if selfcal_library[target][band][vis]['baseband'][baseband]['nspws']> maxspws_per_bb:
                   maxspws_per_bb=selfcal_library[target][band][vis]['baseband'][baseband]['nspws']+0.0

             selfcal_plan[target][band][vis]['solint_settings']={}
             for solint in selfcal_plan[target][band]['solints']:
                gaincal_combine=''
                filename_append=''
                selfcal_plan[target][band][vis]['solint_settings'][solint]={}
                selfcal_plan[target][band][vis]['solint_settings'][solint]['preapply_this_gaintable']=False
                selfcal_plan[target][band][vis]['solint_settings'][solint]['gaincal_preapply_gaintable']=[]
                selfcal_plan[target][band][vis]['solint_settings'][solint]['gaincal_spwmap']=[]
                selfcal_plan[target][band][vis]['solint_settings'][solint]['gaincal_combine']={}
                selfcal_plan[target][band][vis]['solint_settings'][solint]['filename_append']={}
                selfcal_plan[target][band][vis]['solint_settings'][solint]['gaincal_interpolate']=[]
                selfcal_plan[target][band][vis]['solint_settings'][solint]['applycal_gaintable']=[]
                selfcal_plan[target][band][vis]['solint_settings'][solint]['applycal_spwmap']=[]
                selfcal_plan[target][band][vis]['solint_settings'][solint]['spwmap_for_mode']={}
                selfcal_plan[target][band][vis]['solint_settings'][solint]['applycal_interpolate']=applycal_interp
                selfcal_plan[target][band][vis]['solint_settings'][solint]['final_mode']=''
                selfcal_plan[target][band][vis]['solint_settings'][solint]['accepted_gaintable']=''
                selfcal_plan[target][band][vis]['solint_settings'][solint]['modes_to_attempt']=[]
                selfcal_plan[target][band][vis]['solint_settings'][solint]['gaincal_gaintype']='T'
                min_SNR_spw=get_min_SNR_spw(selfcal_plan[target][band]['solint_snr_per_spw'][solint])
                min_SNR_bb=get_min_SNR_spw(selfcal_plan[target][band]['solint_snr_per_bb'][solint])
                selfcal_plan[target][band][vis]['solint_settings'][solint]['modes_to_attempt'].append('combinespw')
                if 'spw' not in selfcal_plan[target][band][vis]['inf_EB_gaincal_combine']:
                    if min_SNR_spw > 2.0 : 
                       selfcal_plan[target][band][vis]['solint_settings'][solint]['modes_to_attempt'].append('per_spw')
                       #selfcal_plan[target][band][vis]['solint_settings'][solint]['preapply_this_gaintable']=True    # leave default to off and have it decide after eval
                    if min_SNR_bb > 2.0 and maxspws_per_bb > 1.0:  # only do the per baseband solutions if there are more than 1
                       selfcal_plan[target][band][vis]['solint_settings'][solint]['modes_to_attempt'].append('per_bb')
                       #selfcal_plan[target][band][vis]['solint_settings'][solint]['preapply_this_gaintable']=True    # leave default to off and have it decide after eval
                    if '_ap' in solint:
                       selfcal_plan[target][band][vis]['solint_settings'][solint]['solmode']='ap'
                    else:
                       selfcal_plan[target][band][vis]['solint_settings'][solint]['solmode']='p'
                    if solint == 'inf_EB':
                       selfcal_plan[target][band][vis]['solint_settings'][solint]['gaincal_gaintype']='G'
                for mode in selfcal_plan[target][band][vis]['solint_settings'][solint]['modes_to_attempt']:
                    if mode =='combinespw':
                       gaincal_combine+='spw'
                       filename_append='combinespw'
                       selfcal_plan[target][band][vis]['solint_settings'][solint]['spwmap_for_mode']['combinespw']=selfcal_library[target][band][vis]['spwmap']
                    if mode == 'per_spw':
                       gaincal_combine=''
                       filename_append='per_spw'
                       selfcal_plan[target][band][vis]['solint_settings'][solint]['spwmap_for_mode']['per_spw']=[]
                    if mode == 'per_bb':
                       gaincal_combine='spw'
                       filename_append='per_bb'
                       selfcal_plan[target][band][vis]['solint_settings'][solint]['spwmap_for_mode']['combinespw']=selfcal_library[target][band][vis]['baseband_spwmap']
                    if 'inf_EB' in solint:
                       if gaincal_combine!='':
                          gaincal_combine+=','
                       gaincal_combine+='scan'
                    selfcal_plan[target][band][vis]['solint_settings'][solint]['gaincal_combine'][mode]=gaincal_combine
                    selfcal_plan[target][band][vis]['solint_settings'][solint]['filename_append'][mode]=filename_append
                      
                        
                      
                  

            
            #for fid in selfcal_library[target][band]['sub-fields']:
            #   selfcal_plan[target][band][vis][fid]['solint_settings'][solint]['gaincal_preapply_gaintable']={}
            #   selfcal_plan[target][band][vis][fid]['solint_settings'][solint]['gaincal_spwmap']={}
            #   selfcal_plan[target][band][vis][fid]['solint_settings'][solint]['gaincal_interpolate']={}
            #   selfcal_plan[target][band][vis][fid]['solint_settings'][solint]['applycal_gaintable']={}
            #   selfcal_plan[target][band][vis][fid]['solint_settings'][solint]['applycal_spwmap']={}
            #   selfcal_plan[target][band][vis][fid]['solint_settings'][solint]['applycal_interpolate']={}
            #   selfcal_plan[target][band][vis][fid]['solint_settings'][solint]['fallback']={}
            #   selfcal_plan[target][band][vis][fid]['solint_settings'][solint]['modes_to_attempt']={}
            

def set_clean_thresholds(selfcal_library, selfcal_plan, dividing_factor=-99.0, rel_thresh_scaling='log10', telescope='ALMA'):
    for target in selfcal_library:
        for band in selfcal_library[target].keys():
            if selfcal_library[target][band]['meanfreq'] <8.0e9 and (dividing_factor ==-99.0):
               dividing_factor_band=40.0
            elif (dividing_factor ==-99.0):
               dividing_factor_band=15.0

            # restricts initial nsigma to be at least 5
            nsigma_init=np.max([selfcal_library[target][band]['SNR_NF_orig']/dividing_factor_band,5.0])

            # count number of amplitude selfcal solints, repeat final clean depth of phase-only for amplitude selfcal
            n_ap_solints=sum(1 for solint in selfcal_plan[target][band]['solints'] if 'ap' in solint)

            if rel_thresh_scaling == 'loge':
                selfcal_library[target][band]['nsigma'] = np.append(np.exp(np.linspace(np.log(nsigma_init),np.log(3.0),\
                        len(selfcal_plan[target][band]['solints'])-n_ap_solints)),np.array([np.exp(np.log(3.0))]*n_ap_solints))
            elif rel_thresh_scaling == 'linear':
                selfcal_library[target][band]['nsigma'] = np.append(np.linspace(nsigma_init,3.0,len(selfcal_plan[target][band]['solints'])-\
                        n_ap_solints),np.array([3.0]*n_ap_solints))
            else: #implicitly making log10 the default
                selfcal_library[target][band]['nsigma'] = np.append(10**np.linspace(np.log10(nsigma_init),np.log10(3.0),\
                        len(selfcal_plan[target][band]['solints'])-n_ap_solints),np.array([10**(np.log10(3.0))]*n_ap_solints))

            if telescope=='ALMA' or telescope =='ACA': #or ('VLA' in telescope) 
               sensitivity=get_sensitivity(selfcal_library[target][band]['vislist'],selfcal_library[target][band],target,virtual_spw='all',\
                       imsize=selfcal_library[target][band]['imsize'],cellsize=selfcal_library[target][band]['cellsize'])
               if band =='Band_9' or band == 'Band_10':   # adjust for DSB noise increase
                  sensitivity=sensitivity*4.0 
               if ('VLA' in telescope):
                  sensitivity=sensitivity*0.0 # empirical correction, VLA estimates for sensitivity have tended to be a factor of ~3 low
            else:
               sensitivity=0.0
            selfcal_library[target][band]['thresholds']=selfcal_library[target][band]['nsigma']*sensitivity

