import numpy as np
import numpy 
import scipy.stats
import scipy.signal
import math
import os
import glob
import sys

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

from .selfcal_helpers import *

def generate_weblog(sclib,selfcal_plan,directory='weblog'):
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
   bands = np.unique(np.concatenate([list(sclib[target].keys()) for target in sclib])).tolist()
   bands_string=', '.join([str(elem) for elem in bands])
   htmlOut.writelines(''+bands_string+'\n')
   htmlOut.writelines('<h2>Solints to Attempt:</h2>\n')
   for target in selfcal_plan:
       for band in selfcal_plan[target]:
          solints_string=', '.join([str(elem) for elem in selfcal_plan[target][band]['solints']])
          htmlOut.writelines('<br>'+target+', '+band+': '+solints_string)

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
         if 'Empty model' not in sclib[target][band]['Stop_Reason']:
            render_selfcal_solint_summary_table(htmlOut,sclib,target,band,selfcal_plan)

         # PER SPW STATS TABLE
         if 'per_spw_stats' in sclib[target][band].keys():
            render_spw_stats_summary_table(htmlOut,sclib,target,band)

   # Close main weblog file
   htmlOut.writelines('</body>\n')
   htmlOut.writelines('</html>\n')
   htmlOut.close()
   
   # Pages for each solint
   if 'Empty model' not in sclib[target][band]['Stop_Reason']:
      render_per_solint_QA_pages(sclib,selfcal_plan,bands,directory=directory)
 

def render_summary_table(htmlOut,sclib,target,band,directory='weblog'):
         plot_image(sanitize_string(target)+'_'+band+'_final.image.tt0',\
                      directory+'/images/'+sanitize_string(target)+'_'+band+'_final.image.tt0.png', zoom=1.0 if directory=="weblog" else 1)
         image_stats=imstat(sanitize_string(target)+'_'+band+'_final.image.tt0')
         
         plot_image(sanitize_string(target)+'_'+band+'_initial.image.tt0',\
                      directory+'/images/'+sanitize_string(target)+'_'+band+'_initial.image.tt0.png',min_val=image_stats['min'][0],max_val=image_stats['max'][0], zoom=1.0 if directory=="weblog" else 1) 
         os.system('rm -rf '+sanitize_string(target)+'_'+band+'_final_initial_div_final.image.tt0 '+sanitize_string(target)+'_'+band+'_final_initial_div_final.temp.image.tt0')

         ### Hacky way to suppress stuff outside mask in ratio images.
         immath(imagename=[sanitize_string(target)+'_'+band+'_final.image.tt0',sanitize_string(target)+'_'+band+'_initial.image.tt0',sanitize_string(target)+'_'+band+'_final.mask'],\
                mode='evalexpr',expr='((IM0-IM1)/IM0)*IM2',outfile=sanitize_string(target)+'_'+band+'_final_initial_div_final.temp.image.tt0')
         immath(imagename=[sanitize_string(target)+'_'+band+'_final_initial_div_final.temp.image.tt0'],\
                mode='evalexpr',expr='iif(IM0==0.0,-99.0,IM0)',outfile=sanitize_string(target)+'_'+band+'_final_initial_div_final.image.tt0')
         plot_image(sanitize_string(target)+'_'+band+'_final_initial_div_final.image.tt0',\
                      directory+'/images/'+sanitize_string(target)+'_'+band+'_final_initial_div_final.image.tt0.png',\
                       min_val=-1.0,max_val=1.0, zoom=1.0 if directory=="weblog" else 1) 
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

def render_selfcal_solint_summary_table(htmlOut,sclib,target,band,selfcal_plan):
         #  SELFCAL SUMMARY TABLE   
         vislist=sclib[target][band]['vislist']
         solint_list=selfcal_plan[target][band]['solints']
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
                    if key in sclib[target][band][vislist[len(vislist)-1]][solint]:
                     if sclib[target][band][vislist[len(vislist)-1]][solint]['Pass'] == False:
                        line+='    <td><font color="red">{}</font> {}</td>\n'.format('Fail',sclib[target][band][vislist[len(vislist)-1]][solint]['Fail_Reason'])
                     elif sclib[target][band][vislist[len(vislist)-1]][solint]['Pass'] == 'None':
                        line+='    <td><font color="green">{}</font> {}</td>\n'.format('Not attempted',sclib[target][band][vislist[len(vislist)-1]][solint]['Fail_Reason'])
                     else:
                        line+='    <td><font color="blue">{}</font></td>\n'.format('Pass')
                    else:
                        line+='    <td><font color="green">{}</font></td>\n'.format('None')
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
            for quantity in ['Nsols_with_preflagged_data','Flagged_Sols_with_preflagged_data','Frac_Flagged_with_preflagged_data','Nsols_without_preflagged_data','Flagged_Sols_without_preflagged_data','Frac_Flagged_without_preflagged_data','Frac_sols_not_equiv_zero_per_spw','Frac_sols_not_equiv_zero_per_bb','SPW_Combine_Mode']:
               line='<tr bgcolor="#ffffff">\n    <td>'+quantity+'</td>\n'
               for solint in solint_list:
                  if solint in vis_keys and sclib[target][band][vis][solint]['Pass'] != 'None' and 'gaintable' in sclib[target][band][vis][solint]:
                     # only evaluate last gaintable not the pre-apply table
                     #gaintable=sclib[target][band][vis][solint]['gaintable'][len(sclib[target][band][vis][solint]['gaintable'])-1]
                     #nflagged_sols, nsols=get_sols_flagged_solns(gaintable)
                     final_mode=selfcal_plan[target][band][vis]['solint_settings'][solint]['final_mode']
                     nflagged_sols_total=np.sum(selfcal_plan[target][band][vis]['solint_settings'][solint]['nflags'][final_mode])
                     nsols_total=np.sum(selfcal_plan[target][band][vis]['solint_settings'][solint]['ntotal'][final_mode])
                     nflagged_sols=np.sum(selfcal_plan[target][band][vis]['solint_settings'][solint]['nflags_non_apriori'][final_mode])
                     nsols=np.sum(selfcal_plan[target][band][vis]['solint_settings'][solint]['ntotal_non_apriori'][final_mode])
                     #nflagged_sols, nsols=get_sols_flagged_solns(gaintable)
                     frac_flagged_sols=nflagged_sols/nsols
                     frac_flagged_sols_total=nflagged_sols_total/nsols_total
                     if 'per_spw' in selfcal_plan[target][band][vis]['solint_settings'][solint]['non_zero_fraction'].keys():
                        nonzero_frac_per_spw='{:0.3f}'.format(selfcal_plan[target][band][vis]['solint_settings'][solint]['non_zero_fraction']['per_spw'])
                     else:
                        nonzero_frac_per_spw='...'
                     if 'per_bb' in selfcal_plan[target][band][vis]['solint_settings'][solint]['non_zero_fraction'].keys():
                        nonzero_frac_per_bb='{:0.3f}'.format(selfcal_plan[target][band][vis]['solint_settings'][solint]['non_zero_fraction']['per_bb'])
                     else:
                        nonzero_frac_per_bb='...'
                     if quantity =='Nsols_with_preflagged_data':
                        line+='<td>'+str(nsols_total)+'</td>\n'
                     if quantity =='Flagged_Sols_with_preflagged_data':
                        line+='<td>'+str(nflagged_sols_total)+'</td>\n'
                     if quantity =='Frac_Flagged_with_preflagged_data':
                        if nsols_total > 0:
                           line+='<td>'+'{:0.3f}'.format(frac_flagged_sols_total)+'</td>\n'
                        else:
                           line+='<td>...</td>\n'
                     if quantity =='Nsols_without_preflagged_data':
                        line+='<td>'+str(nsols)+'</td>\n'
                     if quantity =='Flagged_Sols_without_preflagged_data':
                        line+='<td>'+str(nflagged_sols)+'</td>\n'
                     if quantity =='Frac_Flagged_without_preflagged_data':
                        if nsols > 0:
                           line+='<td>'+'{:0.3f}'.format(frac_flagged_sols)+'</td>\n'
                        else:
                           line+='<td>...</td>\n'
                     if quantity =='Frac_sols_not_equiv_zero_per_spw':
                        line+='<td>'+nonzero_frac_per_spw+'</td>\n'
                     if quantity =='Frac_sols_not_equiv_zero_per_bb':
                        line+='<td>'+nonzero_frac_per_bb+'</td>\n'

                     if quantity =='SPW_Combine_Mode':
                        solint_index=selfcal_plan[target][band]['solints'].index(solint)
                        if sclib[target][band][vis][solint]['final_mode'] == 'combinespw' or \
                                sclib[target][band][vis][solint]['final_mode'] == 'combinespw_fallback':
                           gc_combine_mode='Combine SPW'
                        if sclib[target][band][vis][solint]['final_mode'] == 'combinespwpol':
                           gc_combine_mode='Combine SPW & Pol'
                        if sclib[target][band][vis][solint]['final_mode'] == 'per_bb':
                           gc_combine_mode='Per Baseband'
                        if sclib[target][band][vis][solint]['final_mode'] == 'per_spw':
                           gc_combine_mode='Per SPW'
                        if 'fallback' in sclib[target][band][vis][solint].keys() and sclib[target][band][vis][solint]['fallback']=='spwmap':
                           gc_combine_mode='Per SPW + SPW Mapping'
                        line+='<td><font color="red">'+gc_combine_mode+'</font></td>\n'
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

def render_per_solint_QA_pages(sclib,selfcal_plan,bands,directory='weblog'):
  ## Per Solint pages
   targets=list(sclib.keys())
   for target in targets:
      bands_obsd=list(sclib[target].keys())
      for band in bands_obsd:
         if sclib[target][band]['final_solint'] == 'None':
            final_solint_index=0
         else:
            final_solint_index=selfcal_plan[target][band]['solints'].index(sclib[target][band]['final_solint']) 

         vislist=sclib[target][band]['vislist']
         index_addition=1
         if sclib[target][band]['final_solint'] != selfcal_plan[target][band]['solints'][-1] and sclib[target][band]['final_solint'] != 'None':
            index_addition=2
         # if it's a dataset where inf_EB == inf, make sure to take out the assumption that there would be an 'inf' solution
         if 'inf' not in selfcal_plan[target][band]['solints']:
            index_addition=index_addition-1
         #add an additional check to make sure the final index will be in the array
         if final_solint_index+index_addition-1 > (len(selfcal_plan[target][band]['solints'])-1):
            index_addition=(final_solint_index+index_addition-1) - (len(selfcal_plan[target][band]['solints'])-1)
         

         final_solint_to_plot=selfcal_plan[target][band]['solints'][final_solint_index+index_addition-1]
         keylist=sclib[target][band][vislist[0]].keys()
         if index_addition == 2 and final_solint_to_plot not in keylist:
           index_addition=index_addition-1


         
         #for i in range(final_solint_index+index_addition):
         for i in range(len(selfcal_plan[target][band]['solints'])):

            if selfcal_plan[target][band]['solints'][i] not in keylist or sclib[target][band][vislist[len(vislist)-1]][selfcal_plan[target][band]['solints'][i]]['Pass'] == 'None':
               continue
            htmlOutSolint=open(directory+'/'+target+'_'+band+'_'+selfcal_plan[target][band]['solints'][i]+'.html','w')
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
               if selfcal_plan[target][band]['solints'][j] not in keylist:
                  continue
               solints_string+='<a href="'+target+'_'+band+'_'+selfcal_plan[target][band]['solints'][j]+'.html">'+selfcal_plan[target][band]['solints'][j]+'  </a><br>\n'
            htmlOutSolint.writelines('<br>Solints: '+solints_string)

            htmlOutSolint.writelines('<h3>Solint: '+selfcal_plan[target][band]['solints'][i]+'</h3>\n')       
            keylist_top=sclib[target][band].keys()
            htmlOutSolint.writelines('<a href="index.html#'+target+'_'+band+'">Back to Main Target/Band</a><br>\n')


            #must select last key for pre Jan 14th runs since they only wrote pass to the last MS dictionary entry
            if "Pass" in sclib[target][band][vislist[len(vislist)-1]][selfcal_plan[target][band]['solints'][i]]:
                passed=sclib[target][band][vislist[len(vislist)-1]][selfcal_plan[target][band]['solints'][i]]['Pass']
            else:
                passed = 'None'

            '''
            if (i > final_solint_index) or ('Estimated_SNR_too_low_for_solint' not in sclib[target][band]['Stop_Reason']):
               htmlOut.writelines('<h4>Passed: <font color="red">False</font></h4>\n')
            elif 'Stop_Reason' in keylist_top:
               if (i == final_solint_index) and ('Estimated_SNR_too_low_for_solint' not in sclib[target][band]['Stop_Reason']):
                    htmlOut.writelines('<h4>Passed: <font color="red">False</font></h4>\n') 
            else:
               htmlOut.writelines('<h4>Passed: <font color="blue">True</font></h4>\n')
            '''
            if passed == 'None':
               htmlOutSolint.writelines('<h4>Passed: <font color="green">N/A</font></h4>\n')
            elif passed:
               htmlOutSolint.writelines('<h4>Passed: <font color="blue">True</font></h4>\n')
            else:
               htmlOutSolint.writelines('<h4>Passed: <font color="red">False</font></h4>\n')
            if 'Empty model' in sclib[target][band]['Stop_Reason']:
               htmlOutSolint.writelines('Empty model image, no gains solved<br>\n')
               htmlOutSolint.writelines('</body>\n')
               htmlOutSolint.writelines('</html>\n')
               htmlOutSolint.close()
               continue
            htmlOutSolint.writelines('Pre and Post Selfcal images with scales set to Post image<br>\n')
            plot_image(sanitize_string(target)+'_'+band+'_'+selfcal_plan[target][band]['solints'][i]+'_'+str(i)+'_post.image.tt0',\
                      directory+'/images/'+sanitize_string(target)+'_'+band+'_'+selfcal_plan[target][band]['solints'][i]+'_'+str(i)+'_post.image.tt0.png', \
                      zoom=1.0 if directory=="weblog" else 1) 
            image_stats=imstat(sanitize_string(target)+'_'+band+'_'+selfcal_plan[target][band]['solints'][i]+'_'+str(i)+'_post.image.tt0')
            plot_image(sanitize_string(target)+'_'+band+'_'+selfcal_plan[target][band]['solints'][i]+'_'+str(i)+'.image.tt0',\
                      directory+'/images/'+sanitize_string(target)+'_'+band+'_'+selfcal_plan[target][band]['solints'][i]+'_'+str(i)+'.image.tt0.png',min_val=image_stats['min'][0],max_val=image_stats['max'][0], \
                      zoom=1.0 if directory=="weblog" else 1) 

            htmlOutSolint.writelines('<a href="images/'+sanitize_string(target)+'_'+band+'_'+selfcal_plan[target][band]['solints'][i]+'_'+str(i)+'.image.tt0.png"><img src="images/'+sanitize_string(target)+'_'+band+'_'+selfcal_plan[target][band]['solints'][i]+'_'+str(i)+'.image.tt0.png" ALT="pre-SC-solint image" WIDTH=400 HEIGHT=400></a>\n')
            htmlOutSolint.writelines('<a href="images/'+sanitize_string(target)+'_'+band+'_'+selfcal_plan[target][band]['solints'][i]+'_'+str(i)+'_post.image.tt0.png"><img src="images/'+sanitize_string(target)+'_'+band+'_'+selfcal_plan[target][band]['solints'][i]+'_'+str(i)+'_post.image.tt0.png" ALT="pre-SC-solint image" WIDTH=400 HEIGHT=400></a><br>\n')
            htmlOutSolint.writelines('Post SC SNR: {:0.3f}'.format(sclib[target][band][vislist[0]][selfcal_plan[target][band]['solints'][i]]['SNR_post'])+'<br>Pre SC SNR: {:0.3f}'.format(sclib[target][band][vislist[0]][selfcal_plan[target][band]['solints'][i]]['SNR_pre'])+'<br><br>\n')
            htmlOutSolint.writelines('Post SC RMS: {:0.7f}'.format(sclib[target][band][vislist[0]][selfcal_plan[target][band]['solints'][i]]['RMS_post'])+' Jy/beam<br>Pre SC RMS: {:0.7f}'.format(sclib[target][band][vislist[0]][selfcal_plan[target][band]['solints'][i]]['RMS_pre'])+' Jy/beam<br>\n')
            htmlOutSolint.writelines('Post Beam: {:0.3f}"x{:0.3f}" {:0.3f} deg'.format(sclib[target][band][vislist[0]][selfcal_plan[target][band]['solints'][i]]['Beam_major_post'],sclib[target][band][vislist[0]][selfcal_plan[target][band]['solints'][i]]['Beam_minor_post'],sclib[target][band][vislist[0]][selfcal_plan[target][band]['solints'][i]]['Beam_PA_post'])+'<br>\n')
            htmlOutSolint.writelines('Pre Beam: {:0.3f}"x{:0.3f}" {:0.3f} deg'.format(sclib[target][band][vislist[0]][selfcal_plan[target][band]['solints'][i]]['Beam_major_pre'],sclib[target][band][vislist[0]][selfcal_plan[target][band]['solints'][i]]['Beam_minor_pre'],sclib[target][band][vislist[0]][selfcal_plan[target][band]['solints'][i]]['Beam_PA_pre'])+'<br><br>\n')


            if 'inf_EB' in selfcal_plan[target][band]['solints'][i]:
               htmlOutSolint.writelines('<h3>Phase vs. Frequency Plots:</h3>\n')
            else:
               htmlOutSolint.writelines('<h3>Phase vs. Time Plots:</h3>\n')
            for vis in vislist:
               htmlOutSolint.writelines('<h4>MS: '+vis+'</h4>\n')
               if 'gaintable' not in sclib[target][band][vis][selfcal_plan[target][band]['solints'][i]]:
                    htmlOutSolint.writelines('No gaintable available <br><br>')
                    continue
               ant_list=get_ant_list(vis)
               gaintable=sclib[target][band][vis][selfcal_plan[target][band]['solints'][i]]['gaintable'][len(sclib[target][band][vis][selfcal_plan[target][band]['solints'][i]]['gaintable'])-1]
               print('******************'+gaintable+'***************')
               final_mode=selfcal_plan[target][band][vis]['solint_settings'][selfcal_plan[target][band]['solints'][i]]['final_mode']

               nflagged_sols_total=np.sum(selfcal_plan[target][band][vis]['solint_settings'][selfcal_plan[target][band]['solints'][i]]['nflags'][final_mode])
               nsols_total=np.sum(selfcal_plan[target][band][vis]['solint_settings'][selfcal_plan[target][band]['solints'][i]]['ntotal'][final_mode])
               nflagged_sols=np.sum(selfcal_plan[target][band][vis]['solint_settings'][selfcal_plan[target][band]['solints'][i]]['nflags_non_apriori'][final_mode])
               nsols=np.sum(selfcal_plan[target][band][vis]['solint_settings'][selfcal_plan[target][band]['solints'][i]]['ntotal_non_apriori'][final_mode])
               #nflagged_sols, nsols=get_sols_flagged_solns(gaintable)
               frac_flagged_sols=nflagged_sols/nsols
               frac_flagged_sols_total=nflagged_sols_total/nsols_total
               #plot_ants_flagging_colored(directory+'/images/plot_ants_'+gaintable+'.png',vis,gaintable)
               plot_ants_flagging_colored_from_dict(directory+'/images/plot_ants_'+gaintable+'.png',sclib[target][band][vis],selfcal_plan[target][band][vis],selfcal_plan[target][band]['solints'][i],final_mode,vis)
               htmlOutSolint.writelines('<a href="images/plot_ants_'+gaintable+'.png"><img src="images/plot_ants_'+gaintable+'.png" ALT="antenna positions with flagging plot" WIDTH=400 HEIGHT=400></a>')
               if 'unflagged_lbs' in sclib[target][band][vis][selfcal_plan[target][band]['solints'][i]]:
                   unflag_failed_antennas(vis, gaintable.replace('.g','.pre-pass.g'), \
                           selfcal_plan[target][band][vis]['solint_settings'][selfcal_plan[target][band]['solints'][i]]['gaincal_return_dict'][sclib[target][band][vis][selfcal_plan[target][band]['solints'][i]]['final_mode']], sclib[target][band]['telescope'], \
                           flagged_fraction=0.25, \
                           spwmap=sclib[target][band][vis][selfcal_plan[target][band]['solints'][i]]['unflag_spwmap'], \
                           plot=True, plot_directory=directory+'/images/')
                   htmlOutSolint.writelines('\n<a href="images/'+gaintable.replace('.g.','.pre-pass.pass')+'.png"><img src="images/'+gaintable.replace('.g','.pre-pass.pass')+'.png" ALT="Long baseline unflagging thresholds" HEIGHT=400></a><br>\n')
               else:
                   htmlOutSolint.writelines('<br>\n')
               htmlOutSolint.writelines('Gain Solution Stats without pre-flagged data<br>')
               htmlOutSolint.writelines('N Gain solutions: {:0.0f}<br>'.format(nsols))
               htmlOutSolint.writelines('Flagged solutions: {:0.0f}<br>'.format(nflagged_sols))
               htmlOutSolint.writelines('Fraction Flagged Solutions: {:0.3f} <br><br>'.format(frac_flagged_sols))
               htmlOutSolint.writelines('Gain Solution Stats with pre-flagged data<br>')
               htmlOutSolint.writelines('N Gain solutions: {:0.0f}<br>'.format(nsols_total))
               htmlOutSolint.writelines('Flagged solutions: {:0.0f}<br>'.format(nflagged_sols_total))
               htmlOutSolint.writelines('Fraction Flagged Solutions: {:0.3f} <br><br>'.format(frac_flagged_sols_total))
               print(sclib[target][band][vis][selfcal_plan[target][band]['solints'][i]]['final_mode'])
               if sclib[target][band][vis][selfcal_plan[target][band]['solints'][i]]['final_mode'] == 'combinespw' or \
                       sclib[target][band][vis][selfcal_plan[target][band]['solints'][i]]['final_mode'] == 'combinespw_fallback':
                  gc_combine_mode='Combine SPW'
               if sclib[target][band][vis][selfcal_plan[target][band]['solints'][i]]['final_mode'] == 'combinespwpol':
                  gc_combine_mode='Combine SPW & Pol'
               if sclib[target][band][vis][selfcal_plan[target][band]['solints'][i]]['final_mode'] == 'per_bb':
                  gc_combine_mode='Per Baseband'
               if sclib[target][band][vis][selfcal_plan[target][band]['solints'][i]]['final_mode'] == 'per_spw':
                  gc_combine_mode='Per SPW'
               if 'fallback' in sclib[target][band][vis][selfcal_plan[target][band]['solints'][i]].keys() and sclib[target][band][vis][selfcal_plan[target][band]['solints'][i]]['fallback']=='spwmap':
                  gc_combine_mode='Per SPW + SPW Mapping'
               if 'fallback' in sclib[target][band][vis][selfcal_plan[target][band]['solints'][i]].keys() and sclib[target][band][vis][selfcal_plan[target][band]['solints'][i]]['fallback'] == 'combinespwpol':
                  gc_combine_mode='Combine SPW & Pol'
               htmlOutSolint.writelines('<h4>Gaincal Combine Mode: <font color="red">'+gc_combine_mode+'</font></h4>\n')
               htmlOutSolint.writelines('<h4>Applycal SPW Map: ['+' '.join(map(str,sclib[target][band][vis][selfcal_plan[target][band]['solints'][i]]['spwmap']))+']</h4>\n')


               for ant in ant_list:
                  sani_target=sanitize_string(target)
                  if 'inf_EB' in selfcal_plan[target][band]['solints'][i]:
                     xaxis='frequency'
                  else:
                     xaxis='time'
                  if 'delay' in selfcal_plan[target][band]['solints'][i]:
                     xaxis='frequency'
                     yaxis='delay'
                     plotrange=[0,0,0,0]
                  if 'ap' in selfcal_plan[target][band]['solints'][i]:
                     yaxis='amp'
                     plotrange=[0,0,0,2.0]
                  elif 'delay' not in selfcal_plan[target][band]['solints'][i]:
                     yaxis='phase'
                     plotrange=[0,0,-180,180]
                  try:
                     plotms(gridrows=2,plotindex=0,rowindex=0,vis=gaintable,xaxis=xaxis, yaxis=yaxis,showgui=False,\
                         xselfscale=True,plotrange=plotrange, antenna=ant,customflaggedsymbol=True,title=ant+' '+yaxis,\
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
