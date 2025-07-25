import numpy as np
from .selfcal_helpers import *

def evaluate_subfields_to_gaincal(selfcal_library, target, band, solint, iteration, solmode, solints, selfcal_plan, 
        minsnr_to_proceed, allow_gain_interpolation=False):

     sani_target=sanitize_string(target)

     # Fields that don't have any mask in the primary beam should be removed from consideration, as their models are likely bad.
     new_fields_to_selfcal = []
     for fid in selfcal_library['sub-fields-to-selfcal']:
         os.system('rm -rf test*.mask')
         tmp_SNR_NF,tmp_RMS_NF=estimate_near_field_SNR(sani_target+'_field_'+str(fid)+'_'+band+'_'+solint+'_'+str(iteration)+'.image.tt0', \
                 las=selfcal_library['LAS'], mosaic_sub_field=True, save_near_field_mask=False)

         immath(imagename=[sani_target+"_field_"+str(fid)+"_"+band+"_"+solint+"_"+str(iteration)+".image.tt0",\
                 sani_target+"_field_"+str(fid)+"_"+band+"_"+solint+"_"+str(iteration)+".pb.tt0",\
                 sani_target+"_field_"+str(fid)+"_"+band+"_"+solint+"_"+str(iteration)+".mospb.tt0"], outfile="test.mask", \
                 expr="IIF(IM0*IM1/IM2 > "+str(5*tmp_RMS_NF)+", 1., 0.)")

         bmaj = ''.join(np.array(list(imhead(sani_target+"_field_"+str(fid)+"_"+band+"_"+solint+"_"+str(iteration)+".image.tt0", \
                 mode="get", hdkey="bmaj").values())[::-1]).astype(str))
         bmin = ''.join(np.array(list(imhead(sani_target+"_field_"+str(fid)+"_"+band+"_"+solint+"_"+str(iteration)+".image.tt0", \
                 mode="get", hdkey="bmin").values())[::-1]).astype(str))
         bpa = ''.join(np.array(list(imhead(sani_target+"_field_"+str(fid)+"_"+band+"_"+solint+"_"+str(iteration)+".image.tt0", \
                 mode="get", hdkey="bpa").values())[::-1]).astype(str))

         imsmooth("test.mask", kernel="gauss", major=bmaj, minor=bmin, pa=bpa, outfile="test.smoothed.mask")

         immath(imagename=["test.smoothed.mask",sani_target+"_field_"+str(fid)+"_"+band+"_"+solint+"_"+str(iteration)+".mask"], \
                 outfile="test.smoothed.truncated.mask", expr="IIF(IM0 > 0.01 || IM1 > 0., 1., 0.)")

         original_intflux = get_intflux(sani_target+"_field_"+str(fid)+"_"+band+"_"+solint+"_"+str(iteration)+".image.tt0", \
                 rms=tmp_RMS_NF, maskname=sani_target+"_field_"+str(fid)+"_"+band+"_"+solint+"_"+str(iteration)+".mask", \
                 mosaic_sub_field=True)[0]
         updated_intflux = get_intflux(sani_target+"_field_"+str(fid)+"_"+band+"_"+solint+"_"+str(iteration)+".image.tt0", \
                 rms=tmp_RMS_NF, maskname="test.smoothed.truncated.mask", mosaic_sub_field=True)[0]
         os.system('rm -rf test*.mask')


         if not checkmask(sani_target+'_field_'+str(fid)+'_'+band+'_'+solint+'_'+str(iteration)+'.image.tt0'):
             print("Removing field "+str(fid)+" from gaincal because there is no signal within the primary beam.")
             skip_reason = "No signal"
         elif selfcal_plan[fid]['solint_snr_per_field'][solints[iteration]] < minsnr_to_proceed and \
                 solint not in ['inf_EB','scan_inf']:
             print("Removing field "+str(fid)+" from gaincal because the estimated solint snr is too low.")
             skip_reason = "Estimated SNR"
         elif updated_intflux > 1.25 * original_intflux:
             print("Removing field "+str(fid)+" from gaincal because there appears to be significant flux missing from the model.")
             skip_reason = "Missing flux"
         else:
             new_fields_to_selfcal.append(fid)

         if fid not in new_fields_to_selfcal and solint != "inf_EB" and not allow_gain_interpolation:
             for vis in selfcal_library[fid]['vislist']:
                 #selfcal_library[fid][vis][solint]['interpolated_gains'] = True
                 #selfcal_library[fid]['Stop_Reason'] = "Gaincal solutions would be interpolated"
                 selfcal_library[fid]['Stop_Reason'] = skip_reason
                 selfcal_library[fid][vis][solint]['Pass'] = "None"
                 selfcal_library[fid][vis][solint]['Fail_Reason'] = skip_reason

     return new_fields_to_selfcal




def evaluate_subfields_after_gaincal(selfcal_library, target, band, solint, iteration, solmode, allow_gain_interpolation=False):

     new_fields_to_selfcal = selfcal_library['sub-fields-to-selfcal'].copy()

     sani_target=sanitize_string(target)

     if selfcal_library['obstype'] == 'mosaic' and ((solint != "inf_EB" and not allow_gain_interpolation) or \
             (allow_gain_interpolation and "inf" not in solint)):
        # With gaincal done and bad fields removed from gain tables if necessary, check whether any fields should no longer be selfcal'd
        # because they have too much interpolation.
        for vis in selfcal_library['vislist']:
            ## If an EB had no fields to gaincal on, remove all fields in that EB from being selfcal'd as there is no calibration available
            ## in this EB.
            if np.intersect1d(selfcal_library['sub-fields-to-gaincal'],\
                    list(selfcal_library['sub-fields-fid_map'][vis].keys())).size == 0:
                for fid in np.intersect1d(new_fields_to_selfcal,list(selfcal_library['sub-fields-fid_map'][vis].keys())):
                    new_fields_to_selfcal.remove(fid)

                    selfcal_library[fid]['Stop_Reason'] = 'No viable calibrator fields in at least 1 EB'
                    for v in selfcal_library[fid]['vislist']:
                        selfcal_library[fid][v][solint]['Pass'] = 'None'
                        if 'Fail_Reason' in selfcal_library[fid][v][solint]:
                            selfcal_library[fid][v][solint]['Fail_Reason'] += '; '
                        else:
                            selfcal_library[fid][v][solint]['Fail_Reason'] = ''
                        selfcal_library[fid][v][solint]['Fail_Reason'] += 'No viable fields'
                continue
            ## NEXT TO DO: check % of flagged solutions - DONE, see above
            ## After that enable option for interpolation through inf - DONE
            tb.open(sani_target+'_'+vis+'_'+band+'_'+solint+'_'+str(iteration)+'_'+solmode[iteration]+'_'+
                    selfcal_library[vis][solint]['final_mode']+'.g')
            fields = tb.getcol("FIELD_ID")
            scans = tb.getcol("SCAN_NUMBER")

            for fid in np.intersect1d(new_fields_to_selfcal,list(selfcal_library['sub-fields-fid_map'][vis].keys())):
                if solint == "scan_inf":
                    msmd.open(vis)
                    scans_for_field = []
                    cals_for_scan = []
                    total_cals_for_scan = []
                    for incl_scan in selfcal_library[vis][solint]['include_scans']:
                        scans_array = np.array(incl_scan.split(",")).astype(int)
                        fields_for_scans = msmd.fieldsforscans(scans_array)

                        if selfcal_library['sub-fields-fid_map'][vis][fid] in fields_for_scans:
                            scans_for_field.append(np.intersect1d(scans_array, np.unique(scans)))
                            cals_for_scan.append((scans == scans_for_field[-1]).sum() if scans_for_field[-1] in scans else 0.)
                            #total_cals_for_scan.append(len(msmd.antennasforscan(scans_for_field[-1])))
                            total_cals_for_scan.append(len(msmd.antennanames()))

                    if sum(cals_for_scan) / sum(total_cals_for_scan) < 0.75:
                        new_fields_to_selfcal.remove(fid)

                    msmd.close()
                else:
                    if selfcal_library['sub-fields-fid_map'][vis][fid] not in fields:
                        new_fields_to_selfcal.remove(fid)

                if fid not in new_fields_to_selfcal:
                    # We need to update all the EBs, not just the one that failed.
                    for v in selfcal_library[fid]['vislist']:
                        selfcal_library[fid][v][solint]['Pass'] = 'None'
                        if allow_gain_interpolation:
                            selfcal_library[fid][v][solint]['Fail_Reason'] = 'Interpolation beyond inf'
                        else:
                            selfcal_library[fid][v][solint]['Fail_Reason'] = 'Bad gaincal solutions'


            tb.close()
     elif selfcal_library['obstype'] == 'mosaic' and solint == "inf_EB":
        ## If an EB had no fields to gaincal on, remove all fields in that EB from being selfcal'd as there is no calibration available
        ## in this EB.
        for vis in selfcal_library['vislist']:
            if np.intersect1d(selfcal_library['sub-fields-to-gaincal'],\
                    list(selfcal_library['sub-fields-fid_map'][vis].keys())).size == 0:
                for fid in np.intersect1d(new_fields_to_selfcal,list(selfcal_library['sub-fields-fid_map'][vis].keys())):
                    new_fields_to_selfcal.remove(fid)

                    selfcal_library[fid]['Stop_Reason'] = 'No viable calibrator fields for inf_EB in at least 1 EB'
                    for v in selfcal_library[fid]['vislist']:
                        selfcal_library[fid][v][solint]['Pass'] = 'None'
                        selfcal_library[fid][v][solint]['Fail_Reason'] = 'No viable inf_EB fields'

     return new_fields_to_selfcal

