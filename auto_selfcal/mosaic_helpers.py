import numpy as np
from .selfcal_helpers import *

def evaluate_subfields_to_gaincal(vis, selfcal_library, target, band, solint, iteration, solmode, solints, selfcal_plan, 
        minsnr_to_proceed, allow_gain_interpolation=False,):

     sani_target=sanitize_string(target)

     # Fields that don't have any mask in the primary beam should be removed from consideration, as their models are likely bad.
     new_fields_to_selfcal = []
     for fid in selfcal_library[vis]['sub-fields-to-selfcal']:
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

         print('Checking with flux threshold {:0.2f}'.format(selfcal_library['flux_threshold']))
         if not checkmask(sani_target+'_field_'+str(fid)+'_'+band+'_'+solint+'_'+str(iteration)+'.image.tt0'):
             print("Removing field "+str(fid)+" from gaincal because there is no signal within the primary beam.")
             skip_reason = "No signal"
         elif selfcal_plan[fid][vis]['solint_snr_per_field'][solints[iteration]] < minsnr_to_proceed and \
                 selfcal_plan[vis]['solint_settings'][solint]['sub-name'] not in ['inf_EB','scan_inf']:
             print("Removing field "+str(fid)+" from gaincal because the estimated solint snr is too low.")
             skip_reason = "Estimated SNR"
         elif updated_intflux > selfcal_library['flux_threshold'] * original_intflux:
             print("Removing field "+str(fid)+" from gaincal because there appears to be significant flux missing from the model.")
             print("Original Flux: ",original_intflux, "Per-field Flux: ",updated_intflux)
             skip_reason = "Missing flux"
         else:
             new_fields_to_selfcal.append(fid)

         if fid not in new_fields_to_selfcal and solint != "inf_EB" and not allow_gain_interpolation:
             #selfcal_library[fid][vis][solint]['interpolated_gains'] = True
             #selfcal_library[fid]['Stop_Reason'] = "Gaincal solutions would be interpolated"
             selfcal_library[fid][vis]['Stop_Reason'] = skip_reason
             selfcal_library[fid][vis][solint]['Pass'] = "None"
             selfcal_library[fid][vis][solint]['Fail_Reason'] = skip_reason

     return new_fields_to_selfcal




def evaluate_subfields_after_gaincal(vis, selfcal_library, selfcal_plan, target, band, solint, iteration, solmode, allow_gain_interpolation=False):

     new_fields_to_selfcal = selfcal_library[vis]['sub-fields-to-selfcal'].copy()

     sani_target=sanitize_string(target)

     if selfcal_library['obstype'] == 'mosaic' and ((solint != "inf_EB" and not allow_gain_interpolation) or \
             (allow_gain_interpolation and "inf" not in solint)):
        # With gaincal done and bad fields removed from gain tables if necessary, check whether any fields should no longer be selfcal'd
        # because they have too much interpolation.
        #for vis in selfcal_library['vislist']:
        if True:
            ## If an EB had no fields to gaincal on, remove all fields in that EB from being selfcal'd as there is no calibration available
            ## in this EB.
            if np.intersect1d(selfcal_library[vis]['sub-fields-to-gaincal'],\
                    list(selfcal_library['sub-fields-fid_map'][vis].keys())).size == 0:
                for fid in np.intersect1d(new_fields_to_selfcal,list(selfcal_library['sub-fields-fid_map'][vis].keys())):
                    new_fields_to_selfcal.remove(fid)

                    selfcal_library[fid][vis]['Stop_Reason'] = 'No viable calibrator fields in at least 1 EB'
                    selfcal_library[fid][vis][solint]['Pass'] = 'None'
                    if 'Fail_Reason' in selfcal_library[fid][vis][solint]:
                        selfcal_library[fid][vis][solint]['Fail_Reason'] += '; '
                    else:
                        selfcal_library[fid][vis][solint]['Fail_Reason'] = ''
                    selfcal_library[fid][vis][solint]['Fail_Reason'] += 'No viable fields'
                return new_fields_to_selfcal
            ## NEXT TO DO: check % of flagged solutions - DONE, see above
            ## After that enable option for interpolation through inf - DONE
            tb.open(sani_target+'_'+vis+'_'+band+'_'+solint+'_'+str(iteration)+'_'+solmode[iteration]+'_'+
                    selfcal_library[vis][solint]['final_mode']+'.g')
            fields = tb.getcol("FIELD_ID")
            scans = tb.getcol("SCAN_NUMBER")

            for fid in np.intersect1d(new_fields_to_selfcal,list(selfcal_library['sub-fields-fid_map'][vis].keys())):
                if selfcal_plan[vis]['solint_settings'][solint]['sub-name'] == "scan_inf":
                    msmd.open(vis)
                    cals_for_scan = []
                    total_cals_for_scan = []
                    for incl_scan in selfcal_library[vis][solint]['include_scans']:
                        scans_array = np.array(incl_scan.split(",")).astype(int)
                        fields_for_scans = msmd.fieldsforscans(scans_array)

                        if selfcal_library['sub-fields-fid_map'][vis][fid] in fields_for_scans:
                            scans_for_field = np.intersect1d(scans_array, np.unique(scans))
                            if scans_for_field.size == 0:
                                cals_for_scan.append(0)
                            else:
                                cals_for_scan.append((scans == scans_for_field[-1]).sum())
                            total_cals_for_scan.append(len(msmd.antennanames()))

                    if sum(cals_for_scan) / sum(total_cals_for_scan) < 0.75:
                        new_fields_to_selfcal.remove(fid)

                    msmd.close()
                else:
                    if selfcal_library['sub-fields-fid_map'][vis][fid] not in fields:
                        new_fields_to_selfcal.remove(fid)

                if fid not in new_fields_to_selfcal:
                    selfcal_library[fid][vis][solint]['Pass'] = 'None'
                    if allow_gain_interpolation:
                        selfcal_library[fid][vis][solint]['Fail_Reason'] = 'Interpolation beyond inf'
                    else:
                        selfcal_library[fid][vis][solint]['Fail_Reason'] = 'Bad gaincal solutions'


            tb.close()
     elif selfcal_library['obstype'] == 'mosaic' and solint == "inf_EB":
        ## If an EB had no fields to gaincal on, remove all fields in that EB from being selfcal'd as there is no calibration available
        ## in this EB.
        #for vis in selfcal_library['vislist']:
        if True:
            if np.intersect1d(selfcal_library[vis]['sub-fields-to-gaincal'],\
                    list(selfcal_library['sub-fields-fid_map'][vis].keys())).size == 0:
                for fid in np.intersect1d(new_fields_to_selfcal,list(selfcal_library['sub-fields-fid_map'][vis].keys())):
                    new_fields_to_selfcal.remove(fid)

                    selfcal_library[fid][vis]['Stop_Reason'] = 'No viable calibrator fields for inf_EB in at least 1 EB'
                    selfcal_library[fid][vis][solint]['Pass'] = 'None'
                    selfcal_library[fid][vis][solint]['Fail_Reason'] = 'No viable inf_EB fields'

     print("new_fields_to_selfcal", new_fields_to_selfcal)

     return new_fields_to_selfcal


def scan_inf_scan_combine(selfcal_library, vis, target, gaincalibrator_dict, guess_scan_combine=False):
    if len(gaincalibrator_dict[vis]) > 0:
        print("Determining scan_inf from calibrator scans in full MS")
        scans = []
        intents = []
        times = []
        for t in gaincalibrator_dict[vis].keys():
            scans += [gaincalibrator_dict[vis][t]["scans"]]
            intents += [np.repeat(gaincalibrator_dict[vis][t]["intent"],gaincalibrator_dict[vis][t]["scans"].size)]
            times += [gaincalibrator_dict[vis][t]["times"]]
        
        times = np.concatenate(times)
        order = np.argsort(times)
        times = times[order]
        
        scans = np.concatenate(scans)[order]
        intents = np.concatenate(intents)[order]

        is_gaincalibrator = intents == "phase"
        scans = scans[is_gaincalibrator]

        msmd.open(vis)
        if selfcal_library['telescope'] == 'ALMA' or selfcal_library['telescope'] == 'ACA':
            scan_ids_for_target = msmd.scansforfield(target)
        elif 'VLA' in selfcal_library['telescope']:
            scan_ids_for_target=np.array([],dtype=int)
            for fid in selfcal_library['sub-fields']:
                if fid in selfcal_library['sub-fields-fid_map'][vis].keys():
                    field_id=selfcal_library['sub-fields-fid_map'][vis][fid]
                    scan_ids_for_target=np.append(scan_ids_for_target,msmd.scansforfield(field_id))

            scan_ids_for_target.sort() # sort scans since they will be out of order
        include_scans = []
        for iscan in range(scans.size-1):
            scan_group = np.intersect1d(scan_ids_for_target, 
                    np.array(list(range(scans[iscan]+1, scans[iscan+1])))).astype(str)
            if scan_group.size > 0:
                include_scans.append(",".join(scan_group))
        # PIPE-2741: if there are any scans before the first gain calibrator scan or after the last gain calibrator scan, catch them.
        if scans.size > 0:
            extra_scans = scan_ids_for_target[scan_ids_for_target > max(scans)]
            if extra_scans.size > 0:
                include_scans.append(','.join(extra_scans.astype(str)))
            extra_scans = scan_ids_for_target[scan_ids_for_target < min(scans)]
            if extra_scans.size > 0:
                include_scans.append(','.join(extra_scans.astype(str)))
        msmd.close()
    elif guess_scan_combine:
        print("Determining scan_inf from guessing where the calibrator scans were")
        msmd.open(vis)
        include_scans = []

        #to guess at scan_inf combination for VLA look for breaks in the consecutive
        #scan numbers and assume that the break is due to a calibrator scan
        #Fetch scans for scan inf by collecting the field ids and running msmd.scansforfield
        if 'VLA' in selfcal_library['telescope']:
            scans=np.array([],dtype=int)
            for fid in selfcal_library['sub-fields']:
                if fid in selfcal_library['sub-fields-fid_map'][vis].keys():
                    field_id=selfcal_library['sub-fields-fid_map'][vis][fid]
                    scans=np.append(scans,msmd.scansforfield(field_id))

            scans.sort() # sort scans since they will be out of order
            scan_group=''
            for iscan in range(scans.size):
                if len(include_scans) > 0:
                    if str(scans[iscan]) in include_scans[-1]:
                        continue
                if scan_group == '':
                    scan_group = str(int(scans[iscan]))

                if iscan < scans.size-1:
                    if scans[iscan+1] == scans[iscan]+1:
                        scan_group += ","+str(int(scans[iscan+1]))
                    else:
                        include_scans.append(scan_group)
                        scan_group=''
                else: #write out the last scan group to include_scans
                    if scan_group != '':
                        include_scans.append(scan_group)
                        scan_group=''

        #guess scan_inf combination by getting all the scans for targets and do a simple grouping
        if selfcal_library['telescope'] == 'ALMA' or selfcal_library['telescope'] == 'ACA':
            scans = msmd.scansforfield(target)

            for iscan in range(scans.size):
                if len(include_scans) > 0:
                    if str(scans[iscan]) in include_scans[-1]:
                        continue

                scan_group = str(scans[iscan])

                if iscan < scans.size-1:
                    if msmd.fieldsforscan(scans[iscan+1]).size < msmd.fieldsforscan(scans[iscan]).size/3:
                        scan_group += ","+str(scans[iscan+1])

                include_scans.append(scan_group)

        msmd.close()
    else:
        print("Not guessing where calibration scans are and justincluding all scans")
        msmd.open(vis)
        if selfcal_library['telescope'] == 'ALMA' or selfcal_library['telescope'] == 'ACA':
            include_scans = [str(scan) for scan in msmd.scansforfield(target)]
        elif 'VLA' in selfcal_library['telescope']:
            scans=np.array([],dtype=int)
            for fid in selfcal_library['sub-fields']:
                if fid in selfcal_library['sub-fields-fid_map'][vis].keys():
                    field_id=selfcal_library['sub-fields-fid_map'][vis][fid]
                    scans=np.append(scans,msmd.scansforfield(field_id))

            scans.sort() # sort scans since they will be out of order      
            include_scans = [str(scan) for scan in scans]             
        msmd.close()

    return include_scans