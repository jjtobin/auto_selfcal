from .selfcal_helpers import *

def get_image_stats(image, mask, backup_mask, selfcal_library, use_nfmask, solint, suffix, mosaic_sub_field=False, spw='all'):
    ##
    ## Do the assessment of the post- (and pre-) selfcal images.
    ##
    SNR, RMS = estimate_SNR(image, maskname=mask, mosaic_sub_field=mosaic_sub_field)
    if use_nfmask:
       SNR_NF,RMS_NF = estimate_near_field_SNR(image, maskname=mask, las=selfcal_library['LAS'], mosaic_sub_field=mosaic_sub_field)
       if RMS_NF < 0 and backup_mask != '':
           SNR_NF, RMS_NF = estimate_near_field_SNR(image, maskname=backup_mask, las=selfcal_library['LAS'], mosaic_sub_field=mosaic_sub_field)
    else:
       SNR_NF, RMS_NF = SNR, RMS

    if suffix in ['dirty','orig','initial','final']:
        vislist = selfcal_library['vislist']
    else:
        vislist = selfcal_library['vislist-to-gaincal']

    for vis in vislist:
       if suffix in ['dirty','orig','initial','final']:
           if spw == 'all':
               update_dict = selfcal_library
           else:
               update_dict = selfcal_library['per_spw_stats'][spw]
       else:
           update_dict = selfcal_library[vis][solint]

       ##
       ## record self cal results/details for this solint
       ##
       update_dict['SNR_'+suffix]=SNR.copy()
       update_dict['RMS_'+suffix]=RMS.copy()
       update_dict['SNR_NF_'+suffix]=SNR_NF.copy()
       update_dict['RMS_NF_'+suffix]=RMS_NF.copy()

       header=imhead(imagename=image)
       update_dict['Beam_major_'+suffix]=header['restoringbeam']['major']['value']
       update_dict['Beam_minor_'+suffix]=header['restoringbeam']['minor']['value']
       update_dict['Beam_PA_'+suffix]=header['restoringbeam']['positionangle']['value'] 

       if checkmask(imagename=mask):
           update_dict['intflux_'+suffix], update_dict['e_intflux_'+suffix] = get_intflux(image, RMS, maskname=mask,
                   mosaic_sub_field=mosaic_sub_field)
       elif backup_mask != '' and checkmask(imagename=backup_mask):
           update_dict['intflux_'+suffix], update_dict['e_intflux_'+suffix] = get_intflux(image, RMS, maskname=backup_mask,
                   mosaic_sub_field=mosaic_sub_field)
       else:
           update_dict['intflux_'+suffix], update_dict['e_intflux_'+suffix] = -99.0, -99.0

       if suffix in ['dirty','orig','initial','final']:
           break

    return SNR, RMS, SNR_NF, RMS_NF
