from casatasks import applycal, flagmanager, clearcal, uvcontsub
from .selfcal_helpers import sanitize_string, parse_contdotdat, get_spwnum_refvis,flagchannels_from_contdotdat,get_fitspw_dict, get_telescope
from casatools import msmetadata as msmdtool
import numpy as np
import casatasks
import pickle
import os

msmd = msmdtool()

def applycal_to_orig_MSes(selfcal_library='selfcal_library.pickle', write_only=True, suffix=''):
    """
    Apply self-calibration solutions derived from auto_selfcal back to the original, unaveraged, MSes.

    Parameters
    ----------
        selfcal_library : int, dict
            Name of the file containing the selfcal_library results from auto_selfcal,
            or the selfcal_library dictionary itself. Default: 'selfcal_library.pickle'
        write_only : bool
            Only write a file that can be executed to apply the calibrations, but do not apply. Defaut: True
        suffix (str): Suffix of the file to be applied to. Default: ''

    Returns
    -------
        None
    """

    # First we need to load in the relevant results files.

    if type(selfcal_library) == str:
        with open(selfcal_library, 'rb') as handle:
            selfcal_library=pickle.load(handle)
    elif type(selfcal_library) == dict:
        pass
    else:
        print("Keyword argument 'selfcal_library' must either be a string or dictionary.")
        return

    # Open a file that will log the applycal calls to be made, that can be run later to apply calibrations.

    applyCalOut=open('applycal_to_orig_MSes.py','w')

    # Clear any pre-existing calibrations in the original MS files.

    vislist = []
    for target in selfcal_library:
        for band in selfcal_library[target]:
            vislist += selfcal_library[target][band]['vislist']

    vislist = np.unique(vislist)

    if not write_only:
        for vis in vislist:
            tb.open(vis,nomodify=Flase)
            cols=tb.colnames()
            if 'CORRECTED_DATA' in cols:
                tb.removecols('CORRECTED_DATA')
            tb.close()
    
    for vis in vislist:
        applyCalOut.writelines('tb.open("'+vis+'",nomodify=False)\n')
        applyCalOut.writelines('if "CORRECTED_DATA" in tb.colnames():\n')
        applyCalOut.writelines('    tb.removecols("CORRECTED_DATA")\n')
        applyCalOut.writelines('tb.close()\n')

    # Loop through the targets and apply.

    for target in selfcal_library:
       for band in selfcal_library[target].keys():
          if selfcal_library[target][band]['SC_success']:
             for vis in selfcal_library[target][band]['vislist']: 
                solint=selfcal_library[target][band]['final_solint']
                iteration=selfcal_library[target][band][vis][solint]['iteration']    

                # Write the line to the command log

                line='applycal(vis="'+selfcal_library[target][band]['original_vislist_map'][vis].replace('.ms',suffix+'.ms')+\
                        '",gaintable='+str(selfcal_library[target][band][vis]['gaintable_final'])+\
                        ',interp='+str(selfcal_library[target][band][vis]['applycal_interpolate_final'])+\
                        ', calwt=False,spwmap='+str(selfcal_library[target][band][vis]['spwmap_final'])+\
                        ', applymode="'+selfcal_library[target][band][vis]['applycal_mode_final']+\
                        '",field="'+target+'",spw="'+selfcal_library[target][band][vis]['spws_orig']+'")\n'

                applyCalOut.writelines(line)

                if not write_only:
                   # First we need to restore the flags to the state they were in prior to self-calibration.

                   if os.path.exists(selfcal_library[target][band]['original_vislist_map'][vis].replace('.ms',suffix+'.ms')+".flagversions/flags.starting_flags"):
                      flagmanager(vis=selfcal_library[target][band]['original_vislist_map'][vis].replace('.ms',suffix+'.ms'), mode = 'restore', versionname = 'starting_flags', 
                            comment = 'Flag states at start of reduction')
                   else:
                      flagmanager(vis=selfcal_library[target][band]['original_vislist_map'][vis].replace('.ms',suffix+'.ms'),mode='save',versionname='before_final_applycal')

                   # Now apply

                   applycal(vis=selfcal_library[target][band]['original_vislist_map'][vis].replace('.ms',suffix+'.ms'),\
                        gaintable=selfcal_library[target][band][vis]['gaintable_final'],\
                        interp=selfcal_library[target][band][vis]['applycal_interpolate_final'], 
                        calwt=False,spwmap=[selfcal_library[target][band][vis]['spwmap_final']],\
                        applymode=selfcal_library[target][band][vis]['applycal_mode_final'],
                        field=target,spw=selfcal_library[target][band][vis]['spws_orig'])

    applyCalOut.close()



def uvcontsub_orig_MSes(selfcal_library="selfcal_library.pickle", write_only=True, suffix=''):
    """
    Do continuum subtraction of the original MS files.

    Parameters
    ----------
        selfcal_library : str, dict
            Name of the file containing the selfcal_library results from auto_selfcal,
            or the selfcal_library dictionary itself. Default: 'selfcal_library.pickle'
        write_only : bool
            Only write a file that can be executed to apply the calibrations, but do not apply. Defaut: True
        suffix (str): Suffix of the file to be applied to. Default: ''
    """

    # First we need to load in the relevant results files.

    if type(selfcal_library) == str:
        with open(selfcal_library, "rb") as handle:
            selfcal_library = pickle.load(handle)
    elif type(selfcal_library) == dict:
        pass
    else:
        print(
            "Keyword argument 'selfcal_library' must either be a string or dictionary."
        )
        return

    # Load in the name of the telescope that the data is from.

    for target in selfcal_library:
        for band in selfcal_library[target]:
            telescope = get_telescope(selfcal_library[target][band]['vislist'][0])
            break
        break

    # Do continuum subtraction of the original MS files.

    casaversion = casatasks.version()

    if casaversion[0] > 6 or (
        casaversion[0] == 6
        and (casaversion[1] > 5 or (casaversion[1] == 5 and casaversion[2] >= 2))
    ):
        # new uvcontsub format only works in CASA >=6.5.2
        if os.path.exists("cont.dat"):
            contsub_dict = {}
            n_sc_successes = 0

            for target in selfcal_library:
                sani_target = sanitize_string(target)
                for band in selfcal_library[target].keys():
                    if selfcal_library[target][band]['SC_success']:
                        n_sc_successes += 1

                    contdotdat = parse_contdotdat("cont.dat", target)
                    if len(contdotdat['ranges']) == 0:
                        selfcal_library[target][band]["Found_contdotdat"] = False

                    spwvisref = get_spwnum_refvis(selfcal_library[target][band]["vislist"], target, contdotdat, 
                            dict(zip(selfcal_library[target][band]["vislist"],
                            [selfcal_library[target][band][vis]["spwsarray"] for vis in 
                            selfcal_library[target][band]["vislist"]])), use_names=telescope in ['ALMA','ACA'])

                    for vis in selfcal_library[target][band]["vislist"]:
                        if selfcal_library[target][band]['original_vislist_map'][vis] not in contsub_dict:
                            contsub_dict[selfcal_library[target][band]['original_vislist_map'][vis]]={}
                
                        msmd.open(vis)
                        field_num_array = msmd.fieldsforname(target)
                        msmd.close()
                        for fieldnum in field_num_array:
                            contsub_dict[selfcal_library[target][band]['original_vislist_map'][vis]][str(fieldnum)] = \
                                get_fitspw_dict(selfcal_library[target][band]['original_vislist_map'][vis],
                                target, selfcal_library[target][band][vis]["spwsarray"],
                                selfcal_library[target][band]["vislist"], selfcal_library[target][band]['original_vislist_map'][spwvisref], 
                                contdotdat, telescope)
                            print(contsub_dict[selfcal_library[target][band]['original_vislist_map'][vis]][str(fieldnum)])

            print(contsub_dict)

            if n_sc_successes == 0:
                datacolumn='data'
            else:
                datacolumn='corrected' 

            uvcontsubOut = open("uvcontsub_orig_MSes.py", "w")
            for vis in selfcal_library[target][band]["vislist"]:
                line = 'uvcontsub(vis="'+selfcal_library[target][band]['original_vislist_map'][vis].replace('.ms',suffix+'.ms')+\
                    '", spw="'+selfcal_library[target][band][vis]["spws"]+\
                    '",fitspec='+str(contsub_dict[selfcal_library[target][band]['original_vislist_map'][vis]])+\
                    ', outputvis="'+selfcal_library[target][band]['original_vislist_map'][vis].replace(".ms", ".contsub.ms")+\
                    '",datacolumn="'+datacolumn+'")\n'

                uvcontsubOut.writelines(line)

                if not write_only:
                    exec(line)

            uvcontsubOut.close()

    else:  
        # old uvcontsub formatting, requires splitting out per target, new one is much better
        if os.path.exists("cont.dat"):
            uvcontsubOut = open("uvcontsub_orig_MSes_old.py", "w")

            line = "import os\n"
            uvcontsubOut.writelines(line)

            for target in selfcal_library:
                sani_target = sanitize_string(target)
                for band in selfcal_library[target].keys():
                    contdotdat = parse_contdotdat("cont.dat", target)
                    if len(contdotdat['ranges']) == 0:
                        selfcal_library[target][band]["Found_contdotdat"] = False

                    spwvisref = get_spwnum_refvis(selfcal_library[target][band]["vislist"], target, contdotdat,
                        dict(zip(selfcal_library[target][band]["vislist"], [selfcal_library[target][band][vis]["spwsarray"]
                                for vis in selfcal_library[target][band]["vislist"]])), use_names=telescope in ['ALMA','ACA'])

                    for vis in selfcal_library[target][band]["vislist"]:
                        contdot_dat_flagchannels_string = flagchannels_from_contdotdat(selfcal_library[target][band]['original_vislist_map'][vis],
                            target, selfcal_library[target][band][vis]["spwsarray"], selfcal_library[target][band]["vislist"],
                            selfcal_library[target][band]['original_vislist_map'][spwvisref], contdotdat, telescope, return_contfit_range=True)

                        line = 'uvcontsub(vis="' + selfcal_library[target][band]['original_vislist_map'][vis].replace('.ms',suffix+'.ms') + \
                            '", outputvis="'+sani_target+"_"+vis.replace(".selfcal", "".replace(".ms", ".contsub.ms")) + \
                            '",field="' + target + \
                            '", spw="' + selfcal_library[target][band][vis]["spws"] + \
                            '",fitspec="' + contdot_dat_flagchannels_string + \
                            '", combine="spw")\n'

                        uvcontsubOut.writelines(line)

                        if not write_only:
                            exec(line)

            uvcontsubOut.close()
