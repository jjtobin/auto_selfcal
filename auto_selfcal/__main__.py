# Mac builds of CASA lack MPI and error without this try/except
try:
   from casampi.MPIEnvironment import MPIEnvironment   
   parallel=MPIEnvironment.is_mpi_enabled
except:
   parallel=False

from .auto_selfcal import auto_selfcal
from .regenerate_weblog import regenerate_weblog
from .split_calibrated_final import split_calibrated_final
from .original_ms_helpers import applycal_to_orig_MSes, uvcontsub_orig_MSes
import argparse
import glob
import ast
import sys

parser = argparse.ArgumentParser(
                    prog='auto_selfcal',
                    description='Run automated self-calibration on a collection of MS files.')

parser.add_argument('--vislist', nargs='*', default=[])

parser.add_argument('--action', default='run')

parser.add_argument('--spectral_average', default=True, type=bool)
parser.add_argument('--do_amp_selfcal', default=True)
parser.add_argument('--usermask', default={}, type=ast.literal_eval)  # require that it is a CRTF region (CASA region format)
parser.add_argument('--usermodel', default={}, type=ast.literal_eval) 
parser.add_argument('--uniform_solints', default=False)
parser.add_argument('--inf_EB_gaincal_combine', default='scan', type=str)  # should we get rid of this option?
parser.add_argument('--inf_EB_gaintype', default='G', type=str)
parser.add_argument('--inf_EB_override', action='store_true')
parser.add_argument('--optimize_spw_combine', default=True, type=bool)      # if False, will not attempt per spw or per baseband solutions for any solint except inf_EB
parser.add_argument('--gaincal_minsnr', default=2.0, type=float)
parser.add_argument('--gaincal_unflag_minsnr', default=5.0, type=float)
parser.add_argument('--minsnr_to_proceed', default=2.95, type=float)
parser.add_argument('--spectral_solution_fraction', default=0.25, type=float)
parser.add_argument('--delta_beam_thresh', default=0.05, type=float)
parser.add_argument('--apply_cal_mode_default', default='calflag', type=str)
parser.add_argument('--unflag_only_lbants', action='store_true')
parser.add_argument('--unflag_only_lbants_onlyap', action='store_true')
parser.add_argument('--calonly_max_flagged', default=0.0, type=float)
parser.add_argument('--second_iter_solmode', default= "", type=str)
parser.add_argument('--unflag_fb_to_prev_solint', action='store_true')
parser.add_argument('--rerank_refants', action='store_true')
parser.add_argument('--allow_gain_interpolation', action='store_true')
parser.add_argument('--guess_scan_combine', default=False, type=bool)
parser.add_argument('--aca_use_nfmask', action='store_true')
parser.add_argument('--allow_cocal', action='store_true')
parser.add_argument('--scale_fov', default=1.0, type=float)   # option to make field of view larger than the default
parser.add_argument('--rel_thresh_scaling', default='log10', type=str)  #can set to linear, log10, or loge (natural log)
parser.add_argument('--dividing_factor', default=-99.0, type=float)  # number that the peak SNR is divided by to determine first clean threshold -99.0 uses default
parser.add_argument('--check_all_spws', action='store_true')   # generate per-spw images to check phase transfer did not go poorly for narrow windows
parser.add_argument('--apply_to_target_ms', action='store_true') # apply final selfcal solutions back to the input _target.ms files
parser.add_argument('--uvcontsub_target_ms', action='store_true') # apply final selfcal solutions back to the input _target.ms files
parser.add_argument('--sort_targets_and_EBs', action='store_true')
parser.add_argument('--run_findcont', action='store_true')
parser.add_argument('--debug', action='store_true')

parser.add_argument('--exit', action='store_true')

args = parser.parse_args()

if args.exit:
    pass
elif args.action == "run":
    auto_selfcal(parallel=parallel, **vars(args))
elif args.action == "prepare_data":
    split_calibrated_final(vislist=args.vislist, overwrite=True)
elif args.action == "regenerate_weblog":
    regenerate_weblog()
elif args.action == "apply":
    applycal_to_orig_MSes(write_only=False)
elif args.action == "contsub":
    uvcontsub_orig_MSes(write_only=False)

sys.exit()
