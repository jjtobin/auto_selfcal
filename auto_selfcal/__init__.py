try:
    import os
    if os.getenv("CASAPATH") is not None:
        import casampi.private.start_mpi

    def setup_monolithic_CASA(pytest=False):
        casa_path = os.getenv("CASAPATH").split(" ")[0]
        os.system(f'cd {casa_path}/bin; ln -s casa auto_selfcal')

        if pytest:
            try:
                import pytest
                os.system(f'cd {casa_path}/bin; ln -s casa pytest')
                os.system(f'{casa_path}/bin/pip3 install --upgrade pytest')
            except:
                pass
except:
    pass

from .auto_selfcal import auto_selfcal
from .regenerate_weblog import regenerate_weblog
from .split_calibrated_final import split_calibrated_final
from .original_ms_helpers import applycal_to_orig_MSes, uvcontsub_orig_MSes
