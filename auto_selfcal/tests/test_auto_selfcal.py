try:
    import submitit
    import sys
except:
    pass

import pytest
from auto_selfcal import auto_selfcal
import numpy as np
import pickle
import glob
import os

@pytest.mark.benchmark
@pytest.mark.parametrize(
    "dataset",
    [
        pytest.param("Band8-7m-2", id="Band8-7m-2"),
        pytest.param("mixed_spw_EBs", id="mixed_spw_EBs"),
        pytest.param('2019.1.00377.S', id='2019.1.00377.S'),
        pytest.param('2018.1.01089.S', id='2018.1.01089.S'),
        pytest.param('2017.1.01413.S', id='2017.1.01413.S'),
        pytest.param('2019.1.00261.L-IRAS32', id='2019.1.00261.L-IRAS32'),
        pytest.param('M82-C-conf-C-band', id='M82-C-conf-C-band'),
        pytest.param('m-L-Q', id='m-L-Q'),
        pytest.param('m-S-to-A', id='m-S-to-A'),
        pytest.param('2017.1.00983.S', id='2017.1.00983.S'),
        pytest.param('2016.1.00053.S', id='2016.1.00053.S'),
        pytest.param('2017.1.00404.S', id='2017.1.00404.S'),
        pytest.param('2013.1.00031.S', id='2013.1.00031.S'),
        pytest.param('apriori_flagged_spws', id='apriori_flagged_spws'),
        pytest.param('2022.1.00738.S', id='2022.1.00738.S'),
        pytest.param('2023.1.00905.S', id='2023.1.00905.S'),
    ]
)
def test_benchmark(tmp_path, dataset):
    d = tmp_path
    os.chdir(d)

    starting_MS_files = glob.glob(f"/lustre/cv/projects/SRDP/selfcal-prototyping/datasets/{dataset}/*.ms")
    for msfile in starting_MS_files:
        if ".selfcal." in msfile:
            continue
        os.system(f"cp -r {msfile} .")

    os.system(f"cp -r /lustre/cv/projects/SRDP/selfcal-prototyping/datasets/{dataset}/cont.dat .")
    os.system(f"cp -r /lustre/cv/projects/SRDP/selfcal-prototyping/datasets/{dataset}/selfcal_library_reference.pickle .")

    ex = submitit.SlurmExecutor(folder=".", python=f"OMP_NUM_THREADS=1 xvfb-run -d mpirun -n 8 {sys.executable}")
    ex.update_parameters(partition="batch2", nodes=1, ntasks_per_node=8, cpus_per_task=1, use_srun=False, time=10080, \
            mem="128gb", job_name=dataset)

    job = ex.submit(auto_selfcal, sort_targets_and_EBs=True, weblog=True, parallel=True)
    job.wait()

    assert job.state in ['DONE','COMPLETED']

    with open('selfcal_library_reference.pickle', 'rb') as handle:
        selfcal_library1 = pickle.load(handle)
    with open('selfcal_library.pickle', 'rb') as handle:
        selfcal_library2 = pickle.load(handle)

    difference_count = compare_two_dictionaries(selfcal_library1, selfcal_library2, tolerance=1e-3, exclude=['vislist_orig'])

    assert difference_count == 0

@pytest.mark.ghtest
@pytest.mark.parametrize(
    "zip_file,link",
    [
        pytest.param("2018.1.01284.S_HOPS-384.tar.gz", 'https://nrao-my.sharepoint.com/:u:/g/personal/psheehan_nrao_edu/Ea8NGWjlNptNnyqg_62xxmcB0lk64IpB7Dd7AgfltnNkXQ?e=B3I68J&download=1', id="2018.1.01284.S_HOPS-384"),
        pytest.param("Band8-7m-2.tar.gz", 'https://nrao-my.sharepoint.com/:u:/g/personal/psheehan_nrao_edu/EScVSXH9JHRIt2P-9fwxVRYBsy70G94cKxv00HTv44Pdug?e=PnC9Yb&download=1', id="Band8-7m-2"),
        pytest.param("M82-C-conf-C-band_small.tar.gz", 'https://nrao-my.sharepoint.com/:u:/g/personal/psheehan_nrao_edu/EZP4KRsBbthPksh6_kKsNI4Bm3m4L-1QhF4zCUDXhO73Lg?e=WkczgB&download=1', id="M82-C-conf-C-band_small"),
    ]
)
def test_on_github(tmp_path, request, zip_file, link):
    d = tmp_path
    os.chdir(d)
    if 'https' in link:
        os.system(f'wget "{link}" -O {zip_file}')
    else:
        os.system(f'cp {link}/{zip_file} .')
    os.system(f'tar xf {zip_file}')
    os.system(f'rm -rf {zip_file}')

    auto_selfcal(sort_targets_and_EBs=True, weblog=True)

    os.system('rm -rf *.ms*') # Delete MS files as space is limited on GitHub.

    with open('selfcal_library_reference.pickle', 'rb') as handle:
        selfcal_library1 = pickle.load(handle)
    with open('selfcal_library.pickle', 'rb') as handle:
        selfcal_library2 = pickle.load(handle)

    difference_count = compare_two_dictionaries(selfcal_library1, selfcal_library2, tolerance=0.001)

    assert difference_count == 0

# Utility functions for comparing results.

def compare_values(list1, list2, tol=1e-3):
    if type(list1) == list or type(list1) == np.ndarray:
        if len(list1) != len(list2):
            return False
        elif len(list1) == 0:
            return True
        else:
            return np.all([compare_values(list1[i], list2[i], tol=tol) for i in range(len(list1))])
    elif type(list1) == str or type(list1) == np.str_ or type(list1) == bool:
        return list1 == list2
    else:
        if list1 == 0:
            return abs(list2) < tol
        else:
            return abs(list1 - list2) < abs(list1*tol)

def compare_two_dictionaries(dictionary1, dictionary2, path=[], exclude=[], tolerance=1e-3):
    difference_count = 0

    all_keys = np.unique(list(dictionary1.keys()) + list(dictionary2.keys()))
    intersect_keys = np.intersect1d(list(dictionary1.keys()), list(dictionary2.keys()))
    for key in all_keys:
        if key in exclude:
            continue

        if key not in intersect_keys:
            if key not in dictionary1:
                print('/'.join([str(p) for p in path])+"/"+key+" not in dictionary1")
            else:
                print('/'.join([str(p) for p in path])+"/"+key+" not in dictionary2")

            difference_count += 1

            continue

        if key not in dictionary1 and int(key) in dictionary1:
            key = int(key)

        if type(dictionary1[key]) == dict:
            difference_count += compare_two_dictionaries(dictionary1[key], dictionary2[key], path.copy()+[key], exclude=exclude, tolerance=tolerance)
        else:
            value1 = np.array(dictionary1[key])[np.argsort(dictionary1['vislist'])] if key in ['spws_per_vis','vislist'] else dictionary1[key]
            value2 = np.array(dictionary2[key])[np.argsort(dictionary2['vislist'])] if key in ['spws_per_vis','vislist'] else dictionary2[key]
            #value1 = np.array(dictionary1[key])[np.argsort(dictionary1['vislist'])] if key in ['spws_per_vis'] else dictionary1[key]
            #value2 = np.array(dictionary2[key])[np.argsort(dictionary2['vislist'])] if key in ['spws_per_vis'] else dictionary2[key]

            if key == 'gaincal_combine':
                value1 = dictionary1[key].split(',')
                value1.sort()
                value2 = dictionary2[key].split(',')
                value2.sort()

            if not compare_values(value1, value2, tol=tolerance):
                print('/'.join([str(p) for p in path])+"/"+key, dictionary1[key], dictionary2[key])
                difference_count += 1

    return difference_count
