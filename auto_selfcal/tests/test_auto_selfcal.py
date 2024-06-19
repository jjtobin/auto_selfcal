from auto_selfcal import auto_selfcal
import os
import glob

def test_mixed_spw_EBs(tmp_path):
    d = tmp_path / "mixed_spw_EBs"
    d.mkdir()
    os.chdir(d)
    os.system("cp -r /lustre/cv/projects/SRDP/selfcal-prototyping/datasets/mixed_spw_EBs/*_targets.ms .")
    os.system("cp -r /lustre/cv/projects/SRDP/selfcal-prototyping/datasets/mixed_spw_EBs/cont.dat .")

    vislist = glob.glob("*_targets.ms")
    auto_selfcal(vislist)

    assert os.path.exists("weblog")

def test_2018_1_01284_S_HOPS_384(tmp_path):
    d = tmp_path / "2018.1.01284.S_HOPS-384"
    d.mkdir()
    os.chdir(d)
    os.system("cp -r /lustre/cv/projects/SRDP/selfcal-prototyping/Patrick/datasets/2018.1.01284.S_HOPS-384/*_targets.ms .")

    vislist = glob.glob("*_targets.ms")
    auto_selfcal(vislist)

    assert os.path.exists("weblog")
