from auto_selfcal import auto_selfcal
import os

#def test_mixed_spw_EBs(tmp_path):
#    d = tmp_path / "mixed_spw_EBs"
#    d.mkdir()
#    os.chdir(d)
#    os.system('wget "https://nrao-my.sharepoint.com/:u:/g/personal/psheehan_nrao_edu/EbUGCua1e95MvPhx3FpfKScBfG5KnxfaIGtk5_JA_Ghw6w?e=OB7Atq&download=1" -O mixed_spw_EBs.tar.gz')
#    os.system('tar xf mixed_spw_EBs.tar.gz')

#    auto_selfcal()

#    assert os.path.exists("weblog")

def test_2018_1_01284_S_HOPS_384(tmp_path):
    d = tmp_path / "2018.1.01284.S_HOPS-384"
    d.mkdir()
    os.chdir(d)
    os.system('wget "https://nrao-my.sharepoint.com/:u:/g/personal/psheehan_nrao_edu/ESCyveu27lBBnVHU00RBKuoBAsUZkXXBAZqJyqieCmeFqQ?e=Wpf4ce&download=1" -O 2018.1.01284.S_HOPS-384.tar.gz')
    os.system('tar xf 2018.1.01284.S_HOPS-384.tar.gz')

    auto_selfcal()

    assert os.path.exists("weblog")
