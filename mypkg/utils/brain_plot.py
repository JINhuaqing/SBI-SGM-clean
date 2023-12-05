import numpy as np
from pathlib import Path
import pandas as pd

_cur_dir = Path(__file__).parent
_dat_dir = _cur_dir/"../../data"

# Get reorder indeix
# dk86.***.txt is the order for the MEG data
# DK68_labelorder.xlsx is the order for the template
_dk_brainnet = pd.read_excel(_dat_dir/"DK68_labelorder.xlsx")
with open(_dat_dir/'dk86atlas_regions_python.txt') as f:
    _dk_python = f.readlines()
    
_brainnet_order = _dk_brainnet["Order in Brainstorm"]
_dk_python_renamed = []
_brainnet_renamed = []

for i in range(len(_dk_python)):
    sp1 = _dk_python[i].split('=')
    sp2 = sp1[1].split('_')
    _dk_python_renamed.append((sp2[0] + ' ' + sp2[1]).lower().strip())

for i in range(len(_brainnet_order)):
    _brainnet_renamed.append((_brainnet_order[i].replace("'","")).lower())

d = {x: a for a, x in enumerate(_dk_python_renamed[0:68])}
_orderedind = np.array([d[x] for x in _brainnet_renamed])

def reorderU(wU):
    wUreorder = wU.copy()
    wUreorder = wU[_orderedind]
    return wUreorder

with open(_dat_dir/'BNVtemplate_DK68.txt', "r") as tf:
    _DKtmplateRaw = tf.readlines()
_DKtmplate = np.array([int(x.strip()) for x in _DKtmplateRaw])

def U2BrainVec(wU):
    emVec = np.zeros_like(_DKtmplate, dtype=np.float64)
    for idx in range(1, 69):
        emVec[_DKtmplate==idx] = wU[idx-1]
    return emVec