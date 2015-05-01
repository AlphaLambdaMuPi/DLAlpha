from os.path import join as pjoin
from settings import *

phomap = {}
with open(pjoin(PATH['phones'], '48_39.map')) as f:
    for ln in f:
        ls = ln.split()
        phomap[ls[0]] = ls[1]

with open(pjoin(PATH['phones'], '48_idx_chr.map')) as f:
    for ln in f:
        ls = ln.split()
        phomap[ls[0]] = (int(ls[1]), phomap[ls[0]], ls[2])

phonemes = phomap.keys()
invphomap = {}

for ph in phomap:
    invphomap[phomap[ph][0]] = ph

def ph2id(p):
    return phomap[p][0]

def ph2c(p):
    return phomap[p][2]

def id2ph(p):
    return invphomap[p]

def ph49238(p):
    return phomap[p][1]

def ph48239(p):
    return phomap[p][1]

def get_maps():
    return phomap, invphomap
