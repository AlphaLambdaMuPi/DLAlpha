from read_input import read_map, read_map_39

def pho_init():
    global phomap, phomap39, invphomap, labels
    phomap = read_map()
    phomap39 = read_map_39()
    labels = phomap.keys()
    invphomap = {}

    for ph in phomap:
        invphomap[phomap[ph][0]] = ph

def ph2id(p):
    return phomap[p][0]

def ph2c(p):
    return phomap[p][1]

def id2ph(p):
    return invphomap[p]

def ph49238(p):
    return phomap39[p]

def get_maps():
    return phomap, phomap39, invphomap, labels
