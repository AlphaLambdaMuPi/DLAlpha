import numpy as np
from phomap import ph2id

def flatten(inp):
    res = [(fr[1]+fr[2], fr[3]) for n in inp for fr in n[1]]
    return res

def pair_to_np(inp):
    X, Y = zip(*inp)
    resX = numpy.asarray(X)
    resY = numpy.asarray(map(ph2id, Y))
    return resX, resY

