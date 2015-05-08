import numpy as np
from phomap import ph2id
from theano import config

def flatten(inp):
    res = [(fr[1]+fr[2], fr[3]) for n in inp for fr in n[1]]
    return res

def pair_to_np(inp):
    X, Y = zip(*inp)
    resX = np.asarray(X)
    resY = np.asarray(list(map(ph2id, Y)))
    return resX, resY

def y_to_01(inp, K):
    res = np.asarray( [1 * (np.arange(K) == y) for y in inp] )
    return res

