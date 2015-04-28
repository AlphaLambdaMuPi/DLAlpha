from settings import *
import logging
import re
import shelve
import numpy as np
import random
logger = logging.getLogger()

with shelve.open(SHELVE['train']) as sh:
    train_names = sh['names']

with shelve.open(SHELVE['test']) as sh:
    test_names = sh['names']

def read_train_by_group(cnt = None, rnd=True, ls = None):
    
    if ls is None and cnt is None:
        raise ValueError('ls and cnt are both None.')

    res = []
    if ls is None:
        if rnd:
            ls = random.sample(range(len(train_names)), cnt)
        else:
            ls = train_names[:cnt]

    with shelve.open(SHELVE['train']) as sh:
        for i in ls:
            name = train_names[i]
            res.append((name, sh[name]))

    return res


