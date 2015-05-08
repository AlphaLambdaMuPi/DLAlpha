import numpy as np
import re

N = 1943
mat = np.zeros((N+1, N+1))
regex = re.compile('(\w+)_\d+')

with open('state_train.lab') as f:
    f.readline()

    l = f.readline()
    _,c = l.strip().split(',')
    last_name = regex.match(_).group(1)
    cur = int(c)
    mat[N, cur] += 1.0

    while True:
        l = f.readline()
        if not l: break
        _,c = l.strip().split(',')
        name = regex.match(_).group(1)
        nxt = int(c)
        if name == last_name:
            mat[cur,nxt] += 1.0
        else:
            mat[cur, N] += 1.0
            mat[N, nxt] += 1.0
            last_name = name
        cur = nxt
    mat[cur, N] += 1.0

np.save('state_prob.npy', mat)



    
