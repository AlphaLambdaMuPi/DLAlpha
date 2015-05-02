import shelve
from settings import *

p1 = '../output/Strange_Dropout_0502_104714/test.out'
p2 = '../output/Strange_Dropout_0502_104714/final_hmm.out'

with shelve.open(SHELVE['test']) as s:
    with open(p1) as f1, open(p2, 'w') as f2:
        for name in s['names']:
            ln = len(s[name])
            for i in range(1, ln+1):
                z = f1.readline().strip('\n')
                f2.write(('{}_{},{}\n').format(name, i, z))

