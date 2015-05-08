import numpy as np
from utils import *
from read_input import *

PATH = '../svm-python3-kai/prob.csv'

def readfeature(max_lines = 10**10):
    cnt = 0
    feature = []
    with open(PATH) as f:
        for line in f:
            line = line.rstrip('\n')
            datas = line.split()
            datas = datas[0:1] + list(map(float, datas[1:]))
            feature.append(datas)
            cnt += 1
            if cnt >= max_lines:
                break

    return feature

def clr(s):
    ret = []
    lst = ''
    cnt = 0
    for x in s:
        if x != lst:
            if cnt >= 3:
                ret.append(lst)
            lst = x
            cnt = 0
        cnt += 1
    if cnt >= 3:
        ret.append(lst)
    return ret

feature = readfeature(10000)

pho_init()
mp = {}
for f in feature:
    lab = f[0]
    feat = f[1:]
    am = np.argmax(feat)
    ph = ph49238(id2ph(am))
    # print(lab, ph)
    utt = lab[:lab.rfind('_')]
    if utt not in mp:
        mp[utt] = []
    mp[utt].append(ph)

rmd = read_models(40)
rmd = [answer(y) for x, y in rmd]

ret = []
for utt in mp:
    mu = mp[utt]
    mu = clr(mu)
    ans = answer(mu)
    ret.append((utt, ans))
ret.sort()

totdist = 0
totcnt = 0

f = open('yaps.out', 'w')
f.write('id,phone_sequence\n')
for l, y1 in ret:
    f.write('{},{}\n'.format(l, y1))

for [l, y1], y2 in zip(ret, rmd):
    dist = delta(y1, y2)
    print(l, dist)
    print(y1)
    print(y2)
    
    totdist += dist
    totcnt += 1

avgdist = totdist / totcnt
print('Avg dist', avgdist)


