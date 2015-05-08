from read_input import read_map, read_map_39
from phone_maps import *
from settings import *

def phi(dt, mp):
    PHONES = LABEL_DIM
    FBANKS = FEATURE_DIM
    DIMS = PHONES * FBANKS + PHONES * PHONES
    res = [0.0] * DIMS

    def label_id(l):
        return mp[l][0] 
    def label_feature_id(l, f):
        return l * FBANKS + f
    def trans_id(a, b):
        return PHONES * FBANKS + a * PHONES + b

    __, fe, la = zip(*dt)
    last_lid = -1
    for i in range(len(__)):
        lid = label_id(la[i])

        for j in range(FBANKS):
            res[label_feature_id(lid, j)] += fe[i][j]

        if last_lid != -1:
            res[trans_id(last_lid, lid)] += 1.0

        last_lid = lid
    return res

def varpsi(X, Y, mp):
    PHONES = LABEL_DIM
    FBANKS = FEATURE_DIM
    DIMS = PHONES * FBANKS + PHONES * PHONES
    res = [0.0] * DIMS

    def label_id(l):
        return mp[l][0] 
    def label_feature_id(l, f):
        return l * FBANKS + f
    def trans_id(a, b):
        return PHONES * FBANKS + a * PHONES + b

    last_lid = -1
    for i in range(len(Y)):
        lid = label_id(Y[i])

        for j in range(FBANKS):
            res[label_feature_id(lid, j)] += X[i*FBANKS + j]

        if last_lid != -1:
            res[trans_id(last_lid, lid)] += 1.0

        last_lid = lid

    # for i in range(len(res)):
        # res[i] /= 50000
    return res

def delta(s1, s2):
    pv = list(range(len(s2)+1))
    for i, c1 in enumerate(s1):
        cv = [i+1]
        for j, c2 in enumerate(s2):
            cv.append(min(pv[j+1]+1, cv[j]+1, pv[j]+(c1 != c2)))
        pv = cv
    return pv[-1]


def answer(l, SN = True):
    if SN:
        return answer39(l)
    seq = [ph2c(x) for x in l]
    s = ''.join(seq)
    s = s.strip(ph2c('sil'))

    if s == '':
        return ''
    ans = s[0]
    for c in s:
        if c != ans[-1]: ans += c

    return ans

def answer39(l):
    seq = [ph2c(ph49238(x)) for x in l]
    s = ''.join(seq)
    s = s.strip(ph2c('sil'))

    if s == '':
        return ''
    ans = s[0]
    for c in s:
        if c != ans[-1]: ans += c

    return ans
