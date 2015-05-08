from settings import *
import logging
import re
import shelve
import numpy as np
logger = logging.getLogger()

def read_feature(max_lines = 10**10):

    cnt = 0
    feature = []
    with open(DATA_PATH['fbank']) as f:
        for line in f:
            line = line.rstrip('\n')
            datas = line.split()
            datas = datas[0:1] + list(map(float, datas[1:]))
            feature.append(datas)
            cnt += 1
            if cnt >= max_lines:
                break

    logger.info('Read %d datas feature...', cnt)

    return feature

def read_feature_by_group():
    '''
        res[group_name] = [
         (1, [fbanks features]),
         (2, [fbanks features]),...
     ]
     '''

    fe = read_feature()
    nid = 0
    res = {}
    while nid < len(fe):
        gr = []
        cur_name, __ = feature_group(fe[nid][0])
        i = nid
        while i < len(fe) and cur_name == feature_group(fe[i][0])[0]:
            pr = (feature_group(fe[i][0])[1] , fe[i][1:])
            gr.append(pr)
            i += 1
        res[cur_name] = gr
        nid = i
    return res

regex = re.compile(r'(\w+)_(\d+)')
def feature_group(name):
    
    match = regex.fullmatch(name)
    return match.group(1), int(match.group(2))

def read_label(max_lines = 10**10):

    cnt = 0
    res = []
    with open(DATA_PATH['label']) as f:
        for line in f:
            line = line.rstrip('\n')
            datas = line.split(',')
            res.append(datas)
            if len(datas) != 2:
                raise ValueError('zz')
            cnt += 1
            if cnt >= max_lines:
                break

    logger.info('Read %d datas label...', cnt)
    return res

def read_label_dict():
    '''
        res[(name, id)] = ans
    '''
    ls = read_label()
    res = {}
    for data in ls:
        pr = feature_group(data[0])
        res[pr] = data[1]

    return res

def read_feature_label(test=False):
    fe = read_feature_by_group()
    la = read_label_dict()
    res = {}

    for k, v in fe.items():
        res[k] = []
        for x in v:
            fid = x[0]
            ans = la[k, fid]
            dt = (x[0], x[1], ans)
            res[k].append(dt)

    return res

def read_train_datas(count = 100, start = 0):
    '''
        The prefered way to get train data...
        Total groups is about 3xxx (count = 100 ~ 3%)
    '''
    rt = []
    with shelve.open(SHELVE_FILE_NAME, 'r') as d:
        data = d['names']
        data.sort()
        l = start
        r = start + count
        for i in range(l, min(r, len(data))):
            rt.append((data[i], d[data[i]]))
    return rt

def read_train_datas_by_name(name):
    '''
        The prefered way to get train data...
        Total groups is about 3xxx (count = 100 ~ 3%)
    '''
    with shelve.open(SHELVE_FILE_NAME, 'r') as d:
        rt = d[name]

    return rt

    
mpres = {}
def read_map():
    '''
        format: res['<phone>'] = (id, answer_char)
    '''
    if mpres: return mpres
    res = {}
    with open(DATA_PATH['48_idx_chr']) as f:
        for line in f:
            line = line.rstrip('\n')
            datas = line.split()
            res[datas[0]] = int(datas[1]), datas[2]
    logger.info('Read map done ... ')
    global mpres
    mpres = res
    return mpres

mp39res = {}
def read_map_39():
    '''
        format: res['<48_phone>'] = '<39_phone>'
    '''
    if mp39res: return mp39res
    res = {}
    with open(DATA_PATH['48_39']) as f:
        for line in f:
            line = line.rstrip('\n')
            datas = line.split('\t')
            res[datas[0]] = datas[-1]
    logger.info('Read 39 map done ... ')
    global mp39res
    mp39res = res
    return mp39res

def read_models(count = 100, start = 0):
    d = read_train_datas(count, start)
    mp = read_map()
    res = []
    for p in d:
        ls = p[1]
        X = []
        Y = []
        for i in range(len(ls)):
            X.extend(ls[i][1])
            Y.append(ls[i][2])
        res.append((X, Y))
    return res
    
def read_tmodels(count = 100):
    f = open(DATA_PATH['test'])
    mp = read_map()
    res = []
    X = []
    Y = []
    labs = []
    lastlab = ''

    cnt = 0
    for line in f:
        aa = line.strip('\n').split()
        lb = aa[0]
        stl = len(lb.split('_')[-1])
        lb_pre = lb[:-stl-1]
        if lb_pre != lastlab:
            if cnt >= count:
                break
            cnt += 1
            if lastlab != '':
                res.append((X, Y))
            lastlab = lb_pre
            labs.append(lb_pre)
            X = []
            Y = []
        feat = [float(i) for i in aa[1:]]
        X.extend(feat)
        Y.append('aa')
    res.append((X, Y))

    f.close()
    return labs, res

def read_hw1input(path, Y):
    f = open(path)
    ln = [x.strip('\n').split(',')[1] for x in f]
    cnt = 0
    res = []
    for y in Y:
        l = len(y)
        res.append(ln[cnt:cnt+l])
        cnt += l

    f.close()
    return res

def read_hw1_matrix(path, Y):
    f = open(path)
    ln = [[float(y) for y in x.strip('\n').split()[1:]] for x in f]
    ln = np.array(ln)

    cnt = 0
    res = []
    for y in Y:
        l = len(y)
        res.append(ln[cnt:cnt+l,:])
        cnt += l

    f.close()
    return res
