from settings import *
import shelve
import numpy as np
import os
from os.path import join as pjoin
from progressbar import ProgressBar
import logging
import random
import h5py
from fuel.datasets.hdf5 import Hdf5Dataset, H5PYDataset
lgr = logging.getLogger()

def init(valp, shuffle, normalize, prefix, concat, limit):
    if prefix == 'konkon' or prefix == 'concon':
        prefix = ''
    else: prefix += '_'
    from phomap import ph2id
    numpy_path = pjoin(PATH['data'], 'numpy')
    fuel_path = pjoin(PATH['data'], 'fuel')
    for p in (numpy_path, fuel_path):
        if not os.path.isdir(p):
            lgr.info('Path {} not found, mkdir!'.format(p))
            os.mkdir(p)

    points = [(-1, 1), (0, 1)]

    lgr.info('start building numpy datas, becareful for swap out! (need 4G)')
    lgr.info('Build train data.')
    with shelve.open(SHELVE['train']) as f:
        names = f['names']
        if shuffle: random.shuffle(names)
        fet = []
        lab = []
        with ProgressBar(maxval=min(len(names), limit)) as progbar:
            cnt = 0
            for n in names:
                dt = f[n]
                for fr in dt:
                    fet.append(fr[1] + fr[2])
                labs = [fr[3] for fr in dt]
                labsl = len(labs)
                i = 0
                while i < labsl:
                    j = i
                    while j < labsl and labs[i] == labs[j]:
                        j += 1
                    ql = j - i
                    getl = max(ql//5, 1)
                    a1 = [[1]] * getl if i != 0 else [[0]]*getl
                    a2 = [[0]] * (ql - getl * 2)
                    a3 = [[1]] * getl if j != labsl else [[0]]*getl
                    lab.extend(a1+a2+a3)
                    i = j
                cnt += 1
                if cnt >= limit: break
                progbar.update(cnt)


    print(lab)
    #print(fe_array[:5], np.mean(fe_array, axis=

    tr_n = int(len(fet) * (1 - valp))

    
    train_features = np.asarray(fet, np.float32)
    train_targets = np.asarray(lab, np.int64)

    if normalize:
        train_features = ((train_features - np.mean(train_features, axis=0))
                / (np.std(train_features, axis=0)) + 1E-2)

    mult = len(range(-concat[0], concat[0]+1, concat[1]))
    def save_h5py(tn, start, stop):
        cf = train_features[start:stop]
        ct = train_targets[start:stop]
        np.save(pjoin(numpy_path, prefix+tn+'_features.npy'), cf)
        np.save(pjoin(numpy_path, prefix+tn+'_targets.npy'), ct)
        h5 = h5py.File(pjoin(fuel_path, prefix+tn+'.hdf5'), mode='w')
        h5_features = h5.create_dataset(
            'features', (cf.shape[0], cf.shape[1]*mult)
            , dtype='float32')
        lenf = stop - start
        with ProgressBar(maxval=lenf) as progbar:
            for i in range(lenf):
                arr = []
                for j in range(-concat[0], concat[0]+1, concat[1]):
                    arr.extend(cf[(i-j)%lenf])
                h5_features[i] = np.asarray(arr)
                progbar.update(i)

        h5_targets = h5.create_dataset(
            'targets', ct.shape, dtype='uint8')
        h5_targets[...] = ct
        h5_features.dims[0].label = 'batch'
        h5_features.dims[1].label = 'feature'
        h5_targets.dims[0].label = 'batch'
        h5_targets.dims[1].label = 'index'

        split_dict = {
            tn: {'features': (0, stop-start), 'targets': (0, stop-start)},
            #'validate': {'features': (tr_n, len(fet)), 'targets': (tr_n, len(fet))},
        }
        h5.attrs['split'] = H5PYDataset.create_split_array(split_dict)
        h5.flush()
        h5.close()

    save_h5py('train', 0, tr_n)
    save_h5py('validate', tr_n, train_features.shape[0])
    del fet, lab

    lgr.info('Build test data.')
    with shelve.open(SHELVE['test']) as f:
        names = f['names']
        fet = []
        with ProgressBar(maxval=len(names)) as progbar:
            cnt = 0
            for n in names:
                dt = f[n]
                for fr in dt:
                    fet.append(fr[1] + fr[2])
                progbar.update(cnt)

    feat = np.asarray(fet, np.float32)
    lenf = len(feat)
    features = np.empty((feat.shape[0], feat.shape[1]*mult))

    with ProgressBar(maxval=lenf) as progbar:
        for i in range(lenf):
            arr = []
            for j in range(-concat[0], concat[0]+1, concat[1]):
                arr.extend(feat[(i-j)%lenf])
            features[i] = np.asarray(arr)
            progbar.update(i)

    np.save(pjoin(numpy_path, prefix+'test_features.npy'), features)



if __name__ == '__main__':
    init(0.1, True, True, 'border', [3, 1], 2000)
