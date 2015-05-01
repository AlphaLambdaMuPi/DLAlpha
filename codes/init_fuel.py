from settings import *
import shelve
import numpy as np
import os
from os.path import join as pjoin
from phomap import ph2id
from progressbar import ProgressBar
import logging
import random
import h5py
from fuel.datasets.hdf5 import Hdf5Dataset, H5PYDataset
lgr = logging.getLogger()

def init(valp, shuffle, normalize, prefix):
    numpy_path = pjoin(PATH['data'], 'numpy')
    fuel_path = pjoin(PATH['data'], 'fuel')
    for p in (numpy_path, fuel_path):
        if not os.path.isdir(p):
            lgr.info('Path {} not found, mkdir!'.format(p))
            os.mkdir(p)


    lgr.info('start building numpy datas, becareful for swap out! (need 4G)')
    lgr.info('Build train data.')
    with shelve.open(SHELVE['train']) as f:
        names = f['names']
        if shuffle: random.shuffle(names)
        fet = []
        lab = []
        with ProgressBar(maxval=len(names)) as progbar:
            cnt = 0
            for n in names:
                tfet = []
                tlab = []
                dt = f[n]
                for fr in dt:
                    tfet.append(fr[1] + fr[2])
                    tlab.append([ph2id(fr[3])])
                cnt += 1
                for i in range(len(tfet)):
                    ff = []
                    for j in range(-2, 2+1):
                        z = (i+j*2) % len(tlab)
                        ff.extend(tfet[z])
                    fet.append(ff)
                    lab.append(tlab[i])
                progbar.update(cnt)


    #print(fe_array[:5], np.mean(fe_array, axis=

    tr_n = int(len(fet) * (1 - valp))

    
    train_features = np.asarray(fet, np.float32)
    train_targets = np.asarray(lab, np.uint8)

    if normalize:
        train_features = ((train_features - np.mean(train_features, axis=0))
                / (np.std(train_features, axis=0)) + 1E-2)

    def save_h5py(tn, start, stop):
        cf = train_features[start:stop]
        ct = train_targets[start:stop]
        np.save(pjoin(numpy_path, prefix+tn+'_features.npy'), cf)
        np.save(pjoin(numpy_path, prefix+tn+'_targets.npy'), ct)
        h5 = h5py.File(pjoin(fuel_path, prefix+tn+'.hdf5'), mode='w')
        h5_features = h5.create_dataset(
            'features', cf.shape, dtype='float32')
        h5_features[...] = cf
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
        lab = []
        with ProgressBar(maxval=len(names)) as progbar:
            cnt = 0
            for n in names:
                dt = f[n]
                for fr in dt:
                    fet.append(fr[1] + fr[2])
                cnt += 1
                if cnt >= 1: break
                progbar.update(cnt)

    features = np.asarray(fet, np.float32)
    labels = np.asarray(lab, np.int32)
    np.save(pjoin(numpy_path, prefix+'test_features.npy'), features)



if __name__ == '__main__':
    init(0.1, True, True, '')
