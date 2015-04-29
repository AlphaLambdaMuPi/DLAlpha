from settings import *
import shelve
import numpy as np
import os
from os.path import join as pjoin
from phomap import ph2id
from progressbar import ProgressBar
import logging
import h5py
from fuel.datasets.hdf5 import Hdf5Dataset, H5PYDataset
lgr = logging.getLogger()

def init(valp, wr):
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
        fet = []
        lab = []
        with ProgressBar(maxval=len(names)) as progbar:
            cnt = 0
            for n in names:
                dt = f[n]
                for fr in dt:
                    fet.append(fr[1] + fr[2])
                    lab.append([ph2id(fr[3])])
                cnt += 1
                if cnt >= 400: break
                progbar.update(cnt)

    tr_n = int(len(fet) * (1 - valp))
    #print(len(fet), len(lab))
    train_features = np.asarray(fet, np.float32)
    train_targets = np.asarray(lab, np.uint8)
    #print(len(fet), len(lab), train_targets.shape, train_features.shape)
    #return
    np.save(pjoin(numpy_path, 'train_features.npy'), train_features)
    np.save(pjoin(numpy_path, 'train_targets.npy'), train_targets)
    h5 = h5py.File(pjoin(fuel_path, 'train.hdf5'), mode='w')
    h5_features = h5.create_dataset(
        'features', train_features.shape, dtype='float32')
    h5_features[...] = train_features
    h5_targets = h5.create_dataset(
        'targets', train_targets.shape, dtype='uint8')
    h5_targets[...] = train_targets
    h5_features.dims[0].label = 'batch'
    h5_features.dims[1].label = 'feature'
    h5_targets.dims[0].label = 'batch'
    h5_targets.dims[1].label = 'index'

    #split_array = np.empty(4,
        #[('split', 'a', 10), ('source', 'a', 10),
         #('start', np.int64, 1), ('stop', np.int64, 1),
         #('available', np.bool, 1), ('comment', 'a', 1)])
    #split_array[0:2]['split'] = 'train'.encode('utf8')
    #split_array[2:4]['split'] = 'validate'.encode('utf8')
    #split_array[0:4:2]['source'] = 'features'.encode('utf8')
    #split_array[1:4:2]['source'] = 'targets'.encode('utf8')
    #split_array[0:2]['start'] = 0
    #split_array[0:2]['stop'] = tr_n
    #split_array[2:4]['start'] = tr_n
    #split_array[2:4]['stop'] = len(fet)
    #split_array[:]['available'] = True
    #split_array[:]['comment'] = '.'
    #h5.attrs['split'] = split_array
    split_dict = {
        'train': {'features': (0, tr_n), 'targets': (0, tr_n)},
        'validate': {'features': (tr_n, len(fet)), 'targets': (tr_n, len(fet))},
    }
    #split_dict = {
        #'train': {'features': (0, tr_n), 'targets': (0, tr_n)},
    #}
    h5.attrs['split'] = H5PYDataset.create_split_array(split_dict)
    h5.flush()
    h5.close()
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
    np.save(pjoin(numpy_path, 'test_features.npy'), features)

    if wr:
        p = pjoin(os.path.expanduser('~'), '.fuelrc')
        with open(p, 'w') as f:
            f.write('data_path: {}'.format(fuel_path))





if __name__ == '__main__':
    init(0.1, True)
