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

def init(valp, shuffle, normalize, prefix, concat, limit, state):
    limit = limit[0]
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


    lgr.info('start building numpy datas, becareful for swap out! (need 4G)')
    lgr.info('Build train data.')
    shelve_type = 'train_state' if state else 'train'

    # tr_n = int(len(fet) * (1 - valp))
    tr_n = 0

    spo = 0
    sp_offset = []
    sp_offsetp = []
    with shelve.open(SHELVE[shelve_type]) as f:
        names = f['names']
        utt = {}
        for n in names:
            u, s = n.split('_')
            if u not in utt:
                utt[u] = []
            utt[u].append(s)
        speakers = list(utt.keys())
        speakers.sort()
        if shuffle: random.shuffle(speakers)
        fet = []
        lab = []
        spc = min(len(speakers), limit)
        with ProgressBar(maxval=spc) as progbar:
            cnt = 0
            fcnt = 0
            for i, s in enumerate(speakers):
                if i == int(spc * (1-valp)):
                    tr_n = fcnt
                    spo = fcnt
                    sp_offsetp = sp_offset
                    sp_offset = []
                for u in utt[s]:
                    n = s + '_' + u
                    dt = f[n]
                    for fr in dt:
                        fet.append(fr[1] + fr[2])
                        if not state:
                            lab.append([ph2id(fr[3])])
                        else:
                            lab.append([fr[3]])
                        fcnt += 1
                cnt += 1
                progbar.update(cnt)
                sp_offset.append(fcnt - spo)
                if cnt >= limit: break

    sp_offset = np.array(sp_offset)
    sp_offsetp = np.array(sp_offsetp)

    #print(fe_array[:5], np.mean(fe_array, axis=

    np.save(pjoin(numpy_path, prefix+'train_spoffset.npy'), sp_offsetp)
    np.save(pjoin(numpy_path, prefix+'validate_spoffset.npy'), sp_offset)

    
    train_features = np.asarray(fet, np.float32)
    train_targets = np.asarray(lab, np.uint16)

    tr_mean = np.mean(train_features, axis=0)
    tr_std = np.std(train_features, axis=0) + 1E-2

    if normalize:
        train_features = (train_features - tr_mean) / tr_std

    mult = len(range(-concat[0], concat[0]+1, concat[1]))
    def save_h5py(tn, start, stop, save_feat=False):
        cf = train_features[start:stop]
        ct = train_targets[start:stop]
        if not save_feat:
            np.save(pjoin(numpy_path, prefix+tn+'_features.npy'), cf)
        np.save(pjoin(numpy_path, prefix+tn+'_targets.npy'), ct)
        h5 = h5py.File(pjoin(fuel_path, prefix+tn+'.hdf5'), mode='w')
        h5_features = h5.create_dataset(
            'features', (cf.shape[0], cf.shape[1]*mult)
            , dtype='float32')
        lenf = stop - start
        if save_feat:
            tot_feat = []
        with ProgressBar(maxval=lenf) as progbar:
            for i in range(lenf):
                arr = []
                for j in range(-concat[0], concat[0]+1, concat[1]):
                    arr.extend(cf[(i-j)%lenf])
                arr = np.asarray(arr)
                h5_features[i] = arr
                if save_feat:
                    tot_feat.append(arr)
                progbar.update(i)
        if save_feat:
            tot_feat = np.array(tot_feat)
            np.save(pjoin(numpy_path, prefix+tn+'_features.npy'), tot_feat)

        h5_targets = h5.create_dataset(
            'targets', ct.shape, dtype='uint16')
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

    save_h5py('train', 0, tr_n, True)
    save_h5py('validate', tr_n, train_features.shape[0], True)
    del fet, lab

    lgr.info('Build test data.')
    spo = 0
    sp_offset = []
    with shelve.open(SHELVE['test']) as f:
        names = f['names']
        fet = []
        cnt = 0
        cur_name = "concon"
        with ProgressBar(maxval=len(names)) as progbar:
            num = 0
            for n in names:
                u, s = n.split('_')
                if u != cur_name:
                    sp_offset.append(num)
                    cur_name = u
                dt = f[n]
                for fr in dt:
                    fet.append(fr[1] + fr[2])
                    num += 1
                progbar.update(cnt)
            sp_offset.append(num)

    feat = np.asarray(fet, np.float32)
    lenf = len(feat)
    features = np.empty((feat.shape[0], feat.shape[1]*mult))

    if normalize:
        feat = (feat - tr_mean) / tr_std

    with ProgressBar(maxval=lenf) as progbar:
        for i in range(lenf):
            arr = []
            for j in range(-concat[0], concat[0]+1, concat[1]):
                arr.extend(feat[(i-j)%lenf])
            features[i] = np.asarray(arr)
            progbar.update(i)


    np.save(pjoin(numpy_path, prefix+'test_features.npy'), features)
    np.save(pjoin(numpy_path, prefix+'test_spoffset.npy'), np.asarray(sp_offset[1:]))

if __name__ == '__main__':
    init(0.1, True, True, 'concat', [4, 2])
