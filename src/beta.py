import numpy as np
import h5py
from fuel.datasets.hdf5 import Hdf5Dataset, H5PYDataset

def make_data(xxt, yyt, path):
    h5 = h5py.File(path, mode='w')
    h5_features = h5.create_dataset('features', xxt.shape, dtype='float32')
    h5_features[...] = xxt
    h5_targets = h5.create_dataset('targets', yyt.shape, dtype='uint16')
    h5_targets[...] = yyt
    h5_features.dims[0].label = 'batch'
    h5_features.dims[1].label = 'feature'
    h5_targets.dims[0].label = 'batch'
    h5_targets.dims[1].label = 'index'
    split_dict = {
        'train': {'features': (0, xxt.shape[0]), 'targets': (0, yyt.shape[0])},
    }
    h5.attrs['split'] = H5PYDataset.create_split_array(split_dict)
    h5.flush()
    h5.close()

fet = np.load('newfeat_train.npy').astype('float32')
lab = np.load('../data/numpy/utt3_train_targets.npy')
lab = lab.reshape((lab.shape[0], 1)).astype('float32')
path = 'utt4_train_features.npy'
make_data(fet, lab, path)
