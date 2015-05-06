import numpy as np
import pickle
import theano.tensor as T
from theano import config, scan, function
from blocks.bricks import MLP, Rectifier, Softmax, Linear, Maxout
from blocks.bricks.recurrent import *
from blocks.initialization import IsotropicGaussian, Constant
import fuel, blocks
from fuel.datasets.hdf5 import H5PYDataset
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme, ShuffledScheme
from blocks.initialization import IsotropicGaussian, Constant
from blocks.algorithms import *
from blocks.bricks import WEIGHT
from blocks.roles import INPUT, DROPOUT, PARAMETER, OUTPUT
from blocks.filter import VariableFilter
from blocks.bricks.cost import CategoricalCrossEntropy, MisclassificationRate
from blocks.graph import ComputationGraph, apply_dropout
from blocks.main_loop import MainLoop
from blocks.extensions import FinishAfter, Printing
from blocks.extensions.plot import Plot
from blocks.extensions.monitoring import DataStreamMonitoring
from os.path import join as pjoin
from settings import *
from phomap import ph48239, id2ph, ph2id, state239
from profile import BaseExecutor
import h5py
from fuel.datasets.hdf5 import Hdf5Dataset, H5PYDataset

pfx = 'utt'

class Executor(BaseExecutor):
    def __init__(self):
        NAME = 'Strange_Dropout'
        path = PATH['numpy']+'/'+pfx+'_test_features.npy'
        self.test_file = path
        super().__init__(name=NAME, test_file=path)
        pass

    def get_io(self):
        return self.x, self.y_hat_prob

    def start(self):
        x = T.matrix('features', config.floatX)
        y = T.imatrix('targets')

        self.x = x

        DIMS = [108*5, 1000, 1000, 1000, 1000, 48]
        NUMS = [1, 1, 1, 1, 1, 1]
        FUNCS = [
            Rectifier, 
            Rectifier, 
            Rectifier, 
            Rectifier, 
            # Rectifier, 
            # Maxout(num_pieces=5),
            # Maxout(num_pieces=5),
            # Maxout(num_pieces=5),
            # SimpleRecurrent,
            # SimpleRecurrent,
            # SimpleRecurrent,
            Softmax,
        ]

        def lllistool(i, inp, func):
            l = Linear(input_dim=DIMS[i], output_dim=DIMS[i+1] * NUMS[i+1], 
                       weights_init=IsotropicGaussian(std=DIMS[i]**(-0.5)), 
                       biases_init=IsotropicGaussian(std=DIMS[i]**(-0.5)),
                       name='Lin{}'.format(i))
            l.initialize()
            func.name='Fun{}'.format(i)
            if func == SimpleRecurrent:
                gong = func(dim=DIMS[i+1], activation=Rectifier(), weights_init=IsotropicGaussian(std=(DIMS[i]+DIMS[i+1])**(-0.5)))
            else:
                gong = func()
            ret = gong.apply(l.apply(inp))
            return ret

        satW = theano.shared(np.identity(108).astype(config.floatX))
        satC = theano.shared(np.zeros((108, )).astype(config.floatX))
        # satW2 = theano.shared(np.identity(108).astype(config.floatX))
        # satC2 = theano.shared(np.zeros((108, )).astype(config.floatX))

        xx = []
        for i in range(DIMS[0]//108):
            ans = T.dot(x[:,i*108:(i+1)*108], satW) + satC
            # ans = Rectifier().apply(ans)
            # ans = T.dot(ans, satW2) + satC2
            xx.append(ans)
        xx = T.concatenate(xx, axis=1)

        oup = xx
        for i in range(len(DIMS)-1):
            oup = lllistool(i, oup, FUNCS[i])
        y_hat = oup

        self.y_hat_prob = y_hat

        cost = CategoricalCrossEntropy().apply(y.flatten(), y_hat).astype(config.floatX)

        cg = ComputationGraph(cost)
        orig_cg = cg
        ips = VariableFilter(roles=[INPUT])(cg.variables)
        ops = VariableFilter(roles=[OUTPUT])(cg.variables)
        cg = apply_dropout(cg, ips[0:2:1], 0.2)
        cg = apply_dropout(cg, ips[2:-2:1], 0.5)
        cost = cg.outputs[0]

        cost.name = 'cost'

        mps = theano.shared(np.array([ph2id(ph48239(id2ph(t))) for t in range(48)]))
        # mps = theano.shared(np.array([ph2id(state239(t)) for t in range(1943)]))
        z_hat = T.argmax(y_hat, axis=1)

        y39,_ = scan(fn=lambda t: mps[t], outputs_info=None, sequences=[y.flatten()])
        y_hat39,_ = scan(fn=lambda t: mps[t], outputs_info=None, sequences=[z_hat])

        self.y_hat39 = y_hat39

        lost01 = (T.sum(T.neq(y_hat39, y39)) / y39.shape[0]).astype(config.floatX)
        lost01.name = '0/1 loss'
        lost23 = (T.sum(T.neq(y_hat39, y39)) / y39.shape[0]).astype(config.floatX)
        #lost23 = MisclassificationRate().apply(y39, y_hat39).astype(config.floatX)
        lost23.name = '2/3 loss'


        Ws = VariableFilter(roles=[WEIGHT])(cg.variables)
        # print(cg.parameters)
        norms = sum(w.norm(2) for w in Ws)
        norms.name = 'norms'
        path = pjoin(PATH['fuel'], pfx+'_train.hdf5')
        # data = H5PYDataset(path, which_set='train', load_in_memory=True, subset=slice(0, 100000))
        data = H5PYDataset(path, which_set='train', load_in_memory=True)
        data_v = H5PYDataset(pjoin(PATH['fuel'], pfx+'_validate.hdf5'), which_set='validate', load_in_memory=True)
        num = data.num_examples
        data_stream = DataStream(data, iteration_scheme=ShuffledScheme(
                        num, batch_size=128))
        data_stream_v = DataStream(data_v, iteration_scheme=SequentialScheme(
                        data_v.num_examples, batch_size=128))
        algo = GradientDescent(cost=cost, params=cg.parameters, step_rule=CompositeRule([Momentum(0.002, 0.9)]))
        monitor = DataStreamMonitoring( variables=[cost, lost01, norms],
                data_stream=data_stream)
        monitor_v = DataStreamMonitoring( variables=[lost23],
                data_stream=data_stream_v)
        plt = Plot('AlpALP', channels=[['0/1 loss', '2/3 loss']], after_epoch=True)
        main_loop = MainLoop(data_stream = data_stream, 
                algorithm=algo, 
                extensions=[monitor, monitor_v, FinishAfter(after_n_epochs=2000), Printing(), plt])
        
        main_loop.run()

        # return
        ##############

        vpath = PATH['numpy']+'/'+pfx+'_validate_features.npy'
        xt = np.load(vpath)
        vpath = PATH['numpy']+'/'+pfx+'_validate_targets.npy'
        ryt = np.load(vpath)
        vpath = PATH['numpy']+'/'+pfx+'_validate_spoffset.npy'
        sp_offset = np.load(vpath)
        tmppath = PATH['fuel']+'/alaltmp.hdf5'

        func = theano.function([x], z_hat)

        start = 0
        totaldiff = 0
        tcnt = 0
        for i in sp_offset:
            end = i
            satW.set_value(np.identity(108).astype(config.floatX))
            satC.set_value(np.zeros((108, )).astype(config.floatX))
            # satW2.set_value(np.identity(108).astype(config.floatX))
            # satC2.set_value(np.zeros((108, )).astype(config.floatX))
            xxt = xt[start:end]
            ryyt = ryt[start:end]
            yyt = func(xxt)
            yyt = yyt.reshape((yyt.shape[0], 1))

            score = np.sum(ryyt != yyt) / yyt.shape[0]
            print(score)

            h5 = h5py.File(tmppath, mode='w')
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

            # params = [satW, satC, satW2, satC2]
            params = [satW, satC]
            data_sat = H5PYDataset(tmppath, which_set='train', load_in_memory=True)
            data_stream = DataStream(data_sat, iteration_scheme=ShuffledScheme(
                        data_sat.num_examples, batch_size=64))
            algo = GradientDescent(cost=cost, params=params, step_rule=CompositeRule([Momentum(0.003, 0.9)]))

            monitor_v = DataStreamMonitoring( variables=[lost23, cost],
                data_stream=data_stream)
            main_loop = MainLoop(data_stream = data_stream, algorithm=algo, 
                extensions=[FinishAfter(after_n_epochs=50)])
                # extensions=[monitor_v, FinishAfter(after_n_epochs=500), Printing()])
        
            main_loop.run()


            yyt = func(xxt)
            yyt = yyt.reshape((yyt.shape[0], 1))
            score2 = np.sum(ryyt != yyt) / yyt.shape[0]
            diff = score2 - score
            totaldiff += diff
            tcnt += 1
            print(score2, diff, totaldiff / tcnt)

            start = end
