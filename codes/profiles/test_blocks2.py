import numpy as np
import pickle
import theano.tensor as T
from theano import config, scan
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
from phomap import ph48239, id2ph, ph2id
from profile import BaseExecutor

class Executor(BaseExecutor):
    def __init__(self):
        NAME = 'Strange_Dropout'
        super().__init__(name=NAME, test_file='c42_test_features.npy')
        pass

    def get_io(self):
        return self.x, self.y_hat39

    def start(self):
        x = T.matrix('features', config.floatX)
        y = T.imatrix('targets')

        self.x = x

        DIMS = [108*5, 1000, 1000, 1000, 48]
        NUMS = [1, 1, 1, 1, 1]
        FUNCS = [
            # Rectifier(), 
            # Rectifier(), 
            # Rectifier(), 
            # Rectifier(), 
            # Rectifier(), 
            # Maxout(num_pieces=5),
            # Maxout(num_pieces=5),
            # Maxout(num_pieces=5),
            # SimpleRecurrent,
            # SimpleRecurrent,
            # SimpleRecurrent,
            Softmax
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

        oup = x
        for i in range(len(DIMS)-1):
            oup = lllistool(i, oup, FUNCS[i])
        y_hat = oup

        cost = CategoricalCrossEntropy().apply(y.flatten(), y_hat).astype(config.floatX)

        cg = ComputationGraph(cost)
<<<<<<< HEAD
        ips = VariableFilter(roles=[INPUT])(cg.variables)
        ops = VariableFilter(roles=[OUTPUT])(cg.variables)
        cg = apply_dropout(cg, ips[0:2:1], 0.2)

        cg = apply_dropout(cg, ips[2:-2:1], 0.5)
        cost = cg.outputs[0]


        cg = ComputationGraph(cost)
=======
        orig_cg = cg
>>>>>>> Alpa
        ips = VariableFilter(roles=[INPUT])(cg.variables)
        ops = VariableFilter(roles=[OUTPUT])(cg.variables)
        cg = apply_dropout(cg, ips[0:2:1], 0.2)
        cg = apply_dropout(cg, ips[2:-2:1], 0.5)
        cost = cg.outputs[0]

        cost.name = 'cost'

        mps = theano.shared(np.array([ph2id(ph48239(id2ph(t))) for t in range(48)]))
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
        norms = sum(w.norm(2) for w in Ws)
        norms.name = 'norms'
        path = pjoin(PATH['fuel'], 'r42train.hdf5')
        data = H5PYDataset(path, which_set='train', load_in_memory=True)#, subset=slice(0, 100000))
        # data = H5PYDataset(path, which_set='train', load_in_memory=True)
        data_v = H5PYDataset(pjoin(PATH['fuel'], 'r42validate.hdf5'), which_set='validate', load_in_memory=True)
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
        plt = Plot('Alp', channels=[['0/1 loss', '2/3 loss']], after_epoch=True)
        main_loop = MainLoop(data_stream = data_stream, 
                algorithm=algo, 
                extensions=[monitor, monitor_v, FinishAfter(after_n_epochs=2000), Printing(), plt])
        
        main_loop.run()

<<<<<<< HEAD
=======
        pfile = open('beta.pkl', 'wb')
        pickle.dump(orig_cg, pfile)
        pickle.dump(cg, pfile)
        pfile.close()

>>>>>>> Alpa
