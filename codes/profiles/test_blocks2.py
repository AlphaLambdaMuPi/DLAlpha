import numpy as np
import theano.tensor as T
from theano import config, scan
from blocks.bricks import MLP, Rectifier, Softmax
from blocks.bricks.recurrent import *
from blocks.initialization import IsotropicGaussian, Constant
import fuel, blocks
from fuel.datasets.hdf5 import H5PYDataset
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme, ShuffledScheme
from blocks.initialization import IsotropicGaussian, Constant
from blocks.algorithms import *
from blocks.bricks import WEIGHT
from blocks.roles import INPUT, DROPOUT
from blocks.filter import VariableFilter
from blocks.bricks.cost import CategoricalCrossEntropy, MisclassificationRate
from blocks.graph import ComputationGraph, apply_dropout
from blocks.main_loop import MainLoop
from blocks.extensions import FinishAfter, Printing
from blocks.extensions.monitoring import DataStreamMonitoring
from os.path import join as pjoin
from settings import *
from phomap import ph48239, id2ph, ph2id

class Executor:
    def __init__(self):
        pass

    def start(self):
        x = T.matrix('features', config.floatX)
        y = T.imatrix('targets')
        mlp = MLP(activations = [
            Rectifier(name='r0'), 
            Rectifier(name='r1'), 
            Rectifier(name='r2'), 
            # Rectifier(name='r3'), 
            Softmax(name='rs')
        ],
             dims=[108*7, 200, 200, 200, 48], weights_init=IsotropicGaussian(std=0.05, mean=0), biases_init=IsotropicGaussian(std=0.1))
        # mlp = SimpleRecurrent(dim=200, activation=Rectifier(), weights_init=IsotropicGaussian(std=0.1))
        y_hat = mlp.apply(x)
        # y_hat = Softmax().apply(mlp.apply(x))
        cost = CategoricalCrossEntropy().apply(y.flatten(), y_hat).astype(config.floatX)
        cost.name = 'cost'

        mps = theano.shared(np.array([ph2id(ph48239(id2ph(t))) for t in range(48)]))
        print(mps.get_value(), y.flatten(), y_hat)
        z_hat = T.argmax(y_hat, axis=1)

        y39,_ = scan(fn=lambda t: mps[t], outputs_info=None, sequences=[y.flatten()])
        y_hat39,_ = scan(fn=lambda t: mps[t], outputs_info=None, sequences=[z_hat])

        lost01 = T.neq(y_hat39, y39).astype(config.floatX) / y39.shape[0]
        lost01.name = '0/1 loss'
        lost23 = T.neq(y_hat39, y39).astype(config.floatX) / y39.shape[0]
        #lost23 = MisclassificationRate().apply(y39, y_hat39).astype(config.floatX)
        lost23.name = '2/3 loss'
        cg = ComputationGraph(cost)

        # inputs = VariableFilter(roles=[WEIGHT])(cg.variables)
        # cg = apply_dropout(cg, inputs, 0.5)
        # cost = cg.outputs[0]

        Ws = VariableFilter(roles=[WEIGHT])(cg.variables)
        norms = sum(w.norm(2) for w in Ws)
        norms.name = 'norms'
        mlp.initialize()
        path = pjoin(PATH['fuel'], 'train.hdf5')
        data = H5PYDataset(path, which_set='train', load_in_memory=True, subset=slice(0, 100))
        # data = H5PYDataset(path, which_set='train', load_in_memory=Tr5e)
        data_v = H5PYDataset(pjoin(PATH['fuel'], 'validate.hdf5'), which_set='validate', load_in_memory=True)
        num = data.num_examples
        data_stream = DataStream(data, iteration_scheme=ShuffledScheme(
                        num, batch_size=128))
        data_stream_v = DataStream(data_v, iteration_scheme=SequentialScheme(
                        data_v.num_examples, batch_size=128))
        algo = GradientDescent(cost=cost, params=cg.parameters, step_rule=CompositeRule([Momentum(0.001, 0.9)]))
        monitor = DataStreamMonitoring( variables=[cost, lost01, norms],
                data_stream=data_stream)
        monitor_v = DataStreamMonitoring( variables=[lost23],
                data_stream=data_stream_v)
        main_loop = MainLoop(data_stream = data_stream, 
                algorithm=algo, 
                extensions=[monitor, monitor_v, FinishAfter(after_n_epochs=2000), Printing()])
        
        main_loop.run()
