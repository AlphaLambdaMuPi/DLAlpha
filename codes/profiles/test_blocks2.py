import theano.tensor as T
from theano import config
from blocks.bricks import MLP, Rectifier, Softmax
from blocks.initialization import IsotropicGaussian, Constant
import fuel, blocks
from fuel.datasets.hdf5 import H5PYDataset
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme, ShuffledScheme
from blocks.initialization import IsotropicGaussian, Constant
from blocks.algorithms import GradientDescent, Scale, AdaDelta, StepClipping, CompositeRule
from blocks.bricks import WEIGHT
from blocks.filter import VariableFilter
from blocks.bricks.cost import CategoricalCrossEntropy, MisclassificationRate
from blocks.graph import ComputationGraph
from blocks.main_loop import MainLoop
from blocks.extensions import FinishAfter, Printing
from blocks.extensions.monitoring import DataStreamMonitoring
from os.path import join as pjoin
from settings import *

class Executor:
    def __init__(self):
        pass

    def start(self):
        x = T.matrix('features', config.floatX)
        y = T.imatrix('targets')
        mlp = MLP(activations = [Rectifier(name='r0'), Rectifier(name='r1'), Softmax(name='r2')],
             dims=[108, 200, 200, 48], weights_init=IsotropicGaussian(std=0.1, mean=0), biases_init=IsotropicGaussian(std=0.1))
        y_hat = mlp.apply(x)
        cost = CategoricalCrossEntropy().apply(y.flatten(), y_hat)
        lost01 = MisclassificationRate().apply(y.flatten(), y_hat)
        cg = ComputationGraph(cost)
        Ws = VariableFilter(roles=[WEIGHT])(cg.variables)
        norms = Ws[0].norm(2) + Ws[1].norm(2) + Ws[2].norm(2)
        norms.name = 'norms'
        mlp.initialize()
        path = pjoin(PATH['fuel'], 'train.hdf5')
        data = H5PYDataset(path, which_set='train')
        data_v = H5PYDataset(pjoin(PATH['fuel'], 'validate.hdf5'), which_set='validate')
        num = data.num_examples
        data_stream = DataStream(data, iteration_scheme=SequentialScheme(
                        data.num_examples, batch_size=128))
        data_stream_v = DataStream(data_v, iteration_scheme=SequentialScheme(
                        data_v.num_examples, batch_size=128))
        algo = GradientDescent(cost=cost, params=cg.parameters, step_rule=CompositeRule([AdaDelta()]))
        monitor = DataStreamMonitoring( variables=[cost, lost01, norms],
                data_stream=data_stream)
        monitor_v = DataStreamMonitoring( variables=[lost01],
                data_stream=data_stream_v)
        main_loop = MainLoop(data_stream = data_stream, 
                algorithm=algo, 
                extensions=[monitor, monitor_v, FinishAfter(after_n_epochs=30), Printing()])
        
        main_loop.run()
