import theano.tensor as T
from theano import config
from blocks.bricks import MLP, Rectifier, Softmax
from blocks.initialization import IsotropicGaussian, Constant
import fuel, blocks
from fuel.datasets.hdf5 import H5PYDataset
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme
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
        self.x = T.matrix('features', config.floatX)
        self.y = T.imatrix('targets')
        self.mlp = MLP(activations = [Rectifier(name='r0'), Rectifier(name='r2'), Softmax(name='r1')],
                  dims=[108, 200, 200, 48], weights_init=IsotropicGaussian(std=0.1, mean=0), biases_init=IsotropicGaussian(std=0.1))
        self.y_hat = self.mlp.apply(self.x)
        self.cost = CategoricalCrossEntropy().apply(self.y.flatten(), self.y_hat)
        self.lost01 = MisclassificationRate().apply(self.y.flatten(), self.y_hat)
        self.cg = ComputationGraph(self.cost)
        self.Ws = VariableFilter(roles=[WEIGHT])(self.cg.variables)
        self.norms = self.Ws[0].norm(2) + self.Ws[1].norm(2) + self.Ws[2].norm(2)
        self.norms.name = 'norms'
        self.mlp.initialize()

    def start(self):
        path = pjoin(PATH['fuel'], 'train.hdf5')
        data = H5PYDataset(path, which_set='train')
        num = data.num_examples
        data_t = H5PYDataset(path, which_set='train', subset=slice(0, int(num*0.9)))
        #data_v = H5PYDataset(path, which_set='train', subset=slice(int(num*0.1), num))
        data_v = data_t
        #data = fuel.datasets.MNIST('train')
        #data_v = fuel.datasets.MNIST('test')
        self.data_stream = DataStream(data, iteration_scheme=SequentialScheme(
                        data.num_examples, batch_size=128))
        self.algo = GradientDescent(cost=self.cost, params=self.cg.parameters, step_rule=CompositeRule([AdaDelta()]))
        monitor = DataStreamMonitoring( variables=[self.lost01, self.norms],
                data_stream=self.data_stream)
        main_loop = MainLoop(data_stream = self.data_stream, 
                algorithm=self.algo, 
                extensions=[monitor, FinishAfter(after_n_epochs=30), Printing()])
        main_loop.run()
            
    def read(self):
        return read_train_by_group(100)

    def preproc(self, inp):
        return pair_to_np(flatten(inp))

    def preprocX(self, inp):
        return inp

    def preprocY(self, inp):
        return y_to_01(inp, 48)

    def split_validate(self, X, Y):
        n = X.shape[0]
        tr_n = n * self.alpha
        return X[:tr_n], Y[:tr_n], X[tr_n:], Y[tr_n:]
        
