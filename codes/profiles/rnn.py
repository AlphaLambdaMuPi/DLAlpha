import numpy as np
import pickle
import theano.tensor as T
from theano import config, scan, shared
from blocks.bricks import MLP, Rectifier, Softmax, Linear, Maxout, Identity
from blocks.bricks.recurrent import *
from blocks.bricks.sequence_generators import *
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

CONCON = None
HMM_RATIO = 0.01
SLEN = 256
BNUM = 4

from blocks.bricks.recurrent import BaseRecurrent, recurrent
class FeedbackRNN(BaseRecurrent):
    def __init__(self, dim, **kwargs):
        super(FeedbackRNN, self).__init__(**kwargs)
        self.dim = dim
        self.first_recurrent_layer = SimpleRecurrent(
            dim=self.dim, activation=Identity(), name='first_recurrent_layer',
            weights_init=initialization.Identity())
        self.second_recurrent_layer = SimpleRecurrent(
            dim=self.dim, activation=Identity(), name='second_recurrent_layer',
            weights_init=initialization.Identity())
        self.children = [self.first_recurrent_layer,
                         self.second_recurrent_layer]

    @recurrent(sequences=['inputs'], contexts=[],
               states=['first_states', 'second_states'],
               outputs=['first_states', 'second_states'])
    def apply(self, inputs, first_states=None, second_states=None):
        first_h = self.first_recurrent_layer.apply(
            inputs=inputs, states=first_states + second_states, iterate=False)
        second_h = self.second_recurrent_layer.apply(
            inputs=first_h, states=second_states, iterate=False)
        return first_h, second_h

    def get_dim(self, name):
        return (self.dim if name in ('inputs', 'first_states', 'second_states')
                else super(FeedbackRNN, self).get_dim(name))

def trim(x, cut=False):
    ret = []
    lst = -1
    cnt = 0
    for i in x:
        if i == lst:
            cnt += 1
            continue
        if (not cut) or cnt >= 3:
            ret.append(lst)
        lst = i
        cnt = 1
    if (not cut) or cnt >= 3:
        ret.append(lst)
    return ret

def Yimumu(inp, y):
    wlh = shared(np.zeros((48, 48)).astype(config.floatX))

    ws = T.log(T.sum(T.exp(wlh), axis=1))
    lw = - (wlh - ws)
    p0 = T.ones((inp.shape[1], ), dtype='int32') * 36
    t0 = T.zeros((inp.shape[1], ))
    def func(y, prv, pb):
        alp = T.sum(lw[prv,y])
        return [y, pb + alp]

    [trash, s2], _ = scan(fn=func, 
                          sequences=y, 
                          outputs_info=[p0, t0],
                          )
    j2 = HMM_RATIO * T.sum(s2) / inp.shape[1] / inp.shape[0]
    return j2, wlh

class Yimumu_Decode(theano.Op):
    # Properties attribute
    __props__ = ()

    def make_node(self, yh, wlh):
        # print(yh, wlh)
        yh = theano.tensor.as_tensor_variable(yh)
        wlh = theano.tensor.as_tensor_variable(wlh)
        # yt = theano.tensor.as_tensor_variable(yt)
        return theano.Apply(self, [yh, wlh], [T.vector(dtype='int32').type()])

    # Python implementation:
    def perform(self, node, inputs_storage, output_storage):
        yh = inputs_storage[0]
        trans = inputs_storage[1]
        # yt = inputs_storage[2]
        ret = []
        # print(yh.shape, trans.shape)

        B = yh.shape[1]
        K = yh.shape[2]
        for b in range(B):
            yp = yh[:,b,:]
            lgprob = np.zeros((K, 1))
            lst = []

            for i in range(yp.shape[0]):
                p = lgprob + trans + yp[i,:] 
                lst.append(np.argmax(p, axis=0))
                lgprob = np.max(p, axis=0).reshape((K, 1))

            y = []
            now = np.argmax(lgprob)
            y.append(now)
            for i in range(yp.shape[0]-1, 0, -1):
                now = lst[i][now]
                y.append(now)
            ret.append(y[::-1])
        output_storage[0][0] = np.array(ret).flatten().astype('int32')

class EditDistance(theano.Op):
    # Properties attribute
    __props__ = ()

    def make_node(self, y, yh):
        y = theano.tensor.as_tensor_variable(y)
        yh = theano.tensor.as_tensor_variable(yh)
        return theano.Apply(self, [y, yh], [T.scalar(dtype=config.floatX).type()])

    # Python implementation:
    def perform(self, node, inputs_storage, output_storage):
        y = inputs_storage[0]
        yh = inputs_storage[1]
        SL = y.shape[0] // BNUM
        y = y.reshape((BNUM, SL))
        yh = yh.reshape((BNUM, SL))

        ans = 0
        for b in range(BNUM):
            s1 = trim(y[b,:])
            s2 = trim(yh[b,:], False)
            pv = list(range(len(s2)+1))
            for i, c1 in enumerate(s1):
                cv = [i+1]
                for j, c2 in enumerate(s2):
                    cv.append(min(pv[j+1]+1, cv[j]+1, pv[j]+(c1 != c2)))
                pv = cv
            ans += pv[-1] / len(s1)
        ans /= BNUM

        output_storage[0][0] = np.array(ans).astype('float32')

class Executor:
    def __init__(self):
        pass

    def start(self):
        xx = T.matrix('features', config.floatX)
        yy = T.imatrix('targets')
        zm = BNUM*(xx.shape[0]//BNUM)
        x = xx[:zm].reshape((BNUM, zm//BNUM, xx.shape[1])).dimshuffle(1, 0, 2)
        y = yy[:zm].reshape((BNUM, zm//BNUM)).dimshuffle(1, 0)
        # x = xx[:zm].reshape((zm//16, 16, xx.shape[1]))
        # y = yy[:zm].reshape((zm//16, 16))

        DIMS = [108*5, 200, 200, 200, 48]
        NUMS = [1, 1, 1, 1, 1]
        # DIMS = [108*5, 48]
        # NUMS = [1, 1]
        FUNCS = [
            Rectifier, 
            Rectifier, 
            Rectifier, 
            # Rectifier, 
            # Rectifier, 
            # Maxout(num_pieces=5),
            # Maxout(num_pieces=5),
            # Maxout(num_pieces=5),
            # SimpleRecurrent,
            # SimpleRecurrent,
            # SimpleRecurrent,

            # SequenceGenerator,

            # Softmax,
            None,
        ]

        def lllistool(i, inp, func):
            sdim = DIMS[i]
            if func == SimpleRecurrent:
                sdim = DIMS[i] + DIMS[i+1]
            l = Linear(input_dim=DIMS[i], output_dim=DIMS[i+1] * NUMS[i+1], 
                       weights_init=IsotropicGaussian(std=sdim**(-0.5)), 
                       biases_init=IsotropicGaussian(std=sdim**(-0.5)),
                       name='Lin{}'.format(i))
            l.initialize()
            if func == SimpleRecurrent:
                gong = func(dim=DIMS[i+1], activation=Rectifier(), weights_init=IsotropicGaussian(std=sdim**(-0.5)))
                gong.initialize()
                ret = gong.apply(l.apply(inp))
            elif func == SequenceGenerator:
                gong = func(
                    readout=None, 
                    transition=SimpleRecurrent(dim=100, activation=Rectifier(), weights_init=IsotropicGaussian(std=0.1)))
                ret = None
            elif func == None:
                ret = l.apply(inp)
            else:
                gong = func()
                ret = gong.apply(l.apply(inp))
            return ret

        oup = x
        for i in range(len(DIMS)-1):
            oup = lllistool(i, oup, FUNCS[i])
        y_hat = oup

        y_rsp = y.reshape((y.shape[0]*y.shape[1],))
        y_dsf_rsp = y.dimshuffle(1, 0).reshape((y.shape[0]*y.shape[1],))
        yh_rsp = y_hat.reshape((y_hat.shape[0]*y_hat.shape[1], y_hat.shape[2]))
        yh_dsf_rsp = y_hat.dimshuffle(1, 0, 2).reshape((y_hat.shape[0]*y_hat.shape[1], y_hat.shape[2]))
        sfmx = Softmax().apply(yh_rsp)

        # cost = CategoricalCrossEntropy().apply(y, y_hat).astype(config.floatX)

        j, wlh = Yimumu(y_hat, y)
        cost = CategoricalCrossEntropy().apply(y_rsp, sfmx) + j
        # cost = CategoricalCrossEntropy().apply(y_rsp, sfmx)
        cost = cost.astype(config.floatX)

        cg = ComputationGraph(cost)
        orig_cg = cg
        ips = VariableFilter(roles=[INPUT])(cg.variables)
        ops = VariableFilter(roles=[OUTPUT])(cg.variables)
        # print(ips, ops)
        # cg = apply_dropout(cg, ips[0:2:1], 0.2)
        # cg = apply_dropout(cg, ips[2:-2:1], 0.5)
        # cost = cg.outputs[0].astype(config.floatX)

        cost.name = 'cost'

        mps = theano.shared(np.array([ph2id(ph48239(id2ph(t))) for t in range(48)]))
        # z_hat = T.argmax(yh_dsf_rsp, axis=1)
        z_hat = Yimumu_Decode()(y_hat, wlh)

        y39,_ = scan(fn=lambda t: mps[t], outputs_info=None, sequences=[y_dsf_rsp])
        y_hat39,_ = scan(fn=lambda t: mps[t], outputs_info=None, sequences=[z_hat])

        lost01 = (T.sum(T.neq(y_hat39, y39)) / y39.shape[0]).astype(config.floatX)
        lost01.name = '0/1 loss'
        lost23 = (T.sum(T.neq(y_hat39, y39)) / y39.shape[0]).astype(config.floatX)
        lost23.name = '2/3 loss'

        edit01 = EditDistance()(y39, y_hat39).astype(config.floatX)
        edit01.name = '0/1 edit'
        edit23 = EditDistance()(y39, y_hat39).astype(config.floatX)
        edit23.name = '2/3 edit'


        Ws = cg.parameters
        Ws = Ws + [wlh]
        print(list(Ws))
        norms = sum(w.norm(2) for w in Ws)
        norms = norms.astype(config.floatX)
        norms.name = 'norms'
        path = pjoin(PATH['fuel'], 'train.hdf5')
        data = H5PYDataset(path, which_set='train', load_in_memory=True, subset=slice(0, 100000))
        # data = H5PYDataset(path, which_set='train', load_in_memory=True)
        data_v = H5PYDataset(pjoin(PATH['fuel'], 'validate.hdf5'), which_set='validate', load_in_memory=True)
        num = data.num_examples
        data_stream = DataStream(data, iteration_scheme=SequentialScheme(
                        num, batch_size=SLEN*BNUM))
        data_stream_v = DataStream(data_v, iteration_scheme=SequentialScheme(
                        data_v.num_examples, batch_size=SLEN*BNUM))
        algo = GradientDescent(cost=cost, params=Ws, step_rule=CompositeRule([
            Momentum(0.005, 0.9)
            # AdaDelta()
        ]))
        monitor = DataStreamMonitoring( variables=[cost, lost01, edit01, norms],
                data_stream=data_stream)
        monitor_v = DataStreamMonitoring( variables=[lost23, edit23],
                data_stream=data_stream_v)
        plt = Plot('AlpEditHMMDropout', channels=[['0/1 loss', '2/3 loss'], ['0/1 edit', '2/3 edit']], after_epoch=True)
        main_loop = MainLoop(data_stream = data_stream, 
                algorithm=algo, 
                extensions=[monitor, monitor_v, FinishAfter(after_n_epochs=2000), Printing(), plt])

        main_loop.run()

        pfile = open('beta.pkl', 'wb')
        pickle.dump(orig_cg, pfile)
        pickle.dump(wlh, pfile)
        pfile.close()

    def end(self):
        pass

