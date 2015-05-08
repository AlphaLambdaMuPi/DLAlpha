import numpy as np
import pickle
import shelve
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
from phomap import ph48239, id2ph, ph2id, ph2c, state239

CONCON = None
HMM_RATIO = 0.01
SLEN = 64
BNUM = 4
LABEL = 1943 #+1

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
            if lst != -1:
                ret.append(lst)
        lst = i
        cnt = 1
    if (not cut) or cnt >= 3:
        ret.append(lst)
    if ret[0] == 'L':
        ret = ret[1:]
    if len(ret) > 0 and ret[-1] == 'L':
        ret = ret[:-1]
    # print(ret)
    return ret

def Yimumu(inp, y):
    wlh = shared(np.zeros((LABEL, LABEL)).astype(config.floatX))

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

def CTC_cost(y, yh):
    yy = y.dimshuffle(1, 0)
    yyh = yh.dimshuffle(1, 0, 2)
    yyh = T.exp(yyh)
    yyh /= T.sum(yyh, axis=2, keepdims=True)
    yyh = T.log(yyh)

    def func(al, be):
        ga = TrimOp()(al)
        num = ga[0]
        ga = ga[1:]

        # initdp = T.concatenate((T.ones(1), T.zeros((ga.shape[0],))), axis=0)
        initdp = T.concatenate((T.zeros(1), -10000*T.ones((ga.shape[0],))), axis=0)

        def func2(se, prv, s):
            # nxt = T.concatenate((T.zeros(1), prv[:-1]))
            # see = T.concatenate((T.zeros(1), se[s]))
            nxt = T.concatenate((-10000*T.ones(1), prv[:-1]))
            see = T.concatenate((-10000*T.ones(1), se[s]))
            ze = prv + se[-1]
            zz = nxt + see
            sd = prv + see
            mx = T.maximum(ze, T.maximum(zz, sd))
            dp = mx + T.log(T.exp(ze-mx) + T.exp(zz-mx) + T.exp(sd-mx))
            # dp = ze + T.log(1 + ezzze + seese)
            # dp = T.log(T.exp(prv + se[-1]) + T.exp(nxt + see))
            # dp = prv * se[-1] + nxt * see
            # dp = theano.printing.Print('Yap')(dp)
            return dp

        sc, _ = scan(fn=func2, 
                          sequences=be, 
                          outputs_info=initdp,
                          non_sequences=ga,
                          )
        # retval = -T.log(sc[-1][num]) / SLEN
        retval = -sc[-1][num] / SLEN
        # retval = theano.printing.Print('YapYap')(retval)
        return retval
    sc2, _ = scan(fn=func, sequences=[yy, yyh])
    return T.mean(sc2)

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

class CTC_Decode(theano.Op):
    # Properties attribute
    __props__ = ()

    def make_node(self, yh):
        # print(yh, wlh)
        yh = theano.tensor.as_tensor_variable(yh)
        # yt = theano.tensor.as_tensor_variable(yt)
        return theano.Apply(self, [yh, ], [T.vector(dtype='int32').type()])

    # Python implementation:
    def perform(self, node, inputs_storage, output_storage):
        yh = inputs_storage[0]
        ret = []

        B = yh.shape[1]
        K = yh.shape[2]
        for b in range(B):
            yp = yh[:,b,:]
            lst = yp.argmax(axis=1)
            reallst = []
            prv = ph2id('sil')
            for i in range(len(lst)):
                if lst[i] == ph2id('concon'):
                    lst[i] = prv
                else:
                    prv = lst[i]
                    reallst.append(lst[i])
            print(' '.join(list(map(id2ph, reallst))))
            ret.append(lst)
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

class TrimOp(theano.Op):
    # Properties attribute
    __props__ = ()

    def make_node(self, y):
        y = theano.tensor.as_tensor_variable(y)
        return theano.Apply(self, [y], [y.astype('int32').type()])

    # Python implementation:
    def perform(self, node, inputs_storage, output_storage):
        y = inputs_storage[0]
        q = trim(y)
        sdlen = max(0, 100-len(q))
        q = np.concatenate((np.array([len(q)]), q, np.zeros(sdlen)))
        output_storage[0][0] = np.array(q).astype('int32')

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

        DIMS = [108*9, 300, 300, 300, LABEL]
        NUMS = [1, 1, 1, 1, 1]
        # DIMS = [108*5, 48]
        # NUMS = [1, 1]
        FUNCS = [
            # Rectifier, 
            # Rectifier, 
            # Rectifier, 
            # Rectifier, 
            # Rectifier, 
            # Maxout(num_pieces=5),
            # Maxout(num_pieces=5),
            # Maxout(num_pieces=5),
            # SimpleRecurrent,
            # SimpleRecurrent,
            # SimpleRecurrent,
            # SimpleRecurrent,
            LSTM,
            LSTM,
            LSTM,

            # SequenceGenerator,

            # Softmax,
            None,
        ]

        def lllistool(i, inp, func):
            if func == LSTM:
                NUMS[i+1] *= 4
            sdim = DIMS[i]
            if func == SimpleRecurrent or func == LSTM:
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
            elif func == LSTM:
                gong = func(dim=DIMS[i+1], activation=Tanh(), weights_init=IsotropicGaussian(std=sdim**(-0.5)))
                gong.initialize()
                print(inp)
                ret, _ = gong.apply(
                    l.apply(inp), 
                    T.zeros((inp.shape[1], DIMS[i+1])),
                    T.zeros((inp.shape[1], DIMS[i+1])),
                )
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

        # j, wlh = Yimumu(y_hat, y)
        # cost = CategoricalCrossEntropy().apply(y_rsp, sfmx) + j
        cost = CategoricalCrossEntropy().apply(y_rsp, sfmx)
        # cost_p = cost_p.astype(config.floatX)
        # cost = CTC_cost(y, y_hat)
        cost = cost.astype(config.floatX)

        cg = ComputationGraph(cost)
        # cg_p = ComputationGraph(cost_p)
        orig_cg = cg
        ips = VariableFilter(roles=[INPUT])(cg.variables)
        ops = VariableFilter(roles=[OUTPUT])(cg.variables)
        # print(ips, ops)
        # cg = apply_dropout(cg, ips[0:2:1], 0.2)
        # cg = apply_dropout(cg, ips[2:-2:1], 0.5)
        # cost = cg.outputs[0].astype(config.floatX)

        cost.name = 'cost'

        # mps = theano.shared(np.array([ph2id(ph48239(id2ph(t))) for t in range(48)]))
        mps = theano.shared(np.array([ph2id(state239(t)) for t in range(1943)]))
        # yh_dsf_rsp = theano.printing.Print('YapYapYap')(yh_dsf_rsp)
        # z_hat = T.argmax(yh_dsf_rsp[:,:-1], axis=1)
        z_hat = T.argmax(yh_dsf_rsp, axis=1)
        # z_hat = theano.printing.Print('Yap')(z_hat)
        # z_hat = Yimumu_Decode()(y_hat, wlh)
        # z_hat_hat = CTC_Decode()(y_hat)

        y39,_ = scan(fn=lambda t: mps[t], outputs_info=None, sequences=[y_dsf_rsp])
        y_hat39,_ = scan(fn=lambda t: mps[t], outputs_info=None, sequences=[z_hat])
        y_hat_hat39 = y_hat39
        # y_hat_hat39,_ = scan(fn=lambda t: mps[t], outputs_info=None, sequences=[z_hat_hat])
        # trm = TrimOp()(y_hat_hat39)
        # trm = trm[1:1+trm[0]]
        # trm = theano.printing.Print('Trm')(trm)

        lost01 = (T.sum(T.neq(y_hat39, y39)) / y39.shape[0]).astype(config.floatX)
        lost01.name = '0/1 loss'
        lost23 = (T.sum(T.neq(y_hat39, y39)) / y39.shape[0]).astype(config.floatX)
        lost23.name = '2/3 loss'

        edit01 = EditDistance()(y39, y_hat_hat39).astype(config.floatX) #+ T.sum(trm) * 1E-10
        # edit01 = edit01.astype(config.floatX)
        edit01.name = '0/1 edit'
        edit23 = EditDistance()(y39, y_hat_hat39).astype(config.floatX)
        edit23.name = '2/3 edit'


        Ws = cg.parameters
        # Ws = Ws + [wlh]
        # print(list(Ws)
        norms = sum(w.norm(2) for w in Ws)
        norms = norms.astype(config.floatX)
        norms.name = 'norms'
        path = pjoin(PATH['fuel'], 'utt3_train.hdf5')
        # data = H5PYDataset(path, which_set='train', load_in_memory=True, subset=slice(0, 100000))
        data = H5PYDataset(path, which_set='train', load_in_memory=True)
        data_v = H5PYDataset(pjoin(PATH['fuel'], 'utt3_validate.hdf5'), which_set='validate', load_in_memory=True)
        num = data.num_examples
        data_stream = DataStream(data, iteration_scheme=ShuffledScheme(
                        num, batch_size=SLEN*BNUM))
        data_stream_v = DataStream(data_v, iteration_scheme=SequentialScheme(
                        data_v.num_examples, batch_size=SLEN*BNUM))
        algo = GradientDescent(cost=cost, params=Ws, step_rule=CompositeRule([
            Momentum(0.005, 0.9)
            # AdaDelta()
        ]))
        # algo_p = GradientDescent(cost=cost_p, params=cg_p.parameters, step_rule=CompositeRule([
            # Momentum(0.01, 0.9)
            # # AdaDelta()
        # ]))
        monitor = DataStreamMonitoring( variables=[cost, lost01, edit01, norms],
                data_stream=data_stream)
        monitor_v = DataStreamMonitoring( variables=[lost23, edit23],
                data_stream=data_stream_v)

        # main_loop_p = MainLoop(data_stream = data_stream, 
                # algorithm=algo_p, 
                # extensions=[monitor, monitor_v, FinishAfter(after_n_epochs=10), Printing(), plt])
        # main_loop_p.run()

        main_loop = MainLoop(data_stream = data_stream, 
                algorithm=algo, 
                extensions=[monitor, monitor_v, FinishAfter(after_n_epochs=20), Printing()])

        main_loop.run()

        # pfile = open('zzz.pkl', 'wb')
        # pickle.dump(orig_cg, pfile)
        # # pickle.dump(wlh, pfile)
        # pfile.close()
        
        ################

        test_feat = np.load(pjoin(PATH['numpy'], 'utt3_test_features.npy')).astype(config.floatX)
        func = theano.function([xx], y_hat.astype(config.floatX))
        test_hat = []
        for i in range(19):
            tmp = func(test_feat[i*10000:(i+1)*10000])
            tmp = tmp.transpose((1, 0, 2)).reshape((tmp.shape[0]*tmp.shape[1], tmp.shape[2]))
            test_hat.append(tmp)
        test_hat = np.concatenate(test_hat, axis=0)
        test_hat = np.concatenate((test_hat, np.zeros((2, LABEL))), axis=0)

        alpha = T.tensor3(config.floatX)
        beta = alpha.argmax(axis=2)
        # beta = alpha[:,:,:-1].argmax(axis=2)
        # beta = Yimumu_Decode()(alpha, wlh)
        # beta = CTC_Decode()(alpha)
        func2 = theano.function([alpha], beta)

        lens = []
        tags = []
        with shelve.open(SHELVE['test']) as f:
            names = f['names']
            for n in names:
                lens.append(len(f[n]))
                for i in range(lens[-1]):
                    tags.append(n+'_'+str(i+1))

        seq = []
        seq2 = []
        nowcnt = 0
        for i in lens:
            nxt = nowcnt + i
            cur_hat = test_hat[nowcnt:nxt].reshape((i, 1, LABEL)).astype(config.floatX)
            nowcnt = nxt
            fc2 = func2(cur_hat).flatten()
            fc3 = []
            fc4 = []
            for j in fc2:
                fc3.append(state239(j))
                fc4.append(ph2c(state239(j)))
            seq.append(fc3)
            seq2.append(''.join(trim(fc4)))

        seq_flat = np.concatenate(seq)
        with open('hw1_outz.txt', 'w') as f:
            f.write('id,prediction\n')
            for t, i in zip(tags, seq_flat):
                f.write(t+','+i+'\n')

        with open('hw2_outz.txt', 'w') as f:
            f.write('id,phone_sequence\n')
            for n, i in zip(names, seq2):
                f.write(n+','+i+'\n')

    def end(self):
        pass

