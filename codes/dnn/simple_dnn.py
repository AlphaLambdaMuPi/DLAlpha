import numpy as np
import theano, theano.tensor as T
from theano import shared
from utils.functions import sigmoid


class SigmoidLayer():
    def __init__(self, n_in, n_out, inp, rng):
        W = np.asarray(
            rng.uniform(
                low=-4.0 * np.sqrt(6. / (n_in + n_out)),
                high=4.0 * np.sqrt(6. / (n_in + n_out)),
                size=(n_in, n_out)
            ),
            dtype=theano.config.floatX
        )
        self.W = shared(value=W, name='W', borrow=True)
        b = np.asarray(
            rng.uniform(
                low=-4.0 * np.sqrt(6. / (n_in + n_out)),
                high=4.0 * np.sqrt(6. / (n_in + n_out)),
                size=n_out
            ),
            dtype=theano.config.floatX
        )
        b = np.zeros(n_out)
        
        self.b = shared(value=b, name='b', borrow=True)
        
        self.out = T.nnet.sigmoid(T.dot(inp, self.W) + self.b)
        return
        #self.out = T.tanh(T.dot(inp, self.W) + self.b)

class OutputSigmoidLayer(SigmoidLayer):
    def __init__(self, n_in, n_out, inp, rng):
        super().__init__(n_in, n_out, inp, rng)
        self.out = T.dot(inp, self.W) + self.b
        self.pred = T.argmax(self.out, axis=1)

    def cost(self, y):
        rp = T.nnet.softmax(self.out)
        return -T.sum(y * T.log(rp))

class SimpleDnn():
    def __init__(self, dims, rng): 
        self.dims = dims
        self.layers = []
        self.params = []
        self.x = T.matrix('x', theano.config.floatX)
        self.y = T.matrix('y', theano.config.floatX)

        for i in range(len(self.dims)-1):
            if i != len(self.dims) - 2:
                self.layers.append(
                    SigmoidLayer(
                        n_in = self.dims[i],
                        n_out = self.dims[i+1],
                        inp = self.x if i == 0 else self.layers[-1].out,
                        rng = rng
                    )
                )
            else:
                self.layers.append(
                    OutputSigmoidLayer(
                        n_in = self.dims[i],
                        n_out = self.dims[i+1],
                        inp = self.x if i == 0 else self.layers[-1].out,
                        rng = rng
                    )
                )
            self.params.extend((self.layers[-1].W, self.layers[-1].b))

        self.cost = self.layers[-1].cost(self.y)
        self.pred = self.layers[-1].pred

        
