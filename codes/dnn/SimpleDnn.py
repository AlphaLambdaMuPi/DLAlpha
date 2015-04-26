import numpy as np
import theano, theano.tensor as T
from theano import shared
from utils import sigmoid

class OutputSigmoidLayer(SigmoidLayer):
    def __init__(self, n_in, n_out, inp, rng):
        super().__init__()
        self.p_y = T.nnet.softmax(self.out)
        self.pred = T.argmax(self.p_y, axis=1)

    def cost(self):
        return -T.mean(T.log(self.p_y)[T.arange(y.shape[0]), y])

class SigmoidLayer():
    def __init__(self, n_in, n_out, inp, rng):
        W = np.asarray(
            rng.uniform(
                low=-4.0 * numpy.sqrt(6. / (n_in + n_out)),
                high=4.0 * numpy.sqrt(6. / (n_in + n_out)),
                size=(n_in, n_out)
            ),
            dtype=theano.config.floatX
        )
        self.W = shared(value=W, name='W', borrow=True)
        b = np.asarray(
            rng.uniform(
                low=-4.0 * numpy.sqrt(6. / (n_in + n_out)),
                high=4.0 * numpy.sqrt(6. / (n_in + n_out)),
                size=n_out
            ),
            dtype=theano.config.floatX
        )
        self.b = shared(value=b, name='b', borrow=True)
        
        self.inp = inp
        self.out = sigmoid(T.dot(inp, self.W) + self.b)

class SimpleDnn():
    def __init__(self, dims, rng): 
        self.dims = dims
        self.layers = []
        self.params = []
        self.x = T.matrix('x')
        self.y = T.ivector('y')

        for i in range(self.dims-1):
            if i != self.dims - 2:
                self.layers.append(
                    SigmoidLayer(
                        n_in = self.dims[i],
                        n_out = self.dims[i+1],
                        inp = x if i == 0 else self.layers[-1].out,
                        rng = rng
                    )
                )
            else:
                self.layers.append(
                    OutputSigmoidLayer(
                        n_in = self.dims[i],
                        n_out = self.dims[i+1],
                        inp = self.layers[-1].out,
                        rng = rng
                    )
                )
            self.params.extend((self.layers[-1].W, self.layers[-1].b))

        self.cost = self.layers[-1].cost(y)

        
