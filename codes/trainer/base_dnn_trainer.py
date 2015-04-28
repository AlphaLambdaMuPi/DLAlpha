import numpy as np
import theano, theano.tensor as T
from theano import shared, config
import time

class BaseDnnTrainer:
    def __init__(self, tX, tY, vX, vY, dnn, eta, degen, batch_size):
        self.tX = tX
        self.tY = tY
        self.tS = np.argmax(tY, axis=1)
        self.vX = vX
        self.vY = vY
        self.vS = np.argmax(vY, axis=1)
        self.dnn = dnn
        self.eta = shared(np.asarray(eta), config.floatX)
        self.cost = self.dnn.cost
        self.pred = self.dnn.pred
        self.batch_size = batch_size

        self.grads = [T.grad(self.dnn.cost, p) for p in self.dnn.params]
        updates = [
                (p, p-eta*gd) for p, gd in zip(self.dnn.params, self.grads)
            ]

        updates.append((self.eta, self.eta*degen))

        self.train_function = theano.function(
            inputs = [dnn.x, dnn.y],
            outputs = [self.cost, self.pred],
            updates = updates
        )

        self.pred_function = theano.function(
            inputs = [dnn.x],
            outputs = self.pred
        )

    def start(self):
        
        train_n = self.tX.shape[0]
        val_n = self.vX.shape[0]
        batch_num = train_n // self.batch_size
        epoched = 0

        while True:
            epoched += 1
            acc_n_t = 0
            for i in range(batch_num):
                l, r = i*self.batch_size, (i+1)*self.batch_size
                if i == batch_num - 1: r = train_n

                c, res = self.train_function(self.tX[l:r], self.tY[l:r])
                acc_n_t += np.sum(res == self.tS[l:r])

            res = self.pred_function(self.vX)
            acc_n_v = np.sum(res == self.vS)

            print('Epoch #{0}: A_in = {1:.6f}, A_out = {2:.6f}'.format(
                epoched, acc_n_t / train_n, acc_n_v / val_n ))
