import numpy as np
import theano, theano.tensor as T
from theano import shared

class BaseDnnTrainer:
    def __init__(self, tX, tY, vX, vY, dnn, eta, batch_size):
        self.tX = tX
        self.tY = tY
        self.vX = vX
        self.vY = vY
        self.dnn = dnn
        self.eta = eta
        self.cost = self.dnn.cost

        self.grads = [T.grad(cost, p) for p in self.dnn.params]
        updates = [
                (p, p-eta*gd) for p, g in zip(self.dnn.params, self.grads)
            ]
        train_function = theano.function(
            inputs = [idx],
            outputs = cost,
            updates = updates,
            givens = {
                dnn.x: self.tX[idx*batch_size: (idx+1)*batch_size],
                dnn.y: self.tY[idx*batch_size: (idx+1)*batch_size],
            }
        )

    def start():
        
        batch_num = self.tX.shape[0] // self.batch_size
        epoched = 0

        while epoched < 20:
            epoched += 1
            for i in range(batch_num):
                c = self.train_function(i)
                print(c)
