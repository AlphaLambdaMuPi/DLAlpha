from read_input import read_train_by_group
from trainer.base_dnn_trainer import BaseDnnTrainer
from dnn.simple_dnn import SimpleDnn
import numpy as np
from preproc import flatten, pair_to_np, y_to_01
from theano import config

np.set_printoptions(threshold=np.nan)

class Executor:
    def __init__(self):
        self.dims = [500, 500, 500]
        self.alpha = 0.9
        pass

    def start(self):
        inp = self.read()
        
        X, Y = self.preproc(inp)
        pX = self.preprocX(X)
        pY = self.preprocY(Y)
        
        tX, tY, vX, vY = self.split_validate(pX, pY)

        self.rng = np.random.RandomState(12)
        dims = [tX.shape[1]] + self.dims + [tY.shape[1]]
        #dims = [2, 2]
        self.dnn = SimpleDnn(dims, self.rng)
        self.trainer = BaseDnnTrainer(
                tX = tX,
                tY = tY,
                vX = vX,
                vY = vY,
                dnn = self.dnn,
                eta = 0.005,
                degen = 1,
                batch_size = 128
            )
        self.trainer.start()
            
    def read(self):
        return read_train_by_group(100)

    def preproc(self, inp):
        return pair_to_np(flatten(inp))

    def preprocX(self, inp):
        return inp.astype(config.floatX)

    def preprocY(self, inp):
        return y_to_01(inp, 48).astype(config.floatX)

    def split_validate(self, X, Y):
        n = X.shape[0]
        tr_n = n * self.alpha
        return X[:tr_n], Y[:tr_n], X[tr_n:], Y[tr_n:]
        


    
