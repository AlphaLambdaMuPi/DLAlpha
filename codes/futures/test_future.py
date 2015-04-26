from read_input import read_train_by_group
from trainer import BaseDnnTrainer
from dnn import SimpleDnn
import numpy as np

class Executor:
    def __init__(self):
        pass

    def start(self):
        inp = self.read()
        
        X, Y = self.preproc(inp)
        pX = self.preprocX(X)
        
        tX, tY, vX, vY = self.split_validate(pX, Y)

        self.rng = np.random.RandomState(1234)
        dims = [tX.shape[1], 200, tY.shape[0]]
        self.dnn = SimpleDnn(X.shape[0], x, self.rng)
        self.trainer = BaseDnnTrainer(
                tX = tX,
                tY = tY,
                vX = vX,
                vY = vY,
                dnn = self.dnn,
                eta = 0.01,
                batch_size = 50,
            )
        self.trainer.start()
            
    def read(self):
        return read_train_by_group(10)

    def preproc(self, inp):
        return pair_to_np(flatten(inp))

    def preprocX(self, inp):
        return inp


    
