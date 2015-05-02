import theano
from theano import config
import numpy as np
from settings import *
from datetime import datetime
import os

class BaseExecutor:
    def __init__(self, name, test_file):
        now = datetime.now()
        now_str = now.strftime('%m%d_%H%M%S')
        self.name = name + '_' + now_str
        self.path = os.path.join(PATH['output'], self.name)
        os.mkdir(self.path)

        path = os.path.join(PATH['numpy'], test_file)
        if not os.path.isfile(path):
            raise ValueError('Test file not found!')
        self.test_file = path

    def get_io(self):
        raise NotImplementedError('self.get_io() not implemented')

    def start(self):
        self.x = T.matrix('features', config.floatX)
        self.y = T.imatrix('targets')
        raise NotImplementedError('self.start() not implemented')

    def end(self):
        self.predict_test()

    def predict_test(self):
        x, y = self.get_io()
        fun = theano.function([x], y)

        test_feature = np.load(self.test_file).astype(config.floatX)
        result = fun(test_feature)
        np.save(os.path.join(self.path, 'result.npy'), result)
        argm = np.argmax(result, axis=1)

        #from phomap import id2ph

        #answer = []
        #for r in argm:
            #answer.append(id2ph(r))

        #with open(os.path.join(self.path, 'test.out'), 'w') as f:
            #f.write('\n'.join(answer))
