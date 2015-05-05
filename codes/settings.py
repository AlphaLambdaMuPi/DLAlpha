import logging, sys, os
from os.path import dirname, abspath
from os.path import join as pjoin

PATH = {}
PATH['codes'] = dirname(abspath(__file__))
PATH['root'] = abspath(pjoin(PATH['codes'], os.pardir))
PATH['data'] = abspath(pjoin(PATH['root'], 'data'))
PATH['mfcc'] = abspath(pjoin(PATH['data'], 'mfcc'))
PATH['fbank'] = abspath(pjoin(PATH['data'], 'fbank'))
PATH['label'] = abspath(pjoin(PATH['data'], 'label'))
PATH['state_label'] = abspath(pjoin(PATH['data'], 'state_label'))
PATH['phones'] = abspath(pjoin(PATH['data'], 'phones'))
PATH['shelve'] = abspath(pjoin(PATH['data'], 'shelve'))
PATH['fuel'] = abspath(pjoin(PATH['data'], 'fuel'))
PATH['numpy'] = abspath(pjoin(PATH['data'], 'numpy'))
PATH['output'] = abspath(pjoin(PATH['root'], 'output'))

SHELVE = {}
SHELVE['train'] = abspath(pjoin(PATH['shelve'], 'train'))
SHELVE['train_state'] = abspath(pjoin(PATH['shelve'], 'train_state'))
SHELVE['test'] = abspath(pjoin(PATH['shelve'], 'test'))

LOG_LEVEL = logging.DEBUG

#SVM_PATH = os.path.join(ROOT_DIR, '..', 'svm-python3-kai')
#SVM_LEARN_PATH = os.path.join(SVM_PATH, 'svm_python_learn')
#SVM_CLASSIFY_PATH = os.path.join(SVM_PATH, 'svm_python_classify')
#if not os.path.isdir(OUTPUT_PATH):
    #os.makedirs(OUTPUT_PATH)


