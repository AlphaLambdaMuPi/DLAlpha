import logging, sys, os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, '../../data')
LOG_FILE_NAME = os.path.join(ROOT_DIR, 'log/hw2.log')
SHELVE_FILE_NAME = os.path.join(ROOT_DIR, 'shelve/train')
SHELVE_TEST_FILE_NAME = os.path.join(ROOT_DIR, 'shelve/test')
LOG_LEVEL = logging.DEBUG
TAR_FILE = os.path.join(ROOT_DIR, 'hw2.tgz')

DATA_PATH = {}
DATA_PATH['fbank'] = os.path.join(DATA_DIR, 'fbank', 'train.ark')
# DATA_PATH['fbank'] = 'prob.csv'
DATA_PATH['test'] = os.path.join(DATA_DIR, 'fbank', 'test.ark')
DATA_PATH['label'] = os.path.join(DATA_DIR, 'label', 'train.lab')
DATA_PATH['48_idx_chr'] = os.path.join(DATA_DIR, 'phones', '48_idx_chr.map')
DATA_PATH['48_39'] = os.path.join(DATA_DIR, 'phones', '48_39.map')

OUTPUT_PATH = os.path.join(ROOT_DIR, '..', 'output')

SVM_PATH = os.path.join(ROOT_DIR, '..', 'svm-python3-kai')
SVM_LEARN_PATH = os.path.join(SVM_PATH, 'svm_python_learn')
SVM_CLASSIFY_PATH = os.path.join(SVM_PATH, 'svm_python_classify')
if not os.path.isdir(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

FEATURE_DIM = 69
# FEATURE_DIM = 48
LABEL_DIM = 48

def init_log_settings():

    log_parent = os.path.dirname(LOG_FILE_NAME)
    if not os.path.isdir(log_parent):
        os.makedirs(log_parent)

    rotator = logging.handlers.RotatingFileHandler(
                  LOG_FILE_NAME, maxBytes=65536, backupCount=5)
    logging.basicConfig(handlers=[rotator],
                        format='[%(asctime)s.%(msecs)d] %(module)s - %(levelname)s : %(message)s',
                        datefmt='%H:%M:%S',
                        level=LOG_LEVEL)
    logger = logging.getLogger()

    formatter = logging.Formatter(fmt = '[%(asctime)s.%(msecs)d] %(module)s - %(levelname)s : %(message)s'
                                  ,datefmt = '%H:%M:%S')
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    logger.addHandler(console)

