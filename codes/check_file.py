from urllib.request import urlretrieve
import tarfile, os
import logging
from settings import *
import shelve
from read_input import read_feature_label

logger = logging.getLogger()

def check_file():

    def do_check():
        if os.path.isdir(DATA_DIR):
            logger.info('It seems like you have the datas...')
        else:
            logger.warning('data/ dir missing, try extract from hw2.tgz...')
            return False

        for k, v in DATA_PATH.items():
            if not os.path.isfile(v):
                logger.warning('%s missing, try extract from hw2.tgz', k)
                return False
            logger.info('%s OK', v)

        return True

    if not do_check():
        try:
            with tarfile.open(TAR_FILE) as tar:
                data_dir = os.path.dirname(__file__)
                tar.extractall(path=DATA_DIR)
        except OSError:
            logger.error('File {} not found, please download' 
                    'from kaggle and extract it here'.format(os.path.abspath(TAR_FILE)))
            return
        except Exception as e:
            logger.error('Fatal Error.... {}'.format(e))
            return

        logger.info('Data Extract Complete!')

def load_shelve(test=False):
    SFN = SHELVE_FILE_NAME
    try:
        logger.debug('Start open shelve')
        with shelve.open(SFN, 'r') as d:
            res = d['names']
            logger.info('Shelve has {} group datas'.format(len(res)))
    except Exception as e:
        
        shelve_parent = os.path.dirname(SFN)
        if not os.path.isdir(shelve_parent):
            logger.info('Make dir %s', shelve_parent)
            os.makedirs(shelve_parent)

        with shelve.open(SFN, 'c') as d:
            logger.warning('shelve broken or not built, rebuilding...')
            dt = read_feature_label()
            name_list = []
            for k, v in dt.items():
                d[k] = v
                name_list.append(k)
            d['names'] = name_list
            res = d['names']
    
    logger.debug('Done loading shelve')
    return res

