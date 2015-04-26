import logging
from settings import *
from os.path import join as pjoin, abspath, isfile, isdir
import shelve
import re

lg = logging.getLogger()

def check_files():

    files = [
        ('data', 'fbank/test.ark'),
        ('data', 'fbank/train.ark'),
        ('data', 'mfcc/test.ark'),
        ('data', 'mfcc/train.ark'),
        ('data', 'label/train.lab'),
        ('data', 'phones/48_39.map'),
        ('data', 'phones/48_idx_chr.map'),
    ]

    for f in files:
        file_dir = pjoin(PATH[f[0]], f[1])
        if isfile(file_dir):
            lg.info('Checking file {}: OK!'.format(file_dir))
        else:
            lg.error('File {} not found'.format(file_dir))
            return False

    return True

r = re.compile(r'(\w+)_(\d+)')
def get_group(s):
    mat = r.fullmatch(s)
    return mat.group(1), int(mat.group(2))

def build_shelve(name, fg):
    lg.info('Build shelve {}...'.format(name))
    shel = pjoin(PATH['shelve'], name)
    fbank = pjoin(PATH['fbank'], name + '.ark')
    mfcc = pjoin(PATH['mfcc'], name + '.ark')
    print(shel, fbank)
    
    if fg:
        lg.debug('Start loading answer label.')
        label = pjoin(PATH['label'], name + '.lab')
        answer = {}
        with open(label) as f:
            for ln in f:
                n, a = ln.strip('\n').split(',')
                pr = get_group(n)
                answer[pr] = a
    lg.debug('Load answer label done.')

    with shelve.open(shel) as sh:
        with open(fbank) as ff, open(mfcc) as fm:
            cnt = 0
            scnt = 0
            cur_name = ''
            cur_list = []
            names = []
            while True:
                cnt += 1
                lnf = ff.readline().strip('\n')
                lnm = fm.readline().strip('\n')

                if not lnf:
                    sh[cur_name] = cur_list
                    break

                fbank_data = lnf.split()
                mfcc_data = lnm.split()
                assert(fbank_data[0] == mfcc_data[0])
                nm, fr = get_group(fbank_data[0])
                if nm != cur_name:
                    sh[cur_name] = cur_list
                    cur_name = nm
                    cur_list = []
                    names.append(cur_name)
                    scnt += 1

                fbank_features = list(map(float, fbank_data[1:]))
                mfcc_features = list(map(float, mfcc_data[1:]))
                ls = [fr, fbank_features, mfcc_features]
                if fg: ls.append(answer[(nm, fr)])
                cur_list.append(ls)

                if cnt % 100000 == 0:
                    lg.info('Load {} frames, {} sentences.'.format(cnt, scnt))
            sh['names'] = names
    
    lg.info('Load {} frame datas, {} sencences.'.format(cnt, scnt))
    

def check_shelve():
    if not isdir(PATH['shelve']):
        lg.warning('Making directory: {}'.format(PATH['shelve']))
        os.makedirs(PATH['shelve'])

    shelves = [
        ('shelve', 'train'),
        ('shelve', 'test'),
    ]

    for s in shelves:
        file_dir = pjoin(PATH[s[0]], s[1])
        if isfile(file_dir):
            lg.info('Checking shelve {}: OK!'.format(file_dir))
        else:
            lg.warning( ('Shelve {} not found, press any key to rebuild '
                         '(Becareful swap out!)').format(file_dir) )
            input()

            build_shelve(s[1], s[1] == 'train')


def check_output():
    if not isdir(PATH['output']):
        lg.warning('Making directory: {}'.format(PATH['output']))
        os.makedirs(PATH['output'])

def init():
    check_files() and check_shelve() and check_output()






