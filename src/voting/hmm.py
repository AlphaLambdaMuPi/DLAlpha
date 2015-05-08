import numpy as np
import shelve
from progressbar import ProgressBar
import re

remps = {}
with open('48_idx_chr.map') as mp:
    while True:
        l = mp.readline()
        if not l: break
        a, idx, b = l.split()
        remps[int(idx)] = a
mps = {}
with open('48_39.map') as mp:
    while True:
        l = mp.readline()
        if not l: break
        a, b = l.split()
        mps[a] = b

def build():
    mat = np.zeros((49, 49))

    from phomap import ph2id
    with shelve.open(SHELVE['train']) as sh:
        names = sh['names']
        with ProgressBar(maxval=len(names)) as prog:
            for cnt in range(len(names)):
                cur = sh[names[cnt]]
                mat[48, ph2id(cur[0][3])] += 1
                for i in range(len(cur)-1):
                    f, t = ph2id(cur[i][3]), ph2id(cur[i+1][3])
                    mat[f,t] += 1
                mat[ph2id(cur[len(cur)-1][3]), 48] += 1
                cnt += 1
                prog.update(cnt)

    np.save('prob.npy', mat)


def hmm(probs, mat):
    probs = np.hstack((np.array(probs), np.full((len(probs), 1), -np.inf)))
    probs = np.vstack((probs, np.hstack((np.full(48, -np.inf), np.array([0])))))

    dps = np.empty((len(probs)+1, 49))
    link = np.empty((len(probs), 49), dtype='int64')
    dps[0] = np.hstack((np.full(48, -np.inf), np.array([0])))

    for i in range(len(probs)):
        last = dps[i]
        transp = mat + last[:,None]
        link[i] = np.argmax(transp, axis=0)
        dps[i+1] = np.max(transp, axis=0) + probs[i]

    best = np.argmax(dps[len(probs)])
    res = [best]

    for i in range(len(probs)-1, 0, -1):
        res.append(link[i][res[-1]])
    res = res[1:]

    return res[::-1]

def main():
    #path = '../output/Strange_Dropout_0502_145049/result.npy'
    #prob = np.load(path)
    mat = np.load('prob.npy')
    #mat = mat * 5 + 1
    mat = mat / np.sum(mat, axis=1)[:,None]
    mat = np.log(mat) 
    reg = re.compile('(\w+)_(\d+)')
    A = np.log(np.load('ans_hat.prob.npy'))

    with open('submit_prob.csv') as f, open('hw1hmm_eta.out', 'w') as fw:
        f.readline()
        fw.write('id,prediction\n')

        cnt = 0
        with ProgressBar(maxval=596) as prog:
            l = f.readline()
            pos = 0
            acc = 1
            while l:
                ls = l.split()
                res = reg.search(ls[0])
                n = name = res.group(1)
                while n == name:
                    acc += 1
                    l = f.readline()
                    ls = l.split()
                    if not l: break
                    res = reg.search(l)
                    n = res.group(1)

                fin = hmm(A[pos:acc], mat)
                for i in range(len(probs)):
                    fw.write('{}_{},{}\n'.format(name, i+1, mps[remps[fin[i]]]))
                cnt += 1
                prog.update(cnt)

if __name__ == '__main__':
    main()





