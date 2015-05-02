from settings import *
import numpy as np
import shelve
from progressbar import ProgressBar


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
    probs = np.hstack((probs, np.zeros((len(probs), 1))))

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

    return res[::-1]

def main():
    path = '../output/Strange_Dropout_0502_145049/result.npy'
    prob = np.load(path)
    mat = np.load('prob.npy')
    mat = mat / np.sum(mat, axis=1)[:,None]
    mat = np.log(mat)

    from phomap import id2ph, ph49238, ph2id
    from utils import answer
    with shelve.open(SHELVE['test']) as sh, open('res4.out', 'w') as f:
        f.write('id,prediction\n')
        names = sh['names']

        acc = 0
        cnt = 0
        with ProgressBar(maxval=len(names)) as prog:
            for name in names:
                curl = len(sh[name])
                #r = hmm(prob[acc:acc+curl], mat)
                r = np.argmax(prob[acc:acc+curl], axis=1)

                lastlab = -1
                i = 0
                qq = []
                q = 0
                while i < len(r):
                    j = i
                    while j < len(r) and r[j] == r[i]:
                        j += 1
                    if j - i > 2 or lastlab == -1:
                        qq.append(id2ph(r[i]))
                        #for k in range(i, j):
                            #f.write('{}_{},{}\n'.format(name, k+1, ph49238(id2ph(r[k]))))
                        lastlab = r[i]
                    else:
                        pass
                        #for k in range(i, j):
                            #f.write('{}_{},{}\n'.format(name, k+1, ph49238(id2ph(lastlab))))
                    i = j
                f.write('{},{}\n'.format(name, answer(qq)))
                acc += curl
                cnt += 1
                prog.update(cnt)

if __name__ == '__main__':
    main()





