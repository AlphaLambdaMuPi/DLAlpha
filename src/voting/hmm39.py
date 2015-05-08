import numpy as np
import shelve
from progressbar import ProgressBar
import re
from phomap import ph2id, ph48239, id2ph


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
    mat = np.load('prob_r.npy')
    for i in range(48):
        j = ph2id(ph48239(id2ph(i)))
        if i != j:
            mat[j,:] += mat[i,:]
            mat[:,j] += mat[:,i]
    mat = mat / np.sum(mat, axis=1)[:,None]
    mat += 0.2
    mat = np.log(mat) 
    #mat *= 0
    A = np.load('voting_prob.npy')
    A = np.log(A)
    reg = re.compile('(\w+)_(\d+)')

    with open('submit_prob.csv') as f, open('albegadee.out', 'w') as fw:
        fw.write('id,prediction\n')

        cnt = 0
        with ProgressBar(maxval=596) as prog:
            l = f.readline()
            pos = acc = 0
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
                #probs = np.asarray(probs)
                #fin = np.argmax(A[pos:acc], axis=1)
                for i in range(acc-pos):
                    fw.write('{}_{},{}\n'.format(name, i+1, id2ph(fin[i])))
                pos = acc
                cnt += 1
                prog.update(cnt)

if __name__ == '__main__':
    main()





