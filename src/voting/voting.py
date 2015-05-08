import numpy as np
from phomap import ph2id
FILES = [
    'hw1_out.txt',
    'hw1_outb.txt',
    'hw1_pred.out',
    'submit.csv', 
    # 'res3.out',

    'hw1hmm_state_good.out',
    'hw1hmm_state_span.out',
    'hw1hmm_state_2000.out',
    'hw1hmm_state_013.out',
    'hw1hmm_state_logmix3.out',
    # 'hw1hmm_state_mix3.out',
    # 'hw1hmm_state_logmix3.out',
         ]

answer = {}

for fn in FILES:
    with open(fn) as f:
        f.readline()
        while True:
            l = f.readline().strip('\n')
            if not l: break
            a, b = l.split(',')
            if a not in answer:
                answer[a] = [b]
            else:
                answer[a].append(b)

max_answer = {}
prd = [0] * (len(FILES)+1)
diff = [0] * len(FILES)
df2 = [ [0] * len(FILES) for i in range(len(FILES)) ]
probs = []

with open(FILES[0]) as f:
    l = f.readline()
    while True:
        l = f.readline()
        if not l: break
        a, b = l.split(',')
        probs.append([0] * 48)
        for j in range(len(FILES)):
            probs[-1][ph2id(answer[a][j])] += 1.0/len(FILES)
        #if answer[a].count(max_answer[a]) < len(FILES) - 1:
            #max_answer[a] = 'concon'

probs = np.asarray(probs)
np.save('voting_prob.npy', probs)
