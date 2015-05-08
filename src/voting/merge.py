import numpy as np
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

for a in answer:
    for i in range(len(answer[a])):
        max_answer[a] = max(set(answer[a]), key=answer[a].count)
        # for j in range(len(FILES)):
            # if max_answer[a] != answer[a][j]: diff[j] += 1
            # for k in range(len(FILES)):
                # if answer[a][j] != answer[a][k]:
                    # df2[j][k] += 1
        prd[answer[a].count(max_answer[a])] += 1
        # if answer[a].count(max_answer[a]) < len(FILES) // 2:
            # max_answer[a] = 'concon'

print(prd, diff)
print(np.array(df2))

with open(FILES[0]) as f, open('max1.out', 'w') as fw:
    l = f.readline()
    fw.write(l)
    while True:
        l = f.readline().strip('\n')
        if not l: break
        a, b = l.split(',')
        fw.write('{},{}\n'.format(a, max_answer[a]))



