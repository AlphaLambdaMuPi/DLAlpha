import numpy as np
fn1 = 'max2.out'
fn2 = 'minout.out'
s = [0, 0, 0]
cnt = 0

def edd(s1, s2):
    if len(s1) < len(s2):
        return edd(s2, s1)
 
    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)
 
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1 # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + 1       # than s2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

remps = {}
with open('48_idx_chr.map') as mp:
    while True:
        l = mp.readline()
        if not l: break
        a, idx, b = l.strip('\n').split()
        remps[b] = int(idx)

prob = np.load('prob_r.npy')
#prob = prob * 10 + 1
prob /= np.sum(prob, axis=1)
prob = np.log(prob)

def nice(s):
    res = 0
    for i in range(len(s)-1):
        res += prob[remps[s[i]], remps[s[i+1]]]
    return res

win = cnt = 0
with open(fn1) as f1, open(fn2) as f2, open('minout2.out', 'w') as fw:
    l1 = f1.readline()
    l2 = f2.readline()
    fw.write(l1)
    while True:
        l1 = f1.readline()
        if not l1: break
        a1, b1 = l1.strip().split(',')
        l2 = f2.readline()
        a2, b2 = l2.strip().split(',')
        s[0] += len(b1)
        s[1] += len(b2)

        s[2] += edd(b1,b2)
        cnt += 1
        #if(len(b1) <= len(b2)): fw.write(l1)
        #else: fw.write(l2)

        if(nice(b1) <= nice(b2)):
            win += 1
            fw.write(l2)
        else:
            fw.write(l1)
            
        #print(nice(b1), nice(b2))
        cnt += 1
    

print(s[0]/cnt, s[1]/cnt, s[2]/cnt, win, cnt-win)
