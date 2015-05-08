import re


mps = {}
with open('48_idx_chr.map_b') as mp:
    while True:
        l = mp.readline()
        if not l: break
        a, _, b = l.split()
        mps[a] = b
with open('48_39.map') as mp:
    while True:
        l = mp.readline()
        if not l: break
        a, b = l.split()
        mps[a] = mps[b]

def ph2c(x):
    return mps[x]

def answer(l):
    seq = [ph2c(x) for x in l if x != 'concon']
    s = ''.join(seq)
    s = s.strip(ph2c('sil'))

    if s == '':
        return ''
    ansz = ans = s[0]
    ls = 'concon'
    t = 0
    for c in s:
        if c != ans[-1]: ans += c
        if c != ansz[-1]: 
            if ls == c:
                if t == 2:
                    ansz += c
                else:
                    t += 1
            else:
                t = 1
                ls = c
    #if ans != ansz: print(ans, ansz, sep='\n')
    return ansz

r = re.compile('(\w+)_\d+,(\w+)')
qq = []
with open('albegadee.out') as f, open('mnmn.out', 'w') as fw:
    f.readline()
    fw.write('id,phone_sequence\n')
    l = f.readline()
    while l:
        res = r.search(l)
        n = name = res.group(1)
        ls = []
        while n == name:
            ans = res.group(2)
            ls.append(ans)
            l = f.readline()
            if not l: break
            res = r.search(l)
            n = res.group(1)
        fin = answer(ls)
        qq.append(len(fin))
        fw.write('{},{}\n'.format(name, fin))

print( sum(qq)/len(qq) )


        
