import numpy as np

nlist = []
ls = [None] * 48 + [48]

with open('48_idx_chr.map') as of:
    while True:
        ol = of.readline()
        if not ol: break

        t,_,_ = ol.split()
        nlist.append(t)

with open('48_idx_chr.map_b') as of:
    while True:
        ol = of.readline()
        if not ol: break

        t,x,_ = ol.split()
        ls[int(x)] = nlist.index(t)


ls[48] = 48
print(ls)
swp = np.asarray(ls)
mat = np.load('prob.npy')
print(mat[37][47])
#np.save('prob_r.npy', mat[swp][swp])

res = np.empty(mat.shape)
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        res[i][j] = mat[swp[i]][swp[j]]

np.save('prob_r.npy', res)


