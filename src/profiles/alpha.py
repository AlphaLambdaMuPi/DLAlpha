def alpha():
    with shelve.open('train') as f:
        names = f['names']
        fet = []
        lab = []
        for n in names:
            dt = f[n]
            for fr in dt:
                fet.append(fr[1] + fr[2])
                lab.append([ph2id(fr[3])])

    fet = fet[-10000:]
    lab = lab[-10000:]

    train_features = np.asarray(fet, np.float32)
    train_targets = np.asarray(lab, np.uint8)

    cf = ((train_features - np.mean(train_features, axis=0))
        / (np.std(train_features, axis=0)) + 1E-2)

    lenf = len(lab)
    alray = []
    concat = (4, 2)
    for i in range(lenf):
        arr = []
        for j in range(-concat[0], concat[0]+1, concat[1]):
            arr.extend(cf[(i-j)%lenf])
        alray.append(arr)

    alray = np.array(alray).astype(config.floatX)
    return alray, train_targets

def beta():
    with shelve.open('test') as f:
        names = f['names']
        fet = []
        for n in names:
            dt = f[n]
            for fr in dt:
                fet.append(fr[1] + fr[2])

    train_features = np.asarray(fet, np.float32)

    cf = ((train_features - np.mean(train_features, axis=0))
        / (np.std(train_features, axis=0)) + 1E-2)

    lenf = cf.shape[0]
    alray = []
    concat = (4, 2)
    for i in range(lenf):
        arr = []
        for j in range(-concat[0], concat[0]+1, concat[1]):
            arr.extend(cf[(i-j)%lenf])
        alray.append(arr)

    alray = np.array(alray).astype(config.floatX)
    return alray

def wrt(arr, tags, path):
    g = open(path, 'w')
    g.write('Id,prediction\n')
    for i, x in enumerate(arr):
        g.write('{},{}\n'.format(tags[i], ph48239(id2ph(x))))
    g.close()
    
def wrt_prob(arr, tags, path):
    g = open(path, 'w')
    for i, x in enumerate(arr):
        g.write('{} {}\n'.format(tags[i], ' '.join([str(y) for y in x])))
    g.close()

    