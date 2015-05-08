"""A module that SVM^python interacts with to do its evil bidding."""
# Thomas Finley, tfinley@gmail.com

import os, sys
import svmapi
sys.path.insert(0, os.path.join(os.getcwd(), '../codes'))
# import imp
# svmapi = imp.load_dynamic('svmapi',
   # '/scratch/step5/chroot/usr/lib/python3.4/site-packages/svmapi.cpython-34.so')

import numpy as np
import random
import logging, logging.handlers
import sys
from settings import *
from shared import init
from check_file import check_file
from read_input import read_train_datas, read_map, read_models, read_tmodels, \
    read_map_39, read_hw1_matrix
from utils import varpsi, answer, delta, answer39
from timer import Timer
from phone_maps import *
pho_init()
phomap, phomap39, invphomap, labels = get_maps()

phomap = read_map()
phomap39 = read_map_39()
labels = list(phomap.keys())
label_list = [''] * LABEL_DIM
builtin_llabels = []
builtin_loutputs = []
builtin_rawoutputs = []
builtin_39outputs = []
hw1_big_matrix = None
# bigphi = None
global_y = []
gstartnum = 0
real_losses = []
timer = Timer()

def parse_parameters(sparm):
    """Sets attributes of sparm based on command line arguments.
    
    This gives the user code a chance to change sparm based on the
    custom command line arguments.  The custom command line arguments
    are stored in sparm.argv as a list of strings.  The custom command
    lines are stored in '--option', then 'value' sequence.

    If this function is not implemented, any custom command line
    arguments are ignored and sparm remains unchanged."""
    sparm.arbitrary_parameter = 'I am an arbitrary parameter!'

def parse_parameters_classify(attribute, value):
    """Process a single custom command line argument for the classifier.

    This gives the user code a chance to change the state of the
    A
    classifier based on a single custom command line argument, e.g.,
    one that begins with two dashes.  This function will be called
    multiple times if there are multiple custom command line
    arguments.

    If this function is not implemented, any custom command line
    arguments are ignored."""
    print('Got a custom command line argument %s %s' % (attribute, value))

def read_examples(filename, sparm):
    """Reads and returns x,y example pairs from a file.

    This reads the examples contained at the file at path filename and
    returns them as a sequence.  Each element of the sequence should
    be an object 'e' where e[0] and e[1] is the pattern (x) and label
    (y) respectively.  Specifically, the intention is that the element
    be a two-element tuple containing an x-y pair."""
    # We're not actually reading a file in this sample binary
    # classification task, but rather just returning a contrived
    # problem for learning.  The correct hypothesis would obviously
    # tend to have a positive weight for the first feature, and a
    # negative weight for the 4th feature.

    from random import random, randint
    init()
    global gstartnum
    gstartnum = 0
    # d = read_models(gstartnum + 5000)
    bll, d = read_tmodels(3000)
    global builtin_llabels
    builtin_llabels = bll

    global global_y
    global_y = [y for x, y in d]

    d = d[gstartnum:]

    # global bigphi
    # iks = np.concatenate([x for x, y in d], axis=0)
    # igraca = np.concatenate([y for x, y in d])
    # bigphi = varpsi(iks, igraca, phomap)
    # print(bigphi)

    # for i in range(len(d)):
        # for j in range(len(d[i][0])):
            # d[i][0][j] /= 30

    # thePsi = np.zeros(48*48+69*48)
    # for i, j in d:
        # thePsi += np.abs(varpsi(i, j, phomap))
    # for i in range(len(thePsi)//1000):
        # print(sum(thePsi[i*1000:(i+1)*1000]), end=' ')

    return d
    #return [([1,1,0,0], 1), ([1,0,1,0], 1), ([0,1,0,1],-1),
    #        ([0,0,1,1],-1), ([1,0,0,0], 1), ([0,0,0,1],-1)]

def init_model(sample, sm, sparm):
    """Initializes the learning model.

    Initialize the structure model sm.  The sm.size_psi must be set to
    the number of features.  The ancillary purpose is to add any
    information to sm that is necessary from the user code
    perspective.  This function returns nothing."""
    # In our binary classification task, we've encoded a pattern as a
    # list of four features.  We just want a linear rule, so we have a
    # weight corresponding to each feature.  We also add one to allow
    # for a last "bias" feature.
    sm.size_psi = FEATURE_DIM*LABEL_DIM+LABEL_DIM*LABEL_DIM

def init_constraints(sample, sm, sparm):
    """Initializes special constraints.

    Returns a sequence of initial constraints.  Each constraint in the
    returned sequence is itself a sequence with two items (the
    intention is to be a tuple).  The first item of the tuple is a
    document object.  The second item is a number, indicating that the
    inner product of the feature vector of the document object with
    the linear weights must be greater than or equal to the number
    (or, in the nonlinear case, the evaluation of the kernel on the
    feature vector with the current model must be greater).  This
    initializes the optimization problem by allowing the introduction
    of special constraints.  Typically no special constraints are
    necessary.  A typical constraint may be to ensure that all feature
    weights are positive.

    Note that the slack id must be set.  The slack IDs 1 through
    len(sample) (or just 1 in the combined constraint option) are used
    by the training examples in the sample, so do not use these if you
    do not intend to share slack with the constraints inferred from
    the training data.

    The default behavior is equivalent to returning an empty list,
    i.e., no constraints."""
    import svmapi

    if False:
        # Just some example cosntraints.
        c, d = svmapi.Sparse, svmapi.Document
        # Return some really goofy constraints!  Normally, if the SVM
        # is allowed to converge normally, the second and fourth
        # features are 0 and -1 respectively for sufficiently high C.
        # Let's make them be greater than 1 and 0.2 respectively!!
        # Both forms of a feature vector (sparse and then full) are
        # shown.
        return [(d([c([(1,1)])],slackid=len(sample)+1),   1),
                (d([c([0,0,0,1])],slackid=len(sample)+1),.2)]
    # Encode positivity constraints.  Note that this constraint is
    # satisfied subject to slack constraints.
    constraints = []
    # for i in range(sm.size_psi):
        # # Create a sparse vector which selects out a single feature.
        # sparse = svmapi.Sparse([(i,1)])
        # # The left hand side of the inequality is a document.
        # lhs = svmapi.Document([sparse], costfactor=1, slackid=i+1+len(sample))
        # # Append the lhs and the rhs (in this case 0).
        # constraints.append((lhs, 0))
    return constraints

def classify_example(x, sm, sparm, hw1_matrix=None):
    """Given a pattern x, return the predicted label."""
    # Believe it or not, this is a dot product.  The last element of
    # sm.w is assumed to be the weight associated with the bias
    # feature as explained earlier.


    ql = list(sm.w)
    obs = np.array(ql[:FEATURE_DIM*LABEL_DIM]).reshape((LABEL_DIM, FEATURE_DIM))
    trans = np.array(ql[FEATURE_DIM*LABEL_DIM:]).reshape((LABEL_DIM, LABEL_DIM))

    LEN = len(x) // FEATURE_DIM
    xx = np.array(x).reshape((LEN, FEATURE_DIM))
    xxt = np.dot(xx, obs.T)

    global global_y
    global hw1_big_matrix
    global gstartnum
    if hw1_big_matrix is None:
        hw1_big_matrix = read_hw1_matrix('submit_prob.csv', global_y)
    hw1_matrix = hw1_big_matrix[gstartnum + len(builtin_loutputs)]
    if hw1_matrix is not None:
        xxt = hw1_matrix

    # global bigphi
    # trans = np.array(bigphi[FEATURE_DIM*LABEL_DIM:]).reshape((LABEL_DIM, LABEL_DIM)) / 1500

    y = []
    lgprob = np.zeros((LABEL_DIM,1))
    lst = []

    for i in range(LEN):
        p = lgprob + trans + xxt[i,:] * 1.2
        newlst = np.argmax(p, axis=0)
        lst.append(newlst)
        lgprob = np.max(p, axis=0).reshape((LABEL_DIM,1))

    now = np.argmax(lgprob)
    y.append(now)
    for i in range(LEN-1, 0, -1):
        now = lst[i][now]
        y.append(now)

    y = y[::-1]
    y = [id2ph(i) for i in y]

    # print(answer(y))
    return y

def find_most_violated_constraint(x, y, sm, sparm):
    """Return ybar associated with x's most violated constraint.

    Returns the label ybar for pattern x corresponding to the most
    violated constraint according to SVM^struct cost function.  To
    find which cost function you should use, check sparm.loss_type for
    whether this is slack or margin rescaling (1 or 2 respectively),
    and check sparm.slack_norm for whether the slack vector is in an
    L1-norm or L2-norm in the QP (1 or 2 respectively).

    If this function is not implemented, this function is equivalent
    to 'classify(x, sm, sparm)'.  The optimality guarantees of
    Tsochantaridis et al. no longer hold, since this doesn't take the
    loss into account at all, but it isn't always a terrible
    approximation.  One still technically maintains the empirical
    risk bound condition, but without any regularization."""
    # return ['aa'] * (len(x) // 69)

    ql = list(sm.w)
    obs = np.array(ql[:FEATURE_DIM*LABEL_DIM]).reshape((LABEL_DIM, FEATURE_DIM))
    trans = np.array(ql[FEATURE_DIM*LABEL_DIM:]).reshape((LABEL_DIM, LABEL_DIM))

    LEN = len(x) // FEATURE_DIM
    xx = np.array(x).reshape((LEN, FEATURE_DIM))
    xxt = np.dot(xx, obs.T)

    yy = []
    lgprob = np.zeros((LABEL_DIM,1))
    lst = []

    for i in range(LEN):
        # ylab = np.ones((1,LABEL_DIM))
        # ylab[0,phomap[y[i]][0]] = 0
        alpha = np.array((i == 0) or (y[i] != y[i-1]))
        beta = (~np.identity(LABEL_DIM, dtype=bool)) | (i == 0)
        gamma = np.zeros((1,LABEL_DIM))
        gamma[0,ph2id(y[i])] = 1
        gamma = gamma * (alpha | beta)
        alpha = alpha.astype(float)
        beta = beta.astype(float)

        p = lgprob + trans + xxt[i,:] + 1 * alpha + beta - 2 * gamma
        # p = lgprob + trans + xxt[i,:] + ylab
        newlst = np.argmax(p, axis=0)
        lst.append(newlst)
        lgprob = np.max(p, axis=0).reshape((LABEL_DIM,1))

    now = np.argmax(lgprob)
    yy.append(now)
    for i in range(LEN-1, 0, -1):
        now = lst[i][now]
        yy.append(now)

    yy = yy[::-1]
    yy = [id2ph(i) for i in yy]

    realy = classify_example(x, sm, sparm)

    l2 = loss2(y, realy)

    # print(answer(yy), loss2(y, yy), l2, answer(realy))

    global real_losses
    real_losses.append(l2)
    if len(real_losses) >= 100:
        rls = sum(real_losses) / len(real_losses)
        print('>>> [Loss]({})'.format(rls))
        sys.stdout.flush()
        real_losses = []

    # if random.random() < 0.01:
        # theW = list(sm.w)
        # for i in range(len(theW)//500):
            # print(sum(theW[i*500:(i+1)*500]), end=' ')

    return yy


    # LEN = len(x) // 69
    # # yy = [random.choice(labels) for i in range(LEN)]
    # yy = ['aa'] * LEN
    # return yy;
    # current_cost = loss(y, yy) + wdotphi(x, yy, sm)
    # print('Current_DC', current_cost)

    # for i in range(LEN):
        # if(random.random() > 0.1): continue
        # print(i, '/', LEN)
        # for j in labels:
            # if(random.random() > 0.1): continue
            # yp = yy[:]
            # yp[i] = j
            # new_cost = loss(y, yp) + wdotphi(x, yp, sm)
            # if new_cost > current_cost:
                # yy = yp
                # current_cost = new_cost
                # print('Update_DC', new_cost, yp)
            # # print('hao123', i, j, new_cost)

    # return yy
    # score = classify_example(x,sm,sparm)
    # discy, discny = y*score, -y*score + 1
    # if discy > discny: return y
    # return -y

def find_most_violated_constraint_slack(x, y, sm, sparm):
    """Return ybar associated with x's most violated constraint.

    The find most violated constraint function for slack rescaling.
    The default behavior is that this returns the value from the
    general find_most_violated_constraint function."""
    return find_most_violated_constraint(x, y, sm, sparm)

def find_most_violated_constraint_margin(x, y, sm, sparm):
    """Return ybar associated with x's most violated constraint.

    The find most violated constraint function for margin rescaling.
    The default behavior is that this returns the value from the
    general find_most_violated_constraint function."""
    return find_most_violated_constraint(x, y, sm, sparm)

def psi(x, y, sm=None, sparm=None):
    """Return a feature vector representing pattern x and label y.

    This is the combined feature function, which this returns either a
    svmapi.Sparse object, or sequence of svmapi.Sparse objects (useful
    during kernel evaluations, as all components undergo kernel
    evaluation separately).  There is no default behavior."""
    # In the case of binary classification, psi is just the class (+1
    # or -1) times the feature vector for x, including that special
    # constant bias feature we pretend that we have.

    import svmapi
    
    thePsi = varpsi(x, y, phomap)
    # for i in range(len(thePsi)//500):
        # print(sum(thePsi[i*500:(i+1)*500]), end=' ')

    return svmapi.Sparse(thePsi)

def zero_one_loss(y, ybar):
    cnt = 0
    for i in range(len(y)):
        if phomap39[y[i]] != phomap39[ybar[i]]:
            cnt += 1
    return cnt

def loss2(y, ybar):
    a1 = answer39(y)
    a2 = answer39(ybar)
    return delta(a1, a2)

def loss(y, ybar, sparm=None):
    """Return the loss of ybar relative to the true labeling y.
    
    Returns the loss for the correct label y and the predicted label
    ybar.  In the event that y and ybar are identical loss must be 0.
    Presumably as y and ybar grow more and more dissimilar the
    returned value will increase from that point.  sparm.loss_function
    holds the loss function option specified on the command line via
    the -l option.

    The default behavior is to perform 0/1 loss based on the truth of
    y==ybar."""
    # If they're the same sign, then the loss should be 0.

    cnt = 0
    yl = ''
    ybl = ''
    for i in range(len(y)):
        alpha = (y[i] != yl)
        beta = (ybar[i] != ybl)
        if alpha:
            cnt += 1
            yl = y[i]
        if beta:
            cnt += 1
            ybl = ybar[i]
        if (alpha or beta) and (y[i] == ybar[i]):
            cnt -= 2

    #print('Time used: {}, total used: {}'.format(
    #    timer.get('loss'), timer.get()))
    return cnt

    # cnt = 0
    # for i in range(len(y)):
        # if y[i] != ybar[i]:
            # cnt += 1
    # return cnt

    # a1 = answer(y)
    # a2 = answer(ybar)
    # # print(a1)
    # # print(a2)
    # return delta(a1, a2)

def print_iteration_stats(ceps, cached_constraint, sample, sm,
                          cset, alpha, sparm):
    """Called just before the end of each cutting plane iteration.

    This is called just before the end of each cutting plane
    iteration, primarily to print statistics.  The 'ceps' argument is
    how much the most violated constraint was violated by.  The
    'cached_constraint' argument is true if this constraint was
    constructed from the cache.
    
    The default behavior is that nothing is printed."""
    print('Iter')
    thePsi = list(sm.w)
    for i in range(len(thePsi)//500):
        print(sum(thePsi[i*500:(i+1)*500]), end=' ')

def print_learning_stats(sample, sm, cset, alpha, sparm):
    """Print statistics once learning has finished.
    
    This is called after training primarily to compute and print any
    statistics regarding the learning (e.g., training error) of the
    model on the training sample.  You may also use it to make final
    changes to sm before it is written out to a file.  For example, if
    you defined any non-pickle-able attributes in sm, this is a good
    time to turn them into a pickle-able object before it is written
    out.  Also passed in is the set of constraints cset as a sequence
    of (left-hand-side, right-hand-side) two-element tuples, and an
    alpha of the same length holding the Lagrange multipliers for each
    constraint.

    The default behavior is that nothing is printed."""
    # print('Model learned:', end=' ')
    # print('[',', '.join(['%g'%i for i in sm.w]),']')
    print('Losses:')
    yp = [classify_example(x, sm, sparm) for x, y in sample]
    yy = [y for x, y in sample]
    ls = [loss(y, yb, sparm) for y, yb in zip(yy, yp)]
    ls2 = [loss2(y, yb) for y, yb in zip(yy, yp)]
    avgls = sum(ls) / len(ls)
    avgls2 = sum(ls2) / len(ls2)
    avglen = sum(len(answer(y)) for x, y in sample) / len(ls)
    avglen2 = sum(len(y) for x, y in sample) / len(ls2)
    # print(ls)
    for yn, yc, lm in zip(yp, yy, ls2):
        print(answer(yn), '/', lm)
        print(answer(yc))

    print('Average losses: {} (Lev {}) / {}'.format(
        avgls, avgls2, avglen))

def print_testing_stats(sample, sm, sparm, teststats):
    """Print statistics once classification has finished.

    This is called after all test predictions are made to allow the
    display of any summary statistics that have been accumulated in
    the teststats object through use of the eval_prediction function.

    The default behavior is that nothing is printed."""
    print(teststats)
    # yp = [classify_example(x, sm, sparm) for x, y in sample]
    yp = builtin_rawoutputs
    yy = [y for x, y in sample]
    ls2 = [loss2(y, yb) for y, yb in zip(yy, yp)]
    zols = [zero_one_loss(y, yb) for y, yb in zip(yy, yp)]
    avgls2 = sum(ls2) / len(ls2)
    avgzoloss = sum(zols) / sum(len(y) for y in yy)
    avglen = sum(len(answer(y)) for x, y in sample) / len(ls2)
    for yn, yc, lm in zip(yp, yy, ls2):
        print(answer(yn), '/', lm)
        print(answer(yc))
        print(' '.join(yn))
        print(' '.join(yc))

    print('Levenshtein losses: {} / {}'.format(avgls2, avglen))
    print('0/1 Accuracy: {} %'.format((1 - avgzoloss) * 100))

    f = open('hw2_pred.out', 'w')
    f.write('id,phone_sequence\n')
    for i, j in zip(builtin_llabels, builtin_loutputs):
        f.write(i + ',' + j + '\n')
    f.close()

    f = open('hw1_pred.out', 'w')
    f.write('id,prediction\n')
    for i, j in zip(builtin_llabels, builtin_39outputs):
        for k, kk in enumerate(j):
            f.write(i + '_' + str(k+1) + ',' + kk + '\n')
    f.close()

def eval_prediction(exnum, xxx_todo_changeme, ypred, sm, sparm, teststats):
    """Accumulate statistics about a single training example.

    Allows accumulated statistics regarding how well the predicted
    label ypred for pattern x matches the true label y.  The first
    time this function is called teststats is None.  This function's
    return value will be passed along to the next call to
    eval_prediction.  After all test predictions are made, the last
    value returned will be passed along to print_testing_stats.

    On the first call, that is, when exnum==0, teststats==None.  The
    default behavior is that the function does nothing."""
    (x, y) = xxx_todo_changeme
    if exnum==0: teststats = []
    # print('on example',exnum,'predicted',ypred,'where correct is',y)
    teststats.append(loss(y, ypred, sparm))
    return teststats

def write_model(filename, sm, sparm):
    """Dump the structmodel sm to a file.

    Write the structmodel sm to a file at path filename.

    The default behavior is equivalent to
    'cPickle.dump(sm,bz2.BZ2File(filename,'w'))'."""
    import pickle, bz2
    f = bz2.BZ2File(filename, 'w')
    pickle.dump(sm, f)
    f.close()

def read_model(filename, sparm):
    """Load the structure model from a file.

    Return the structmodel stored in the file at path filename, or
    None if the file could not be read for some reason.

    The default behavior is equivalent to
    'return cPickle.load(bz2.BZ2File(filename))'."""
    import pickle, bz2
    return pickle.load(bz2.BZ2File(filename))

def write_label(y):
    """Write a predicted label to an open file.

    Called during classification, this function is called for every
    example in the input test file.  In the default behavior, the
    label is written to the already open fileptr.  (Note that this
    object is a file, not a string.  Attempts to close the file are
    ignored.)  The default behavior is equivalent to
    'print>>fileptr,y'"""

    ans = answer(y)
    builtin_rawoutputs.append(y)
    builtin_loutputs.append(ans)
    builtin_39outputs.append([phomap39[z] for z in y])

def print_help():
    """Help printed for badly formed CL-arguments when learning.

    If this function is not implemented, the program prints the
    default SVM^struct help string as well as a note about the use of
    the --m option to load a Python module."""

    import svmapi

    print(svmapi.default_help)
    print("This is a help string for the learner!")

def print_help_classify():
    """Help printed for badly formed CL-arguments when classifying.

    If this function is not implemented, the program prints the
    default SVM^struct help string as well as a note about the use of
    the --m option to load a Python module."""
    print("This is a help string for the classifer!")
