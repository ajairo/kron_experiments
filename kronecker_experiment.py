import numpy as np
from rlscore.learner import KronRLS
from rlscore.learner.cg_kron_rls import CGKronRLS
from rlscore.learner.kron_svm import KronSVM
from rlscore.measure import auc
import cPickle
from rlscore.utilities import sparse_kronecker_multiplication_tools_python
from rlscore.learner.abstract_learner import CallbackFunction as CF
from rlscore.learner.cg_kron_rls import LinearPairwiseModel
from rlscore.learner.cg_kron_rls import KernelPairwiseModel
from rlscore.utilities.decomposition import decomposeKernelMatrix as dkm
import random as pyrandom

def dual_rls_objective(a, K1, K2, Y, rowind, colind, lamb):
    #dual form of the objective function for regularized least squares
    #a: current dual solution
    #K1: samples x samples kernel matrix for domain 1
    #K2: samples x samples kernel matrix for domain 2
    #rowind: row indices for training pairs
    #colind: column indices for training pairs
    #lamb: regularization parameter
    P =  sparse_kronecker_multiplication_tools_python.x_gets_C_times_M_kron_N_times_B_times_v(a, K2, K1, rowind, colind, rowind, colind)
    z = (Y - P)
    Ka = sparse_kronecker_multiplication_tools_python.x_gets_C_times_M_kron_N_times_B_times_v(a, K2, K1, rowind, colind, rowind, colind)
    return 0.5*(np.dot(z,z)+lamb*np.dot(a, Ka))

def primal_rls_objective(w, X1, X2, Y, rowind, colind, lamb):
    #primal form of the objective function for regularized least squares
    #w: current primal solution
    #X1: samples x features data matrix for domain 1
    #X2: samples x features data matrix for domain 2
    #rowind: row indices for training pairs
    #colind: column indices for training pairs
    #lamb: regularization parameter
    P = sparse_kronecker_multiplication_tools_python.x_gets_subset_of_A_kron_B_times_v(w, X2, X1.T, colind, rowind)
    z = (Y - P)
    return 0.5*(np.dot(z,z)+lamb*np.dot(w,w))

def dual_svm_objective(a, K1, K2, Y, rowind, colind, lamb):
    #dual form of the objective function for support vector machine
    #a: current dual solution
    #K1: samples x samples kernel matrix for domain 1
    #K2: samples x samples kernel matrix for domain 2
    #rowind: row indices for training pairs
    #colind: column indices for training pairs
    #lamb: regularization parameter
    P =  sparse_kronecker_multiplication_tools_python.x_gets_C_times_M_kron_N_times_B_times_v(a, K2, K1, rowind, colind, rowind, colind)
    z = (1. - Y*P)
    z = np.where(z>0, z, 0)
    Ka = sparse_kronecker_multiplication_tools_python.x_gets_C_times_M_kron_N_times_B_times_v(a, K2, K1, rowind, colind, rowind, colind)
    return 0.5*(np.dot(z,z)+lamb*np.dot(a, Ka))

def primal_svm_objective(w, X1, X2, Y, rowind, colind, lamb):
    #primal form of the objective function for support vector machine
    #w: current primal solution
    #X1: samples x features data matrix for domain 1
    #X2: samples x features data matrix for domain 2
    #rowind: row indices for training pairs
    #colind: column indices for training pairs
    #lamb: regularization parameter
    #P = np.dot(X,v)
    P = sparse_kronecker_multiplication_tools_python.x_gets_subset_of_A_kron_B_times_v(w, X2, X1.T, colind, rowind)
    z = (1. - Y*P)
    z = np.where(z>0, z, 0)
    return 0.5*(np.dot(z,z)+lamb*np.dot(w,w))

def get_drugwise_folds(label_row_inds, label_col_inds, drugcount, foldcount):
    assert len(np.array(label_row_inds).shape) == 1, 'label_row_inds should be one dimensional array'
    row_to_indlist = {}
    rows = sorted(list(set(label_row_inds)))
    for rind in rows:
        alloccs = np.where(np.array(label_row_inds) == rind)[0]
        row_to_indlist[rind] = alloccs
    drugfolds = get_random_folds(drugcount, foldcount)
    folds = []
    for foldind in range(foldcount):
        fold = []
        drugfold = drugfolds[foldind]
        for drugind in drugfold:
            fold = fold + row_to_indlist[drugind].tolist()
        folds.append(fold)
    return folds


def get_targetwise_folds(label_row_inds, label_col_inds, targetcount, foldcount):
    assert len(np.array(label_col_inds).shape) == 1, 'label_col_inds should be one dimensional array'
    col_to_indlist = {}
    cols = sorted(list(set(label_col_inds)))
    for cind in cols:
        alloccs = np.where(np.array(label_col_inds) == cind)[0]
        col_to_indlist[cind] = alloccs
    target_ind_folds = get_random_folds(targetcount, foldcount)
    folds = []
    for foldind in range(foldcount):
        fold = []
        targetfold = target_ind_folds[foldind]
        for targetind in targetfold:
            fold = fold + col_to_indlist[targetind].tolist()
        folds.append(fold)
    return folds

def get_random_folds(tsize, foldcount):
    folds = []
    indices = set(range(tsize))
    foldsize = tsize / foldcount
    leftover = tsize % foldcount
    for i in range(foldcount):
        sample_size = foldsize
        if leftover > 0:
            sample_size += 1
            leftover -= 1
        fold = pyrandom.sample(indices, sample_size)
        indices = indices.difference(fold)
        folds.append(fold)
    
    #assert stuff
    foldunion = set([])
    for find in range(len(folds)):
        fold = set(folds[find])
        assert len(fold & foldunion) == 0, str(find)
        foldunion = foldunion | fold
    assert len(foldunion & set(range(tsize))) == tsize
    
    return folds

def train_primal_kronrls(X1, X2, Y, rowinds, colinds, lamb, X1_test, X2_test, Y_test, rowinds_test = None, colinds_test=None):
    class TestCallback(CF):
        def __init__(self):
            self.iter = 0
        def callback(self, learner):
            X1 = learner.resource_pool['xmatrix1']
            X2 = learner.resource_pool['xmatrix2']
            rowind = learner.label_row_inds
            colind = learner.label_col_inds
            w = learner.W.ravel()
            loss = primal_rls_objective(w, X1, X2, Y, rowind, colind, lamb)
            print "iteration", self.iter
            print "Primal RLS loss", loss
            model = LinearPairwiseModel(learner.W, X1.shape[1], X2.shape[1])
            if rowinds_test == None:
                P = model.predictWithDataMatrices(X1_test, X2_test).ravel()
            else:
                P = model.predictWithDataMatricesAlt(X1_test, X2_test, rowinds_test, colinds_test)
            perf = auc(Y_test, P)
            print "Test set AUC", perf 
            self.iter += 1
    params = {}
    params["xmatrix1"] = X1
    params["xmatrix2"] = X2
    params["label_row_inds"] = rowinds
    params["label_col_inds"] = colinds
    params["train_labels"] = Y
    params['callback'] = TestCallback()
    params['maxiter'] = 100
    learner = CGKronRLS.createLearner(**params)
    learner.solve_linear(lamb)
    model = learner.model
    return model.W

def train_dual_kronrls(K1, K2, Y, rowinds, colinds, lamb, K1_test, K2_test, Y_test, rowinds_test = None, colinds_test=None):
    class TestCallback(CF):
        def __init__(self):
            self.iter = 0
        def callback(self, learner):
            K1 = learner.resource_pool['kmatrix1']
            K2 = learner.resource_pool['kmatrix2']
            rowind = learner.label_row_inds
            colind = learner.label_col_inds
            loss = dual_svm_objective(learner.A, K1, K2, Y, rowind, colind, lamb)
            print "iteration", self.iter
            print "Dual RLS loss", loss
            model = KernelPairwiseModel(learner.A, rowind, colind)
            if rowinds_test == None:
                P = model.predictWithKernelMatrices(K1_test, K2_test).ravel()
            else:
                P = model.predictWithKernelMatrices(K1_test, K2_test, rowinds_test, colinds_test)
            perf = auc(Y_test, P)
            print "Test set AUC", perf
            self.iter += 1
    params = {}
    params["kmatrix1"] = K1
    params["kmatrix2"] = K2
    params["label_row_inds"] = rowinds
    params["label_col_inds"] = colinds
    params["train_labels"] = Y
    params['callback'] = TestCallback()
    params['maxiter'] = 100
    learner = CGKronRLS.createLearner(**params)
    learner.solve_kernel(lamb)
    model = learner.model
    return learner.model

def train_primal_kronsvm(X1, X2, Y, rowinds, colinds, lamb, X1_test, X2_test, Y_test, rowinds_test = None, colinds_test=None, inneriter = 100):
    class TestCallback(CF):
        def __init__(self):
            self.iter = 0
        def callback(self, learner):
            X1 = learner.resource_pool['xmatrix1']
            X2 = learner.resource_pool['xmatrix2']
            rowind = learner.label_row_inds
            colind = learner.label_col_inds
            w = learner.W.ravel()
            loss = primal_svm_objective(w, X1, X2, Y, rowind, colind, lamb)
            print "iteration", self.iter
            print "Primal SVM loss", loss
            model = LinearPairwiseModel(learner.W, X1.shape[1], X2.shape[1])
            if rowinds_test == None:
                P = model.predictWithDataMatrices(X1_test, X2_test).ravel()
            else:
                P = model.predictWithDataMatricesAlt(X1_test, X2_test, rowinds_test, colinds_test)
            perf = auc(Y_test, P)
            print "Test set AUC", perf 
            self.iter += 1
    params = {}
    params["xmatrix1"] = X1
    params["xmatrix2"] = X2
    params["train_labels"] = Y
    params["label_row_inds"] = rowinds
    params["label_col_inds"] = colinds
    params['callback'] = TestCallback()
    params['maxiter'] = 100
    params['inneriter'] = inneriter
    learner = KronSVM.createLearner(**params)
    learner.solve_linear(lamb)
    model = learner.model
    return model.W

def train_dual_kronsvm(K1, K2, Y, rowinds, colinds, lamb, K1_test, K2_test, Y_test, rowinds_test = None, colinds_test=None, inneriter = 100):
    class TestCallback(CF):
        def __init__(self):
            self.iter = 0
        def callback(self, learner):
            K1 = learner.resource_pool['kmatrix1']
            K2 = learner.resource_pool['kmatrix2']
            rowind = learner.label_row_inds
            colind = learner.label_col_inds
            loss = dual_svm_objective(learner.A, K1, K2, Y, rowind, colind, lamb)
            print "iteration", self.iter
            print "Dual SVM loss", loss
            model = KernelPairwiseModel(learner.A, rowind, colind)
            if rowinds_test == None:
                P = model.predictWithKernelMatrices(K1_test, K2_test).ravel()
            else:
                P = model.predictWithKernelMatrices(K1_test, K2_test, rowinds_test, colinds_test)
            perf = auc(Y_test, P)
            print "Test set AUC", perf
            print "zero dual coefficients:", sum(np.isclose(learner.A, 0. )), "out of", len(learner.A) 
            self.iter += 1
    params = {}
    params["kmatrix1"] = K1
    params["kmatrix2"] = K2
    params["train_labels"] = Y
    params["label_row_inds"] = rowinds
    params["label_col_inds"] = colinds
    params['callback'] = TestCallback()
    params['maxiter'] = 100
    params['inneriter'] = inneriter
    learner = KronSVM.createLearner(**params)
    learner.solve_dual(lamb)
    model = learner.dual_model
    return model



def predict(W, X1pred, X2pred):
    P = np.array(np.dot(X1pred, np.dot(W, X2pred.T)))
    return P


def primal_experiment(X1_train, X2_train, Y_train, rowinds_train, colinds_train, X1_test, X2_test, Y_test, rowinds_test=None, colinds_test=None, rls=True, lamb=1.0, inneriter=100):
    if rls:
        W = train_primal_kronrls(X1_train, X2_train, Y_train, rowinds_train, colinds_train, lamb, X1_test, X2_test, Y_test, rowinds_test, colinds_test)
    else:
        W = train_primal_kronsvm(X1_train, X2_train, Y_train, rowinds_train, colinds_train, lamb, X1_test, X2_test, Y_test, rowinds_test, colinds_test, inneriter=inneriter)        

def dual_experiment(K1_train, K2_train, Y_train, rowinds_train, colinds_train, K1_test, K2_test, Y_test, rowinds_test=None, colinds_test=None, rls=True, lamb=1.0, inneriter=100):
    if rls:
        A = train_dual_kronrls(K1_train, K2_train, Y_train, rowinds_train, colinds_train, lamb, K1_test, K2_test, Y_test, rowinds_test, colinds_test)
    else:
        A = train_dual_kronsvm(K1_train, K2_train, Y_train, rowinds_train, colinds_train, lamb, K1_test, K2_test, Y_test, rowinds_test, colinds_test, inneriter=inneriter)        

 
def load_larhoven_data(dataset, primal=True, ssize=50000):
    #dataset: one of ['nr', 'ic', 'gpcr', 'e']
    #primal: whether to load data or kernel matrices
    #ssize: how many training pairs sampled, more pairs means more
    #accurate model, but slower training time
    assert dataset in ['nr', 'ic', 'gpcr', 'e']
    fname =  "data/larhoven/folds/FOLDS-%s-q4" %dataset
    f = open(fname)
    dfolds, tfolds = cPickle.load(f)
    dfold = dfolds[0]
    tfold = tfolds[0]
    Y = np.loadtxt('data/larhoven/'+dataset+'_admat_dgc.txt')
    Y = np.where(Y>=0.5, 1., -1.)
    dtraininds = list(set(range(Y.shape[0])).difference(dfold))
    ttraininds = list(set(range(Y.shape[1])).difference(tfold))
    X1 = np.loadtxt('data/larhoven/'+dataset+'_simmat_dc.txt')
    X2 = np.loadtxt('data/larhoven/'+dataset+'_simmat_dg.txt')
    X1_train = X1[dtraininds, :]
    X2_train = X2[ttraininds, :]
    X1_test = X1[dfold,:]
    X2_test = X2[tfold,:]
    KT = np.mat(X2)
    KT = KT * KT.T
    KD = np.mat(X1)
    KD = KD * KD.T
    K1_train = KD[np.ix_(dtraininds, dtraininds)]
    K2_train = KT[np.ix_(ttraininds, ttraininds)]
    Y_train = Y[np.ix_(dtraininds, ttraininds)]
    K1_test = KD[np.ix_(dfold,dtraininds)]
    K2_test = KT[np.ix_(tfold,ttraininds)]
    Y_test = Y[np.ix_(dfold, tfold)]
    rows = np.random.random_integers(0, K1_train.shape[0]-1, ssize)
    cols = np.random.random_integers(0, K2_train.shape[0]-1, ssize)
    ind = np.ravel_multi_index([rows, cols], (K1_train.shape[0], K2_train.shape[0]))
    Y_train = Y_train.ravel()[ind]
    Y_test = Y_test.ravel()
    if primal:
        return X1_train, X2_train, Y_train, rows, cols, X1_test, X2_test, Y_test
    else:
        return K1_train, K2_train, Y_train, rows, cols, K1_test, K2_test, Y_test

def load_metz_data(primal=True):
    #load outputs to be predicted
    Y = np.loadtxt("data/metz/Y.txt")
    #binarize real-valued outputs
    Y[Y<7.6] = -1.
    Y[Y>=7.6] = 1.
    
    #load drug-target index pairs
    label_row_inds = np.loadtxt("data/metz/label_row_inds.txt", dtype = np.int32)
    label_col_inds = np.loadtxt("data/metz/label_col_inds.txt", dtype = np.int32)
    
    #kernel matrices for drugs and targets
    KT = np.mat(np.loadtxt("data/metz/KT.txt"))
    KD = np.mat(np.loadtxt("data/metz/KD.txt"))
    KT = KT * KT.T
    KD = KD * KD.T

    #Normalize data
    #Singular value decompositions
    S, V = dkm(KD)
    XD = np.multiply(V,S)
    S, V = dkm(KT)
    XT = np.multiply(V,S)
    KD = np.array(KD)
    KT = np.array(KT)
    XD = np.array(XD)
    XT = np.array(XT)   
    dfcount, tfcount = 3, 3
    val_sets = []
    labeled_sets = []
    allindices = range(len(label_row_inds))
    drugfolds = get_drugwise_folds(label_row_inds, label_col_inds, XD.shape[0], dfcount)
    targetfolds = get_targetwise_folds(label_row_inds, label_col_inds, XT.shape[0], tfcount)
    for dfoldind in range(dfcount):
        data_inds_in_drug_fold = drugfolds[dfoldind]
        data_inds_not_in_drug_fold = set(allindices) - set(data_inds_in_drug_fold)
        for tfoldind in range(tfcount):
            data_inds_in_target_fold = targetfolds[tfoldind]
            data_inds_not_in_target_fold = set(allindices) - set(data_inds_in_target_fold)
            fold = sorted(list(set(data_inds_in_drug_fold) & set(data_inds_in_target_fold)))
            val_sets.append(fold)
            labeled_sets.append(sorted(list(data_inds_not_in_drug_fold & data_inds_not_in_target_fold)))
    trainindices = labeled_sets[0]
    valindices = val_sets[0]
    Y_train = Y[trainindices]
    Y_test = Y[valindices]
    if primal:
        return XD, XT, Y_train, label_row_inds[trainindices], label_col_inds[trainindices], XD, XT, Y_test, label_row_inds[valindices], label_col_inds[valindices]
    else:
        return KD, KT, Y_train, label_row_inds[trainindices], label_col_inds[trainindices], KD, KT, Y_test, label_row_inds[valindices], label_col_inds[valindices]         

def experiment1():
    #Primal RLS experiment with the gpcr dataset on
    #van larhoven data
    primal = True
    dataset = "gpcr"
    rls = True
    lamb = 1.
    data = load_larhoven_data(dataset, primal)
    primal_experiment(*data, rls=rls, lamb=lamb) 

def experiment2():
    #Dual SVM experiment with the gpcr dataset on
    #van larhoven data
    primal = False
    dataset = "gpcr"
    rls = False
    lamb = 1.
    data = load_larhoven_data(dataset, primal)
    dual_experiment(*data, rls=rls, lamb=lamb, inneriter=100) 

def experiment3():
    #Primal RLS experiment with Metz data
    primal = True   
    rls = True
    lamb = 1.
    data = load_metz_data(primal)
    primal_experiment(*data, rls=rls, lamb=lamb, inneriter=100)

if __name__=="__main__":
    #This is not a great interface, TODO: make it better
    seed = 10
    np.random.seed(seed)
    pyrandom.seed(seed)
    #Parameters of the experiment
    #primal = True #whether we optimize the primal or the dual
    #dataset = 'gpcr' #name of the dataset when loading a larhoven data
    #rls = True #whether to optimize RLS or SVM
    #lamb = 2.**(1.) #regularization parameter
    #inneriter = 100 #Number of inner iterations for SVM

    #At the moment there are five possible datasets:
    #larhoven: gpcr, ic, nr, e
    #metz data
    #data = load_larhoven_data(dataset, primal)
    #data = load_metz_data(primal)
    #primal_experiment(*data, rls=rls, lamb=lamb, inneriter=100)   
    
    #These are just examples of experiments
    #TODO: save from callbacks to file stuff like test performance, objective
    #function value and number/fraction of nonzero coefficients for dual SVM
    experiment1()
    experiment2()
    experiment3()
