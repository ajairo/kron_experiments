#!/usr/bin/env python
# -*- coding: utf-8 -*-

from timeit import Timer
import matplotlib.pyplot as plt

import random as pyrandom
pyrandom.seed(100)

import numpy as np

from rlscore.learner.cg_kron_rls import CGKronRLS
from rlscore.learner.kron_svm import KronSVM
from rlscore.learner.abstract_learner import CallbackFunction as CF
from rlscore.utilities.decomposition import decomposeKernelMatrix as dkm
from rlscore.measure.cindex_measure import cindex as perfmeasure


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


def get_ldo_folds_with_missing_data(label_row_inds, label_col_inds):
    folds = []
    rows = sorted(list(set(label_row_inds)))
    for rind in rows:
        alloccs = np.where(np.array(label_row_inds) == rind)[0]
        folds.append(alloccs)
    return folds


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


def leave_both_rows_and_columns_out_cv(XD, XT, Y, label_row_inds, label_col_inds):

    dfcount, tfcount = 3, 3
    #print 'leave_both_rows_and_columns_out_cv', dfcount, 'times', tfcount, 'folds'

    #totalfoldcount = dfcount * tfcount
    drugfolds = get_drugwise_folds(label_row_inds, label_col_inds, XD.shape[0], dfcount)
    targetfolds = get_targetwise_folds(label_row_inds, label_col_inds, XT.shape[0], tfcount)
    val_sets = []
    labeled_sets = []
    allindices = range(len(label_row_inds))

    for dfoldind in range(dfcount):
        data_inds_in_drug_fold = drugfolds[dfoldind]
        data_inds_not_in_drug_fold = set(allindices) - set(data_inds_in_drug_fold)
        for tfoldind in range(tfcount):
            data_inds_in_target_fold = targetfolds[tfoldind]
            data_inds_not_in_target_fold = set(allindices) - set(data_inds_in_target_fold)
            fold = sorted(list(set(data_inds_in_drug_fold) & set(data_inds_in_target_fold)))
            val_sets.append(fold)
            labeled_sets.append(sorted(list(data_inds_not_in_drug_fold & data_inds_not_in_target_fold)))
    #general_nfold_cv_no_imputation(XD, XT, Y, label_row_inds, label_col_inds, measure, labeled_sets, val_sets, regparam, rls)
    return (XD, XT, Y, label_row_inds, label_col_inds, labeled_sets, val_sets)


def general_nfold_cv_no_imputation(XD, XT, Y, label_row_inds, label_col_inds, measure, train_sets, val_sets, regparam, rls, incindices = None):

    cvrounds = len(train_sets)
    maxiter = 10
    all_predictions = np.zeros((maxiter, Y.shape[0]))
    print 'general nfold.'
    for foldind in range(cvrounds):
        trainindices = train_sets[foldind]
        valindices = val_sets[foldind]
        class TestCallback(CF):
            def __init__(self):
                self.iter = 0
            def callback(self, learner):
                all_predictions[self.iter][valindices] = np.mat(learner.getModel().predictWithDataMatricesAlt(XD, XT, label_row_inds[valindices], label_col_inds[valindices])).T
                self.iter += 1
        params = {}
        params["xmatrix1"] = XD
        params["xmatrix2"] = XT
        params["train_labels"] = Y[trainindices]
        params["label_row_inds"] = label_row_inds[trainindices]
        params["label_col_inds"] = label_col_inds[trainindices]
        params["maxiter"] = maxiter
        params['callback'] = TestCallback()
        if rls:
            learner = CGKronRLS.createLearner(**params)
        else:
            learner = KronSVM.createLearner(**params)
        #regparam = 2. ** (15)
        learner.solve_linear(regparam)
        print foldind, 'done'
    print
    bestperf = -float('Inf')
    bestparam = None
    for iterind in range(maxiter):
        if incindices == None:
            perf = measure(Y, all_predictions[iterind])
        else:
            perf = measure(Y[incindices], all_predictions[iterind][incindices])
        if perf > bestperf:
            bestperf = perf
            bestparam = iterind
        print iterind, perf
    return bestparam, bestperf, all_predictions


def single_holdout(XD, XT, Y, label_row_inds, label_col_inds, train_sets, val_sets, measure, regparam, rls,
                   maxiter=50, inneriter=100, incindices = None):
    cvrounds = len(train_sets)
    all_predictions = np.zeros((maxiter, Y.shape[0]))
    #print 'general nfold.'
    trainindices = train_sets[0]
    valindices = val_sets[0]
    class TestCallback(CF):
        def __init__(self):
            #self.iter = 0
            self.results = []
        def callback(self, learner):
            P = np.mat(learner.getModel().predictWithDataMatricesAlt(XD, XT, label_row_inds[valindices], label_col_inds[valindices])).T
            #print self.iter, measure(Y[valindices], P)
            self.results.append(measure(Y[valindices], P))
            #self.iter += 1
        def get_results(self):
            return self.results;
    params = {}
    params["xmatrix1"] = XD
    params["xmatrix2"] = XT
    params["train_labels"] = Y[trainindices]
    params["label_row_inds"] = label_row_inds[trainindices]
    params["label_col_inds"] = label_col_inds[trainindices]
    params["maxiter"] = maxiter
    params["inneriter"] = inneriter
    callback = TestCallback()
    params['callback'] = callback
    if rls:
        learner = CGKronRLS.createLearner(**params)
    else:
        learner = KronSVM.createLearner(**params)
    learner.solve_linear(regparam)
    return callback.get_results()


def load_metz_data():
    # Load outputs to be predicted
    Y = np.loadtxt("Y.txt")

    # Binarize real-valued outputs
    Y[Y<7.6] = -1.
    Y[Y>=7.6] = 1.

    # Load drug-target index pairs
    label_row_inds = np.loadtxt("label_row_inds.txt", dtype = np.int32)
    label_col_inds = np.loadtxt("label_col_inds.txt", dtype = np.int32)

    # Kernel matrices for drugs and targets
    KT = np.mat(np.loadtxt("KT.txt"))
    KD = np.mat(np.loadtxt("KD.txt"))
    KT = KT * KT.T
    KD = KD * KD.T

    # Singular value decompositions
    S, V = dkm(KD)
    XD = np.multiply(V,S)
    S, V = dkm(KT)
    XT = np.multiply(V,S)

    # Print XD.shape, XT.shape
    XD = np.array(XD)
    XT = np.array(XT)

    return (XD, XT, Y, label_row_inds, label_col_inds)


def metz_data_experiment(regparam, max_outer_iter, max_inner_iter, rls=True):
    data = load_metz_data()
    # Cross-validation
    split_data = leave_both_rows_and_columns_out_cv(*data)
    params = split_data + (perfmeasure, regparam, rls, max_outer_iter, max_inner_iter)
    print single_holdout(*params)


def artificial_data_experiment():
    maxiter = 1
    m1 = 1000
    m2 = 10000#0
    mm = m1 * m2
    d = 100
    l = 10000#0
    params = {}
    params["xmatrix1"] = np.random.rand(m1, d)
    params["xmatrix2"] = np.random.rand(m2, d)
    labelinds = pyrandom.sample(range(mm), l)
    rows, cols = np.unravel_index(range(mm), (m1, m2))
    label_row_inds = rows[labelinds]
    label_col_inds = cols[labelinds]
    params["train_labels"] = np.mat(labelinds, dtype=np.float64).T
    params["label_row_inds"] = label_row_inds
    params["label_col_inds"] = label_col_inds
    params["maxiter"] = maxiter
    learner = CGKronRLS.createLearner(**params)
    regparam = 2. ** (20)
    learner.solve_linear(regparam)


def create_plot(name, all_results):
    plt.figure()
    plt.ylim(ymin=0., ymax=1.)
    plt.xlabel('iterations')
    plt.ylabel('performance')
    for (results, reg_param) in all_results:
        plt.plot(results, label=u'λ = {0}'.format(reg_param))
    plt.legend()
    plt.title(name)
    plt.savefig('img/{0}.png'.format(name), format='png')
    plt.close()


def performance_experiment(use_rls):
    used_method = 'rls' if use_rls else 'svm'

    data = load_metz_data()
    split_data = leave_both_rows_and_columns_out_cv(*data)

    outer_iter = 500
    inner_iters = [1] if use_rls else [1, 10, 100]
    reg_param_range = [-7, -2, 0, 2, 7, 9] # range(-15, 16)
    reg_params = [('0', 0)] + map(lambda x: ('2^{0}'.format(x), 2**x), reg_param_range)
    for inner_iter in inner_iters:
        all_results = []
        for reg_param in reg_params:
            results = []
            params = split_data + (perfmeasure, reg_param[1], use_rls, outer_iter, inner_iter)
            timer_with_lambda = Timer(
                lambda: results.extend(single_holdout(*params)))
            lambda_perf = timer_with_lambda.timeit(number=1)
            print ('With {0} outer loops and {1} inner loops, the algorithm took in total {2} seconds. Regularization parameter {3}.'
                   .format(outer_iter, inner_iter, lambda_perf, reg_param[0]))
            print 'Results were {0}.'.format(results)
            print results
            all_results.append((results, reg_param[0]))
        plot_name = '{0}'.format(used_method) if use_rls else '{0}-iterations={1}'.format(used_method, inner_iter)
        create_plot(plot_name, all_results)


if __name__=="__main__":
    performance_experiment(use_rls=True)
    performance_experiment(use_rls=False)

    # artificial_data_experiment()
    # metz_data_experiment(15, 100, 10)
