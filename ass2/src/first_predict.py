from __future__ import division
import numpy as np
import scipy.stats.mstats as mstats
from sklearn import svm as sk_svm, neighbors
from sklearn.cross_validation import KFold
from math import inf
from datetime import datetime

DATA_DIR = 'data/'
U_MAT = 'first/unigrams.txt'
B_MAT = 'first/bigrams.txt'
U_MAT_LEX = 'first/unigrams_lex.txt'
B_MAT_LEX = 'first/bigrams_lex.txt'
LABELS = 'first/labels.txt'
SVM_RES = 'svm_prediction.txt'
KNN_RES = 'knn_prediction.txt'

MAX_K = 15 # in KNN model
MAX_C = 10 # in SVM model

def predict():
    now = datetime.now()
    # load matrices
    unigrams = np.loadtxt(DATA_DIR + U_MAT, dtype=(int))
    bigrams = np.loadtxt(DATA_DIR + B_MAT, dtype=(int))
    unigrams_lex = np.loadtxt(DATA_DIR + U_MAT_LEX, dtype=(int))
    bigrams_lex = np.loadtxt(DATA_DIR + B_MAT_LEX, dtype=(int))
    # load labels
    labels = np.loadtxt(DATA_DIR + LABELS, dtype=(int))
    # baseline - always predicts majority
    baseline(labels)
    # svm
    svm(unigrams, labels, 'unigrams')
    svm(bigrams, labels, 'bigrams')
    svm(unigrams_lex, labels, 'unigrams with lexicon')
    svm_res = svm(bigrams_lex, labels, 'bigrams with lexicon')

    #neighbor(unigrams, labels, 'unigrams')
    neighbor(bigrams, labels, 'bigrams')
    neighbor(unigrams_lex, labels, 'unigrams with lexicon')
    knn_res = neighbor(bigrams_lex, labels, 'bigrams with lexicon')

    # save results
    np.savetxt(DATA_DIR + SVM_RES, svm_res, fmt='%i')
    np.savetxt(DATA_DIR + KNN_RES, knn_res, fmt='%i')

    then = datetime.now()
    print('TIME: ', then - now)


def baseline(labels):
    """
    Baseline for testing, predict majority value for any input.
    Usually, the majority is positive feedback.
    """
    mode = mstats.mode(labels).mode[0]
    print('method: BASELINE, ACC: %.3f' %
            (accuracy([mode]*len(labels), labels)))

    print()

def svm(matrix, labels, name):
    # cross validation to fit parameters
    c_best = 1
    acc_best = 0
    kf = KFold(len(labels), n_folds=3, shuffle=True)
    for c in range(1,MAX_C):
        clf = sk_svm.SVC(C=c)
        summ = 0
        for train, test in kf:
            clf.fit(matrix[train], labels[train])
            predicted = clf.predict(matrix[test])
            summ += accuracy(predicted, labels[test])
        if summ/len(kf) > acc_best:
            acc_best = summ/len(kf)
            c_best = c
    print('method: SVM, data: %s testing, ACC: %.3f for c=%i' %
        (name, round(acc_best, 3), c_best))
    # accuracy on training data
    clf = sk_svm.SVC(C=c_best)
    clf.fit(matrix, labels)
    predicted = clf.predict(matrix)
    print('method: SVM, data: %s training, ACC: %.3f' %
            (name, round(accuracy(predicted, labels), 3)))
    print()
    return predicted

def neighbor(matrix, labels, name):
    # cross validation - find best k
    k_best = 1
    acc_best = 0
    kf = KFold(len(labels), n_folds=3, shuffle=True)
    for k in range(2,MAX_K,2):
        clf = neighbors.KNeighborsClassifier(n_neighbors=k)
        summ = 0
        for train, test in kf:
            clf.fit(matrix[train], labels[train])
            predicted = clf.predict(matrix[test])
            summ += accuracy(predicted, labels)
        if summ/len(kf) > acc_best:
            acc_best = summ/len(kf)
            k_best = k
    print('method: KNN, data: %s testing, ACC: %.3f for k=%i' %
        (name, round(acc_best, 3), k_best))
    # acc on training data
    clf = neighbors.KNeighborsClassifier(n_neighbors=k_best)
    clf.fit(matrix, labels)
    predicted = clf.predict(matrix)
    print('method: KNN, data: %s training, ACC: %.3f' %
            (name, round(accuracy(predicted, labels), 3)))
    print()
    return predicted


# --------- help functions --------------

def accuracy(labels1, labels2):
    s = 0
    for i in range(len(labels1)):
        s += labels1[i] == labels2[i]
    return s/len(labels1)



#MAIN
predict()
