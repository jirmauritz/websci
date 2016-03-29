from __future__ import division
import numpy as np
import scipy.stats.mstats as mstats
from sklearn import svm as sk_svm, neighbors
from sklearn.cross_validation import KFold
from math import inf

DATA_DIR = '../data/'
U_MAT = 'first/unigrams.txt'
B_MAT = 'first/bigrams.txt'
LABELS = 'first/labels.txt'

MAX_K = 6#10 # in KNN model
MAX_C = 2#10 # in SVM model

def predict():
    # load matrices
    unigrams = np.loadtxt(DATA_DIR + U_MAT, dtype=(int))
    bigrams = np.loadtxt(DATA_DIR + B_MAT, dtype=(int))
    # load labels
    labels = np.loadtxt(DATA_DIR + LABELS, dtype=(int))
    # baseline - always predicts majority
    baseline(labels)
    # svm
    svm(unigrams, labels, 'unigrams')
    svm(bigrams, labels, 'bigrams')

    neighbor(unigrams, labels, 'unigrams')
    neighbor(bigrams, labels, 'bigrams')
    

def baseline(labels):
    """
    Baseline for testing, predict majority value for any input.
    Usually, the majority is positive feedback.
    """
    mode = mstats.mode(labels).mode[0]
    print('method: BASELINE, ACC: %.2f' %
            (accuracy([mode]*len(labels), labels)))

    print()

def svm(matrix, labels, name):
    clf = sk_svm.SVC()
    # accuracy on training data
    clf.fit(matrix, labels)
    predicted = clf.predict(matrix)
    print('method: SVM, data: %s training, ACC: %.2f' %
            (name, round(accuracy(predicted, labels), 2)))
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
    print('method: SVM, data: %s testing, ACC: %.2f for c=%i' %
        (name, round(acc_best, 2), c_best))
    print()

def neighbor(matrix, labels, name):
    clf = neighbors.KNeighborsClassifier()
    # acc on training data just for check
    clf.fit(matrix, labels)
    predicted = clf.predict(matrix)
    print('method: KNN, data: %s training, ACC: %.2f' %
            (name, round(accuracy(predicted, labels), 2)))
    # cross validation - find best k
    k_best = 1
    acc_best = 0
    kf = KFold(len(labels), n_folds=3, shuffle=True)
    for k in range(5,MAX_K,2):
        clf = neighbors.KNeighborsClassifier(n_neighbors=k)
        summ = 0
        for train, test in kf:
            clf.fit(matrix[train], labels[train])
            predicted = clf.predict(matrix[test])
            summ += accuracy(predicted, labels)
        if summ/len(kf) > acc_best:
            acc_best = summ/len(kf)
            k_best = k
    print('method: KNN, data: %s testing, ACC: %.2f for k=%i' %
        (name, round(acc_best, 2), k_best))
    print()


# --------- help functions --------------

def accuracy(labels1, labels2):
    s = 0
    for i in range(len(labels1)):
        s += labels1[i] == labels2[i]
    return s/len(labels1)



#MAIN
predict()
