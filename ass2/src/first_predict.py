import numpy as np
import scipy.stats.mstats as mstats
from sklearn import svm as sk_svm, neural_network
from sklearn.cross_validation import KFold
from math import sqrt, inf

DATA_DIR = '../data/'
U_MAT = 'first/unigrams.txt'
B_MAT = 'first/bigrams.txt'
LABELS = 'first/labels.txt'

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
    clf.fit(matrix, labels)
    # rmse on training data just for check
    predicted = clf.predict(matrix)
    print('method: SVM, data: %s training, ACC: %.2f' %
            (name, round(accuracy(predicted, labels), 2)))
    # cross validation
    kf = KFold(len(labels), n_folds=5, shuffle=True)
    summ = 0
    for train, test in kf:
        clf.fit(matrix[train], labels[train])
        predicted = clf.predict(matrix[test])
        summ += accuracy(predicted, labels[test])
    print('method: SVM, data: %s testing, ACC: %.2f' %
        (name, round(summ/len(kf), 2)))
    print()


# --------- help functions --------------

def accuracy(labels1, labels2):
    s = 0
    for i in range(len(labels1)):
        s += labels1[i] == labels2[i]
    return s/len(labels1)



#MAIN
predict()
